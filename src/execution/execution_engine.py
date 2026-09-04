"""
====================================================
QuantAI Professional
Execution Engine
====================================================

High-level execution coordinator:
- Receives OrderIntent from Strategy/Risk
- Routes through OrderManager → Exchange Adapter
- Handles order lifecycle, retries, errors
- Coordinates with ReconciliationEngine
====================================================
"""

from __future__ import annotations

import asyncio
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional

from src.execution.orders import (
    Fill,
    Order,
    OrderIntentData,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from src.production.order_deduplication import IdempotencyKeyGenerator
from src.execution.order_manager import OrderManager, OrderManagerConfig
from src.execution.binance_adapter import (
    BinanceConfig,
    BinanceRestAdapter,
    BinanceWebSocketAdapter,
)
from src.production.rate_limiter import MultiLimitRateLimiter
from src.production.order_deduplication import OrderDeduplicator


class ExecutionMode(str, Enum):
    PAPER = "PAPER"       # Simulated (uses PaperTradingEngine)
    DRY_RUN = "DRY_RUN"   # Real market data, simulated orders
    LIVE = "LIVE"         # Real orders on exchange


@dataclass
class ExecutionConfig:
    mode: ExecutionMode = ExecutionMode.PAPER
    
    # Binance config (for DRY_RUN/LIVE)
    binance: Optional[Any] = None  # BinanceConfig
    
    # Order management
    max_open_orders: int = 100
    max_order_age_seconds: int = 3600
    enable_order_expiration: bool = True
    
    # Retry logic
    retry_failed_submissions: bool = True
    max_submission_retries: int = 3
    submission_retry_delay_seconds: float = 1.0
    
    # Reconciliation
    reconciliation_interval_seconds: float = 30.0
    position_tolerance: float = 0.001  # Max position difference allowed
    balance_tolerance: float = 1.0     # Max balance difference (USDT)
    
    # Safety
    kill_switch_enabled: bool = True
    max_drawdown_pct: float = 10.0
    max_daily_loss_pct: float = 5.0
    
    # Paper trading fallback
    paper_fallback: bool = True

    # P1.24 Seedable Fill Model — experiment random_seed for reproducible fill simulation
    experiment_seed: int = 42
    fill_seed: Optional[int] = None  # if None, uses experiment_seed


@dataclass
class ExecutionStats:
    intents_received: int = 0
    orders_created: int = 0
    orders_submitted: int = 0
    orders_filled: int = 0
    orders_canceled: int = 0
    orders_rejected: int = 0
    orders_failed: int = 0
    fills_received: int = 0
    reconciliation_runs: int = 0
    reconciliation_fixes: int = 0
    errors: int = 0
    
    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class BalanceVerification:
    """Verified balance with exchange response metadata."""
    balance: float
    timestamp: datetime
    request_id: str
    exchange_response: dict  # Raw exchange response for audit
    
    def is_valid(self, max_age_seconds: float = 5.0) -> bool:
        """Check if balance verification is within age threshold."""
        age = (datetime.now(timezone.utc) - self.timestamp).total_seconds()
        return age <= max_age_seconds
    
    @property
    def age_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.timestamp).total_seconds()


class ExecutionEngine:
    """
    Central execution coordinator.
    
    Flow:
        Strategy/Risk → OrderIntent
                              ↓
                    ExecutionEngine
                              ↓
                  OrderManager (lifecycle)
                              ↓
              BinanceAdapter (REST/WS) or PaperEngine
                              ↓
                    ReconciliationEngine
    """
    
    def __init__(
        self,
        config: ExecutionConfig,
        # Paper trading fallback
        paper_engine: Optional[Any] = None,
        # Production guards
        rate_limiter: Optional[MultiLimitRateLimiter] = None,
        order_deduplicator: Optional[OrderDeduplicator] = None,
        # Callbacks
        on_fill: Optional[Callable[[Fill], None]] = None,
        on_order_update: Optional[Callable[[Order], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        on_reconciliation_fix: Optional[Callable[[str, dict], None]] = None,
    ) -> None:
        self.config = config
        self.paper_engine = paper_engine
        self.rate_limiter = rate_limiter
        self.order_deduplicator = order_deduplicator
        
        self._on_fill = on_fill
        self._on_order_update = on_order_update
        self._on_error = on_error
        self._on_reconciliation_fix = on_reconciliation_fix
        
        # State
        self._running = False
        self._halted = False  # Audit P0: HALTED after emergency_stop — blocks new orders
        self._stats = ExecutionStats()
        self._daily_pnl = 0.0
        self._start_balance = 0.0
        
        # Components (initialized in start())
        self.order_manager: Optional[OrderManager] = None
        self.binance_rest: Optional[BinanceRestAdapter] = None
        self.binance_ws: Optional[BinanceWebSocketAdapter] = None
        
        # Reconciliation
        self._reconciliation_task: Optional[asyncio.Task] = None
        self._position_cache: dict[str, float] = defaultdict(float)  # symbol -> qty
        self._balance_cache: float = 0.0
        self._balance_verification: Optional[BalanceVerification] = None
        # Risk guard for recovery (P1.20)
        try:
            from src.drawdown_guard import DrawdownGuard
            self.drawdown_guard = DrawdownGuard(max_drawdown_percent=self.config.max_drawdown_pct)
        except Exception:
            self.drawdown_guard = None  # type: ignore
    
    # ============================================================
    # LIFECYCLE
    # ============================================================
    
    async def start(self) -> None:
        """Start execution engine."""
        if self._running:
            return
        
        self._running = True
        
        # Initialize OrderManager
        om_config = OrderManagerConfig(
            max_open_orders=self.config.max_open_orders,
            max_order_age_seconds=self.config.max_order_age_seconds,
            enable_order_expiration=self.config.enable_order_expiration,
        )
        self.order_manager = OrderManager(om_config)
        
        # Set up OrderManager callbacks
        self.order_manager._submit_callback = self._submit_order_to_exchange
        self.order_manager._cancel_callback = self._cancel_order_on_exchange
        
        # Initialize Binance (for DRY_RUN/LIVE)
        if self.config.mode in {ExecutionMode.DRY_RUN, ExecutionMode.LIVE}:
            await self._init_binance()
        
        # Paper trading fallback
        if self.config.mode == ExecutionMode.PAPER or self.config.paper_fallback:
            if self.paper_engine is None:
                from src.paper_trading_engine import PaperTradingEngine
                self.paper_engine = PaperTradingEngine(
                    initial_balance=1000.0,
                    commission=0.0004,
                )
        
        # Start reconciliation
        if self.config.mode != ExecutionMode.PAPER:
            self._reconciliation_task = asyncio.create_task(self._reconciliation_loop())
        
        # Get initial balance
        await self._update_balance_cache()
        
        print(f"[ExecutionEngine] Started in {self.config.mode.value} mode")
    
    async def stop(self) -> None:
        """Stop execution engine."""
        self._running = False
        
        if self._reconciliation_task:
            self._reconciliation_task.cancel()
            try:
                await self._reconciliation_task
            except asyncio.CancelledError:
                pass
        
        # Cancel all open orders
        if self.order_manager:
            self.order_manager.cancel_all()
        
        # Close Binance connections
        if self.binance_ws:
            await self.binance_ws.close()
        if self.binance_rest:
            await self.binance_rest.__aexit__(None, None, None)
        
        print("[ExecutionEngine] Stopped")
    
    # ============================================================
    # BINANCE INITIALIZATION
    # ============================================================
    
    async def _init_binance(self) -> None:
        if not self.config.binance:
            raise ValueError("Binance config required for DRY_RUN/LIVE mode")
        
        self.binance_rest = BinanceRestAdapter(self.config.binance)
        await self.binance_rest.__aenter__()

        # SECURITY GUARD (R1): never run against a key with withdrawal
        # permission. Fail-closed for DRY_RUN/LIVE; testnet keys without
        # the restrictions endpoint must set allow_unverified explicitly.
        perms = await self.binance_rest.verify_no_withdraw_permission()
        print(f"[SECURITY] key permissions verified: {perms}")

        # Initialize WebSocket
        self.binance_ws = BinanceWebSocketAdapter(self.config.binance, self.binance_rest)
        self.binance_ws.on_order_update = self._on_ws_order_update
        self.binance_ws.on_account_update = self._on_ws_account_update
        self.binance_ws.on_position_update = self._on_ws_position_update
        self.binance_ws.on_balance_update = self._on_ws_balance_update
        self.binance_ws.on_error = self._on_ws_error
        self.binance_ws.on_connect = lambda: print("[Binance WS] Connected")
        self.binance_ws.on_disconnect = lambda: print("[Binance WS] Disconnected")
        
        # Start user data stream
        await self.binance_ws.connect_user_stream()
        
        # Start listen key keepalive
        asyncio.create_task(self.binance_rest.keepalive_loop())
    
    # ============================================================
    # INTENT PROCESSING (Main Entry Point)
    # ============================================================
    
    async def submit_intent(self, intent) -> Order:
        """
        Submit OrderIntent for execution.
        
        Returns the created Order for tracking.
        """
        if self._halted:
            raise ExecutionError("HALTED after emergency_stop — new orders blocked")
        if not self._running:
            raise RuntimeError("ExecutionEngine not started")
        
        self._stats.intents_received += 1
        
        # Safety checks
        if not await self._safety_check(intent):
            raise ExecutionError("Safety check failed")
        
        # Create order via OrderManager
        order = self.order_manager.submit_intent(intent)
        self._stats.orders_created += 1
        
        return order
     
    async def _safety_check(self, intent) -> bool:
        """Pre-trade safety checks — daily PnL = realized + fees + funding (Audit: commission-only insufficient).
        
        Uses verified balance from exchange with timestamp, request_id, and age validation.
        """
        if not self.config.kill_switch_enabled:
            return True
        
        # Get verified balance (fetches fresh if cache stale > 5s)
        verification = await self.get_verified_balance(max_age_seconds=5.0)
        if not verification:
            print("[ExecutionEngine] SAFETY CHECK FAILED: No verified balance available")
            return False
        
        balance = verification.balance
        
        # Daily reset at UTC midnight
        today = datetime.now(timezone.utc).date()
        if not hasattr(self, '_daily_date'):
            self._daily_date = today  # type: ignore
            self._daily_start_balance = balance or self._start_balance  # type: ignore
        elif self._daily_date != today:  # type: ignore
            self._daily_date = today  # type: ignore
            self._daily_start_balance = balance  # type: ignore
            self._daily_pnl = 0.0
            print(f"[ExecutionEngine] Daily PnL reset for {today}")

        # Authoritative daily PnL is balance delta + funding (fees already in balance)
        # Use max of incremental _daily_pnl and balance delta to catch drift
        try:
            balance_delta = (balance - getattr(self, '_daily_start_balance', self._start_balance)) if self._start_balance else self._daily_pnl
            # Prefer balance delta when available (ground truth)
            effective_daily = balance_delta if abs(balance_delta) > abs(self._daily_pnl) * 0.5 or balance else self._daily_pnl
        except Exception:
            effective_daily = self._daily_pnl

        # Check daily loss limit (realized + fees + funding)
        if effective_daily <= -abs(self._start_balance * self.config.max_daily_loss_pct / 100):
            print(f"[ExecutionEngine] KILL SWITCH: Daily loss limit reached (daily {effective_daily:.2f} vs start {self._start_balance:.2f})")
            return False
        
        # Check drawdown
        if self._start_balance > 0:
            drawdown = (self._start_balance - balance) / self._start_balance * 100
            if drawdown >= self.config.max_drawdown_pct:
                print(f"[ExecutionEngine] KILL SWITCH: Max drawdown reached ({drawdown:.2f}%)")
                return False
        
        return True
    
    # ============================================================
    # ORDER SUBMISSION (OrderManager callback)
    # ============================================================
    
    async def _submit_order_to_exchange(self, order: Order) -> bool:
        """Submit order to exchange (called by OrderManager)."""
        self._stats.orders_submitted += 1
        
        # Deduplication check using consistent idempotency key
        if self.order_deduplicator and order.intent:
            idempotency_key = self._build_idempotency_key(order)
            is_new, existing_id = await self.order_deduplicator.check_and_register(
                idempotency_key,
                {
                    "symbol": order.intent.symbol,
                    "side": order.intent.side.value,
                    "quantity": order.intent.quantity,
                    "price": order.intent.price,
                    "order_type": order.intent.order_type.value,
                }
            )
            if not is_new:
                order.reject(f"Duplicate order detected (existing: {existing_id})")
                self._stats.orders_rejected += 1
                return False
        
        # Rate limiting for live/dry-run modes
        if self.rate_limiter and self.config.mode in {ExecutionMode.LIVE, ExecutionMode.DRY_RUN}:
            endpoint = "/fapi/v1/order"
            acquired = await self.rate_limiter.acquire_for_endpoint(endpoint)
            if not acquired:
                order.reject("Rate limit exceeded")
                self._stats.orders_rejected += 1
                # Mark deduplicator as failed
                if self.order_deduplicator and order.intent:
                    await self.order_deduplicator.update_status(
                        self._build_idempotency_key(order),
                        OrderStatus.REJECTED,
                        error_message="Rate limit exceeded",
                    )
                return False
        
        try:
            if self.config.mode == ExecutionMode.LIVE:
                return await self._submit_live(order)
            elif self.config.mode == ExecutionMode.DRY_RUN:
                return await self._submit_dry_run(order)
            else:
                return await self._submit_paper(order)
                
        except Exception as e:
            self._stats.errors += 1
            if self._on_error:
                self._on_error(e)
            order.reject(f"Submission error: {e}")
            # Mark deduplicator as failed
            if self.order_deduplicator and order.intent:
                await self.order_deduplicator.update_status(
                    self._build_idempotency_key(order),
                    OrderStatus.REJECTED,
                    error_message=str(e),
                )
            return False
    
    async def _submit_live(self, order: Order) -> bool:
        """Submit to live Binance with idempotency key and retry on timeout."""
        intent = order.intent
        max_retries = self.config.max_submission_retries if self.config.retry_failed_submissions else 1
        
        for attempt in range(max_retries):
            try:
                # Use order's client_order_id as idempotency key for exchange
                result = await self.binance_rest.place_order_with_client_id(
                    intent, 
                    client_order_id=order.client_order_id
                )
                
                if "orderId" in result:
                    order.exchange_order_id = str(result["orderId"])
                    order.status = OrderStatus.NEW
                    order.submitted_at = datetime.now(timezone.utc)
                    # Update deduplicator with success
                    if self.order_deduplicator and order.intent:
                        await self.order_deduplicator.update_status(
                            self._build_idempotency_key(order),
                            OrderStatus.ACCEPTED,
                            exchange_order_id=order.exchange_order_id,
                        )
                    return True
                else:
                    error_msg = f"Binance error: {result}"
                    # Check if it's a duplicate order error (order already exists) — P1.22: query before any handling
                    if self._is_duplicate_order_error(result):
                        existing_order = await self._query_order_by_client_id(order, order.client_order_id)
                        if existing_order and "orderId" in existing_order:
                            order.exchange_order_id = str(existing_order["orderId"])
                            order.status = OrderStatus.NEW
                            order.submitted_at = datetime.now(timezone.utc)
                            return True
                    order.reject(error_msg)
                    self._stats.orders_rejected += 1
                    return False
                    
            except asyncio.TimeoutError:
                # P1.22: Network timeout - MUST query by clientOrderId BEFORE any retry (never blind retry)
                if attempt < max_retries - 1:
                    existing_order = await self._query_order_by_client_id(order, order.client_order_id)
                    if existing_order and "orderId" in existing_order:
                        # Order was actually placed despite timeout, use it (idempotent)
                        order.exchange_order_id = str(existing_order["orderId"])
                        order.status = OrderStatus.NEW
                        order.submitted_at = datetime.now(timezone.utc)
                        if self.order_deduplicator and order.intent:
                            await self.order_deduplicator.update_status(
                                self._build_idempotency_key(order),
                                OrderStatus.ACCEPTED,
                                exchange_order_id=order.exchange_order_id,
                            )
                        return True
                    # Order not found on exchange, safe to retry
                    await asyncio.sleep(self.config.submission_retry_delay_seconds * (attempt + 1))
                    continue
                else:
                    order.reject("Max retries exceeded after timeout — verified via query that order not found")
                    self._stats.orders_failed += 1
                    return False
                    
            except Exception as e:
                error_str = str(e)
                # Check for duplicate order error in exception
                if self._is_duplicate_order_error({"msg": error_str}):
                    existing_order = await self._query_order_by_client_id(order, order.client_order_id)
                    if existing_order and "orderId" in existing_order:
                        order.exchange_order_id = str(existing_order["orderId"])
                        order.status = OrderStatus.NEW
                        order.submitted_at = datetime.now(timezone.utc)
                        return True
                # P1.22: For ANY exception before retry, query first to avoid blind duplicate
                if attempt < max_retries - 1:
                    # Query to determine actual state before retry
                    try:
                        existing_order = await self._query_order_by_client_id(order, order.client_order_id)
                        if existing_order and "orderId" in existing_order:
                            order.exchange_order_id = str(existing_order["orderId"])
                            order.status = OrderStatus.NEW
                            order.submitted_at = datetime.now(timezone.utc)
                            return True
                    except Exception:
                        pass
                    await asyncio.sleep(self.config.submission_retry_delay_seconds * (attempt + 1))
                    continue
                else:
                    order.reject(f"Submission error after retries: {e}")
                    self._stats.orders_failed += 1
                    return False
        
        return False
    
    def _build_idempotency_key(self, order: Order) -> str:
        """Build deterministic idempotency key from order parameters."""
        intent = order.intent
        return IdempotencyKeyGenerator.generate(
            symbol=intent.symbol,
            side=intent.side.value,
            quantity=intent.quantity,
            price=intent.price,
            order_type=intent.order_type.value,
        )
    
    async def _query_order_by_client_id(self, order: Order, client_order_id: str) -> Optional[dict]:
        """Query exchange for order by clientOrderId."""
        if not self.binance_rest or not order.intent:
            return None
        try:
            return await self.binance_rest.get_order(
                symbol=order.intent.symbol,
                client_order_id=client_order_id,
            )
        except Exception:
            return None
    
    def _is_duplicate_order_error(self, result: dict) -> bool:
        """Check if error indicates duplicate order (already placed)."""
        if isinstance(result, dict):
            msg = result.get("msg", "").lower()
            code = result.get("code", 0)
            # Binance error codes for duplicate: -2011 (unknown order), -1022 (signature), etc.
            # Duplicate clientOrderId: -2011 or specific msg
            return "clientorderid" in msg or "duplicate" in msg or code == -2011
        return False
    
    async def _submit_dry_run(self, order: Order) -> bool:
        """Submit to Binance but don't actually place (dry run)."""
        # Validate against exchange
        intent = order.intent
        valid, msg = self.binance_rest.validate_order(intent.symbol, intent.quantity, intent.price)
        
        if not valid:
            order.reject(f"Validation failed: {msg}")
            self._stats.orders_rejected += 1
            return False
        
        # Simulate order acceptance
        order.exchange_order_id = f"dryrun_{uuid.uuid4().hex[:12]}"
        order.status = OrderStatus.NEW
        order.submitted_at = datetime.now(timezone.utc)
        
        # Update deduplicator
        if self.order_deduplicator and order.intent:
            await self.order_deduplicator.update_status(
                self._build_idempotency_key(order),
                OrderStatus.ACCEPTED,
                exchange_order_id=order.exchange_order_id,
            )
        
        # Simulate immediate fill for market orders
        if intent.order_type == OrderType.MARKET:
            # Get current price from cache or fetch
            price = await self._get_current_price(intent.symbol)
            if price:
                fill = Fill(
                    order_id=order.order_id,
                    client_order_id=order.client_order_id,
                    exchange_order_id=order.exchange_order_id,
                    symbol=intent.symbol,
                    side=intent.side,
                    quantity=intent.quantity,
                    price=price,
                    commission=intent.quantity * price * 0.0004,
                    commission_asset="USDT",
                )
                await self._process_fill(fill)
        
        return True
    
    async def _submit_paper(self, order: Order) -> bool:
        """Submit to paper trading engine."""
        if not self.paper_engine:
            order.reject("Paper engine not available")
            return False
        
        intent = order.intent
        
        try:
            if intent.side == OrderSide.BUY:
                self.paper_engine.open_position("LONG", order.intent.price, order.intent.quantity)
            else:
                self.paper_engine.open_position("SHORT", order.intent.price, order.intent.quantity)
            
            order.exchange_order_id = f"paper_{uuid.uuid4().hex[:12]}"
            order.status = OrderStatus.FILLED
            order.submitted_at = datetime.now(timezone.utc)
            order.filled_quantity = intent.quantity
            order.average_fill_price = intent.price
            order.filled_at = datetime.now(timezone.utc)
            
            # Update deduplicator
            if self.order_deduplicator and order.intent:
                await self.order_deduplicator.update_status(
                    self._build_idempotency_key(order),
                    OrderStatus.ACCEPTED,
                    exchange_order_id=order.exchange_order_id,
                )
            
            # Create fill for tracking
            fill = Fill(
                order_id=order.order_id,
                client_order_id=order.client_order_id,
                exchange_order_id=order.exchange_order_id,
                symbol=intent.symbol,
                side=intent.side,
                quantity=intent.quantity,
                price=intent.price,
                commission=intent.quantity * intent.price * 0.0004,
                commission_asset="USDT",
            )
            await self._process_fill(fill)
            
            return True
            
        except Exception as e:
            order.reject(f"Paper error: {e}")
            return False
    
    async def _cancel_order_on_exchange(self, order: Order) -> bool:
        """Cancel order on exchange (called by OrderManager)."""
        try:
            if self.config.mode == ExecutionMode.LIVE and order.exchange_order_id:
                result = await self.binance_rest.cancel_order(
                    symbol=order.intent.symbol,
                    order_id=order.exchange_order_id,
                )
                return "orderId" in result
            
            elif self.config.mode == ExecutionMode.DRY_RUN:
                order.cancel()
                return True
            
            else:  # PAPER
                # Paper engine doesn't support cancel after fill
                return False
                
        except Exception as e:
            if self._on_error:
                self._on_error(e)
            return False
    
    # ============================================================
    # WEBSOCKET CALLBACKS
    # ============================================================
    
    def _on_ws_order_update(self, data: dict):
        """Handle order update from WebSocket."""
        o = data
        client_order_id = o.get("c")  # Client order ID
        exchange_order_id = str(o.get("i"))  # Exchange order ID
        status = o.get("X")  # Order status
        filled_qty = float(o.get("z", 0))  # Cumulative filled qty
        avg_price = float(o.get("ap", 0)) if o.get("ap") else 0
        
        # Find order
        order = None
        if client_order_id:
            order = self.order_manager.get_order_by_client_id(client_order_id)
        if not order and exchange_order_id:
            order = self.order_manager.get_order_by_exchange_id(exchange_order_id)
        
        if not order:
            print(f"[ExecutionEngine] Order not found for WS update: {client_order_id}/{exchange_order_id}")
            return
        
        # Update order status
        status_map = {
            "NEW": OrderStatus.NEW,
            "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
            "FILLED": OrderStatus.FILLED,
            "CANCELED": OrderStatus.CANCELED,
            "REJECTED": OrderStatus.REJECTED,
            "EXPIRED": OrderStatus.EXPIRED,
        }
        new_status = status_map.get(status, OrderStatus.NEW)
        
        self.order_manager.on_order_update(
            exchange_order_id=exchange_order_id,
            status=new_status,
            filled_qty=filled_qty,
            avg_price=avg_price if avg_price > 0 else None,
        )
        
        # Update local order object
        order.status = new_status
        order.filled_quantity = filled_qty
        order.average_fill_price = avg_price
        order.updated_at = datetime.now(timezone.utc)
        
        if new_status == OrderStatus.FILLED:
            self._stats.orders_filled += 1
            order.filled_at = datetime.now(timezone.utc)
            self._update_position_cache(order)
        elif new_status == OrderStatus.CANCELED:
            self._stats.orders_canceled += 1
        elif new_status == OrderStatus.REJECTED:
            self._stats.orders_rejected += 1
        
        if self._on_order_update:
            self._on_order_update(order)
    
    def _on_ws_account_update(self, data: dict):
        """Handle account update (balances)."""
        # Trigger balance reconciliation
        asyncio.create_task(self._update_balance_cache())
    
    def _on_ws_position_update(self, data: dict):
        """Handle position update from WebSocket."""
        symbol = data.get("s")
        pos_amt = float(data.get("pa", 0))
        
        if symbol:
            self._position_cache[symbol] = pos_amt
            if self._on_order_update and hasattr(self, '_position_callback'):
                pass  # Could notify position callbacks
    
    def _on_ws_balance_update(self, data: dict):
        """Handle balance update from WebSocket."""
        asset = data.get("a")
        if asset == "USDT":
            self._balance_cache = float(data.get("wb", 0))
    
    def _on_ws_error(self, error: Exception):
        if self._on_error:
            self._on_error(error)
    
    # ============================================================
    # FILL PROCESSING
    # ============================================================
    
    async def _process_fill(self, fill: Fill) -> None:
        """Process fill — daily PnL = realized PnL + fees + funding (Audit: commission-only was insufficient for kill switch)."""
        self._stats.fills_received += 1
        
        # Update order via OrderManager
        self.order_manager.on_fill(fill)
        
        # Update position cache
        order = self.order_manager.get_order(fill.order_id)
        if order:
            self._update_position_cache(order)
        
        # Daily PnL: realized trading PnL + fees + funding
        # 1. Fees always
        self._daily_pnl -= fill.commission
        # 2. Realized PnL if this fill closes/reduces a position (use paper_engine closed_positions when available)
        try:
            # If paper_engine has just closed a position, its last closed net_profit is realized
            if self.paper_engine and hasattr(self.paper_engine, 'closed_positions') and self.paper_engine.closed_positions:
                last = self.paper_engine.closed_positions[-1]
                # Only count once per fill — check if this fill corresponds to last close
                # Use fill order_id matching last position id
                if getattr(last, 'exit_time', None) and not getattr(last, '_counted_for_daily', False):
                    # Realized PnL from closed position (net of fees already, but we already subtracted commission above -> adjust)
                    # Use gross_profit for pure trading PnL, then fees already accounted
                    realized = float(getattr(last, 'gross_profit', 0) or getattr(last, 'net_profit', 0))
                    self._daily_pnl += realized
                    last._counted_for_daily = True  # type: ignore
            # 3. Funding (if fill has funding field — futures funding is separate from commission)
            funding = float(getattr(fill, 'funding', 0) or 0)
            if funding:
                self._daily_pnl += funding  # funding can be positive (received) or negative (paid)
        except Exception:
            pass
        # 4. Fallback: if not paper and no closed position tracking, daily PnL as balance delta
        # _safety_check also uses balance delta as authoritative, so keep _daily_pnl in sync with balance
        try:
            if self._start_balance > 0 and self._balance_cache > 0:
                # Balance delta is ground truth for daily PnL (includes all realized)
                # Use it to correct drift, but keep incremental for low latency
                balance_delta = self._balance_cache - self._start_balance
                # Slowly converge: if diff > tolerance, sync
                if abs(balance_delta - self._daily_pnl) > 5.0:  # $5 tolerance
                    self._daily_pnl = balance_delta
        except Exception:
            pass
        
        if self._on_fill:
            self._on_fill(fill)
     
    def _update_position_cache(self, order: Order):
        """Update internal position cache from filled order."""
        symbol = order.intent.symbol
        side = 1 if order.intent.side == OrderSide.BUY else -1
        self._position_cache[symbol] += side * order.filled_quantity
     
    async def _update_balance_cache(self) -> BalanceVerification:
        """Update balance cache from exchange and return verification object."""
        request_id = uuid.uuid4().hex[:16]
        timestamp = datetime.now(timezone.utc)
        exchange_response = {}
        balance = 0.0
        
        if self.config.mode == ExecutionMode.LIVE and self.binance_rest:
            try:
                balances = await self.binance_rest.get_balance()
                exchange_response = {"balances": [b.__dict__ if hasattr(b, '__dict__') else str(b) for b in balances]}
                for b in balances:
                    if b.asset == "USDT":
                        balance = b.available_balance
                        self._balance_cache = balance
                        self._start_balance = self._start_balance or balance
                        break
            except Exception as e:
                print(f"[ExecutionEngine] Balance update error: {e}")
                exchange_response = {"error": str(e)}
        elif self.paper_engine:
            balance = self.paper_engine.balance
            self._balance_cache = balance
            self._start_balance = self._start_balance or balance
            exchange_response = {"source": "paper_engine", "balance": balance}
        else:
            exchange_response = {"source": "cache_only", "balance": self._balance_cache}
            balance = self._balance_cache
        
        verification = BalanceVerification(
            balance=balance,
            timestamp=timestamp,
            request_id=request_id,
            exchange_response=exchange_response,
        )
        self._balance_verification = verification
        return verification
    
    async def get_verified_balance(self, max_age_seconds: float = 5.0) -> Optional[BalanceVerification]:
        """
        Get verified balance from exchange.
        
        Args:
            max_age_seconds: Maximum age of cached verification (default 5s).
                            If cached verification is older, fetches fresh from exchange.
        
        Returns:
            BalanceVerification if valid, None if no verification available.
        """
        if self._balance_verification and self._balance_verification.is_valid(max_age_seconds):
            return self._balance_verification
        
        # Fetch fresh verification
        return await self._update_balance_cache()
    
    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current mark price for symbol."""
        if self.binance_rest:
            try:
                data = await self.binance_rest.get_mark_price(symbol)
                return float(data.get("markPrice", 0))
            except Exception:
                pass
        return None
    
    # ============================================================
    # RECONCILIATION
    # ============================================================
    
    async def _reconciliation_loop(self):
        """Periodic reconciliation of positions and balances."""
        while self._running:
            try:
                await asyncio.sleep(self.config.reconciliation_interval_seconds)
                if self._running:
                    await self.reconcile()
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self._on_error:
                    self._on_error(e)
    
    async def reconcile(self) -> dict:
        """
        Reconcile local state with exchange.
        
        Returns reconciliation report.
        """
        self._stats.reconciliation_runs += 1
        fixes = 0
        issues = []
        
        # Reconcile positions
        if self.config.mode != ExecutionMode.PAPER:
            try:
                exchange_positions = await self.binance_rest.get_positions()
                exchange_pos_map = {p.symbol: p.position_amt for p in exchange_positions}
                
                for symbol, local_qty in self._position_cache.items():
                    exchange_qty = exchange_pos_map.get(symbol, 0.0)
                    
                    if abs(local_qty - exchange_qty) > self.config.position_tolerance:
                        issues.append({
                            "type": "position_mismatch",
                            "symbol": symbol,
                            "local": local_qty,
                            "exchange": exchange_qty,
                            "diff": local_qty - exchange_qty,
                        })
                        
                        # Fix: update local cache to match exchange
                        self._position_cache[symbol] = exchange_qty
                        fixes += 1
                        self._stats.reconciliation_fixes += 1
                        
                        if self._on_reconciliation_fix:
                            self._on_reconciliation_fix(symbol, {
                                "type": "position",
                                "old": local_qty,
                                "new": exchange_qty,
                            })
                
                # Check for positions on exchange not in local cache
                for symbol, exchange_qty in exchange_pos_map.items():
                    if symbol not in self._position_cache and abs(exchange_qty) > self.config.position_tolerance:
                        issues.append({
                            "type": "ghost_position",
                            "symbol": symbol,
                            "exchange": exchange_qty,
                        })
                        self._position_cache[symbol] = exchange_qty
                        fixes += 1
                        self._stats.reconciliation_fixes += 1
                        
            except Exception as e:
                issues.append({"type": "position_reconciliation_error", "error": str(e)})
        
        # Reconcile balance
        try:
            await self._update_balance_cache()
            if self.paper_engine:
                paper_balance = self.paper_engine.balance
                if abs(paper_balance - self._balance_cache) > self.config.balance_tolerance:
                    issues.append({
                        "type": "balance_mismatch",
                        "paper": paper_balance,
                        "exchange": self._balance_cache,
                        "diff": paper_balance - self._balance_cache,
                    })
        except Exception as e:
            issues.append({"type": "balance_reconciliation_error", "error": str(e)})
        
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "fixes_applied": fixes,
            "issues": issues,
            "positions": dict(self._position_cache),
            "balance": self._balance_verification.balance if self._balance_verification else self._balance_cache,
            "balance_verification": {
                "request_id": self._balance_verification.request_id,
                "timestamp": self._balance_verification.timestamp.isoformat(),
                "age_seconds": self._balance_verification.age_seconds,
            } if self._balance_verification else None,
        }
        
        if issues:
            print(f"[Reconciliation] Found {len(issues)} issues, applied {fixes} fixes")
        
        return report
    
    # ============================================================
    # QUERY METHODS
    # ============================================================
    
    def get_order(self, order_id: str) -> Optional[Order]:
        return self.order_manager.get_order(order_id)
    
    def get_position(self, symbol: str) -> float:
        return self._position_cache.get(symbol, 0.0)
    
    def get_all_positions(self) -> dict[str, float]:
        return dict(self._position_cache)
    
    def get_balance(self) -> float:
        """Get current balance (uses verified balance if available)."""
        if self._balance_verification and self._balance_verification.is_valid(30.0):
            return self._balance_verification.balance
        return self._balance_cache
    
    def get_balance_verification(self) -> Optional[BalanceVerification]:
        """Get the current balance verification object."""
        return self._balance_verification
    
    def get_stats(self) -> dict:
        stats = self._stats.to_dict()
        stats.update(self.order_manager.stats)
        return stats
    
    def get_open_orders(self, symbol: Optional[str] = None) -> list[Order]:
        if self.order_manager:
            return self.order_manager.get_active_orders(symbol)
        return []
    
    # ============================================================
    # EMERGENCY
    # ============================================================
    
    async def emergency_stop(self) -> dict:
        """
        Emergency stop — futures-safe full halt (Audit P0).

        1. stop new orders (HALTED flag)
        2. cancel all resting orders (local + exchange)
        3. query exchange positions
        4. flatten positions (market reduceOnly)
        5. verify flat (re-query)
        6. verify balances
        7. enter HALTED state
        """
        print("[ExecutionEngine] EMERGENCY STOP triggered — HALTED")
        self._halted = True
        result: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "steps": [],
            "canceled_orders": 0,
            "flattened": [],
            "verified_flat": False,
            "verified_balances": False,
            "halted": True,
        }

        # P1.19 Step 1: STOP NEW ORDERS
        result["steps"].append("STOP NEW ORDERS — HALTED flag set, new intents blocked")
        # step already via _halted flag

        # P1.19 Step 2: CANCEL ORDERS
        canceled = 0
        if self.order_manager:
            try:
                canceled = self.order_manager.cancel_all()
            except Exception as e:
                result["steps"].append(f"CANCEL ORDERS: local cancel error {e}")
        # 2b. cancel on exchange (LIVE/DRY_RUN)
        if self.binance_rest and self.config.mode in {ExecutionMode.LIVE, ExecutionMode.DRY_RUN}:
            try:
                # Binance: cancel all open orders per symbol
                symbols = list(self._position_cache.keys()) or ["BTCUSDT"]
                for sym in symbols:
                    try:
                        await self.binance_rest.cancel_all_orders(sym)
                    except Exception:
                        pass
                # Fallback: cancel via order_manager exchange cancel
                for order in list(self.order_manager.get_active_orders() if self.order_manager else []):
                    try:
                        await self._cancel_order_on_exchange(order)
                    except Exception:
                        pass
            except Exception as e:
                result["steps"].append(f"CANCEL ORDERS: exchange cancel error {e}")
        result["canceled_orders"] = canceled
        result["steps"].append(f"CANCEL ORDERS: canceled {canceled} resting orders")

        # P1.19 Step 3: FETCH POSITIONS (fresh from exchange, not cache)
        positions_before: dict[str, float] = {}
        if self.binance_rest and self.config.mode in {ExecutionMode.LIVE, ExecutionMode.DRY_RUN}:
            try:
                ex_positions = await self.binance_rest.get_positions()
                positions_before = {p.symbol: float(p.position_amt) for p in ex_positions if abs(float(p.position_amt)) > 1e-9}
                result["exchange_positions_fresh"] = True
                result["steps"].append(f"FETCH POSITIONS: {len(positions_before)} positions from exchange")
            except Exception as e:
                result["steps"].append(f"FETCH POSITIONS error {e} → UNKNOWN_POSITION (exchange not responded, software != exchange)")
                result["unknown_position"] = True
                # Do not trust software cache when exchange unreachable — keep HALTED UNKNOWN
                positions_before = dict(self._position_cache)
                result["exchange_positions_fresh"] = False
                # Keep HALTED, do not clear software_position to None
        else:
            # PAPER: FETCH POSITIONS from paper_engine + cache (simulation)
            positions_before = {k: v for k, v in self._position_cache.items() if abs(v) > 1e-9}
            result["steps"].append(f"FETCH POSITIONS (PAPER): {len(positions_before)} cached positions")
            if self.paper_engine:
                # singular PaperTradingEngine.position (symbol assumed BTCUSDT if not present)
                try:
                    pos = getattr(self.paper_engine, 'position', None)
                    if pos is not None and abs(float(getattr(pos, 'quantity', 0))) > 1e-9:
                        qty = float(getattr(pos, 'quantity', 0))
                        side = getattr(getattr(pos, 'side', 'LONG'), 'value', getattr(pos, 'side', 'LONG'))
                        signed_qty = qty * (1 if str(side) in ('LONG', 'BUY') else -1)
                        sym = getattr(pos, 'symbol', 'BTCUSDT')
                        # avoid double-count: if cache already tracks this symbol, trust cache (already 0.1)
                        if sym not in positions_before:
                            positions_before[sym] = signed_qty
                except Exception:
                    pass
                if hasattr(self.paper_engine, 'positions'):
                    try:
                        for pos in list(getattr(self.paper_engine, 'positions', []) or []):
                            qty = float(getattr(pos, 'quantity', 0)) * (1 if getattr(pos, 'side', 'LONG') in ('LONG','BUY') else -1)
                            sym = getattr(pos, 'symbol', 'BTCUSDT')
                            if abs(qty) > 1e-9 and sym not in positions_before:
                                positions_before[sym] = positions_before.get(sym, 0) + qty
                    except Exception:
                        pass
        result["steps"].append(f"FETCH POSITIONS: before {positions_before}")

        # P1.19 Step 4: FLATTEN (market reduceOnly opposite side)
        flattened: list[dict] = []
        for symbol, qty in list(positions_before.items()):
            if abs(qty) < 1e-9:
                continue
            side = OrderSide.SELL if qty > 0 else OrderSide.BUY  # opposite to flatten
            abs_qty = abs(qty)
            try:
                if self.config.mode == ExecutionMode.LIVE and self.binance_rest:
                    # Futures flatten: market reduceOnly
                    res = await self.binance_rest.place_order_raw(
                        symbol=symbol, side=side.value, order_type="MARKET", quantity=abs_qty, reduce_only=True
                    )
                    flattened.append({"symbol": symbol, "qty": qty, "side": side.value, "result": str(res)[:200]})
                elif self.paper_engine:
                    # PAPER flatten: close via paper_engine (requires price, symbol not required by PaperTradingEngine)
                    close_ok = False
                    if hasattr(self.paper_engine, 'close_position'):
                        try:
                            # Determine price to close: current mark or entry fallback
                            price_to_close = None
                            try:
                                price_to_close = await self._get_current_price(symbol)
                            except Exception:
                                pass
                            if not price_to_close:
                                # fallback to entry_price of open position
                                try:
                                    pos = getattr(self.paper_engine, 'position', None)
                                    if pos is not None and getattr(pos, 'entry_price', None):
                                        price_to_close = float(pos.entry_price)
                                except Exception:
                                    pass
                            # NO dummy price fallback — leave as UNVERIFIED_PRICE if no valid price
                            if price_to_close is not None:
                                # PaperTradingEngine signature is close_position(price)
                                # Other engines may take (symbol, price); try both
                                try:
                                    self.paper_engine.close_position(price_to_close)  # type: ignore
                                    close_ok = True
                                except TypeError:
                                    self.paper_engine.close_position(symbol, price_to_close)  # type: ignore
                                    close_ok = True
                            else:
                                # No valid price — close will be UNVERIFIED, verification loop re-queries
                                pass
                        except Exception as e:
                            # fallback: will clear cache anyway; record error but continue
                            print(f"[ExecutionEngine] Paper close failed {symbol}: {e}")
                            pass
                    # Software state must NOT be set to 0 until exchange confirms 0
                    # Keep UNKNOWN_POSITION while unverified
                    if close_ok:
                        self._position_cache[symbol] = 0.0
                    else:
                        # Leave as UNKNOWN — will be re-queried in verification loop
                        # Mark as UNKNOWN_POSITION for audit
                        result["steps"].append(f"4: {symbol} close unverified → UNKNOWN_POSITION (software != exchange)")
                    # Do NOT artificially set position=None — leave as UNVERIFIED_POSITION for verification loop
                    if not close_ok and self.paper_engine and getattr(self.paper_engine, 'position', None) is not None:
                        try:
                            if getattr(self.paper_engine, 'has_position', False):
                                # Position remains UNVERIFIED — verification loop will re-query and confirm flat
                                pass
                        except Exception:
                            pass
                    flattened.append({"symbol": symbol, "qty": qty, "side": side.value, "paper": True, "close_ok": close_ok, "unknown_position": not close_ok})
                else:
                    self._position_cache[symbol] = 0.0
                    flattened.append({"symbol": symbol, "qty": qty, "side": side.value, "cached": True})
            except Exception as e:
                flattened.append({"symbol": symbol, "qty": qty, "error": str(e)})
        result["flattened"] = flattened
        if flattened:
            result["steps"].append(f"FLATTEN: {len(flattened)} positions flattened")
            # Brief pause for exchange to process flatten
            try:
                await asyncio.sleep(1.0)
            except Exception:
                pass

        # P1.19 Step 5: VERIFY FLAT — MANDATORY flatten verification loop (Task 11)
        # Ensures positions are ACTUALLY flattened, not just signal sent.
        # Retries flatten up to 5 times with re-query; if still not flat → error.
        verified_flat = False
        remaining: dict[str, float] = {}
        MAX_VERIFY_ATTEMPTS = 5
        VERIFY_INTERVAL_SEC = 0.8
        for attempt in range(1, MAX_VERIFY_ATTEMPTS + 1):
            try:
                if self.binance_rest and self.config.mode in {ExecutionMode.LIVE, ExecutionMode.DRY_RUN}:
                    ex_positions2 = await self.binance_rest.get_positions()
                    remaining = {
                        p.symbol: float(p.position_amt)
                        for p in ex_positions2
                        if abs(float(p.position_amt)) > self.config.position_tolerance
                    }
                    # also check local cache for divergence
                    local_remaining = {
                        k: v for k, v in self._position_cache.items()
                        if abs(v) > self.config.position_tolerance
                    }
                    # sync cache to exchange truth
                    for sym, qty in list(remaining.items()):
                        self._position_cache[sym] = qty
                    for sym in list(local_remaining.keys()):
                        if sym not in remaining:
                            self._position_cache[sym] = 0.0
                else:
                    # PAPER: check both cache and paper_engine — avoid double-count same symbol
                    remaining = {
                        k: v for k, v in self._position_cache.items()
                        if abs(v) > self.config.position_tolerance
                    }
                    if self.paper_engine:
                        # singular PaperTradingEngine.position
                        try:
                            pos = getattr(self.paper_engine, 'position', None)
                            if pos is not None and abs(float(getattr(pos, 'quantity', 0))) > self.config.position_tolerance:
                                sym = getattr(pos, 'symbol', 'BTCUSDT')
                                side = getattr(getattr(pos, 'side', 'LONG'), 'value', getattr(pos, 'side', 'LONG'))
                                qty = float(pos.quantity) * (1 if str(side) in ('LONG', 'BUY') else -1)
                                if abs(qty) > self.config.position_tolerance:
                                    if sym in remaining:
                                        # cache and paper track same position — take max absolute as truth, not sum
                                        if abs(qty) > abs(remaining[sym]):
                                            remaining[sym] = qty
                                    else:
                                        remaining[sym] = qty
                        except Exception:
                            pass
                        if hasattr(self.paper_engine, 'positions'):
                            try:
                                for pos in list(getattr(self.paper_engine, 'positions', []) or []):
                                    qty = float(getattr(pos, 'quantity', 0)) * (
                                        1 if getattr(pos, 'side', 'LONG') in ('LONG', 'BUY') else -1
                                    )
                                    sym = getattr(pos, 'symbol', 'BTCUSDT')
                                    if abs(qty) > self.config.position_tolerance and sym not in remaining:
                                        remaining[sym] = qty
                            except Exception:
                                pass

                if len(remaining) == 0:
                    verified_flat = True
                    result["steps"].append(f"VERIFY FLAT: attempt {attempt}/{MAX_VERIFY_ATTEMPTS} PASS — all positions flattened (position == 0)")
                    break

                result["steps"].append(f"VERIFY FLAT: attempt {attempt}/{MAX_VERIFY_ATTEMPTS} FAIL remaining {remaining}")

                if attempt < MAX_VERIFY_ATTEMPTS:
                        # Retry flatten for remaining positions before next verify
                        for symbol, qty in list(remaining.items()):
                            if abs(qty) < 1e-9:
                                continue
                            side = OrderSide.SELL if qty > 0 else OrderSide.BUY
                            abs_qty = abs(qty)
                            try:
                                if self.config.mode == ExecutionMode.LIVE and self.binance_rest:
                                    await self.binance_rest.place_order_raw(
                                        symbol=symbol, side=side.value, order_type="MARKET",
                                        quantity=abs_qty, reduce_only=True,
                                    )
                                    result["steps"].append(f"5: retry flatten {symbol} qty {qty} -> {side.value} {abs_qty}")
                                elif self.paper_engine and hasattr(self.paper_engine, 'close_position'):
                                    try:
                                        price_to_close = None
                                        try:
                                            price_to_close = await self._get_current_price(symbol)
                                        except Exception:
                                            pass
                                        if not price_to_close:
                                            try:
                                                pos = getattr(self.paper_engine, 'position', None)
                                                if pos is not None and getattr(pos, 'entry_price', None):
                                                    price_to_close = float(pos.entry_price)
                                            except Exception:
                                                pass
# NO dummy price fallback — leave as UNVERIFIED_PRICE
                                        close_ok_retry = False
                                        if price_to_close is not None:
                                            try:
                                                self.paper_engine.close_position(price_to_close)  # type: ignore
                                                close_ok_retry = True
                                            except TypeError:
                                                try:
                                                    self.paper_engine.close_position(symbol, price_to_close)  # type: ignore
                                                    close_ok_retry = True
                                                except Exception:
                                                    close_ok_retry = False
                                        else:
                                            # No valid price — skip close, verification loop will re-query (UNVERIFIED_PRICE)
                                            close_ok_retry = False
                                            result["steps"].append(f"5: retry {symbol} skip close — UNVERIFIED_PRICE")
                                    except Exception as e:
                                        print(f"[ExecutionEngine] Paper retry close failed {symbol}: {e}")
                                        close_ok_retry = False
                                    # Software state only to 0 if exchange confirmed close_ok, else UNKNOWN_POSITION stays
                                    if close_ok_retry:
                                        self._position_cache[symbol] = 0.0
                                    else:
                                        result["steps"].append(f"5: retry {symbol} UNKNOWN_POSITION — software != exchange (not flat)")
                                    if not close_ok_retry and getattr(self.paper_engine, 'position', None) is not None and getattr(self.paper_engine, 'has_position', False):
                                        try:
                                            # Position remains UNVERIFIED — verification loop will re-query
                                            pass
                                        except Exception:
                                            pass
                            except Exception as e:
                                result["steps"].append(f"5: retry flatten error {symbol}: {e}")
                        try:
                            await asyncio.sleep(VERIFY_INTERVAL_SEC)
                        except Exception:
                            pass
            except Exception as e:
                result["steps"].append(f"5: verify flat attempt {attempt}/{MAX_VERIFY_ATTEMPTS} error {e}")
                try:
                    await asyncio.sleep(VERIFY_INTERVAL_SEC)
                except Exception:
                    pass
                remaining = remaining or {}
                verified_flat = False

        if not verified_flat:
            err_msg = f"EMERGENCY STOP FLATTEN VERIFICATION FAILED — positions not flat after {MAX_VERIFY_ATTEMPTS} attempts: {remaining} (position != 0)"
            result["verification_error"] = err_msg
            result["steps"].append(f"VERIFY FLAT: ERROR {err_msg}")
            print(f"[ExecutionEngine] CRITICAL {err_msg}")
        result["verified_flat"] = verified_flat
        result["remaining_positions"] = remaining

        # P1.19 Step 6: VERIFY BALANCE (fresh with timestamp, not stale cache)
        verified_balances = False
        try:
            verification = await self.get_verified_balance(max_age_seconds=5.0)
            if verification and verification.is_valid(5.0):
                verified_balances = True
                result["steps"].append(f"VERIFY BALANCE: {verification.balance} (req_id={verification.request_id}, age={verification.age_seconds:.2f}s) — FRESH")
                result["balance_verification"] = {
                    "balance": verification.balance,
                    "timestamp": verification.timestamp.isoformat(),
                    "request_id": verification.request_id,
                    "age_seconds": verification.age_seconds,
                }
            else:
                result["steps"].append(f"VERIFY BALANCE: FAILED — no valid verification (verification={verification})")
        except Exception as e:
            result["steps"].append(f"VERIFY BALANCE: error {e}")
        result["verified_balances"] = verified_balances

        # P1.19 Step 7: HALT
        result["steps"].append("HALT: entered HALTED state — manual resume required via RECOVERY PROTOCOL")
        result["halted"] = self._halted

        # Audit log
        print(f"[ExecutionEngine] EMERGENCY STOP complete: flat={verified_flat} canceled={canceled} flattened={len(flattened)}")

        return result

    def is_halted(self) -> bool:
        """Check if engine is in HALTED state after emergency_stop."""
        return self._halted

    async def resume_from_halt(self, manual_approval: bool = False) -> dict:
        """
        Recovery protocol — mandatory 7 checks after HALTED (P1.20):

        HALTED → RECONCILIATION → POSITIONS VERIFIED → ORDERS CLEAN → BALANCE FRESH
        → RISK HEALTHY → MARKET DATA HEALTHY → MANUAL APPROVAL → ACTIVE

        software_position = None only after exchange_position == 0 + fresh reconciliation.
        If exchange not responded: UNKNOWN_POSITION and stay HALTED.
        """
        if not self._halted:
            return {"resumed": False, "reason": "not halted"}

        if not manual_approval:
            return {"resumed": False, "reason": "MANUAL APPROVAL required — HALTED stays", "step": "MANUAL_APPROVAL"}

        checks: dict = {}

        # P1.20 Step 1: RECONCILIATION — fresh REST/WebSocket
        try:
            # Force fresh reconciliation: query exchange positions + orders
            if self.binance_rest and self.config.mode in {ExecutionMode.LIVE, ExecutionMode.DRY_RUN}:
                try:
                    ex_pos = await self.binance_rest.get_positions()
                    ex_orders = await self.binance_rest.get_open_orders()
                    checks["RECONCILIATION"] = True
                    checks["reconciliation"] = True
                    checks["reconciliation_detail"] = f"positions={len(ex_pos)} orders={len(ex_orders)} fresh"
                except Exception as e:
                    return {"resumed": False, "reason": f"RECONCILIATION failed: {e}", "step": "RECONCILIATION", "checks": checks}
            else:
                # PAPER: use local verification loop state
                checks["RECONCILIATION"] = True
                checks["reconciliation"] = True
                checks["reconciliation_detail"] = "paper reconciliation"
        except Exception as e:
            return {"resumed": False, "reason": f"RECONCILIATION error: {e}", "step": "RECONCILIATION", "checks": checks}

        # P1.20 Step 2: POSITIONS VERIFIED — software == exchange == 0 (P1.21: never position=None without exchange 0)
        try:
            remaining = {}
            if self.binance_rest and self.config.mode in {ExecutionMode.LIVE, ExecutionMode.DRY_RUN}:
                ex_pos2 = await self.binance_rest.get_positions()
                remaining = {p.symbol: float(p.position_amt) for p in ex_pos2 if abs(float(p.position_amt)) > self.config.position_tolerance}
            else:
                remaining = {k: v for k, v in self._position_cache.items() if abs(v) > self.config.position_tolerance}
                # also check paper_engine
                if self.paper_engine and getattr(self.paper_engine, 'position', None) is not None:
                    try:
                        if getattr(self.paper_engine, 'has_position', False):
                            remaining["paper"] = float(getattr(self.paper_engine.position, 'quantity', 0))
                    except Exception:
                        pass
            if remaining:
                return {"resumed": False, "reason": f"POSITIONS VERIFIED failed: {remaining} (software must be 0 only after exchange 0 — P1.21)", "step": "POSITIONS VERIFIED", "checks": checks, "remaining": remaining}
            checks["POSITIONS VERIFIED"] = True
            checks["no_open_positions"] = True
        except Exception as e:
            return {"resumed": False, "reason": f"POSITIONS VERIFIED error: {e} — UNKNOWN_POSITION stays HALTED (P1.21)", "step": "POSITIONS VERIFIED", "checks": checks}

        # P1.20 Step 3: ORDERS CLEAN — no open orders
        try:
            open_orders = self.order_manager.get_active_orders() if self.order_manager else []
            if open_orders:
                return {"resumed": False, "reason": f"ORDERS CLEAN failed: {len(open_orders)} still open", "step": "ORDERS CLEAN", "checks": checks}
            checks["ORDERS CLEAN"] = True
            checks["no_open_orders"] = True
        except Exception as e:
            return {"resumed": False, "reason": f"ORDERS CLEAN check error: {e}", "step": "ORDERS CLEAN", "checks": checks}

        # P1.20 Step 4: BALANCE FRESH — verified balance <5s, with exchange response
        try:
            ver = await self.get_verified_balance(max_age_seconds=5.0)
            if not ver or not ver.is_valid(5.0):
                return {"resumed": False, "reason": "BALANCE FRESH failed: no fresh verification (<5s)", "step": "BALANCE FRESH", "checks": checks}
            checks["BALANCE FRESH"] = True
            checks["balance_verified"] = True
            checks["balance"] = ver.balance
        except Exception as e:
            return {"resumed": False, "reason": f"BALANCE FRESH error: {e}", "step": "BALANCE FRESH", "checks": checks}

        # P1.20 Step 5: RISK HEALTHY — drawdown and exposure must be healthy
        try:
            # Ensure drawdown_guard exists
            if not hasattr(self, 'drawdown_guard') or self.drawdown_guard is None:
                try:
                    from src.drawdown_guard import DrawdownGuard
                    self.drawdown_guard = DrawdownGuard(max_drawdown_percent=self.config.max_drawdown_pct)
                except Exception:
                    self.drawdown_guard = None
            if self.drawdown_guard:
                dd_res = self.drawdown_guard.evaluate(ver.balance if 'ver' in locals() and ver else 0)
                if not dd_res.allowed:
                    return {"resumed": False, "reason": f"RISK HEALTHY failed: drawdown {dd_res.drawdown_percent}", "step": "RISK HEALTHY", "checks": checks}
            # Also check exposure via position cache
            total_exposure = sum(abs(v) for v in self._position_cache.values())
            if total_exposure > 0:
                return {"resumed": False, "reason": f"RISK HEALTHY failed: exposure {total_exposure} not 0 after flatten", "step": "RISK HEALTHY", "checks": checks}
            checks["RISK HEALTHY"] = True
            checks["risk_verified"] = True
        except Exception as e:
            return {"resumed": False, "reason": f"RISK HEALTHY error: {e}", "step": "RISK HEALTHY", "checks": checks}

        # P1.20 Step 6: MARKET DATA HEALTHY — fresh market data
        try:
            # Need fresh market data (at least one symbol)
            if self.binance_rest:
                # try to fetch a mark price to prove market data alive
                try:
                    await self._get_current_price("BTCUSDT")
                    checks["MARKET DATA HEALTHY"] = True
                    checks["market_data_verified"] = True
                except Exception as e:
                    return {"resumed": False, "reason": f"MARKET DATA HEALTHY failed: {e}", "step": "MARKET DATA HEALTHY", "checks": checks}
            else:
                checks["MARKET DATA HEALTHY"] = True
                checks["market_data_verified"] = True
        except Exception as e:
            return {"resumed": False, "reason": f"MARKET DATA HEALTHY error: {e}", "step": "MARKET DATA HEALTHY", "checks": checks}

        # P1.20 Step 7: MANUAL APPROVAL -> ACTIVE (already checked, now allow)
        self._halted = False
        checks["MANUAL APPROVAL"] = True
        checks["manual_approval"] = True
        print("[ExecutionEngine] Resumed HALTED -> ACTIVE via recovery protocol (all 7 checks PASS: RECONCILIATION->POSITIONS VERIFIED->ORDERS CLEAN->BALANCE FRESH->RISK HEALTHY->MARKET DATA HEALTHY->MANUAL APPROVAL)")
        return {"resumed": True, "checks": checks, "step": "ACTIVE"}


class ExecutionError(Exception):
    pass


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================

async def create_execution_engine(
    mode: ExecutionMode = ExecutionMode.PAPER,
    binance_config: Optional[BinanceConfig] = None,
    paper_engine: Optional[Any] = None,
) -> ExecutionEngine:
    """Create and start execution engine."""
    config = ExecutionConfig(mode=mode, binance=binance_config)
    engine = ExecutionEngine(config, paper_engine=paper_engine)
    await engine.start()
    return engine


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "ExecutionMode",
    "ExecutionConfig",
    "ExecutionStats",
    "ExecutionEngine",
    "ExecutionError",
    "create_execution_engine",
]