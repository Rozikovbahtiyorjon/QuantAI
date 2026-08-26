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
from src.execution.order_manager import OrderManager, OrderManagerConfig
from src.execution.binance_adapter import (
    BinanceConfig,
    BinanceRestAdapter,
    BinanceWebSocketAdapter,
)


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
        # Callbacks
        on_fill: Optional[Callable[[Fill], None]] = None,
        on_order_update: Optional[Callable[[Order], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        on_reconciliation_fix: Optional[Callable[[str, dict], None]] = None,
    ) -> None:
        self.config = config
        self.paper_engine = paper_engine
        
        self._on_fill = on_fill
        self._on_order_update = on_order_update
        self._on_error = on_error
        self._on_reconciliation_fix = on_reconciliation_fix
        
        # State
        self._running = False
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
        if not self._running:
            raise RuntimeError("ExecutionEngine not started")
        
        self._stats.intents_received += 1
        
        # Safety checks
        if not self._safety_check(intent):
            raise ExecutionError("Safety check failed")
        
        # Create order via OrderManager
        order = self.order_manager.submit_intent(intent)
        self._stats.orders_created += 1
        
        return order
    
    def _safety_check(self, intent) -> bool:
        """Pre-trade safety checks."""
        if not self.config.kill_switch_enabled:
            return True
        
        # Check daily loss limit
        if self._daily_pnl <= -abs(self._start_balance * self.config.max_daily_loss_pct / 100):
            print(f"[ExecutionEngine] KILL SWITCH: Daily loss limit reached ({self._daily_pnl})")
            return False
        
        # Check drawdown
        if self._start_balance > 0:
            drawdown = (self._start_balance - self._balance_cache) / self._start_balance * 100
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
            return False
    
    async def _submit_live(self, order: Order) -> bool:
        """Submit to live Binance."""
        intent = order.intent
        result = await self.binance_rest.place_order(intent)
        
        if "orderId" in result:
            order.exchange_order_id = str(result["orderId"])
            order.status = OrderStatus.NEW
            order.submitted_at = datetime.now(timezone.utc)
            return True
        else:
            order.reject(f"Binance error: {result}")
            self._stats.orders_rejected += 1
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
        """Process fill from any source."""
        self._stats.fills_received += 1
        
        # Update order via OrderManager
        self.order_manager.on_fill(fill)
        
        # Update position cache
        order = self.order_manager.get_order(fill.order_id)
        if order:
            self._update_position_cache(order)
        
        # Update daily PnL
        if fill.side == OrderSide.BUY:
            self._daily_pnl -= fill.commission
        else:
            self._daily_pnl -= fill.commission
        
        if self._on_fill:
            self._on_fill(fill)
    
    def _update_position_cache(self, order: Order):
        """Update internal position cache from filled order."""
        symbol = order.intent.symbol
        side = 1 if order.intent.side == OrderSide.BUY else -1
        self._position_cache[symbol] += side * order.filled_quantity
    
    async def _update_balance_cache(self) -> None:
        """Update balance cache from exchange."""
        if self.config.mode == ExecutionMode.LIVE and self.binance_rest:
            try:
                balances = await self.binance_rest.get_balance()
                for b in balances:
                    if b.asset == "USDT":
                        self._balance_cache = b.available_balance
                        self._start_balance = self._start_balance or b.available_balance
                        break
            except Exception as e:
                print(f"[ExecutionEngine] Balance update error: {e}")
        elif self.paper_engine:
            self._balance_cache = self.paper_engine.balance
            self._start_balance = self._start_balance or self.paper_engine.balance
    
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
            "balance": self._balance_cache,
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
        return self._balance_cache
    
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
        """Emergency stop: cancel all orders, flatten positions."""
        print("[ExecutionEngine] EMERGENCY STOP triggered")
        
        # Cancel all orders
        canceled = 0
        if self.order_manager:
            canceled = self.order_manager.cancel_all()
        
        # In LIVE mode, could add position flattening logic here
        
        return {
            "canceled_orders": canceled,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


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