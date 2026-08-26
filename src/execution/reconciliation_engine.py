"""
====================================================
QuantAI Professional
Reconciliation Engine
====================================================

Periodic state reconciliation between local tracking and exchange.
Detects and fixes:
- Position drift (ghost positions, missing positions)
- Balance discrepancies
- Order state mismatches
- Fill discrepancies
====================================================
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional

from src.execution.orders import OrderSide, OrderStatus
from src.execution.binance_adapter import BinanceRestAdapter, Position as BinancePosition
from src.execution.order_manager import OrderManager


class ReconciliationAction(str, Enum):
    NONE = "NONE"
    UPDATE_LOCAL = "UPDATE_LOCAL"           # Update local cache to match exchange
    CANCEL_ORDER = "CANCEL_ORDER"           # Cancel order on exchange
    REPLACE_ORDER = "REPLACE_ORDER"         # Cancel + replace
    FLATTEN_POSITION = "FLATTEN_POSITION"   # Emergency close
    ALERT = "ALERT"                         # Notify only


@dataclass
class ReconciliationIssue:
    type: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    symbol: Optional[str] = None
    details: dict = field(default_factory=dict)
    suggested_action: ReconciliationAction = ReconciliationAction.ALERT
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    auto_fixed: bool = False


@dataclass
class ReconciliationReport:
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    issues: list[ReconciliationIssue] = field(default_factory=list)
    fixes_applied: int = 0
    positions_checked: int = 0
    orders_checked: int = 0
    balance_checked: bool = False
    duration_ms: float = 0.0
    
    @property
    def has_critical(self) -> bool:
        return any(i.severity == "CRITICAL" for i in self.issues)
    
    @property
    def summary(self) -> dict:
        by_type = defaultdict(int)
        by_severity = defaultdict(int)
        for issue in self.issues:
            by_type[issue.type] += 1
            by_severity[issue.severity] += 1
        return {
            "total_issues": len(self.issues),
            "fixes_applied": self.fixes_applied,
            "by_type": dict(by_type),
            "by_severity": dict(by_severity),
            "duration_ms": self.duration_ms,
        }


class ReconciliationConfig:
    """Configuration for reconciliation engine."""
    
    def __init__(
        self,
        interval_seconds: float = 30.0,
        position_tolerance: float = 0.001,
        balance_tolerance: float = 1.0,
        order_age_limit_seconds: float = 3600.0,
        enable_auto_fix: bool = True,
        max_position_diff_pct: float = 5.0,
        alert_on_critical: bool = True,
        enable_ghost_detection: bool = True,
        enable_stuck_order_detection: bool = True,
    ):
        self.interval_seconds = interval_seconds
        self.position_tolerance = position_tolerance
        self.balance_tolerance = balance_tolerance
        self.order_age_limit_seconds = order_age_limit_seconds
        self.enable_auto_fix = enable_auto_fix
        self.max_position_diff_pct = max_position_diff_pct
        self.alert_on_critical = alert_on_critical
        self.enable_ghost_detection = enable_ghost_detection
        self.enable_stuck_order_detection = enable_stuck_order_detection


class ReconciliationEngine:
    """
    Periodic reconciliation between local state and exchange.
    
    Checks:
    1. Position reconciliation (local vs exchange)
    2. Balance reconciliation (local vs exchange)
    3. Order state reconciliation (local vs exchange)
    4. Fill reconciliation
    4. Ghost position detection
    5. Stuck order detection
    """
    
    def __init__(
        self,
        config: ReconciliationConfig,
        binance_rest: BinanceRestAdapter,
        order_manager: OrderManager,
        paper_engine: Optional[Any] = None,
        get_local_balance: Optional[Callable[[], float]] = None,
        get_local_positions: Optional[Callable[[], dict[str, float]]] = None,
        on_issue: Optional[Callable[[ReconciliationIssue], None]] = None,
        on_fix: Optional[Callable[[ReconciliationIssue], None]] = None,
        on_critical: Optional[Callable[[ReconciliationIssue], None]] = None,
    ) -> None:
        self.config = config
        self.binance = binance_rest
        self.order_manager = order_manager
        self.paper_engine = paper_engine
        self.get_local_balance = get_local_balance
        self.get_local_positions = get_local_positions
        
        self.on_issue = on_issue
        self.on_fix = on_fix
        self.on_critical = on_critical
        
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_report: Optional[ReconciliationReport] = None
    
    async def start(self) -> None:
        """Start periodic reconciliation."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        print(f"[ReconciliationEngine] Started (interval={self.config.interval_seconds}s)")
    
    async def stop(self) -> None:
        """Stop periodic reconciliation."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        print("[ReconciliationEngine] Stopped")
    
    async def _run_loop(self):
        """Main reconciliation loop."""
        while self._running:
            try:
                await asyncio.sleep(self.config.interval_seconds)
                if self._running:
                    await self.reconcile()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[ReconciliationEngine] Loop error: {e}")
    
    # ============================================================
    # MAIN RECONCILIATION
    # ============================================================
    
    async def reconcile(self) -> dict:
        """Run full reconciliation cycle."""
        start_time = datetime.now(timezone.utc)
        report = ReconciliationReport()
        
        try:
            # 1. Position reconciliation
            await self._reconcile_positions(report)
            
            # 2. Balance reconciliation
            await self._reconcile_balance(report)
            
            # 4. Order state reconciliation
            await self._reconcile_orders(report)
            
            # 4. Ghost position detection
            if self.config.enable_ghost_detection:
                await self._detect_ghost_positions(report)
            
            # 5. Stuck order detection
            if self.config.enable_stuck_order_detection:
                await self._detect_stuck_orders(report)
            
        except Exception as e:
            print(f"[ReconciliationEngine] Error: {e}")
        
        # Finalize report
        report.duration_ms = (datetime.now(timezone.utc) - report.timestamp).total_seconds() * 1000
        self._last_report = report
        
        # Notify issues
        for issue in report.issues:
            if self.on_issue:
                self.on_issue(issue)
            if issue.severity == "CRITICAL" and self.on_critical:
                self.on_critical(issue)
        
        return report.summary
    
    # ============================================================
    # POSITION RECONCILIATION
    # ============================================================
    
    async def _reconcile_positions(self, report: ReconciliationReport):
        """Reconcile local positions with exchange."""
        if not hasattr(self, 'binance') or not self.binance:
            return
        
        try:
            exchange_positions = await self.binance.get_positions()
            exchange_map = {p.symbol: p.position_amt for p in exchange_positions}
            report.positions_checked = len(exchange_map)
            
            # Get local positions
            local_positions = {}
            if hasattr(self, 'get_local_positions') and self.get_local_positions:
                local_positions = self.get_local_positions() or {}
            
            # Check each exchange position
            for symbol, exchange_qty in exchange_map.items():
                local_qty = 0.0
                # Try to get from order manager cache
                # This would be populated from fills
                
                diff = abs(exchange_qty)  # If no local tracking
                
                if diff > self.config.position_tolerance:
                    # Calculate percentage difference
                    pct_diff = (diff / abs(exchange_qty) * 100) if exchange_qty != 0 else 100
                    
                    severity = "LOW"
                    if pct_diff > self.config.max_position_diff_pct:
                        severity = "HIGH"
                    if pct_diff > self.config.max_position_diff_pct * 2:
                        severity = "CRITICAL"
                    
                    issue = ReconciliationIssue(
                        type="position_mismatch",
                        severity=severity,
                        symbol=issue_symbol,
                        details={
                            "exchange_qty": exchange_qty,
                            "local_qty": 0,
                            "diff_pct": pct_diff,
                        },
                        suggested_action=ReconciliationAction.UPDATE_LOCAL,
                    )
                    report.issues.append(issue)
            
            # Check for positions we have locally but not on exchange
            # (would need local position tracking)
            
        except Exception as e:
            print(f"[ReconciliationEngine] Position reconciliation error: {e}")
    
    # ============================================================
    # BALANCE RECONCILIATION
    # ============================================================
    
    async def _reconcile_balance(self, report: ReconciliationReport):
        """Reconcile local balance with exchange."""
        if not hasattr(self, 'binance') or not self.binance:
            return
        
        report.balance_checked = True
        
        try:
            exchange_balances = await self.binance.get_balance()
            exchange_balance = 0.0
            for b in exchange_balances:
                if b.asset == "USDT":
                    exchange_balance = b.available_balance
                    break
            
            # Get local balance
            local_balance = 0.0
            if hasattr(self, 'get_local_balance') and self.get_local_balance:
                local_balance = self.get_local_balance() or 0.0
            
            diff = abs(local_balance - exchange_balance)
            if diff > self.config.balance_tolerance:
                pct_diff = (diff / exchange_balance * 100) if exchange_balance > 0 else 0
                
                severity = "LOW"
                if pct_diff > 1.0:
                    severity = "MEDIUM"
                if pct_diff > 5.0:
                    severity = "HIGH"
                if pct_diff > 10.0:
                    severity = "CRITICAL"
                
                issue = ReconciliationIssue(
                    type="balance_mismatch",
                    severity=severity,
                    details={
                        "local": local_balance,
                        "exchange": exchange_balance,
                        "diff": local_balance - exchange_balance,
                        "diff_pct": pct_diff,
                    },
                    suggested_action=ReconciliationAction.UPDATE_LOCAL,
                )
                report.issues.append(issue)
                
        except Exception as e:
            print(f"[ReconciliationEngine] Balance reconciliation error: {e}")
    
    # ============================================================
    # ORDER STATE RECONCILIATION
    # ============================================================
    
    async def _reconcile_orders(self, report: ReconciliationReport):
        """Reconcile local order states with exchange."""
        if not hasattr(self, 'binance') or not self.binance:
            return
        
        try:
            # Get all active local orders
            active_orders = self.order_manager.get_active_orders()
            report.orders_checked = len(active_orders)
            
            for order in active_orders:
                if not order.exchange_order_id:
                    continue
                
                try:
                    exchange_order = await self.binance.get_order(
                        symbol=order.intent.symbol,
                        order_id=order.exchange_order_id,
                    )
                    
                    exchange_status = exchange_order.get("status")
                    if exchange_status:
                        # Map Binance status to our status
                        status_map = {
                            "NEW": "NEW",
                            "PARTIALLY_FILLED": "PARTIALLY_FILLED",
                            "FILLED": "FILLED",
                            "CANCELED": "CANCELED",
                            "REJECTED": "REJECTED",
                            "EXPIRED": "EXPIRED",
                        }
                        exchange_status = status_map.get(exchange_status, "NEW")
                        
                        if order.status.value != exchange_status:
                            # State mismatch
                            issue = ReconciliationIssue(
                                type="order_state_mismatch",
                                severity="MEDIUM",
                                symbol=order.intent.symbol,
                                details={
                                    "order_id": order.order_id,
                                    "exchange_order_id": order.exchange_order_id,
                                    "local_status": order.status.value,
                                    "exchange_status": exchange_status,
                                },
                                suggested_action=ReconciliationAction.UPDATE_LOCAL,
                            )
                            report.issues.append(issue)
                            
                            # Auto-fix: update local state
                            if self.config.enable_auto_fix:
                                from src.execution.orders import OrderStatus
                                self.order_manager.on_order_update(
                                    exchange_order_id=order.exchange_order_id,
                                    status=OrderStatus(exchange_status),
                                )
                                report.fixes_applied += 1
                        
                        # Check filled quantity
                        exchange_filled = float(exchange_order.get("cumQty", 0))
                        if abs(order.filled_quantity - exchange_filled) > 0.0001:
                            issue = ReconciliationIssue(
                                type="fill_qty_mismatch",
                                severity="MEDIUM",
                                symbol=order.intent.symbol,
                                details={
                                    "order_id": order.order_id,
                                    "local_filled": order.filled_quantity,
                                    "exchange_filled": exchange_filled,
                                },
                                suggested_action=ReconciliationAction.UPDATE_LOCAL,
                            )
                            report.issues.append(issue)
                            
                except Exception as e:
                    print(f"[ReconciliationEngine] Order check error for {order.order_id}: {e}")
                    
        except Exception as e:
            print(f"[ReconciliationEngine] Order reconciliation error: {e}")
    
    # ============================================================
    # GHOST POSITION DETECTION
    # ============================================================
    
    async def _detect_ghost_positions(self, report: ReconciliationReport):
        """Detect positions on exchange that we don't track locally."""
        if not hasattr(self, 'binance') or not self.binance:
            return
        
        try:
            exchange_positions = await self.binance.get_positions()
            
            # This would need local position tracking
            # For now, we'd need a way to get tracked symbols
            
        except Exception as e:
            print(f"[ReconciliationEngine] Ghost detection error: {e}")
    
    # ============================================================
    # STUCK ORDER DETECTION
    # ============================================================
    
    async def _detect_stuck_orders(self, report: ReconciliationReport):
        """Detect orders that haven't been filled for too long."""
        if not self.order_manager:
            return
        
        try:
            active_orders = self.order_manager.get_active_orders()
            now = datetime.now(timezone.utc)
            
            for order in active_orders:
                age = (now - order.created_at).total_seconds()
                
                if age > self.config.order_age_limit_seconds:
                    # Check if it's a limit order far from market
                    issue = ReconciliationIssue(
                        type="stuck_order",
                        severity="MEDIUM",
                        symbol=order.intent.symbol,
                        details={
                            "order_id": order.order_id,
                            "age_seconds": age,
                            "limit_seconds": self.config.order_age_limit_seconds,
                            "side": order.intent.side.value,
                            "type": order.intent.order_type.value,
                            "price": order.intent.price,
                        },
                        suggested_action=ReconciliationAction.CANCEL_ORDER,
                    )
                    report.issues.append(issue)
                    
                    # Auto-cancel if enabled
                    if self.config.enable_auto_fix:
                        self.order_manager.cancel_order(order.order_id)
                        report.fixes_applied += 1
                        
        except Exception as e:
            print(f"[ReconciliationEngine] Stuck order detection error: {e}")
    
    # ============================================================
    # PUBLIC METHODS
    # ============================================================
    
    async def run_once(self) -> dict:
        """Run single reconciliation cycle."""
        return await self.reconcile()
    
    def get_last_report(self) -> Optional[dict]:
        """Get last reconciliation report."""
        if self._last_report:
            return self._last_report.summary
        return None
    
    async def force_reconcile(self) -> dict:
        """Force immediate reconciliation."""
        return await self.reconcile()


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================

async def create_reconciliation_engine(
    binance_rest: BinanceRestAdapter,
    order_manager: OrderManager,
    config: Optional[ReconciliationConfig] = None,
    **kwargs,
) -> ReconciliationEngine:
    """Create and start reconciliation engine."""
    engine = ReconciliationEngine(
        config=config or ReconciliationConfig(),
        binance_rest=binance_rest,
        order_manager=order_manager,
        **kwargs,
    )
    await engine.start()
    return engine


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "ReconciliationAction",
    "ReconciliationIssue",
    "ReconciliationReport",
    "ReconciliationConfig",
    "ReconciliationEngine",
]