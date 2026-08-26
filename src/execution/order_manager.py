"""
====================================================
QuantAI Professional
Execution Boundary - Order Manager
====================================================

Manages order lifecycle: creation, submission, tracking, cancellation.
====================================================
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Optional
from uuid import uuid4

from src.execution.orders import (
    Fill,
    Order,
    OrderIntentData,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)


@dataclass
class OrderManagerConfig:
    """Configuration for OrderManager."""
    max_open_orders: int = 100
    max_order_age_seconds: int = 3600  # 1 hour default
    enable_order_expiration: bool = True
    cancel_on_disconnect: bool = True
    retry_failed_submissions: bool = True
    max_submission_retries: int = 3
    submission_retry_delay_seconds: float = 1.0


class OrderManager:
    """
    Central order lifecycle management.
    
    Responsibilities:
    - Create orders from intents
    - Submit orders to exchange adapter
    - Track order state (active, filled, canceled, rejected)
    - Handle fills and partial fills
    - Manage order expiration and cancellation
    - Provide order queries and history
    """
    
    def __init__(
        self,
        config: Optional[OrderManagerConfig] = None,
        submit_callback: Optional[Callable[[Order], bool]] = None,
        cancel_callback: Optional[Callable[[Order], bool]] = None,
    ) -> None:
        self.config = config or OrderManagerConfig()
        self._submit_callback = submit_callback
        self._cancel_callback = cancel_callback
        
        # Order storage
        self._orders: dict[str, Order] = {}  # order_id -> Order
        self._orders_by_client_id: dict[str, str] = {}  # client_order_id -> order_id
        self._orders_by_exchange_id: dict[str, str] = {}  # exchange_order_id -> order_id
        
        # Symbol-indexed views
        self._active_orders_by_symbol: dict[str, set[str]] = defaultdict(set)
        self._fills: list[Fill] = []
        
        # Statistics
        self._stats = {
            "created": 0,
            "submitted": 0,
            "filled": 0,
            "partially_filled": 0,
            "canceled": 0,
            "rejected": 0,
            "expired": 0,
            "errors": 0,
        }
    
    # ============================================================
    # ORDER CREATION
    # ============================================================
    
    def create_order(
        self,
        intent: OrderIntentData,
        client_order_id: Optional[str] = None,
    ) -> Order:
        """Create a new order from intent."""
        order = Order(
            client_order_id=client_order_id or f"quantai_{uuid4().hex[:12]}",
            intent=intent,
        )
        
        self._orders[order.order_id] = order
        self._orders_by_client_id[order.client_order_id] = order.order_id
        
        self._active_orders_by_symbol[intent.symbol].add(order.order_id)
        self._stats["created"] += 1
        
        return order
    
    # ============================================================
    # ORDER SUBMISSION
    # ============================================================
    
    def submit_order(self, order: Order) -> bool:
        """Submit order to exchange via callback."""
        if order.status != OrderStatus.NEW:
            return False
        
        if self._submit_callback is None:
            order.reject("No submit callback configured")
            self._stats["rejected"] += 1
            self._stats["errors"] += 1
            return False
        
        try:
            success = self._submit_callback(order)
            if success:
                order.status = OrderStatus.NEW  # Submitted, awaiting ack
                order.submitted_at = datetime.utcnow()
                order.updated_at = datetime.utcnow()
                self._stats["submitted"] += 1
                return True
            else:
                order.reject("Submission callback returned False")
                self._stats["rejected"] += 1
                self._stats["errors"] += 1
                return False
        except Exception as e:
            order.reject(f"Submission error: {e}")
            self._stats["rejected"] += 1
            self._stats["errors"] += 1
            return False
    
    def submit_intent(
        self,
        intent: OrderIntentData,
        client_order_id: Optional[str] = None,
    ) -> Order:
        """Create and submit order in one call."""
        order = self.create_order(intent, client_order_id)
        self.submit_order(order)
        return order
    
    # ============================================================
    # ORDER UPDATES (from exchange callbacks)
    # ============================================================
    
    def on_order_ack(
        self,
        client_order_id: str,
        exchange_order_id: str,
    ) -> bool:
        """Handle order acknowledgment from exchange."""
        order_id = self._orders_by_client_id.get(client_order_id)
        if not order_id:
            return False
        
        order = self._orders.get(order_id)
        if not order:
            return False
        
        order.exchange_order_id = exchange_order_id
        self._orders_by_exchange_id[exchange_order_id] = order_id
        order.status = OrderStatus.NEW  # Confirmed by exchange
        order.updated_at = datetime.utcnow()
        return True
    
    def on_fill(
        self,
        fill: Fill,
    ) -> bool:
        """Handle fill notification from exchange."""
        # Find order by exchange_order_id or client_order_id
        order_id = None
        if fill.exchange_order_id:
            order_id = self._orders_by_exchange_id.get(fill.exchange_order_id)
        if not order_id and fill.client_order_id:
            order_id = self._orders_by_client_id.get(fill.client_order_id)
        
        if not order_id:
            return False
        
        order = self._orders.get(order_id)
        if not order:
            return False
        
        # Update order
        order.update_fill(
            fill_qty=fill.quantity,
            fill_price=fill.price,
            commission=fill.commission,
            commission_asset=fill.commission_asset,
        )
        
        # Store fill
        self._fills.append(fill)
        
        # Update stats
        if order.status == OrderStatus.FILLED:
            self._stats["filled"] += 1
            self._active_orders_by_symbol[order.intent.symbol].discard(order.order_id)
        elif order.status == OrderStatus.PARTIALLY_FILLED:
            self._stats["partially_filled"] += 1
        
        return True
    
    def on_order_update(
        self,
        exchange_order_id: str,
        status: OrderStatus,
        filled_qty: Optional[float] = None,
        avg_price: Optional[float] = None,
    ) -> bool:
        """Handle order status update from exchange."""
        order_id = self._orders_by_exchange_id.get(exchange_order_id)
        if not order_id:
            return False
        
        order = self._orders.get(order_id)
        if not order:
            return False
        
        order.status = status
        order.updated_at = datetime.utcnow()
        
        if filled_qty is not None:
            order.filled_quantity = filled_qty
        if avg_price is not None:
            order.average_fill_price = avg_price
        
        if status == OrderStatus.FILLED:
            order.filled_quantity = order.intent.quantity
            order.filled_at = datetime.utcnow()
            self._stats["filled"] += 1
            self._active_orders_by_symbol[order.intent.symbol].discard(order.order_id)
        elif status == OrderStatus.CANCELED:
            self._stats["canceled"] += 1
            self._active_orders_by_symbol[order.intent.symbol].discard(order.order_id)
        elif status == OrderStatus.REJECTED:
            self._stats["rejected"] += 1
            self._active_orders_by_symbol[order.intent.symbol].discard(order.order_id)
        elif status == OrderStatus.EXPIRED:
            self._stats["expired"] += 1
            self._active_orders_by_symbol[order.intent.symbol].discard(order.order_id)
        
        return True
    
    # ============================================================
    # ORDER CANCELLATION
    # ============================================================
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel order by internal order_id."""
        order = self._orders.get(order_id)
        if not order:
            return False
        
        if not order.is_active:
            return False
        
        if self._cancel_callback is None:
            # Local cancel only
            order.cancel()
            self._stats["canceled"] += 1
            self._active_orders_by_symbol[order.intent.symbol].discard(order.order_id)
            return True
        
        try:
            success = self._cancel_callback(order)
            if success:
                order.status = OrderStatus.PENDING_CANCEL
                order.updated_at = datetime.utcnow()
                return True
            return False
        except Exception:
            return False
    
    def cancel_by_client_id(self, client_order_id: str) -> bool:
        """Cancel order by client_order_id."""
        order_id = self._orders_by_client_id.get(client_order_id)
        if not order_id:
            return False
        return self.cancel_order(order_id)
    
    def cancel_by_exchange_id(self, exchange_order_id: str) -> bool:
        """Cancel order by exchange_order_id."""
        order_id = self._orders_by_exchange_id.get(exchange_order_id)
        if not order_id:
            return False
        return self.cancel_order(order_id)
    
    def cancel_all_for_symbol(self, symbol: str) -> int:
        """Cancel all active orders for a symbol."""
        order_ids = list(self._active_orders_by_symbol.get(symbol, set()))
        count = 0
        for oid in order_ids:
            if self.cancel_order(oid):
                count += 1
        return count
    
    def cancel_all(self) -> int:
        """Cancel all active orders."""
        all_ids = list(self._orders.keys())
        count = 0
        for oid in all_ids:
            if self._orders[oid].is_active:
                if self.cancel_order(oid):
                    count += 1
        return count
    
    # ============================================================
    # ORDER QUERIES
    # ============================================================
    
    def get_order(self, order_id: str) -> Optional[Order]:
        return self._orders.get(order_id)
    
    def get_order_by_client_id(self, client_order_id: str) -> Optional[Order]:
        order_id = self._orders_by_client_id.get(client_order_id)
        if order_id:
            return self._orders.get(order_id)
        return None
    
    def get_order_by_exchange_id(self, exchange_order_id: str) -> Optional[Order]:
        order_id = self._orders_by_exchange_id.get(exchange_order_id)
        if order_id:
            return self._orders.get(order_id)
        return None
    
    def get_active_orders(self, symbol: Optional[str] = None) -> list[Order]:
        if symbol:
            order_ids = self._active_orders_by_symbol.get(symbol, set())
            return [self._orders[oid] for oid in order_ids if oid in self._orders]
        return [o for o in self._orders.values() if o.is_active]
    
    def get_filled_orders(self, symbol: Optional[str] = None) -> list[Order]:
        orders = [o for o in self._orders.values() if o.status == OrderStatus.FILLED]
        if symbol:
            orders = [o for o in orders if o.intent.symbol == symbol]
        return orders
    
    def get_order_history(
        self,
        symbol: Optional[str] = None,
        status: Optional[OrderStatus] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[Order]:
        orders = list(self._orders.values())
        
        if symbol:
            orders = [o for o in orders if o.intent.symbol == symbol]
        if status:
            orders = [o for o in orders if o.status == status]
        if since:
            orders = [o for o in orders if o.created_at >= since]
        
        orders.sort(key=lambda o: o.created_at, reverse=True)
        return orders[:limit]
    
    def get_fills(
        self,
        symbol: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[Fill]:
        fills = self._fills
        if symbol:
            fills = [f for f in fills if f.symbol == symbol]
        if since:
            fills = [f for f in fills if f.timestamp >= since]
        fills.sort(key=lambda f: f.timestamp, reverse=True)
        return fills[:limit]
    
    # ============================================================
    # EXPIRATION & CLEANUP
    # ============================================================
    
    def expire_old_orders(self, max_age_seconds: Optional[int] = None) -> int:
        """Expire orders older than max_age_seconds."""
        max_age = max_age_seconds or self.config.max_order_age_seconds
        if not self.config.enable_order_expiration:
            return 0
        
        now = datetime.utcnow()
        expired_count = 0
        
        for order in list(self._orders.values()):
            if not order.is_active:
                continue
            
            age = (now - order.created_at).total_seconds()
            if age > max_age:
                order.status = OrderStatus.EXPIRED
                order.updated_at = now
                self._stats["expired"] += 1
                self._active_orders_by_symbol[order.intent.symbol].discard(order.order_id)
                expired_count += 1
        
        return expired_count
    
    # ============================================================
    # STATISTICS
    # ============================================================
    
    @property
    def stats(self) -> dict:
        return self._stats.copy()
    
    def reset_stats(self) -> None:
        self._stats = {k: 0 for k in self._stats}
    
    # ============================================================
    # CLEANUP
    # ============================================================
    
    def clear_completed_orders(self, older_than: Optional[datetime] = None) -> int:
        """Remove terminal orders from memory."""
        cutoff = older_than or (datetime.utcnow() - timedelta(days=7))
        removed = 0
        
        for oid in list(self._orders.keys()):
            order = self._orders[oid]
            if order.is_terminal and order.updated_at < cutoff:
                del self._orders[oid]
                self._orders_by_client_id.pop(order.client_order_id, None)
                if order.exchange_order_id:
                    self._orders_by_exchange_id.pop(order.exchange_order_id, None)
                self._active_orders_by_symbol[order.intent.symbol].discard(oid)
                removed += 1
        
        return removed
    
    def reset(self) -> None:
        """Full reset of order manager."""
        self._orders.clear()
        self._orders_by_client_id.clear()
        self._orders_by_exchange_id.clear()
        self._active_orders_by_symbol.clear()
        self._fills.clear()
        self.reset_stats()


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "OrderManagerConfig",
    "OrderManager",
]