"""
====================================================
QuantAI Professional
Execution Boundary - Order Definitions
====================================================

Core order types and intents for the execution layer.
====================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_MARKET = "STOP_MARKET"
    STOP_LIMIT = "STOP_LIMIT"
    TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"
    TAKE_PROFIT_LIMIT = "TAKE_PROFIT_LIMIT"


class OrderStatus(str, Enum):
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    PENDING_CANCEL = "PENDING_CANCEL"


class TimeInForce(str, Enum):
    GTC = "GTC"  # Good Till Canceled
    IOC = "IOC"  # Immediate Or Cancel
    FOK = "FOK"  # Fill Or Kill
    GTX = "GTX"  # Post Only (Good Till Crossing)


class OrderIntent(str, Enum):
    """High-level intent for order routing."""
    ENTRY = "ENTRY"           # Open new position
    EXIT = "EXIT"             # Close existing position
    STOP_LOSS = "STOP_LOSS"   # Stop loss order
    TAKE_PROFIT = "TAKE_PROFIT"  # Take profit order
    REBALANCE = "REBALANCE"   # Portfolio rebalancing
    HEDGE = "HEDGE"           # Hedging order


@dataclass(frozen=True)
class OrderIntentData:
    """
    Immutable order intent from strategy/risk layer.
    This is what the execution engine receives.
    """
    intent: OrderIntent
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.GTC
    reduce_only: bool = False
    post_only: bool = False
    
    # Risk context
    risk_decision_id: Optional[str] = None
    strategy_signal_id: Optional[str] = None
    
    # Metadata
    metadata: dict = field(default_factory=dict)
    
    def __post_init__(self):
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.order_type in {OrderType.LIMIT, OrderType.STOP_LIMIT, OrderType.TAKE_PROFIT_LIMIT} and self.price is None:
            raise ValueError(f"{self.order_type} requires price")
        if self.order_type in {OrderType.STOP_MARKET, OrderType.STOP_LIMIT, OrderType.TAKE_PROFIT_MARKET, OrderType.TAKE_PROFIT_LIMIT} and self.stop_price is None:
            raise ValueError(f"{self.order_type} requires stop_price")


@dataclass
class Order:
    """
    Mutable order state tracked by OrderManager.
    """
    # Immutable identity
    intent: OrderIntentData
    order_id: str = field(default_factory=lambda: str(uuid4()))
    client_order_id: str = field(default_factory=lambda: f"quantai_{uuid4().hex[:12]}")
    
    # Exchange info
    exchange_order_id: Optional[str] = None
    exchange: str = "binance"
    
    # State
    status: OrderStatus = OrderStatus.NEW
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    
    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    
    # Error tracking
    error_message: Optional[str] = None
    reject_reason: Optional[str] = None
    
    # Fees
    commission_paid: float = 0.0
    commission_asset: Optional[str] = None
    
    def __post_init__(self):
        if self.client_order_id is None:
            self.client_order_id = f"quantai_{uuid4().hex[:12]}"
    
    @property
    def remaining_quantity(self) -> float:
        return self.intent.quantity - self.filled_quantity
    
    @property
    def is_active(self) -> bool:
        return self.status in {OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED, OrderStatus.PENDING_CANCEL}
    
    @property
    def is_terminal(self) -> bool:
        return self.status in {OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED}
    
    @property
    def fill_ratio(self) -> float:
        if self.intent.quantity == 0:
            return 0.0
        return self.filled_quantity / self.intent.quantity
    
    def update_fill(self, fill_qty: float, fill_price: float, commission: float = 0.0, commission_asset: Optional[str] = None):
        """Update order with new fill."""
        if fill_qty <= 0:
            return
        
        # Weighted average price
        total_value = self.average_fill_price * self.filled_quantity + fill_price * fill_qty
        self.filled_quantity += fill_qty
        self.average_fill_price = total_value / self.filled_quantity
        
        self.commission_paid += commission
        if commission_asset:
            self.commission_asset = commission_asset
        
        self.updated_at = datetime.utcnow()
        
        if self.filled_quantity >= self.intent.quantity:
            self.status = OrderStatus.FILLED
            self.filled_at = datetime.utcnow()
        elif self.filled_quantity > 0:
            self.status = OrderStatus.PARTIALLY_FILLED
    
    def cancel(self):
        """Mark order as canceled."""
        if self.is_active:
            self.status = OrderStatus.CANCELED
            self.updated_at = datetime.utcnow()
    
    def reject(self, reason: str):
        """Mark order as rejected."""
        self.status = OrderStatus.REJECTED
        self.reject_reason = reason
        self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> dict:
        return {
            "order_id": self.order_id,
            "client_order_id": self.client_order_id,
            "exchange_order_id": self.exchange_order_id,
            "exchange": self.exchange,
            "symbol": self.intent.symbol,
            "side": self.intent.side.value,
            "type": self.intent.order_type.value,
            "quantity": self.intent.quantity,
            "price": self.intent.price,
            "stop_price": self.intent.stop_price,
            "time_in_force": self.intent.time_in_force.value,
            "reduce_only": self.intent.reduce_only,
            "post_only": self.intent.post_only,
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "remaining_quantity": self.remaining_quantity,
            "average_fill_price": self.average_fill_price,
            "fill_ratio": self.fill_ratio,
            "commission_paid": self.commission_paid,
            "commission_asset": self.commission_asset,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "error_message": self.error_message,
            "reject_reason": self.reject_reason,
        }


@dataclass
class Fill:
    """Individual fill/trade execution."""
    fill_id: str = field(default_factory=lambda: str(uuid4()))
    order_id: str = ""
    client_order_id: str = ""
    exchange_order_id: str = ""
    exchange_trade_id: str = ""
    
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    quantity: float = 0.0
    price: float = 0.0
    commission: float = 0.0
    commission_asset: str = ""
    
    timestamp: datetime = field(default_factory=datetime.utcnow)
    is_maker: bool = False
    
    def to_dict(self) -> dict:
        return {
            "fill_id": self.fill_id,
            "order_id": self.order_id,
            "client_order_id": self.client_order_id,
            "exchange_order_id": self.exchange_order_id,
            "exchange_trade_id": self.exchange_trade_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "price": self.price,
            "commission": self.commission,
            "commission_asset": self.commission_asset,
            "timestamp": self.timestamp.isoformat(),
            "is_maker": self.is_maker,
        }


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "OrderSide",
    "OrderType",
    "OrderStatus",
    "TimeInForce",
    "OrderIntent",
    "OrderIntentData",
    "Order",
    "Fill",
]