"""
Position model — extracted from src/trade_engine.py (Audit §43 split)
Canonical location for Position, PositionSide, PositionStatus, CloseReason.
TradeEngine re-exports for backward compat.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional


class PositionSide(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    BUY = "BUY"
    SELL = "SELL"


class PositionStatus(str, Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"


class CloseReason(str, Enum):
    TAKE_PROFIT = "TAKE_PROFIT"
    STOP_LOSS = "STOP_LOSS"
    TRAILING_STOP = "TRAILING_STOP"
    BREAK_EVEN = "BREAK_EVEN"
    TIME_EXIT = "TIME_EXIT"
    END_OF_BACKTEST = "END_OF_BACKTEST"
    MANUAL = "MANUAL"
    LIQUIDATION = "LIQUIDATION"


@dataclass
class Position:
    id: int
    side: PositionSide
    status: PositionStatus
    entry_time: object
    exit_time: Optional[object] = None
    entry_price: float = 0.0
    exit_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    quantity: float = 0.0
    confidence: float = 0.0
    reason_open: List[str] = field(default_factory=list)
    reason_close: CloseReason = CloseReason.MANUAL
    commission: float = 0.0
    gross_profit: float = 0.0
    net_profit: float = 0.0
    balance_after_close: float = 0.0
    max_profit: float = 0.0
    max_drawdown: float = 0.0
    bars_open: int = 0
    break_even_activated: bool = False
    trailing_activated: bool = False
