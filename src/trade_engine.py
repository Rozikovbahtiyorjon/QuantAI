"""
====================================================
QuantAI Professional v3.0
Professional Trade Engine
====================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import pandas as pd

from config.settings import (
    INITIAL_BALANCE,
    RISK_PERCENT,
    COMMISSION,
    SLIPPAGE,
    MAX_OPEN_POSITIONS,
)

from src.strategy import (
    SignalResult,
    generate_signal_result,
)

from src.risk_manager import (
    calculate_position_size,
)

# ====================================================
# Position Side
# ====================================================

class PositionSide(str, Enum):

    LONG = "BUY"

    SHORT = "SELL"


# ====================================================
# Position Status
# ====================================================

class PositionStatus(str, Enum):

    OPEN = "OPEN"

    CLOSED = "CLOSED"


# ====================================================
# Close Reason
# ====================================================

class CloseReason(str, Enum):

    TAKE_PROFIT = "TAKE_PROFIT"

    STOP_LOSS = "STOP_LOSS"

    TRAILING_STOP = "TRAILING_STOP"

    BREAK_EVEN = "BREAK_EVEN"

    MANUAL = "MANUAL"

    END_OF_BACKTEST = "END_OF_BACKTEST"


# ====================================================
# Position
# ====================================================

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


# ====================================================
# Trade Engine
# ====================================================

class TradeEngine:

    def __init__(self):

        self.balance = INITIAL_BALANCE

        self.equity = INITIAL_BALANCE

        self.positions: List[Position] = []

        self.closed_positions: List[Position] = []

        self.position_counter = 0

        # ====================================================
# Engine Helpers
# ====================================================

    def next_position_id(self) -> int:

        self.position_counter += 1

        return self.position_counter


    def can_open_position(self) -> bool:

        return len(self.positions) < MAX_OPEN_POSITIONS


    def get_open_positions(self):

        return [
            p
            for p in self.positions
            if p.status == PositionStatus.OPEN
        ]


# ====================================================
# Commission
# ====================================================

    def calculate_commission(
        self,
        quantity: float,
        price: float,
    ) -> float:

        return quantity * price * COMMISSION


# ====================================================
# Slippage
# ====================================================

    def apply_slippage(
        self,
        side: PositionSide,
        price: float,
    ) -> float:

        if side == PositionSide.LONG:

            return price * (1 + SLIPPAGE)

        return price * (1 - SLIPPAGE)


# ====================================================
# Open Position
# ====================================================

    def open_position(

        self,

        candle,

        signal: SignalResult,

    ) -> bool:

        if not self.can_open_position():

            return False

        entry_price = self.apply_slippage(

            PositionSide(signal.signal),

            signal.entry,

        )

        quantity = calculate_position_size(

            balance=self.balance,

            risk_percent=RISK_PERCENT,

            entry_price=entry_price,

            stop_loss=signal.stop_loss,

        )

        commission = self.calculate_commission(

            quantity,

            entry_price,

        )

        position = Position(

            id=self.next_position_id(),

            side=PositionSide(signal.signal),

            status=PositionStatus.OPEN,

            entry_time=candle["timestamp"],

            entry_price=round(entry_price, 2),

            stop_loss=round(signal.stop_loss, 2),

            take_profit=round(signal.take_profit, 2),

            quantity=round(quantity, 6),

            confidence=signal.confidence,

            reason_open=list(signal.reasons),

            commission=commission,

        )

        self.positions.append(position)

        return True

        # ====================================================
# Position Update
# ====================================================

    def update_position(

        self,

        position: Position,

        candle,

    ):

        position.bars_open += 1

        high = float(candle["high"])

        low = float(candle["low"])

        close = float(candle["close"])

        atr = float(candle["atr"])

        # ============================================
        # Floating Profit / Drawdown
        # ============================================

        if position.side == PositionSide.LONG:

            floating = (
                close -
                position.entry_price
            ) * position.quantity

        else:

            floating = (
                position.entry_price -
                close
            ) * position.quantity

        position.max_profit = max(
            position.max_profit,
            floating,
        )

        position.max_drawdown = min(
            position.max_drawdown,
            floating,
        )

        # ============================================
        # Break Even
        # ============================================

        if position.side == PositionSide.LONG:

            trigger = (
                position.entry_price +
                atr
            )

            if (
                high >= trigger
                and position.stop_loss < position.entry_price
            ):

                position.stop_loss = round(
                    position.entry_price,
                    2,
                )

        else:

            trigger = (
                position.entry_price -
                atr
            )

            if (
                low <= trigger
                and position.stop_loss > position.entry_price
            ):

                position.stop_loss = round(
                    position.entry_price,
                    2,
                )

        # ============================================
        # Trailing Stop
        # ============================================

        trail = atr * 2.0

        if position.side == PositionSide.LONG:

            new_stop = close - trail

            if new_stop > position.stop_loss:

                position.stop_loss = round(
                    new_stop,
                    2,
                )

        else:

            new_stop = close + trail

            if new_stop < position.stop_loss:

                position.stop_loss = round(
                    new_stop,
                    2,
                )

        # ============================================
        # Exit Conditions
        # ============================================

        if position.side == PositionSide.LONG:

            if low <= position.stop_loss:

                self.close_position(

                    position,

                    candle,

                    position.stop_loss,

                    CloseReason.STOP_LOSS,

                )

                return

            if high >= position.take_profit:

                self.close_position(

                    position,

                    candle,

                    position.take_profit,

                    CloseReason.TAKE_PROFIT,

                )

                return

        else:

            if high >= position.stop_loss:

                self.close_position(

                    position,

                    candle,

                    position.stop_loss,

                    CloseReason.STOP_LOSS,

                )

                return

            if low <= position.take_profit:

                self.close_position(

                    position,

                    candle,

                    position.take_profit,

                    CloseReason.TAKE_PROFIT,

                )

                return

                # ====================================================
# Close Position
# ====================================================

    def close_position(

        self,

        position: Position,

        candle,

        exit_price: float,

        reason: CloseReason,

    ):

        # ============================================
        # Exit price with slippage
        # ============================================

        exit_price = self.apply_slippage(

            position.side,

            exit_price,

        )

        position.exit_time = candle["timestamp"]

        position.exit_price = round(
            exit_price,
            2,
        )

        position.reason_close = reason

        position.status = PositionStatus.CLOSED

        # ============================================
        # Gross Profit
        # ============================================

        if position.side == PositionSide.LONG:

            gross_profit = (

                exit_price

                - position.entry_price

            ) * position.quantity

        else:

            gross_profit = (

                position.entry_price

                - exit_price

            ) * position.quantity

        position.gross_profit = round(
            gross_profit,
            2,
        )

        # ============================================
        # Exit commission
        # ============================================

        exit_commission = self.calculate_commission(

            position.quantity,

            exit_price,

        )

        position.commission += exit_commission

        position.commission = round(
            position.commission,
            4,
        )

        # ============================================
        # Net Profit
        # ============================================

        position.net_profit = round(

            position.gross_profit

            - position.commission,

            2,

        )

        # ============================================
        # Update Balance / Equity
        # ============================================

        self.balance = round(

            self.balance

            + position.net_profit,

            2,

        )

        position.balance_after_close = self.balance
        
        self.equity = self.balance

        # ============================================
        # Archive Position
        # ============================================

        self.closed_positions.append(position)

        if position in self.positions:

            self.positions.remove(position)

        # ============================================
        # Optional Debug
        # ============================================

        print(

            f"[{position.id}] "

            f"{position.side.value} "

            f"{position.reason_close.value} "

            f"PnL={position.net_profit:.2f}$ "

            f"Balance={self.balance:.2f}$"

        )

        # ====================================================
# Main Engine Loop
# ====================================================

    def run(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        start_index = 250

        for i in range(start_index, len(df)):

            history = df.iloc[: i + 1]

            candle = history.iloc[-1]

            # ========================================
            # Update all open positions
            # ========================================

            for position in self.get_open_positions():

                self.update_position(
                    position,
                    candle,
                )

            # ========================================
            # Maximum simultaneous positions
            # ========================================

            if not self.can_open_position():
                continue

            # ========================================
            # Strategy Signal
            # ========================================

            signal = generate_signal_result(
                history,
            )

            if signal.signal == "HOLD":
                continue

            # ========================================
            # Open New Position
            # ========================================

            self.open_position(
                candle,
                signal,
            )

        # ============================================
        # Close remaining positions
        # ============================================

        if len(self.get_open_positions()) > 0:

            last_candle = df.iloc[-1]

            for position in self.get_open_positions()[:]:

                self.close_position(

                    position,

                    last_candle,

                    float(last_candle["close"]),

                    CloseReason.END_OF_BACKTEST,

                )

        return self.to_dataframe()


# ====================================================
# Convert Trades to DataFrame
# ====================================================

    def to_dataframe(self) -> pd.DataFrame:

        rows = []

        for trade in self.closed_positions:

            rows.append({

                "id": trade.id,

                "side": trade.side.value,

                "entry_time": trade.entry_time,

                "exit_time": trade.exit_time,

                "entry": round(trade.entry_price, 2),

                "exit": round(trade.exit_price, 2),

                "stop_loss": round(trade.stop_loss, 2),

                "take_profit": round(trade.take_profit, 2),

                "quantity": round(trade.quantity, 6),

                "confidence": round(trade.confidence, 2),

                "bars": trade.bars_open,

                "gross_profit": round(trade.gross_profit, 2),

                "commission": round(trade.commission, 4),

                "net_profit": round(trade.net_profit, 2),

                "balance": round(
                    trade.balance_after_close,
                    2,
                ),

                "close_reason": trade.reason_close.value,

            })

        if len(rows) == 0:

            return pd.DataFrame(columns=[

                "id",

                "side",

                "entry_time",

                "exit_time",

                "entry",

                "exit",

                "stop_loss",

                "take_profit",

                "quantity",

                "confidence",

                "bars",

                "gross_profit",

                "commission",

                "net_profit",

                "balance",

                "close_reason",

            ])

        return pd.DataFrame(rows)

        # ====================================================
# Engine Statistics
# ====================================================

    @property
    def total_trades(self) -> int:

        return len(self.closed_positions)


    @property
    def winning_trades(self) -> int:

        return sum(

            1

            for p in self.closed_positions

            if p.net_profit > 0

        )


    @property
    def losing_trades(self) -> int:

        return sum(

            1

            for p in self.closed_positions

            if p.net_profit <= 0

        )


    @property
    def total_profit(self) -> float:

        return round(

            sum(

                p.net_profit

                for p in self.closed_positions

            ),

            2,

        )


    @property
    def win_rate(self) -> float:

        if self.total_trades == 0:

            return 0.0

        return round(

            self.winning_trades

            / self.total_trades

            * 100,

            2,

        )


# ====================================================
# Public Runner
# ====================================================

def run_trade_engine(

    df: pd.DataFrame,

) -> pd.DataFrame:

    engine = TradeEngine()

    trades = engine.run(df)

    print()

    print("=" * 60)

    print("TRADE ENGINE REPORT")

    print("=" * 60)

    print(f"Initial Balance : {INITIAL_BALANCE:.2f}")

    print(f"Final Balance   : {engine.balance:.2f}")

    print(f"Net Profit      : {engine.total_profit:.2f}")

    print(f"Trades          : {engine.total_trades}")

    print(f"Wins            : {engine.winning_trades}")

    print(f"Losses          : {engine.losing_trades}")

    print(f"Win Rate        : {engine.win_rate:.2f}%")

    print("=" * 60)

    return trades


# ====================================================
# Module Export
# ====================================================

__all__ = [

    "Position",

    "PositionSide",

    "PositionStatus",

    "CloseReason",

    "TradeEngine",

    "run_trade_engine",

]