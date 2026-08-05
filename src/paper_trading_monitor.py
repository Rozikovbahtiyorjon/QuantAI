"""
=========================================================
QuantAI Professional v5
Paper Trading Monitor

Read-only monitoring layer for paper trading.

This module does NOT:
    - generate Strategy signals
    - open positions
    - close positions
    - modify PaperTradingEngine
    - modify PaperTradingSession
    - connect to Binance
    - execute real orders

It only observes an existing PaperTradingSession
and exposes monitoring statistics.
=========================================================
"""

from __future__ import annotations

from dataclasses import dataclass

from src.paper_trading_session import (
    PaperTradingSession,
)


# =========================================================
# MONITOR SNAPSHOT
# =========================================================

@dataclass
class PaperTradingMonitorSnapshot:
    """
    Immutable snapshot of the current paper-trading state.
    """

    balance: float

    realized_profit: float

    total_steps: int

    opened_positions: int

    closed_positions: int

    current_signal: str

    has_position: bool

    position_side: str | None

    entry_price: float | None

    quantity: float | None

    total_trades: int

    winning_trades: int

    losing_trades: int

    win_rate: float

    return_percent: float


# =========================================================
# MONITOR
# =========================================================

class PaperTradingMonitor:
    """
    Read-only observer of a PaperTradingSession.
    """

    def __init__(
        self,
        session: PaperTradingSession,
    ) -> None:

        if not isinstance(
            session,
            PaperTradingSession,
        ):
            raise TypeError(
                "session must be PaperTradingSession."
            )

        self.session = session

    # =====================================================
    # BASIC STATE
    # =====================================================

    @property
    def balance(self) -> float:
        """
        Current paper balance.
        """

        return float(
            self.session.balance
        )

    @property
    def realized_profit(self) -> float:
        """
        Current realized profit.
        """

        return float(
            self.session.realized_profit
        )

    @property
    def total_steps(self) -> int:
        """
        Number of processed market-data steps.
        """

        return len(
            self.session.steps
        )

    # =====================================================
    # POSITION
    # =====================================================

    @property
    def has_position(self) -> bool:
        """
        Whether a virtual position is currently open.
        """

        return bool(
            self.session.has_position
        )

    @property
    def position_side(self) -> str | None:
        """
        Current position side.
        """

        position = (
            self.session.runner.engine.position
        )

        if position is None:
            return None

        return position.side

    @property
    def entry_price(self) -> float | None:
        """
        Current position entry price.
        """

        position = (
            self.session.runner.engine.position
        )

        if position is None:
            return None

        return float(
            position.entry_price
        )

    @property
    def quantity(self) -> float | None:
        """
        Current position quantity.
        """

        position = (
            self.session.runner.engine.position
        )

        if position is None:
            return None

        return float(
            position.quantity
        )

    # =====================================================
    # POSITION COUNTERS
    # =====================================================

    @property
    def opened_positions(self) -> int:
        """
        Number of positions opened during the session.
        """

        return sum(
            1
            for step in self.session.steps
            if step.position_opened
        )

    @property
    def closed_positions(self) -> int:
        """
        Number of positions closed during the session.
        """

        return sum(
            1
            for step in self.session.steps
            if step.position_closed
        )

    # =====================================================
    # TRADE STATISTICS
    # =====================================================

    @property
    def total_trades(self) -> int:
        """
        Number of completed paper trades.
        """

        return len(
            self.session.runner.engine.trade_history
        )

    @property
    def winning_trades(self) -> int:
        """
        Number of profitable completed trades.
        """

        return sum(
            1
            for trade
            in self.session.runner.engine.trade_history
            if trade.net_profit > 0
        )

    @property
    def losing_trades(self) -> int:
        """
        Number of losing completed trades.
        """

        return sum(
            1
            for trade
            in self.session.runner.engine.trade_history
            if trade.net_profit < 0
        )

    @property
    def win_rate(self) -> float:
        """
        Percentage of profitable completed trades.
        """

        if self.total_trades == 0:
            return 0.0

        return round(
            self.winning_trades
            / self.total_trades
            * 100.0,
            2,
        )

    # =====================================================
    # RETURN
    # =====================================================

    @property
    def return_percent(self) -> float:
        """
        Realized return relative to initial balance.
        """

        initial_balance = float(
            self.session.runner.engine.initial_balance
        )

        if initial_balance <= 0:
            return 0.0

        return round(
            self.realized_profit
            / initial_balance
            * 100.0,
            2,
        )

    # =====================================================
    # CURRENT SIGNAL
    # =====================================================

    @property
    def current_signal(self) -> str:
        """
        Signal generated at the latest processed step.

        Returns HOLD when no steps have been processed.
        """

        steps = self.session.steps

        if not steps:
            return "HOLD"

        signal = steps[-1].signal

        if signal is None:
            return "HOLD"

        return str(
            signal.signal
        )

    # =====================================================
    # SNAPSHOT
    # =====================================================

    def snapshot(
        self,
    ) -> PaperTradingMonitorSnapshot:
        """
        Return a complete monitoring snapshot.
        """

        return PaperTradingMonitorSnapshot(
            balance=round(
                self.balance,
                8,
            ),

            realized_profit=round(
                self.realized_profit,
                8,
            ),

            total_steps=self.total_steps,

            opened_positions=self.opened_positions,

            closed_positions=self.closed_positions,

            current_signal=self.current_signal,

            has_position=self.has_position,

            position_side=self.position_side,

            entry_price=self.entry_price,

            quantity=self.quantity,

            total_trades=self.total_trades,

            winning_trades=self.winning_trades,

            losing_trades=self.losing_trades,

            win_rate=self.win_rate,

            return_percent=self.return_percent,
        )


# =========================================================
# EXPORTS
# =========================================================

__all__ = [
    "PaperTradingMonitorSnapshot",
    "PaperTradingMonitor",
]
