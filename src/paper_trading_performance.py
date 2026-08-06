"""
=========================================================
QuantAI Professional v5
Paper Trading Performance Analytics

Analyzes completed PaperTradingSessionResult objects.

This module does NOT:
    - generate Strategy signals
    - execute trades
    - modify PaperTradingEngine
    - modify PaperTradingSession
    - connect to Binance
    - train ML models

It only calculates performance and risk metrics
from completed paper-trading results.
=========================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import List

from src.paper_trading_engine import PaperTrade
from src.paper_trading_session import PaperTradingSessionResult


# =========================================================
# PERFORMANCE RESULT
# =========================================================

@dataclass
class PaperTradingPerformanceResult:
    """
    Aggregated performance and risk statistics.
    """

    total_trades: int

    winning_trades: int

    losing_trades: int

    win_rate: float

    total_profit: float

    average_trade: float

    average_win: float

    average_loss: float

    profit_factor: float

    cumulative_return: float

    max_drawdown: float

    max_drawdown_percent: float


# =========================================================
# PERFORMANCE ANALYZER
# =========================================================

class PaperTradingPerformance:
    """
    Analyze a completed paper-trading session.
    """

    def __init__(
        self,
        session_result: PaperTradingSessionResult,
    ) -> None:

        if not isinstance(
            session_result,
            PaperTradingSessionResult,
        ):
            raise TypeError(
                "session_result must be "
                "PaperTradingSessionResult."
            )

        self.session_result = session_result

        self.trades: List[PaperTrade] = [
            step.trade
            for step in session_result.steps
            if step.trade is not None
        ]

    # =====================================================
    # BASIC TRADE COUNTS
    # =====================================================

    @property
    def total_trades(self) -> int:
        """
        Number of completed trades.
        """

        return len(self.trades)

    @property
    def winning_trades(self) -> int:
        """
        Number of profitable trades.
        """

        return sum(
            1
            for trade in self.trades
            if trade.net_profit > 0
        )

    @property
    def losing_trades(self) -> int:
        """
        Number of losing trades.
        """

        return sum(
            1
            for trade in self.trades
            if trade.net_profit < 0
        )

    # =====================================================
    # WIN RATE
    # =====================================================

    @property
    def win_rate(self) -> float:
        """
        Percentage of profitable trades.
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
    # PROFIT
    # =====================================================

    @property
    def total_profit(self) -> float:
        """
        Total realized net profit.
        """

        return round(
            sum(
                trade.net_profit
                for trade in self.trades
            ),
            8,
        )

    @property
    def average_trade(self) -> float:
        """
        Average net profit per completed trade.
        """
    
        if self.total_trades == 0:
            return 0.0
    
        return (
            self.total_profit
            / self.total_trades
        )
    
    
    @property
    def average_win(self) -> float:
        """
        Average net profit of winning trades.
        """
    
        winners = [
            trade.net_profit
            for trade in self.trades
            if trade.net_profit > 0
        ]
    
        if not winners:
            return 0.0
    
        return (
            sum(winners)
            / len(winners)
        )
    
    
    @property
    def average_loss(self) -> float:
        """
        Average net loss of losing trades.
    
        Returned as a negative number.
        """
    
        losers = [
            trade.net_profit
            for trade in self.trades
            if trade.net_profit < 0
        ]
    
        if not losers:
            return 0.0
    
        return (
            sum(losers)
            / len(losers)
        )
    
    # =====================================================
    # PROFIT FACTOR
    # =====================================================

    @property
    def profit_factor(self) -> float:
        """
        Gross profits divided by gross losses.

        If there are profits but no losses,
        returns infinity.

        If there are no trades or no profits,
        returns 0.0.
        """

        gross_profit = sum(
            trade.net_profit
            for trade in self.trades
            if trade.net_profit > 0
        )

        gross_loss = abs(
            sum(
                trade.net_profit
                for trade in self.trades
                if trade.net_profit < 0
            )
        )

        if gross_profit == 0:
            return 0.0

        if gross_loss == 0:
            return inf

        return round(
            gross_profit
            / gross_loss,
            8,
        )

    # =====================================================
    # CUMULATIVE RETURN
    # =====================================================

    @property
    def cumulative_return(self) -> float:
        """
        Total return in percentage.

        Example:

            initial = 1000
            final   = 1030

            return = 3.0%
        """

        initial = float(
            self.session_result.initial_balance
        )

        final = float(
            self.session_result.final_balance
        )

        if initial <= 0:
            return 0.0

        return round(
            (
                (final - initial)
                / initial
            )
            * 100.0,
            2,
        )

    # =====================================================
    # EQUITY CURVE
    # =====================================================

    @property
    def equity_curve(self) -> List[float]:
        """
        Build realized equity curve.

        The first value is the initial balance.
        Every subsequent value adds the trade's
        realized net profit.
        """

        equity = float(
            self.session_result.initial_balance
        )

        curve = [equity]

        for trade in self.trades:

            equity += trade.net_profit

            curve.append(
                round(equity, 8)
            )

        return curve

    # =====================================================
    # MAX DRAWDOWN
    # =====================================================

    @property
    def max_drawdown(self) -> float:
        """
        Maximum absolute equity drawdown.

        Returned as a positive monetary value.

        Example:

            peak = 1020
            trough = 980

            max_drawdown = 40
        """

        curve = self.equity_curve

        if not curve:
            return 0.0

        peak = curve[0]
        max_drawdown = 0.0

        for equity in curve:

            if equity > peak:
                peak = equity

            drawdown = peak - equity

            if drawdown > max_drawdown:
                max_drawdown = drawdown

        return round(
            max_drawdown,
            8,
        )

    # =====================================================
    # MAX DRAWDOWN %
    # =====================================================

    @property
    def max_drawdown_percent(self) -> float:
        """
        Maximum drawdown as a percentage
        of the corresponding equity peak.
        """

        curve = self.equity_curve

        if not curve:
            return 0.0

        peak = curve[0]
        max_drawdown_percent = 0.0

        for equity in curve:

            if equity > peak:
                peak = equity

            if peak <= 0:
                continue

            drawdown_percent = (
                (peak - equity)
                / peak
                * 100.0
            )

            if (
                drawdown_percent
                > max_drawdown_percent
            ):
                max_drawdown_percent = (
                    drawdown_percent
                )

        return round(
            max_drawdown_percent,
            2,
        )

    # =====================================================
    # SUMMARY
    # =====================================================

    def summarize(
        self,
    ) -> PaperTradingPerformanceResult:
        """
        Return complete performance statistics.
        """

        return PaperTradingPerformanceResult(
            total_trades=self.total_trades,

            winning_trades=self.winning_trades,

            losing_trades=self.losing_trades,

            win_rate=self.win_rate,

            total_profit=self.total_profit,

            average_trade=self.average_trade,

            average_win=self.average_win,

            average_loss=self.average_loss,

            profit_factor=self.profit_factor,

            cumulative_return=self.cumulative_return,

            max_drawdown=self.max_drawdown,

            max_drawdown_percent=(
                self.max_drawdown_percent
            ),
        )


# =========================================================
# EXPORTS
# =========================================================

__all__ = [
    "PaperTradingPerformanceResult",
    "PaperTradingPerformance",
]
