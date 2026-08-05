"""
=========================================================
QuantAI Professional v5
Walk-Forward Results Analyzer
=========================================================

Aggregates results produced by WalkForwardEngine.

This module does NOT:
    - modify Strategy
    - modify TradeEngine
    - modify BacktestEngine
    - run backtests
    - train ML models

It only analyzes already completed
WalkForwardResult objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.walk_forward_engine import WalkForwardResult


# =========================================================
# AGGREGATED RESULT
# =========================================================

@dataclass
class WalkForwardSummary:
    """
    Aggregated statistics for all walk-forward windows.
    """

    total_windows: int

    total_trades: int

    winning_trades: int

    losing_trades: int

    total_profit: float

    initial_balance: float

    final_balance: float

    profitable_windows: int

    losing_windows: int

    win_rate: float

    cumulative_return: float

    max_drawdown: float

    max_drawdown_percent: float


# =========================================================
# ANALYZER
# =========================================================

class WalkForwardAnalyzer:
    """
    Analyze completed walk-forward results.
    """

    def __init__(
        self,
        results: List[WalkForwardResult],
    ) -> None:

        self.results = list(results)

    # =====================================================
    # BASIC METRICS
    # =====================================================

    @property
    def total_windows(self) -> int:

        return len(self.results)

    @property
    def total_trades(self) -> int:

        return sum(
            result.backtest_result.total_trades
            for result in self.results
        )

    @property
    def winning_trades(self) -> int:

        return sum(
            result.backtest_result.winning_trades
            for result in self.results
        )

    @property
    def losing_trades(self) -> int:

        return sum(
            result.backtest_result.losing_trades
            for result in self.results
        )

    # =====================================================
    # PROFIT
    # =====================================================

    @property
    def total_profit(self) -> float:

        return round(
            sum(
                result.backtest_result.net_profit
                for result in self.results
            ),
            2,
        )

    # =====================================================
    # BALANCE
    # =====================================================

    @property
    def initial_balance(self) -> float:

        if not self.results:
            return 0.0

        return float(
            self.results[0]
            .backtest_result
            .initial_balance
        )

    @property
    def final_balance(self) -> float:

        if not self.results:
            return 0.0

        return float(
            self.results[-1]
            .backtest_result
            .final_balance
        )

    # =====================================================
    # CUMULATIVE RETURN
    # =====================================================

    @property
    def cumulative_return(self) -> float:
        """
        Total return from initial balance to final balance.

        Example:

            Initial = 1000
            Final   = 1030

            Return = 3.0%
        """

        if not self.results:
            return 0.0

        if self.initial_balance == 0:
            return 0.0

        return round(
            (
                self.final_balance
                / self.initial_balance
                - 1.0
            )
            * 100.0,
            2,
        )

    # =====================================================
    # DRAW DOWN
    # =====================================================

    @property
    def max_drawdown(self) -> float:
        """
        Maximum absolute drawdown across walk-forward
        equity values.
        """

        if not self.results:
            return 0.0

        peak = self.initial_balance
        max_drawdown = 0.0

        for result in self.results:

            equity = float(
                result.backtest_result.final_balance
            )

            if equity > peak:
                peak = equity

            drawdown = peak - equity

            if drawdown > max_drawdown:
                max_drawdown = drawdown

        return round(
            max_drawdown,
            2,
        )

    # =====================================================
    # DRAW DOWN PERCENT
    # =====================================================

    @property
    def max_drawdown_percent(self) -> float:
        """
        Maximum drawdown expressed as percentage
        of the previous equity peak.
        """

        if not self.results:
            return 0.0

        peak = self.initial_balance
        max_drawdown_percent = 0.0

        for result in self.results:

            equity = float(
                result.backtest_result.final_balance
            )

            if equity > peak:
                peak = equity

            if peak <= 0:
                continue

            drawdown_percent = (
                (peak - equity)
                / peak
                * 100.0
            )

            if drawdown_percent > max_drawdown_percent:
                max_drawdown_percent = drawdown_percent

        return round(
            max_drawdown_percent,
            2,
        )

    # =====================================================
    # WINDOW PERFORMANCE
    # =====================================================

    @property
    def profitable_windows(self) -> int:

        return sum(
            1
            for result in self.results
            if result.backtest_result.net_profit > 0
        )

    @property
    def losing_windows(self) -> int:

        return sum(
            1
            for result in self.results
            if result.backtest_result.net_profit < 0
        )

    # =====================================================
    # WIN RATE
    # =====================================================

    @property
    def win_rate(self) -> float:

        if self.total_trades == 0:
            return 0.0

        return round(
            self.winning_trades
            / self.total_trades
            * 100.0,
            2,
        )

    # =====================================================
    # SUMMARY
    # =====================================================

    def summarize(self) -> WalkForwardSummary:
        """
        Return aggregated walk-forward statistics.
        """

        return WalkForwardSummary(

            total_windows=self.total_windows,

            total_trades=self.total_trades,

            winning_trades=self.winning_trades,

            losing_trades=self.losing_trades,

            total_profit=self.total_profit,

            initial_balance=self.initial_balance,

            final_balance=self.final_balance,

            profitable_windows=self.profitable_windows,

            losing_windows=self.losing_windows,

            win_rate=self.win_rate,

            cumulative_return=self.cumulative_return,

            max_drawdown=self.max_drawdown,

            max_drawdown_percent=self.max_drawdown_percent,
        )


# =========================================================
# EXPORTS
# =========================================================

__all__ = [
    "WalkForwardSummary",
    "WalkForwardAnalyzer",
]