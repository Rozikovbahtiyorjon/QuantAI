"""
=========================================================
QuantAI Professional v5
Walk-Forward Report

Performance analysis layer for WalkForwardResult.

Responsibilities
----------------

WalkForwardReport:

    - analyzes completed Walk-Forward results
    - calculates window statistics
    - calculates balance statistics
    - calculates profit statistics
    - calculates trade statistics
    - calculates profit factor
    - calculates consistency score
    - identifies best/worst windows
    - produces structured summaries
    - prints a concise terminal report

IMPORTANT
---------

This module does NOT:

    - generate Walk-Forward windows
    - run BacktestEngine
    - train ML models
    - execute trades
    - connect to Binance
    - modify Strategy

It only analyzes an already completed
WalkForwardResult.
=========================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Any, Dict, List, Optional

from src.walk.walk_forward_engine import (
    WalkForwardResult,
    WalkForwardWindowResult,
)


# =========================================================
# REPORT RESULT
# =========================================================


@dataclass
class WalkForwardReportResult:
    """
    Structured summary of a WalkForwardResult.
    """

    total_windows: int

    profitable_windows: int

    losing_windows: int

    flat_windows: int

    initial_balance: float

    final_balance: float

    net_profit: float

    cumulative_return: float

    total_trades: int

    winning_trades: int

    losing_trades: int

    win_rate: float

    average_window_profit: float

    best_window_profit: float

    worst_window_profit: float

    average_trade_profit: float

    profit_factor: float

    consistency_score: float


# =========================================================
# REPORT
# =========================================================


class WalkForwardReport:
    """
    Analyze a completed WalkForwardResult.

    Example
    -------

        result = engine.run(df)

        report = WalkForwardReport(result)

        summary = report.summarize()

        WalkForwardReport.print_report(summary)
    """

    # =====================================================
    # CONSTRUCTOR
    # =====================================================

    def __init__(
        self,
        result: WalkForwardResult,
    ) -> None:

        if not isinstance(
            result,
            WalkForwardResult,
        ):
            raise TypeError(
                "WalkForwardReport requires "
                "a WalkForwardResult."
            )

        self.result = result

    # =====================================================
    # INTERNAL WINDOW ACCESS
    # =====================================================

    @property
    def windows(
        self,
    ) -> List[WalkForwardWindowResult]:
        """
        Return all Walk-Forward windows.
        """

        return self.result.windows

    # =====================================================
    # WINDOW COUNTS
    # =====================================================

    @property
    def total_windows(
        self,
    ) -> int:
        """
        Number of Walk-Forward windows.
        """

        return len(
            self.windows
        )

    @property
    def profitable_windows(
        self,
    ) -> int:
        """
        Number of windows with positive profit.
        """

        return sum(
            1
            for window in self.windows
            if window.backtest_result.net_profit > 0
        )

    @property
    def losing_windows(
        self,
    ) -> int:
        """
        Number of windows with negative profit.
        """

        return sum(
            1
            for window in self.windows
            if window.backtest_result.net_profit < 0
        )

    @property
    def flat_windows(
        self,
    ) -> int:
        """
        Number of windows with zero profit.
        """

        return sum(
            1
            for window in self.windows
            if window.backtest_result.net_profit == 0
        )

    @property
    def window_win_rate(
        self,
    ) -> float:
        """
        Percentage of profitable Walk-Forward windows.

        Returns 0.0 when no windows exist.
        """

        if self.total_windows == 0:
            return 0.0

        return round(
            (
                self.profitable_windows
                / self.total_windows
            )
            * 100.0,
            2,
        )

    # =====================================================
    # BALANCE
    # =====================================================

    @property
    def initial_balance(
        self,
    ) -> float:
        """
        Initial account balance.
        """

        return float(
            self.result.initial_balance
        )

    @property
    def final_balance(
        self,
    ) -> float:
        """
        Final account balance.
        """

        return float(
            self.result.final_balance
        )

    # =====================================================
    # PROFIT
    # =====================================================

    @property
    def net_profit(
        self,
    ) -> float:
        """
        Total net profit.
        """

        return float(
            self.result.net_profit
        )

    @property
    def cumulative_return(
        self,
    ) -> float:
        """
        Cumulative return in percent.

        Example:

            initial = 1000
            final   = 1020

            return = 2.0
        """

        if self.initial_balance == 0:
            return 0.0

        return round(
            (
                self.net_profit
                / self.initial_balance
            )
            * 100.0,
            8,
        )

    @property
    def window_profits(
        self,
    ) -> List[float]:
        """
        Return net profit for every window.
        """

        return [
            float(
                window
                .backtest_result
                .net_profit
            )
            for window in self.windows
        ]

    @property
    def average_window_profit(
        self,
    ) -> float:
        """
        Average profit per Walk-Forward window.
        """

        if self.total_windows == 0:
            return 0.0

        return (
            sum(
                self.window_profits
            )
            / self.total_windows
        )

    @property
    def best_window_profit(
        self,
    ) -> float:
        """
        Highest single-window profit.
        """

        if not self.window_profits:
            return 0.0

        return max(
            self.window_profits
        )

    @property
    def worst_window_profit(
        self,
    ) -> float:
        """
        Lowest single-window profit.
        """

        if not self.window_profits:
            return 0.0

        return min(
            self.window_profits
        )

    # =====================================================
    # TRADE STATISTICS
    # =====================================================

    @property
    def total_trades(
        self,
    ) -> int:
        """
        Total number of trades across all windows.
        """

        return sum(
            window.backtest_result.total_trades
            for window in self.windows
        )

    @property
    def winning_trades(
        self,
    ) -> int:
        """
        Total winning trades.
        """

        return sum(
            window.backtest_result.winning_trades
            for window in self.windows
        )

    @property
    def losing_trades(
        self,
    ) -> int:
        """
        Total losing trades.
        """

        return sum(
            window.backtest_result.losing_trades
            for window in self.windows
        )

    @property
    def win_rate(
        self,
    ) -> float:
        """
        Aggregate trade win rate.
        """

        if self.total_trades == 0:
            return 0.0

        return round(
            (
                self.winning_trades
                / self.total_trades
            )
            * 100.0,
            2,
        )

    @property
    def average_trade_profit(
        self,
    ) -> float:
        """
        Average net profit per trade.
        """

        if self.total_trades == 0:
            return 0.0

        return (
            self.net_profit
            / self.total_trades
        )

    # =====================================================
    # PROFIT FACTOR
    # =====================================================

    @staticmethod
    def _extract_trade_profit(
        trade: Any,
    ) -> Optional[float]:
        """
        Extract net profit from a trade.

        Supports:

            {"net_profit": 10.0}

        and object-style trades exposing:

            trade.net_profit
        """

        if isinstance(
            trade,
            dict,
        ):

            value = trade.get(
                "net_profit"
            )

            if value is None:
                return None

            try:
                return float(value)
            except (
                TypeError,
                ValueError,
            ):
                return None

        value = getattr(
            trade,
            "net_profit",
            None,
        )

        if value is None:
            return None

        try:
            return float(value)
        except (
            TypeError,
            ValueError,
        ):
            return None

    @property
    def profit_factor(
        self,
    ) -> float:
        """
        Calculate profit factor.

        Profit Factor =
            Gross Profit / Gross Loss

        If there is profit but no loss:
            infinity

        If there is no profit:
            0.0
        """

        gross_profit = 0.0

        gross_loss = 0.0

        for window in self.windows:

            trades = (
                window
                .backtest_result
                .trades
            )

            for trade in trades:

                profit = (
                    self._extract_trade_profit(
                        trade
                    )
                )

                if profit is None:
                    continue

                if profit > 0:

                    gross_profit += profit

                elif profit < 0:

                    gross_loss += abs(
                        profit
                    )

        if gross_profit <= 0:
            return 0.0

        if gross_loss <= 0:
            return inf

        return round(
            gross_profit
            / gross_loss,
            8,
        )

    # =====================================================
    # CONSISTENCY
    # =====================================================

    @property
    def consistency_score(
        self,
    ) -> float:
        """
        Percentage of profitable windows.

        This metric intentionally represents
        consistency across Walk-Forward windows,
        not individual trade win rate.
        """

        if self.total_windows == 0:
            return 0.0

        return round(
            (
                self.profitable_windows
                / self.total_windows
            )
            * 100.0,
            2,
        )

    # =====================================================
    # WINDOW SUMMARY
    # =====================================================

    def window_summary(
        self,
    ) -> List[Dict[str, Any]]:
        """
        Return detailed information for every window.
        """

        summary: List[
            Dict[str, Any]
        ] = []

        for window in self.windows:

            backtest = (
                window.backtest_result
            )

            summary.append(
                {
                    "window_id": (
                        window.window_id
                    ),
                    "train_start": (
                        window.train_start
                    ),
                    "train_end": (
                        window.train_end
                    ),
                    "test_start": (
                        window.test_start
                    ),
                    "test_end": (
                        window.test_end
                    ),
                    "train_size": (
                        window.train_size
                    ),
                    "test_size": (
                        window.test_size
                    ),
                    "initial_balance": (
                        backtest.initial_balance
                    ),
                    "final_balance": (
                        backtest.final_balance
                    ),
                    "net_profit": (
                        backtest.net_profit
                    ),
                    "total_trades": (
                        backtest.total_trades
                    ),
                    "winning_trades": (
                        backtest.winning_trades
                    ),
                    "losing_trades": (
                        backtest.losing_trades
                    ),
                    "win_rate": (
                        backtest.win_rate
                    ),
                }
            )

        return summary

    # =====================================================
    # BEST / WORST WINDOW
    # =====================================================

    @property
    def best_window(
        self,
    ) -> Optional[WalkForwardWindowResult]:
        """
        Return the most profitable window.

        Returns None when no windows exist.
        """

        if not self.windows:
            return None

        return max(
            self.windows,
            key=lambda window:
                window
                .backtest_result
                .net_profit,
        )

    @property
    def worst_window(
        self,
    ) -> Optional[WalkForwardWindowResult]:
        """
        Return the least profitable window.

        Returns None when no windows exist.
        """

        if not self.windows:
            return None

        return min(
            self.windows,
            key=lambda window:
                window
                .backtest_result
                .net_profit,
        )

    # =====================================================
    # SUMMARY
    # =====================================================

    def summarize(
        self,
    ) -> WalkForwardReportResult:
        """
        Build a structured Walk-Forward report summary.
        """

        return WalkForwardReportResult(
            total_windows=(
                self.total_windows
            ),

            profitable_windows=(
                self.profitable_windows
            ),

            losing_windows=(
                self.losing_windows
            ),

            flat_windows=(
                self.flat_windows
            ),

            initial_balance=(
                self.initial_balance
            ),

            final_balance=(
                self.final_balance
            ),

            net_profit=(
                self.net_profit
            ),

            cumulative_return=(
                self.cumulative_return
            ),

            total_trades=(
                self.total_trades
            ),

            winning_trades=(
                self.winning_trades
            ),

            losing_trades=(
                self.losing_trades
            ),

            win_rate=(
                self.win_rate
            ),

            average_window_profit=(
                self.average_window_profit
            ),

            best_window_profit=(
                self.best_window_profit
            ),

            worst_window_profit=(
                self.worst_window_profit
            ),

            average_trade_profit=(
                self.average_trade_profit
            ),

            profit_factor=(
                self.profit_factor
            ),

            consistency_score=(
                self.consistency_score
            ),
        )

    # =====================================================
    # PRINT REPORT
    # =====================================================

    @staticmethod
    def print_report(
        summary: WalkForwardReportResult,
    ) -> None:
        """
        Print a compact Walk-Forward performance report.

        The output is intentionally kept compact so that
        it fits comfortably inside a normal terminal.
        """

        if not isinstance(
            summary,
            WalkForwardReportResult,
        ):
            raise TypeError(
                "print_report requires "
                "a WalkForwardReportResult."
            )

        print()

        print("=" * 70)
        print(
            "QUANTAI WALK-FORWARD "
            "PERFORMANCE REPORT"
        )
        print("=" * 70)

        print(
            f"Initial Balance   : "
            f"{summary.initial_balance:.2f}"
        )

        print(
            f"Final Balance     : "
            f"{summary.final_balance:.2f}"
        )

        print(
            f"Net Profit        : "
            f"{summary.net_profit:.2f}"
        )

        print(
            f"Cumulative Return : "
            f"{summary.cumulative_return:.2f}%"
        )

        print("-" * 70)

        print(
            f"Windows           : "
            f"{summary.total_windows}"
        )

        print(
            f"Profitable        : "
            f"{summary.profitable_windows}"
        )

        print(
            f"Losing            : "
            f"{summary.losing_windows}"
        )

        print(
            f"Flat              : "
            f"{summary.flat_windows}"
        )

        print(
            f"Consistency Score  : "
            f"{summary.consistency_score:.2f}%"
        )

        print("-" * 70)

        print(
            f"Total Trades      : "
            f"{summary.total_trades}"
        )

        print(
            f"Winning Trades    : "
            f"{summary.winning_trades}"
        )

        print(
            f"Losing Trades     : "
            f"{summary.losing_trades}"
        )

        print(
            f"Win Rate          : "
            f"{summary.win_rate:.2f}%"
        )

        print(
            f"Avg Trade Profit  : "
            f"{summary.average_trade_profit:.2f}"
        )

        print(
            f"Profit Factor     : "
            f"{summary.profit_factor}"
        )

        print("-" * 70)

        print(
            f"Avg Window Profit : "
            f"{summary.average_window_profit:.2f}"
        )

        print(
            f"Best Window       : "
            f"{summary.best_window_profit:.2f}"
        )

        print(
            f"Worst Window      : "
            f"{summary.worst_window_profit:.2f}"
        )

        print("=" * 70)


# =========================================================
# CONVENIENCE FUNCTION
# =========================================================


def create_walk_forward_report(
    result: WalkForwardResult,
) -> WalkForwardReportResult:
    """
    Create a structured Walk-Forward report.

    Parameters
    ----------
    result:
        Completed WalkForwardResult.

    Returns
    -------
    WalkForwardReportResult
        Calculated performance summary.
    """

    report = WalkForwardReport(
        result
    )

    return report.summarize()


# =========================================================
# EXPORTS
# =========================================================

__all__ = [
    "WalkForwardReport",
    "WalkForwardReportResult",
    "create_walk_forward_report",
]
