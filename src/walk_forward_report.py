"""
=========================================================
QuantAI Professional v5
Walk-Forward Report

Formats WalkForwardSummary into a readable report.

This module does NOT:
    - run backtests
    - modify Strategy
    - modify TradeEngine
    - modify BacktestEngine
    - calculate indicators
    - train ML models

It only formats already calculated
WalkForwardSummary data.
=========================================================
"""

from __future__ import annotations

from dataclasses import dataclass

from src.walk_forward_analyzer import WalkForwardSummary


# =========================================================
# REPORT
# =========================================================

@dataclass
class WalkForwardReport:
    """
    Human-readable walk-forward report.
    """

    summary: WalkForwardSummary

    # =====================================================
    # TEXT
    # =====================================================

    def to_text(self) -> str:
        """
        Return formatted text report.
        """

        s = self.summary

        lines = [
            "",
            "=" * 60,
            "QUANTAI WALK-FORWARD REPORT",
            "=" * 60,
            "",
            f"Total Windows       : {s.total_windows}",
            f"Total Trades        : {s.total_trades}",
            f"Winning Trades      : {s.winning_trades}",
            f"Losing Trades       : {s.losing_trades}",
            f"Win Rate            : {s.win_rate:.2f}%",
            "",
            f"Initial Balance     : {s.initial_balance:.2f}",
            f"Final Balance       : {s.final_balance:.2f}",
            f"Total Profit        : {s.total_profit:.2f}",
            f"Cumulative Return   : {s.cumulative_return:.2f}%",
            "",
            f"Profitable Windows  : {s.profitable_windows}",
            f"Losing Windows      : {s.losing_windows}",
            "",
            f"Max Drawdown        : {s.max_drawdown:.2f}",
            f"Max Drawdown %      : {s.max_drawdown_percent:.2f}%",
            "",
            "=" * 60,
        ]

        return "\n".join(lines)

    # =====================================================
    # PRINT
    # =====================================================

    def print_report(self) -> None:
        """
        Print formatted report.
        """

        print(self.to_text())


# =========================================================
# CONVENIENCE FUNCTION
# =========================================================

def create_walk_forward_report(
    summary: WalkForwardSummary,
) -> WalkForwardReport:
    """
    Create a WalkForwardReport from a summary.
    """

    return WalkForwardReport(summary=summary)


# =========================================================
# EXPORTS
# =========================================================

__all__ = [
    "WalkForwardReport",
    "create_walk_forward_report",
]