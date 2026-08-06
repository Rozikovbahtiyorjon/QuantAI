"""
=========================================================
QuantAI Professional v5
Paper Trading Report

Generates human-readable reports from
PaperTradingPerformance analytics.

This module does NOT:
    - generate Strategy signals
    - execute trades
    - modify PaperTradingEngine
    - modify PaperTradingSession
    - modify PaperTradingPerformance
    - connect to Binance
    - train ML models

It only converts performance analytics
into structured and text reports.
=========================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from src.paper_trading_performance import (
    PaperTradingPerformance,
    PaperTradingPerformanceResult,
)


# =========================================================
# REPORT RESULT
# =========================================================

@dataclass
class PaperTradingReportResult:
    """
    Structured paper-trading report.
    """

    performance: PaperTradingPerformanceResult

    status: str

    summary: str

    metrics: Dict[str, float]


# =========================================================
# REPORT
# =========================================================

class PaperTradingReport:
    """
    Generate reports from PaperTradingPerformance.
    """

    def __init__(
        self,
        performance: PaperTradingPerformance,
    ) -> None:

        if not isinstance(
            performance,
            PaperTradingPerformance,
        ):
            raise TypeError(
                "performance must be "
                "PaperTradingPerformance."
            )

        self.performance = performance

    # =====================================================
    # STATUS
    # =====================================================

    @property
    def status(self) -> str:
        """
        Determine overall performance status.

        PROFITABLE:
            total profit > 0

        LOSS:
            total profit < 0

        BREAK_EVEN:
            total profit == 0
        """

        profit = self.performance.total_profit

        if profit > 0:
            return "PROFITABLE"

        if profit < 0:
            return "LOSS"

        return "BREAK_EVEN"

    # =====================================================
    # METRICS
    # =====================================================

    @property
    def metrics(self) -> Dict[str, float]:
        """
        Return key performance metrics.
        """

        return {
            "total_trades": float(
                self.performance.total_trades
            ),
            "winning_trades": float(
                self.performance.winning_trades
            ),
            "losing_trades": float(
                self.performance.losing_trades
            ),
            "win_rate": float(
                self.performance.win_rate
            ),
            "total_profit": float(
                self.performance.total_profit
            ),
            "average_trade": float(
                self.performance.average_trade
            ),
            "average_win": float(
                self.performance.average_win
            ),
            "average_loss": float(
                self.performance.average_loss
            ),
            "profit_factor": float(
                self.performance.profit_factor
            ),
            "cumulative_return": float(
                self.performance.cumulative_return
            ),
            "max_drawdown": float(
                self.performance.max_drawdown
            ),
            "max_drawdown_percent": float(
                self.performance.max_drawdown_percent
            ),
        }

    # =====================================================
    # SUMMARY
    # =====================================================

    def summary(self) -> str:
        """
        Generate a human-readable performance summary.
        """

        performance = self.performance

        return (
            "Paper Trading Report\n"
            "====================\n"
            f"Status: {self.status}\n"
            f"Total trades: "
            f"{performance.total_trades}\n"
            f"Winning trades: "
            f"{performance.winning_trades}\n"
            f"Losing trades: "
            f"{performance.losing_trades}\n"
            f"Win rate: "
            f"{performance.win_rate:.2f}%\n"
            f"Total profit: "
            f"{performance.total_profit:.8f}\n"
            f"Average trade: "
            f"{performance.average_trade:.8f}\n"
            f"Average win: "
            f"{performance.average_win:.8f}\n"
            f"Average loss: "
            f"{performance.average_loss:.8f}\n"
            f"Profit factor: "
            f"{performance.profit_factor}\n"
            f"Cumulative return: "
            f"{performance.cumulative_return:.2f}%\n"
            f"Max drawdown: "
            f"{performance.max_drawdown:.8f}\n"
            f"Max drawdown %: "
            f"{performance.max_drawdown_percent:.2f}%"
        )

    # =====================================================
    # BUILD REPORT
    # =====================================================

    def generate(
        self,
    ) -> PaperTradingReportResult:
        """
        Generate a structured report.
        """

        return PaperTradingReportResult(
            performance=self.performance.summarize(),
            status=self.status,
            summary=self.summary(),
            metrics=self.metrics,
        )

    # =====================================================
    # TEXT REPORT
    # =====================================================

    def to_text(self) -> str:
        """
        Return the report as plain text.
        """

        return self.summary()


# =========================================================
# EXPORTS
# =========================================================

__all__ = [
    "PaperTradingReportResult",
    "PaperTradingReport",
]
