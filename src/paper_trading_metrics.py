from __future__ import annotations

from dataclasses import dataclass

from src.paper_trading_pipeline import (
    PaperTradingPipelineResult,
)


@dataclass(frozen=True)
class PaperTradingMetrics:
    """
    Standardized metrics extracted from a completed
    PaperTradingPipelineResult.
    """

    initial_balance: float
    final_balance: float
    realized_profit: float
    return_percent: float

    total_steps: int
    opened_positions: int
    closed_positions: int

    open_positions: int

    average_profit_per_closed_position: float


class PaperTradingMetricsCalculator:
    """
    Calculate standardized paper-trading metrics.

    This class does not:

        - execute trades;
        - generate signals;
        - connect to exchanges;
        - modify PaperTradingEngine;
        - modify PaperTradingSession;
        - modify PaperTradingPipeline;
        - modify PaperTradingValidator.
    """

    @staticmethod
    def calculate(
        result: PaperTradingPipelineResult,
    ) -> PaperTradingMetrics:
        """
        Calculate metrics from a completed
        PaperTradingPipelineResult.
        """

        if not isinstance(
            result,
            PaperTradingPipelineResult,
        ):
            raise TypeError(
                "result must be "
                "PaperTradingPipelineResult."
            )

        initial_balance = float(
            result.initial_balance
        )

        final_balance = float(
            result.final_balance
        )

        realized_profit = float(
            result.realized_profit
        )

        return_percent = float(
            result.return_percent
        )

        total_steps = int(
            result.total_steps
        )

        opened_positions = int(
            result.opened_positions
        )

        closed_positions = int(
            result.closed_positions
        )

        open_positions = (
            opened_positions
            - closed_positions
        )

        if closed_positions > 0:

            average_profit = (
                realized_profit
                / closed_positions
            )

        else:

            average_profit = 0.0

        return PaperTradingMetrics(
            initial_balance=initial_balance,
            final_balance=final_balance,
            realized_profit=round(
                realized_profit,
                8,
            ),
            return_percent=round(
                return_percent,
                2,
            ),
            total_steps=total_steps,
            opened_positions=opened_positions,
            closed_positions=closed_positions,
            open_positions=open_positions,
            average_profit_per_closed_position=round(
                average_profit,
                8,
            ),
        )


def calculate_paper_trading_metrics(
    result: PaperTradingPipelineResult,
) -> PaperTradingMetrics:
    """
    Convenience function for calculating
    paper-trading metrics.
    """

    return PaperTradingMetricsCalculator.calculate(
        result
    )


__all__ = [
    "PaperTradingMetrics",
    "PaperTradingMetricsCalculator",
    "calculate_paper_trading_metrics",
]