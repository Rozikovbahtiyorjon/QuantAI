from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class MonteCarloResult:
    simulations: int
    periods: int
    initial_balance: float
    mean_final_balance: float
    median_final_balance: float
    percentile_5_final_balance: float
    percentile_25_final_balance: float
    percentile_75_final_balance: float
    percentile_95_final_balance: float
    mean_max_drawdown: float
    median_max_drawdown: float
    percentile_95_max_drawdown: float
    probability_of_profit: float
    probability_of_loss: float
    probability_of_drawdown_exceeding_limit: float
    final_balance_distribution: list[float]
    max_drawdown_distribution: list[float]


class MonteCarloEngine:
    def __init__(
        self,
        initial_balance: float = 1000.0,
        simulations: int = 1000,
        seed: Optional[int] = 42,
        drawdown_limit: float = 0.20,
    ) -> None:
        if initial_balance <= 0:
            raise ValueError(
                "initial_balance must be greater than zero."
            )

        if simulations <= 0:
            raise ValueError(
                "simulations must be greater than zero."
            )

        if drawdown_limit < 0:
            raise ValueError(
                "drawdown_limit cannot be negative."
            )

        if drawdown_limit >= 1:
            raise ValueError(
                "drawdown_limit must be less than 1."
            )

        self.initial_balance = float(initial_balance)
        self.simulations = int(simulations)
        self.seed = seed
        self.drawdown_limit = float(drawdown_limit)

        self._result: Optional[MonteCarloResult] = None

    def run(
        self,
        equity_curve: Sequence[float],
        initial_balance: Optional[float] = None,
    ) -> MonteCarloResult:
        values = self._validate_equity_curve(
            equity_curve
        )

        initial = (
            self.initial_balance
            if initial_balance is None
            else float(initial_balance)
        )

        if initial <= 0:
            raise ValueError(
                "initial_balance must be greater than zero."
            )

        returns = np.diff(values) / values[:-1]

        rng = np.random.default_rng(
            self.seed
        )

        periods = int(returns.size)

        final_balances = np.empty(
            self.simulations,
            dtype=float,
        )

        max_drawdowns = np.empty(
            self.simulations,
            dtype=float,
        )

        for simulation_index in range(
            self.simulations
        ):
            sampled_returns = rng.choice(
                returns,
                size=periods,
                replace=True,
            )

            path = np.empty(
                periods + 1,
                dtype=float,
            )

            path[0] = initial

            path[1:] = (
                initial
                * np.cumprod(
                    1.0 + sampled_returns
                )
            )

            final_balances[
                simulation_index
            ] = path[-1]

            max_drawdowns[
                simulation_index
            ] = self._calculate_max_drawdown(
                path
            )

        probability_of_profit = float(
            np.mean(
                final_balances > initial
            )
            * 100.0
        )

        probability_of_loss = float(
            np.mean(
                final_balances < initial
            )
            * 100.0
        )

        probability_of_drawdown_exceeding_limit = float(
            np.mean(
                max_drawdowns
                > self.drawdown_limit
            )
            * 100.0
        )

        result = MonteCarloResult(
            simulations=self.simulations,
            periods=periods,
            initial_balance=round(
                initial,
                8,
            ),
            mean_final_balance=round(
                float(
                    np.mean(
                        final_balances
                    )
                ),
                8,
            ),
            median_final_balance=round(
                float(
                    np.median(
                        final_balances
                    )
                ),
                8,
            ),
            percentile_5_final_balance=round(
                float(
                    np.percentile(
                        final_balances,
                        5,
                    )
                ),
                8,
            ),
            percentile_25_final_balance=round(
                float(
                    np.percentile(
                        final_balances,
                        25,
                    )
                ),
                8,
            ),
            percentile_75_final_balance=round(
                float(
                    np.percentile(
                        final_balances,
                        75,
                    )
                ),
                8,
            ),
            percentile_95_final_balance=round(
                float(
                    np.percentile(
                        final_balances,
                        95,
                    )
                ),
                8,
            ),
            mean_max_drawdown=round(
                float(
                    np.mean(
                        max_drawdowns
                    )
                ),
                8,
            ),
            median_max_drawdown=round(
                float(
                    np.median(
                        max_drawdowns
                    )
                ),
                8,
            ),
            percentile_95_max_drawdown=round(
                float(
                    np.percentile(
                        max_drawdowns,
                        95,
                    )
                ),
                8,
            ),
            probability_of_profit=round(
                probability_of_profit,
                6,
            ),
            probability_of_loss=round(
                probability_of_loss,
                6,
            ),
            probability_of_drawdown_exceeding_limit=round(
                probability_of_drawdown_exceeding_limit,
                6,
            ),
            final_balance_distribution=[
                round(
                    float(value),
                    8,
                )
                for value in final_balances
            ],
            max_drawdown_distribution=[
                round(
                    float(value),
                    8,
                )
                for value in max_drawdowns
            ],
        )

        self._result = result

        return result

    @staticmethod
    def _validate_equity_curve(
        equity_curve: Sequence[float],
    ) -> np.ndarray:
        if isinstance(
            equity_curve,
            (str, bytes),
        ):
            raise TypeError(
                "equity_curve must be a sequence of numeric values."
            )

        try:
            values = np.asarray(
                list(equity_curve),
                dtype=float,
            )
        except (TypeError, ValueError):
            raise TypeError(
                "equity_curve must contain numeric values."
            ) from None

        if values.ndim != 1:
            raise ValueError(
                "equity_curve must be one-dimensional."
            )

        if values.size < 2:
            raise ValueError(
                "equity_curve must contain at least two values."
            )

        if not np.all(
            np.isfinite(values)
        ):
            raise ValueError(
                "equity_curve must contain only finite values."
            )

        if np.any(values <= 0):
            raise ValueError(
                "equity_curve values must be greater than zero."
            )

        return values

    @staticmethod
    def _calculate_max_drawdown(
        equity_curve: np.ndarray,
    ) -> float:
        peaks = np.maximum.accumulate(
            equity_curve
        )

        drawdowns = 1.0 - (
            equity_curve / peaks
        )

        return float(
            np.max(drawdowns)
        )

    @property
    def result(
        self,
    ) -> Optional[MonteCarloResult]:
        return self._result

    def reset(self) -> None:
        self._result = None


__all__ = [
    "MonteCarloResult",
    "MonteCarloEngine",
]