from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from experimental.src.monte_carlo_engine import (
    MonteCarloEngine,
    MonteCarloResult,
)
from experimental.src.stress_test_engine import (
    StressTestEngine,
    StressTestResult,
)


@dataclass(frozen=True)
class PortfolioStressMonteCarloResult:
    monte_carlo: MonteCarloResult
    stress_test: StressTestResult


class PortfolioStressMonteCarloEngine:
    """Integrates Monte Carlo analysis with portfolio stress testing."""

    def __init__(
        self,
        initial_balance: float = 1000.0,
        simulations: int = 1000,
        seed: int | None = 42,
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

        self._result: PortfolioStressMonteCarloResult | None = None

    def run(
        self,
        equity_curve: Sequence[float],
        trade_pnls: Sequence[float],
        scenario: str = "BASE",
        slippage_cost_per_trade: float = 0.0,
        commission_cost_per_trade: float = 0.0,
        slippage_multiplier: float = 1.0,
        commission_multiplier: float = 1.0,
    ) -> PortfolioStressMonteCarloResult:
        equity_values = [float(value) for value in equity_curve]
        pnl_values = [float(value) for value in trade_pnls]

        if not equity_values:
            raise ValueError(
                "equity_curve cannot be empty."
            )

        if not pnl_values:
            raise ValueError(
                "trade_pnls cannot be empty."
            )

        if any(value <= 0 for value in equity_values):
            raise ValueError(
                "equity_curve values must be greater than zero."
            )

        monte_carlo_engine = MonteCarloEngine(
            simulations=self.simulations,
            seed=self.seed,
            drawdown_limit=self.drawdown_limit,
        )

        stress_engine = StressTestEngine(
            initial_balance=self.initial_balance,
        )

        monte_carlo_result = monte_carlo_engine.run(
            equity_values,
            initial_balance=self.initial_balance,
        )

        effective_slippage_multiplier = (
            float(slippage_multiplier) + 1.0
        )
        effective_commission_multiplier = (
            float(commission_multiplier) + 1.0
        )

        stress_result = stress_engine.run_scenario(
            pnl_values,
            slippage_cost_per_trade=slippage_cost_per_trade,
            commission_cost_per_trade=commission_cost_per_trade,
            slippage_multiplier=effective_slippage_multiplier,
            commission_multiplier=effective_commission_multiplier,
            scenario=scenario,
        )

        result = PortfolioStressMonteCarloResult(
            monte_carlo=monte_carlo_result,
            stress_test=stress_result,
        )

        self._result = result

        return result

    @property
    def result(
        self,
    ) -> PortfolioStressMonteCarloResult | None:
        return self._result

    def reset(self) -> None:
        self._result = None


__all__ = [
    "PortfolioStressMonteCarloResult",
    "PortfolioStressMonteCarloEngine",
]