from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class StressTestResult:
    initial_balance: float
    final_balance: float
    total_return: float
    total_profit: float
    max_drawdown: float
    max_drawdown_percent: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    scenario: str


class StressTestEngine:
    """Apply deterministic stress assumptions to a sequence of trade PnLs."""

    def __init__(
        self,
        initial_balance: float = 1000.0,
        slippage_multiplier: float = 1.0,
        commission_multiplier: float = 1.0,
    ) -> None:
        if initial_balance <= 0:
            raise ValueError("initial_balance must be greater than zero.")

        if slippage_multiplier < 0:
            raise ValueError("slippage_multiplier cannot be negative.")

        if commission_multiplier < 0:
            raise ValueError("commission_multiplier cannot be negative.")

        self.initial_balance = float(initial_balance)
        self.slippage_multiplier = float(slippage_multiplier)
        self.commission_multiplier = float(commission_multiplier)

    @staticmethod
    def _validate_trades(
        trades: Iterable[float],
    ) -> list[float]:
        values = list(trades)

        if not values:
            raise ValueError("trades cannot be empty.")

        validated: list[float] = []

        for value in values:
            if isinstance(value, bool):
                raise TypeError("trade PnL values must be numeric.")

            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "trade PnL values must be numeric."
                ) from exc

            if number != number or number in (
                float("inf"),
                float("-inf"),
            ):
                raise ValueError(
                    "trade PnL values must be finite."
                )

            validated.append(number)

        return validated

    @staticmethod
    def _validate_non_negative(
        value: float,
        name: str,
    ) -> float:
        if isinstance(value, bool):
            raise TypeError(f"{name} must be numeric.")

        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"{name} must be numeric."
            ) from exc

        if number < 0:
            raise ValueError(
                f"{name} cannot be negative."
            )

        return number

    @staticmethod
    def _drawdown(
        equity_curve: Sequence[float],
    ) -> tuple[float, float]:
        peak = float(equity_curve[0])
        max_drawdown = 0.0

        for equity in equity_curve:
            value = float(equity)

            if value > peak:
                peak = value

            drawdown = peak - value

            if drawdown > max_drawdown:
                max_drawdown = drawdown

        if peak > 0:
            percent = (
                max_drawdown
                / peak
            ) * 100.0
        else:
            percent = 0.0

        return max_drawdown, percent

    def run(
        self,
        trades: Iterable[float],
        *,
        slippage_cost_per_trade: float = 0.0,
        commission_cost_per_trade: float = 0.0,
        scenario: str = "BASELINE",
    ) -> StressTestResult:
        values = self._validate_trades(trades)

        slippage = self._validate_non_negative(
            slippage_cost_per_trade,
            "slippage_cost_per_trade",
        )

        commission = self._validate_non_negative(
            commission_cost_per_trade,
            "commission_cost_per_trade",
        )

        if not isinstance(scenario, str) or not scenario.strip():
            raise ValueError(
                "scenario must be a non-empty string."
            )

        per_trade_cost = (
            slippage * self.slippage_multiplier
            + commission * self.commission_multiplier
        )
        
        balance = self.initial_balance
        equity_curve = [round(balance, 8)]

        winning_trades = 0
        losing_trades = 0

        for pnl in values:
            net_pnl = pnl - per_trade_cost
            balance += net_pnl

            equity_curve.append(
                round(balance, 8)
            )

            if net_pnl > 0:
                winning_trades += 1

            elif net_pnl < 0:
                losing_trades += 1

        total_trades = len(values)

        total_profit = (
            balance
            - self.initial_balance
        )

        total_return = (
            total_profit
            / self.initial_balance
        ) * 100.0

        win_rate = (
            winning_trades
            / total_trades
        ) * 100.0

        max_drawdown, max_drawdown_percent = (
            self._drawdown(equity_curve)
        )

        return StressTestResult(
            initial_balance=round(
                self.initial_balance,
                8,
            ),
            final_balance=round(
                balance,
                8,
            ),
            total_return=round(
                total_return,
                8,
            ),
            total_profit=round(
                total_profit,
                8,
            ),
            max_drawdown=round(
                max_drawdown,
                8,
            ),
            max_drawdown_percent=round(
                max_drawdown_percent,
                8,
            ),
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=round(
                win_rate,
                8,
            ),
            scenario=scenario.strip(),
        )

    def run_scenario(
        self,
        trades: Iterable[float],
        *,
        slippage_cost_per_trade: float = 0.0,
        commission_cost_per_trade: float = 0.0,
        slippage_multiplier: float | None = None,
        commission_multiplier: float | None = None,
        scenario: str = "STRESS",
    ) -> StressTestResult:
        values = self._validate_trades(trades)

        slippage = self._validate_non_negative(
            slippage_cost_per_trade,
            "slippage_cost_per_trade",
        )

        commission = self._validate_non_negative(
            commission_cost_per_trade,
            "commission_cost_per_trade",
        )

        effective_slippage = (
            self.slippage_multiplier
            if slippage_multiplier is None
            else self._validate_non_negative(
                slippage_multiplier,
                "slippage_multiplier",
            )
        )

        effective_commission = (
            self.commission_multiplier
            if commission_multiplier is None
            else self._validate_non_negative(
                commission_multiplier,
                "commission_multiplier",
            )
        )

        engine = StressTestEngine(
            initial_balance=self.initial_balance,
            slippage_multiplier=effective_slippage,
            commission_multiplier=effective_commission,
        )

        return engine.run(
            values,
            slippage_cost_per_trade=slippage,
            commission_cost_per_trade=commission,
            scenario=scenario,
        )

    def compare(
        self,
        trades: Iterable[float],
        scenarios: Sequence[dict[str, object]],
    ) -> list[StressTestResult]:
        values = self._validate_trades(trades)

        if not scenarios:
            raise ValueError(
                "scenarios cannot be empty."
            )

        results: list[StressTestResult] = []

        for config in scenarios:
            if not isinstance(config, dict):
                raise TypeError(
                    "each scenario must be a dictionary."
                )

            results.append(
                self.run_scenario(
                    values,
                    slippage_cost_per_trade=float(
                        config.get(
                            "slippage_cost_per_trade",
                            0.0,
                        )
                    ),
                    commission_cost_per_trade=float(
                        config.get(
                            "commission_cost_per_trade",
                            0.0,
                        )
                    ),
                    slippage_multiplier=(
                        None
                        if "slippage_multiplier" not in config
                        else float(
                            config["slippage_multiplier"]
                        )
                    ),
                    commission_multiplier=(
                        None
                        if "commission_multiplier" not in config
                        else float(
                            config["commission_multiplier"]
                        )
                    ),
                    scenario=str(
                        config.get(
                            "scenario",
                            "STRESS",
                        )
                    ),
                )
            )

        return results


__all__ = [
    "StressTestResult",
    "StressTestEngine",
]