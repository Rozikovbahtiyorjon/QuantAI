from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.paper_trading_pipeline import PaperTradingPipelineResult
from src.paper_trading_validator import (
    PaperTradingValidationResult,
    validate_paper_trading,
)
from src.performance_analyzer import (
    PerformanceAnalyzer,
    PerformanceResult,
)


@dataclass
class PaperTradingQualityGateResult:
    passed: bool
    validation: PaperTradingValidationResult
    performance: PerformanceResult
    errors: list[str]
    warnings: list[str]


class PaperTradingQualityGate:
    def __init__(
        self,
        initial_balance: float = 1000.0,
        minimum_win_rate: float = 0.0,
        minimum_profit_factor: float = 0.0,
        maximum_drawdown_percent: float | None = None,
    ) -> None:
        if initial_balance <= 0:
            raise ValueError(
                "initial_balance must be greater than zero."
            )

        if minimum_win_rate < 0 or minimum_win_rate > 100:
            raise ValueError(
                "minimum_win_rate must be between 0 and 100."
            )

        if minimum_profit_factor < 0:
            raise ValueError(
                "minimum_profit_factor cannot be negative."
            )

        if (
            maximum_drawdown_percent is not None
            and maximum_drawdown_percent < 0
        ):
            raise ValueError(
                "maximum_drawdown_percent cannot be negative."
            )

        self.initial_balance = float(initial_balance)
        self.minimum_win_rate = float(minimum_win_rate)
        self.minimum_profit_factor = float(
            minimum_profit_factor
        )

        self.maximum_drawdown_percent = (
            None
            if maximum_drawdown_percent is None
            else float(maximum_drawdown_percent)
        )

    def _validate_thresholds(
        self,
        performance: PerformanceResult,
    ) -> tuple[list[str], list[str]]:
        errors: list[str] = []
        warnings: list[str] = []

        if performance.win_rate < self.minimum_win_rate:
            errors.append(
                "Win rate is below the configured minimum."
            )

        if performance.profit_factor != float("inf"):
            if (
                performance.profit_factor
                < self.minimum_profit_factor
            ):
                errors.append(
                    "Profit factor is below the configured minimum."
                )

        if (
            self.maximum_drawdown_percent is not None
            and performance.max_drawdown_percent
            > self.maximum_drawdown_percent
        ):
            errors.append(
                "Maximum drawdown percent exceeds "
                "the configured limit."
            )

        if performance.total_trades == 0:
            warnings.append(
                "No completed trades are available "
                "for performance analysis."
            )

        return errors, warnings

    @staticmethod
    def _trades_from_result(
        result: PaperTradingPipelineResult,
    ) -> pd.DataFrame:
        session_result = result.session_result

        if session_result is None:
            return pd.DataFrame(
                columns=["net_profit"]
            )

        steps: Any = getattr(
            session_result,
            "steps",
            [],
        )

        if isinstance(steps, pd.DataFrame):
            data = steps.copy(deep=True)

        elif isinstance(steps, list):
            data = pd.DataFrame(steps)

        else:
            data = pd.DataFrame()

        if "net_profit" not in data.columns:
            data["net_profit"] = 0.0

        return data

    def evaluate(
        self,
        result: PaperTradingPipelineResult,
    ) -> PaperTradingQualityGateResult:
        if not isinstance(
            result,
            PaperTradingPipelineResult,
        ):
            raise TypeError(
                "result must be PaperTradingPipelineResult."
            )

        validation = validate_paper_trading(
            result
        )

        trades = self._trades_from_result(
            result
        )

        analyzer = PerformanceAnalyzer(
            initial_balance=self.initial_balance,
        )

        performance = analyzer.analyze(
            trades
        )

        errors = list(
            validation.errors
        )

        warnings = list(
            validation.warnings
        )

        (
            threshold_errors,
            threshold_warnings,
        ) = self._validate_thresholds(
            performance
        )

        errors.extend(
            threshold_errors
        )

        warnings.extend(
            threshold_warnings
        )

        passed = (
            validation.valid
            and len(errors) == 0
        )

        return PaperTradingQualityGateResult(
            passed=passed,
            validation=validation,
            performance=performance,
            errors=errors,
            warnings=warnings,
        )


def evaluate_paper_trading_quality(
    result: PaperTradingPipelineResult,
    initial_balance: float = 1000.0,
    minimum_win_rate: float = 0.0,
    minimum_profit_factor: float = 0.0,
    maximum_drawdown_percent: float | None = None,
) -> PaperTradingQualityGateResult:
    gate = PaperTradingQualityGate(
        initial_balance=initial_balance,
        minimum_win_rate=minimum_win_rate,
        minimum_profit_factor=minimum_profit_factor,
        maximum_drawdown_percent=maximum_drawdown_percent,
    )

    return gate.evaluate(
        result
    )


__all__ = [
    "PaperTradingQualityGateResult",
    "PaperTradingQualityGate",
    "evaluate_paper_trading_quality",
]