from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

from src.quantai_production_model_runtime_execution import (
    QuantAIProductionModelRuntimeExecution,
)


@dataclass
class PredictionHealthSnapshot:
    total_predictions: int
    successful_predictions: int
    failed_predictions: int
    success_rate: float
    failure_rate: float
    last_model_identifier: Optional[str]
    last_prediction: Any = None

    @property
    def healthy(self) -> bool:
        return (
            self.total_predictions > 0
            and self.failed_predictions == 0
        )


@dataclass
class PredictionHealthResult:
    healthy: bool
    prediction: Any = None
    model_identifier: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    snapshot: Optional[PredictionHealthSnapshot] = None


class QuantAIProductionModelRuntimeMonitoring:
    """Inference health and prediction quality monitor."""

    def __init__(
        self,
        execution: QuantAIProductionModelRuntimeExecution,
        max_failure_rate: float = 0.0,
    ) -> None:
        if not isinstance(
            execution,
            QuantAIProductionModelRuntimeExecution,
        ):
            raise TypeError(
                "execution must be a "
                "QuantAIProductionModelRuntimeExecution."
            )

        if isinstance(max_failure_rate, bool) or not isinstance(
            max_failure_rate,
            (int, float),
        ):
            raise TypeError(
                "max_failure_rate must be numeric."
            )

        if not 0.0 <= float(max_failure_rate) <= 1.0:
            raise ValueError(
                "max_failure_rate must be between 0.0 and 1.0."
            )

        self._execution = execution
        self._max_failure_rate = float(
            max_failure_rate
        )
        self._total_predictions = 0
        self._successful_predictions = 0
        self._failed_predictions = 0
        self._last_model_identifier: Optional[str] = None
        self._last_prediction: Any = None

    @property
    def execution(
        self,
    ) -> QuantAIProductionModelRuntimeExecution:
        return self._execution

    @property
    def max_failure_rate(self) -> float:
        return self._max_failure_rate

    @property
    def total_predictions(self) -> int:
        return self._total_predictions

    @property
    def successful_predictions(self) -> int:
        return self._successful_predictions

    @property
    def failed_predictions(self) -> int:
        return self._failed_predictions

    @property
    def failure_rate(self) -> float:
        if self._total_predictions == 0:
            return 0.0

        return (
            self._failed_predictions
            / self._total_predictions
        )

    @property
    def success_rate(self) -> float:
        if self._total_predictions == 0:
            return 0.0

        return (
            self._successful_predictions
            / self._total_predictions
        )

    @property
    def is_healthy(self) -> bool:
        return (
            self._total_predictions > 0
            and self.failure_rate
            <= self._max_failure_rate
        )

    def _snapshot(self) -> PredictionHealthSnapshot:
        return PredictionHealthSnapshot(
            total_predictions=self._total_predictions,
            successful_predictions=(
                self._successful_predictions
            ),
            failed_predictions=self._failed_predictions,
            success_rate=self.success_rate,
            failure_rate=self.failure_rate,
            last_model_identifier=(
                self._last_model_identifier
            ),
            last_prediction=self._last_prediction,
        )

    def monitor(
        self,
        data: Any,
    ) -> PredictionHealthResult:
        result = self._execution.execute(data)

        self._total_predictions += 1

        if result.success:
            self._successful_predictions += 1
            self._last_model_identifier = (
                result.model_identifier
            )
            self._last_prediction = result.prediction

            warnings = list(result.warnings)

            if self.failure_rate > self._max_failure_rate:
                warnings.append(
                    "Inference failure rate exceeds "
                    "the configured threshold."
                )

            return PredictionHealthResult(
                healthy=self.is_healthy,
                prediction=result.prediction,
                model_identifier=(
                    result.model_identifier
                ),
                warnings=warnings,
                snapshot=self._snapshot(),
            )

        self._failed_predictions += 1

        self._last_model_identifier = (
            result.model_identifier
        )

        warnings = list(result.warnings)

        if self.failure_rate > self._max_failure_rate:
            warnings.append(
                "Inference failure rate exceeds "
                "the configured threshold."
            )

        return PredictionHealthResult(
            healthy=False,
            model_identifier=result.model_identifier,
            errors=list(result.errors),
            warnings=warnings,
            snapshot=self._snapshot(),
        )

    def monitor_batch(
        self,
        data_items: List[Any],
    ) -> List[PredictionHealthResult]:
        if not isinstance(data_items, list):
            raise TypeError(
                "data_items must be a list."
            )

        return [
            self.monitor(item)
            for item in data_items
        ]

    def snapshot(self) -> PredictionHealthSnapshot:
        return self._snapshot()

    def reset(self) -> None:
        self._total_predictions = 0
        self._successful_predictions = 0
        self._failed_predictions = 0
        self._last_model_identifier = None
        self._last_prediction = None


__all__ = [
    "PredictionHealthSnapshot",
    "PredictionHealthResult",
    "QuantAIProductionModelRuntimeMonitoring",
]