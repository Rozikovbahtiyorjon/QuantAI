from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

from src.quantai_production_model_runtime_binding import (
    QuantAIProductionModelRuntimeBinding,
)


@dataclass
class ProductionInferenceResult:
    success: bool
    model_identifier: Optional[str] = None
    prediction: Any = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return self.success and not self.errors


class QuantAIProductionModelRuntimeExecution:
    """Safe prediction gateway for the active production model."""

    def __init__(
        self,
        binding: QuantAIProductionModelRuntimeBinding,
    ) -> None:
        if not isinstance(
            binding,
            QuantAIProductionModelRuntimeBinding,
        ):
            raise TypeError(
                "binding must be a "
                "QuantAIProductionModelRuntimeBinding."
            )

        self._binding = binding
        self._prediction_count = 0
        self._failure_count = 0

    @property
    def binding(
        self,
    ) -> QuantAIProductionModelRuntimeBinding:
        return self._binding

    @property
    def prediction_count(self) -> int:
        return self._prediction_count

    @property
    def failure_count(self) -> int:
        return self._failure_count

    @property
    def is_ready(self) -> bool:
        return self._binding.is_bound

    def execute(
        self,
        data: Any,
    ) -> ProductionInferenceResult:
        if not self._binding.is_bound:
            self._failure_count += 1

            return ProductionInferenceResult(
                success=False,
                errors=[
                    "No active production model is bound."
                ],
            )

        if data is None:
            self._failure_count += 1

            return ProductionInferenceResult(
                success=False,
                model_identifier=self._binding.identifier,
                errors=[
                    "Inference input data must not be None."
                ],
            )

        try:
            result = self._binding.predict(
                data
            )

        except Exception as exc:
            self._failure_count += 1

            return ProductionInferenceResult(
                success=False,
                model_identifier=(
                    self._binding.identifier
                ),
                errors=[
                    (
                        "Inference execution failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
                ],
            )

        if not result.success:
            self._failure_count += 1

            return ProductionInferenceResult(
                success=False,
                model_identifier=result.identifier,
                errors=list(result.errors),
                warnings=list(result.warnings),
            )

        if result.prediction is None:
            self._failure_count += 1

            return ProductionInferenceResult(
                success=False,
                model_identifier=result.identifier,
                errors=[
                    (
                        "Production model returned "
                        "no prediction."
                    )
                ],
                warnings=list(result.warnings),
            )

        self._prediction_count += 1

        return ProductionInferenceResult(
            success=True,
            model_identifier=result.identifier,
            prediction=result.prediction,
            warnings=list(result.warnings),
        )

    def execute_batch(
        self,
        data_items: List[Any],
    ) -> List[ProductionInferenceResult]:
        if not isinstance(data_items, list):
            raise TypeError(
                "data_items must be a list."
            )

        return [
            self.execute(item)
            for item in data_items
        ]

    def reset_counters(self) -> None:
        self._prediction_count = 0
        self._failure_count = 0


__all__ = [
    "ProductionInferenceResult",
    "QuantAIProductionModelRuntimeExecution",
]