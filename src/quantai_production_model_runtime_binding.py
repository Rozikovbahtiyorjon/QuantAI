from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

from src.quantai_production_model_registry_integration import (
    QuantAIProductionModelRegistryIntegration,
)


@dataclass
class ProductionModelBindingResult:
    success: bool
    model: Any = None
    identifier: Optional[str] = None
    prediction: Any = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def bound(self) -> bool:
        return self.success and self.model is not None


class QuantAIProductionModelRuntimeBinding:
    """Binds the active production champion model to runtime execution."""

    def __init__(
        self,
        registry_integration: QuantAIProductionModelRegistryIntegration,
    ) -> None:
        if not isinstance(
            registry_integration,
            QuantAIProductionModelRegistryIntegration,
        ):
            raise TypeError(
                "registry_integration must be a "
                "QuantAIProductionModelRegistryIntegration."
            )

        self._registry_integration = registry_integration
        self._model: Any = None
        self._identifier: Optional[str] = None

    @property
    def registry_integration(
        self,
    ) -> QuantAIProductionModelRegistryIntegration:
        return self._registry_integration

    @property
    def model(self) -> Any:
        return self._model

    @property
    def identifier(self) -> Optional[str]:
        return self._identifier

    @property
    def is_bound(self) -> bool:
        return self._model is not None

    def bind(self) -> ProductionModelBindingResult:
        result = (
            self._registry_integration.resolve_champion()
        )

        if not result.success:
            self._model = None
            self._identifier = None

            return ProductionModelBindingResult(
                success=False,
                errors=list(result.errors),
                warnings=list(result.warnings),
            )

        self._model = result.active_model
        self._identifier = result.active_identifier

        return ProductionModelBindingResult(
            success=True,
            model=self._model,
            identifier=self._identifier,
            warnings=list(result.warnings),
        )

    def activate_and_bind(
        self,
    ) -> ProductionModelBindingResult:
        result = (
            self._registry_integration.activate_champion()
        )

        if not result.success:
            self._model = None
            self._identifier = None

            return ProductionModelBindingResult(
                success=False,
                errors=list(result.errors),
                warnings=list(result.warnings),
            )

        self._model = result.active_model
        self._identifier = result.active_identifier

        return ProductionModelBindingResult(
            success=True,
            model=self._model,
            identifier=self._identifier,
            warnings=list(result.warnings),
        )

    def predict(
        self,
        data: Any,
    ) -> ProductionModelBindingResult:
        if not self.is_bound:
            return ProductionModelBindingResult(
                success=False,
                errors=[
                    "No production model is bound to runtime."
                ],
            )

        predictor = getattr(
            self._model,
            "predict",
            None,
        )

        if not callable(predictor):
            return ProductionModelBindingResult(
                success=False,
                model=self._model,
                identifier=self._identifier,
                errors=[
                    "Bound production model does not expose predict()."
                ],
            )

        try:
            prediction = predictor(data)
        except Exception as exc:
            return ProductionModelBindingResult(
                success=False,
                model=self._model,
                identifier=self._identifier,
                errors=[
                    (
                        "Production model prediction failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
                ],
            )

        return ProductionModelBindingResult(
            success=True,
            model=self._model,
            identifier=self._identifier,
            prediction=prediction,
        )

    def rebind_after_promotion(
        self,
        model_identifier: Optional[str] = None,
    ) -> ProductionModelBindingResult:
        result = (
            self._registry_integration.promote_challenger(
                model_identifier
            )
        )

        if not result.success:
            return ProductionModelBindingResult(
                success=False,
                model=self._model,
                identifier=self._identifier,
                errors=list(result.errors),
                warnings=list(result.warnings),
            )

        self._model = result.active_model
        self._identifier = result.active_identifier

        return ProductionModelBindingResult(
            success=True,
            model=self._model,
            identifier=self._identifier,
            warnings=list(result.warnings),
        )

    def rebind_after_rollback(
        self,
    ) -> ProductionModelBindingResult:
        result = (
            self._registry_integration.rollback_champion()
        )

        if not result.success:
            return ProductionModelBindingResult(
                success=False,
                model=self._model,
                identifier=self._identifier,
                errors=list(result.errors),
                warnings=list(result.warnings),
            )

        self._model = result.active_model
        self._identifier = result.active_identifier

        return ProductionModelBindingResult(
            success=True,
            model=self._model,
            identifier=self._identifier,
            warnings=list(result.warnings),
        )

    def unbind(self) -> None:
        self._model = None
        self._identifier = None


__all__ = [
    "ProductionModelBindingResult",
    "QuantAIProductionModelRuntimeBinding",
]