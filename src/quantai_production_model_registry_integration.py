from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class ProductionModelResolutionResult:
    success: bool
    active_model: Any = None
    active_identifier: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return self.success and self.active_model is not None


class QuantAIProductionModelRegistryIntegration:
    """Safe runtime-facing integration for the production model registry."""

    def __init__(
        self,
        registry: Any,
        require_artifact: bool = False,
    ) -> None:
        if registry is None:
            raise TypeError("registry must not be None.")

        required_attributes = (
            "champion",
            "promote",
            "rollback",
            "select_best_challenger",
        )

        for attribute in required_attributes:
            if not hasattr(registry, attribute):
                raise TypeError(
                    f"registry must expose '{attribute}'."
                )

        if not isinstance(require_artifact, bool):
            raise TypeError(
                "require_artifact must be a boolean."
            )

        self._registry = registry
        self._require_artifact = require_artifact
        self._active_model: Any = None
        self._active_identifier: Optional[str] = None

    @property
    def registry(self) -> Any:
        return self._registry

    @property
    def active_model(self) -> Any:
        return self._active_model

    @property
    def active_identifier(self) -> Optional[str]:
        return self._active_identifier

    @property
    def is_active(self) -> bool:
        return self._active_model is not None

    @staticmethod
    def _identifier(model: Any) -> Optional[str]:
        if model is None:
            return None

        identifier = getattr(
            model,
            "identifier",
            None,
        )

        if isinstance(identifier, str) and identifier:
            return identifier

        name = getattr(
            model,
            "name",
            None,
        )

        version = getattr(
            model,
            "version",
            None,
        )

        if isinstance(name, str) and isinstance(version, str):
            return f"{name}:{version}"

        return None

    def _validate_champion(
        self,
        champion: Any,
    ) -> List[str]:
        errors: List[str] = []

        if champion is None:
            errors.append(
                "No champion model is registered."
            )
            return errors

        if self._require_artifact and getattr(
            champion,
            "artifact",
            None,
        ) is None:
            errors.append(
                "Champion model has no production artifact."
            )

        return errors

    def resolve_champion(
        self,
    ) -> ProductionModelResolutionResult:
        champion = self._registry.champion

        errors = self._validate_champion(
            champion
        )

        if errors:
            return ProductionModelResolutionResult(
                success=False,
                errors=errors,
            )

        identifier = self._identifier(
            champion
        )

        if identifier is None:
            return ProductionModelResolutionResult(
                success=False,
                errors=[
                    "Champion model has no supported identifier."
                ],
            )

        return ProductionModelResolutionResult(
            success=True,
            active_model=champion,
            active_identifier=identifier,
        )

    def activate_champion(
        self,
    ) -> ProductionModelResolutionResult:
        result = self.resolve_champion()

        if not result.success:
            return result

        self._active_model = result.active_model
        self._active_identifier = (
            result.active_identifier
        )

        return result

    def promote_challenger(
        self,
        model_identifier: Optional[str] = None,
    ) -> ProductionModelResolutionResult:
        try:
            if model_identifier is None:
                challenger = (
                    self._registry.select_best_challenger()
                )

                model_identifier = self._identifier(
                    challenger
                )

            if not isinstance(
                model_identifier,
                str,
            ) or not model_identifier:
                return ProductionModelResolutionResult(
                    success=False,
                    errors=[
                        (
                            "No valid challenger model "
                            "identifier was provided."
                        )
                    ],
                )

            registry_result = self._registry.promote(
                model_identifier
            )

        except Exception as exc:
            return ProductionModelResolutionResult(
                success=False,
                errors=[
                    (
                        "Challenger promotion failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
                ],
            )

        if hasattr(
            registry_result,
            "success",
        ) and not bool(
            registry_result.success
        ):
            errors = list(
                getattr(
                    registry_result,
                    "errors",
                    [],
                )
                or []
            )

            if not errors:
                errors.append(
                    "Challenger promotion was rejected."
                )

            return ProductionModelResolutionResult(
                success=False,
                errors=[
                    str(error)
                    for error in errors
                ],
                warnings=[
                    str(warning)
                    for warning in (
                        getattr(
                            registry_result,
                            "warnings",
                            [],
                        )
                        or []
                    )
                ],
            )

        return self.activate_champion()

    def rollback_champion(
        self,
    ) -> ProductionModelResolutionResult:
        try:
            registry_result = (
                self._registry.rollback()
            )

        except Exception as exc:
            return ProductionModelResolutionResult(
                success=False,
                errors=[
                    (
                        "Champion rollback failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
                ],
            )

        if hasattr(
            registry_result,
            "success",
        ) and not bool(
            registry_result.success
        ):
            errors = list(
                getattr(
                    registry_result,
                    "errors",
                    [],
                )
                or []
            )

            if not errors:
                errors.append(
                    "Champion rollback was rejected."
                )

            return ProductionModelResolutionResult(
                success=False,
                errors=[
                    str(error)
                    for error in errors
                ],
                warnings=[
                    str(warning)
                    for warning in (
                        getattr(
                            registry_result,
                            "warnings",
                            [],
                        )
                        or []
                    )
                ],
            )

        return self.activate_champion()

    def clear_active_model(self) -> None:
        self._active_model = None
        self._active_identifier = None


__all__ = [
    "ProductionModelResolutionResult",
    "QuantAIProductionModelRegistryIntegration",
]