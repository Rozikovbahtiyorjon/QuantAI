from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ModelVersion:
    name: str
    version: str
    artifact: Any = None
    performance_score: float = 0.0
    stability_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def identifier(self) -> str:
        return f"{self.name}:{self.version}"


@dataclass(frozen=True)
class ModelRegistryEvent:
    action: str
    model_identifier: str
    message: str


@dataclass
class ModelRegistryResult:
    success: bool
    champion: Optional[ModelVersion] = None
    challengers: List[ModelVersion] = field(
        default_factory=list
    )
    errors: List[str] = field(
        default_factory=list
    )
    warnings: List[str] = field(
        default_factory=list
    )
    events: List[ModelRegistryEvent] = field(
        default_factory=list
    )

    @property
    def champion_identifier(self) -> Optional[str]:
        if self.champion is None:
            return None

        return self.champion.identifier


class QuantAIProductionModelRegistry:
    """
    Deterministic production model registry.

    Responsibilities:

        - register model versions
        - maintain the active Champion
        - maintain Challenger versions
        - promote eligible models
        - demote the Champion
        - rollback to the previous Champion
        - prevent unregistering the active Champion
        - select the best eligible Challenger
        - record registry lifecycle events

    This module does not train models and does not execute inference.
    """

    def __init__(
        self,
        minimum_performance_score: float = 0.0,
        minimum_stability_score: float = 0.0,
    ) -> None:
        self._validate_score(
            minimum_performance_score,
            "minimum_performance_score",
        )

        self._validate_score(
            minimum_stability_score,
            "minimum_stability_score",
        )

        self.minimum_performance_score = float(
            minimum_performance_score
        )

        self.minimum_stability_score = float(
            minimum_stability_score
        )

        self._models: Dict[str, ModelVersion] = {}

        self._champion_identifier: Optional[str] = None

        self._previous_champion_identifier: Optional[str] = None

        self._events: List[ModelRegistryEvent] = []

    @staticmethod
    def _validate_score(
        value: float,
        name: str,
    ) -> None:
        if isinstance(value, bool) or not isinstance(
            value,
            (int, float),
        ):
            raise TypeError(
                f"{name} must be numeric."
            )

        if value < 0.0 or value > 1.0:
            raise ValueError(
                f"{name} must be between 0 and 1."
            )

    @staticmethod
    def _validate_model(
        model: ModelVersion,
    ) -> None:
        if not isinstance(
            model,
            ModelVersion,
        ):
            raise TypeError(
                "model must be a ModelVersion instance."
            )

        if (
            not isinstance(model.name, str)
            or not model.name.strip()
        ):
            raise ValueError(
                "model name must be a non-empty string."
            )

        if (
            not isinstance(model.version, str)
            or not model.version.strip()
        ):
            raise ValueError(
                "model version must be a non-empty string."
            )

        for value, field_name in (
            (
                model.performance_score,
                "performance_score",
            ),
            (
                model.stability_score,
                "stability_score",
            ),
        ):
            QuantAIProductionModelRegistry._validate_score(
                value,
                field_name,
            )

    def _record(
        self,
        action: str,
        model: ModelVersion,
        message: str,
    ) -> None:
        self._events.append(
            ModelRegistryEvent(
                action=action,
                model_identifier=model.identifier,
                message=message,
            )
        )

    @property
    def champion(self) -> Optional[ModelVersion]:
        if self._champion_identifier is None:
            return None

        return self._models.get(
            self._champion_identifier
        )

    @property
    def previous_champion(
        self,
    ) -> Optional[ModelVersion]:
        if self._previous_champion_identifier is None:
            return None

        return self._models.get(
            self._previous_champion_identifier
        )

    @property
    def models(self) -> Tuple[ModelVersion, ...]:
        return tuple(
            self._models.values()
        )

    @property
    def events(
        self,
    ) -> Tuple[ModelRegistryEvent, ...]:
        return tuple(
            self._events
        )

    def register(
        self,
        model: ModelVersion,
    ) -> ModelRegistryResult:
        self._validate_model(model)

        if model.identifier in self._models:
            return ModelRegistryResult(
                success=False,
                champion=self.champion,
                challengers=self._challengers(),
                errors=[
                    (
                        f"Model {model.identifier} "
                        "is already registered."
                    )
                ],
                events=list(self._events),
            )

        self._models[
            model.identifier
        ] = model

        self._record(
            "register",
            model,
            "Model version registered.",
        )

        return self._result(
            success=True
        )

    def _eligible(
        self,
        model: ModelVersion,
    ) -> bool:
        return (
            model.performance_score
            >= self.minimum_performance_score
            and model.stability_score
            >= self.minimum_stability_score
        )

    def _challengers(
        self,
    ) -> List[ModelVersion]:
        return [
            model
            for identifier, model
            in self._models.items()
            if identifier != self._champion_identifier
        ]

    def promote(
        self,
        identifier: str,
    ) -> ModelRegistryResult:
        model = self.get(identifier)

        if model is None:
            return self._failure(
                f"Model {identifier} is not registered."
            )

        if not self._eligible(model):
            return self._failure(
                (
                    f"Model {identifier} does not satisfy "
                    "promotion thresholds."
                )
            )

        if self._champion_identifier == identifier:
            return ModelRegistryResult(
                success=True,
                champion=model,
                challengers=self._challengers(),
                warnings=[
                    "Model is already the Champion."
                ],
                events=list(self._events),
            )

        old_champion = self.champion

        self._previous_champion_identifier = (
            old_champion.identifier
            if old_champion is not None
            else None
        )

        self._champion_identifier = identifier

        self._record(
            "promote",
            model,
            "Model promoted to Champion.",
        )

        return self._result(
            success=True
        )

    def demote(
        self,
        identifier: str,
    ) -> ModelRegistryResult:
        model = self.get(identifier)

        if model is None:
            return self._failure(
                f"Model {identifier} is not registered."
            )

        if self._champion_identifier != identifier:
            return self._failure(
                (
                    f"Model {identifier} is not "
                    "the current Champion."
                )
            )

        self._champion_identifier = None

        self._record(
            "demote",
            model,
            "Champion was demoted.",
        )

        return self._result(
            success=True,
            warnings=[
                "No Champion is currently active."
            ],
        )

    def rollback(
        self,
    ) -> ModelRegistryResult:
        previous = self.previous_champion

        if previous is None:
            return self._failure(
                (
                    "No previous Champion is available "
                    "for rollback."
                )
            )

        current = self.champion

        self._champion_identifier = (
            previous.identifier
        )

        self._previous_champion_identifier = (
            current.identifier
            if current is not None
            else None
        )

        self._record(
            "rollback",
            previous,
            "Previous Champion restored.",
        )

        return self._result(
            success=True
        )

    def get(
        self,
        identifier: str,
    ) -> Optional[ModelVersion]:
        return self._models.get(identifier)

    def unregister(
        self,
        identifier: str,
    ) -> ModelRegistryResult:
        model = self.get(identifier)

        if model is None:
            return self._failure(
                f"Model {identifier} is not registered."
            )

        if self._champion_identifier == identifier:
            return self._failure(
                "The active Champion cannot be unregistered."
            )

        del self._models[identifier]

        self._record(
            "unregister",
            model,
            "Model version unregistered.",
        )

        return self._result(
            success=True
        )

    def select_best_challenger(
        self,
    ) -> Optional[ModelVersion]:
        eligible = [
            model
            for model in self._challengers()
            if self._eligible(model)
        ]

        if not eligible:
            return None

        return max(
            eligible,
            key=lambda model: (
                model.performance_score,
                model.stability_score,
            ),
        )

    def _result(
        self,
        success: bool,
        warnings: Optional[List[str]] = None,
    ) -> ModelRegistryResult:
        return ModelRegistryResult(
            success=success,
            champion=self.champion,
            challengers=self._challengers(),
            warnings=list(
                warnings or []
            ),
            events=list(
                self._events
            ),
        )

    def _failure(
        self,
        message: str,
    ) -> ModelRegistryResult:
        return ModelRegistryResult(
            success=False,
            champion=self.champion,
            challengers=self._challengers(),
            errors=[
                message
            ],
            events=list(
                self._events
            ),
        )


def create_production_model_registry(
    minimum_performance_score: float = 0.0,
    minimum_stability_score: float = 0.0,
) -> QuantAIProductionModelRegistry:
    return QuantAIProductionModelRegistry(
        minimum_performance_score=minimum_performance_score,
        minimum_stability_score=minimum_stability_score,
    )


__all__ = [
    "ModelVersion",
    "ModelRegistryEvent",
    "ModelRegistryResult",
    "QuantAIProductionModelRegistry",
    "create_production_model_registry",
]