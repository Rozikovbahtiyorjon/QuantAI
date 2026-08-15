from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping


@dataclass(frozen=True)
class IntelligenceComponent:
    name: str
    score: float
    weight: float
    available: bool = True


@dataclass(frozen=True)
class UnifiedMarketIntelligenceSignal:
    score: float
    confidence: float
    direction: str
    regime: str
    risk_level: str
    components_used: int
    components_total: int


class UnifiedMarketIntelligenceLayer:
    DEFAULT_WEIGHTS = {
        "technical": 0.15,
        "ml": 0.20,
        "order_flow": 0.10,
        "derivatives": 0.10,
        "liquidation": 0.10,
        "regime": 0.10,
        "sentiment": 0.08,
        "social_attention": 0.05,
        "sentiment_divergence": 0.05,
        "event_risk": 0.07,
    }

    def __init__(
        self,
        weights: Mapping[str, float] | None = None,
        neutral_score: float = 0.5,
    ) -> None:
        self.weights = self._validate_weights(
            weights if weights is not None else self.DEFAULT_WEIGHTS
        )
        self.neutral_score = self._validate_score(
            neutral_score,
            "neutral_score",
        )

    def evaluate(
        self,
        components: Mapping[str, float],
        *,
        regime: str = "UNKNOWN",
        event_risk: float | None = None,
    ) -> UnifiedMarketIntelligenceSignal:
        if not isinstance(components, Mapping):
            raise TypeError("components must be a mapping.")

        if not isinstance(regime, str):
            raise TypeError("regime must be a string.")

        regime = regime.strip()

        if not regime:
            raise ValueError("regime cannot be empty.")

        validated: list[IntelligenceComponent] = []

        for name, weight in self.weights.items():
            if name not in components:
                continue

            score = self._validate_score(
                components[name],
                f"{name} score",
            )

            validated.append(
                IntelligenceComponent(
                    name=name,
                    score=score,
                    weight=weight,
                )
            )

        if not validated:
            raise ValueError(
                "at least one configured intelligence component is required."
            )

        if event_risk is not None:
            event_risk = self._validate_score(
                event_risk,
                "event_risk",
            )

        total_weight = sum(
            component.weight
            for component in validated
        )

        weighted_score = sum(
            component.score * component.weight
            for component in validated
        ) / total_weight

        risk_level = self._risk_level(
            event_risk
            if event_risk is not None
            else self.neutral_score
        )

        direction = self._direction(
            weighted_score
        )

        confidence = abs(
            weighted_score - self.neutral_score
        ) * 2.0

        return UnifiedMarketIntelligenceSignal(
            score=weighted_score,
            confidence=confidence,
            direction=direction,
            regime=regime,
            risk_level=risk_level,
            components_used=len(validated),
            components_total=len(self.weights),
        )

    @staticmethod
    def _direction(
        score: float,
    ) -> str:
        if score > 0.55:
            return "BULLISH"

        if score < 0.45:
            return "BEARISH"

        return "NEUTRAL"

    @staticmethod
    def _risk_level(
        score: float,
    ) -> str:
        if score >= 0.75:
            return "EXTREME"

        if score >= 0.50:
            return "HIGH"

        if score >= 0.25:
            return "ELEVATED"

        return "NORMAL"

    @classmethod
    def _validate_weights(
        cls,
        weights: Mapping[str, float],
    ) -> dict[str, float]:
        if not isinstance(weights, Mapping):
            raise TypeError(
                "weights must be a mapping."
            )

        if not weights:
            raise ValueError(
                "weights cannot be empty."
            )

        validated: dict[str, float] = {}

        for name, weight in weights.items():
            if not isinstance(name, str):
                raise TypeError(
                    "weight names must be strings."
                )

            name = name.strip()

            if not name:
                raise ValueError(
                    "weight names cannot be empty."
                )

            if (
                isinstance(weight, bool)
                or not isinstance(weight, (int, float))
                or not isfinite(float(weight))
            ):
                raise TypeError(
                    f"weight for '{name}' must be a finite number."
                )

            weight = float(weight)

            if weight <= 0:
                raise ValueError(
                    f"weight for '{name}' must be greater than zero."
                )

            validated[name] = weight

        total = sum(
            validated.values()
        )

        if total <= 0:
            raise ValueError(
                "weight sum must be greater than zero."
            )

        return {
            name: weight / total
            for name, weight in validated.items()
        }

    @staticmethod
    def _validate_score(
        value: float,
        field_name: str,
    ) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(float(value))
        ):
            raise TypeError(
                f"{field_name} must be a finite number."
            )

        value = float(value)

        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"{field_name} must be between 0.0 and 1.0."
            )

        return value


__all__ = [
    "IntelligenceComponent",
    "UnifiedMarketIntelligenceSignal",
    "UnifiedMarketIntelligenceLayer",
]