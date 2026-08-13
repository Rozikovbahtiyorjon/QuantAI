from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class StrategyCondition:
    name: str
    passed: bool
    weight: float = 1.0


@dataclass(frozen=True)
class StrategyDecision:
    action: str
    score: float
    confidence: float
    filters_passed: bool
    entry_score: float
    confirmation_score: float
    risk_allowed: bool
    regime: str


class AdvancedStrategyArchitecture:
    ACTION_BUY = "BUY"
    ACTION_SELL = "SELL"
    ACTION_HOLD = "HOLD"
    ACTION_BLOCK = "BLOCK"

    def __init__(
        self,
        entry_threshold: float = 0.60,
        confirmation_threshold: float = 0.60,
        confidence_threshold: float = 0.60,
    ) -> None:
        self.entry_threshold = self._validate_threshold(
            entry_threshold,
            "entry_threshold",
        )
        self.confirmation_threshold = self._validate_threshold(
            confirmation_threshold,
            "confirmation_threshold",
        )
        self.confidence_threshold = self._validate_threshold(
            confidence_threshold,
            "confidence_threshold",
        )

    def evaluate(
        self,
        *,
        direction: str,
        filters: Mapping[str, bool],
        entry_conditions: Mapping[str, float],
        confirmation: Mapping[str, float],
        confidence: float,
        risk_allowed: bool,
        regime: str,
    ) -> StrategyDecision:
        direction = self._validate_direction(direction)
        filters = self._validate_filters(filters)
        entry_conditions = self._validate_scores(
            entry_conditions,
            "entry_conditions",
        )
        confirmation = self._validate_scores(
            confirmation,
            "confirmation",
        )
        confidence = self._validate_score(
            confidence,
            "confidence",
        )

        if not isinstance(risk_allowed, bool):
            raise TypeError(
                "risk_allowed must be a boolean."
            )

        if not isinstance(regime, str):
            raise TypeError(
                "regime must be a string."
            )

        regime = regime.strip()

        if not regime:
            raise ValueError(
                "regime cannot be empty."
            )

        filters_passed = all(filters.values())

        entry_score = self._average_score(
            entry_conditions
        )

        confirmation_score = self._average_score(
            confirmation
        )

        if not filters_passed:
            action = self.ACTION_BLOCK

        elif not risk_allowed:
            action = self.ACTION_BLOCK

        elif entry_score < self.entry_threshold:
            action = self.ACTION_HOLD

        elif confirmation_score < self.confirmation_threshold:
            action = self.ACTION_HOLD

        elif confidence < self.confidence_threshold:
            action = self.ACTION_HOLD

        elif direction == "LONG":
            action = self.ACTION_BUY

        else:
            action = self.ACTION_SELL

        return StrategyDecision(
            action=action,
            score=self._combined_score(
                entry_score,
                confirmation_score,
                confidence,
            ),
            confidence=confidence,
            filters_passed=filters_passed,
            entry_score=entry_score,
            confirmation_score=confirmation_score,
            risk_allowed=risk_allowed,
            regime=regime,
        )

    @staticmethod
    def _combined_score(
        entry_score: float,
        confirmation_score: float,
        confidence: float,
    ) -> float:
        return (
            entry_score
            + confirmation_score
            + confidence
        ) / 3.0

    @staticmethod
    def _average_score(
        scores: Mapping[str, float],
    ) -> float:
        if not scores:
            return 0.0

        return sum(
            scores.values()
        ) / len(scores)

    @classmethod
    def _validate_filters(
        cls,
        filters: Mapping[str, bool],
    ) -> dict[str, bool]:
        if not isinstance(filters, Mapping):
            raise TypeError(
                "filters must be a mapping."
            )

        if not filters:
            raise ValueError(
                "filters cannot be empty."
            )

        validated: dict[str, bool] = {}

        for name, value in filters.items():
            if not isinstance(name, str):
                raise TypeError(
                    "filter names must be strings."
                )

            name = name.strip()

            if not name:
                raise ValueError(
                    "filter names cannot be empty."
                )

            if not isinstance(value, bool):
                raise TypeError(
                    f"filter '{name}' must be a boolean."
                )

            validated[name] = value

        return validated

    @classmethod
    def _validate_scores(
        cls,
        scores: Mapping[str, float],
        field_name: str,
    ) -> dict[str, float]:
        if not isinstance(scores, Mapping):
            raise TypeError(
                f"{field_name} must be a mapping."
            )

        if not scores:
            raise ValueError(
                f"{field_name} cannot be empty."
            )

        validated: dict[str, float] = {}

        for name, value in scores.items():
            if not isinstance(name, str):
                raise TypeError(
                    f"{field_name} names must be strings."
                )

            name = name.strip()

            if not name:
                raise ValueError(
                    f"{field_name} names cannot be empty."
                )

            validated[name] = cls._validate_score(
                value,
                f"{field_name} '{name}'",
            )

        return validated

    @staticmethod
    def _validate_score(
        value: float,
        field_name: str,
    ) -> float:
        if isinstance(value, bool):
            raise TypeError(
                f"{field_name} must be a number."
            )

        if not isinstance(value, (int, float)):
            raise TypeError(
                f"{field_name} must be a number."
            )

        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(
                f"{field_name} must be between 0.0 and 1.0."
            )

        return float(value)

    @staticmethod
    def _validate_threshold(
        value: float,
        field_name: str,
    ) -> float:
        return AdvancedStrategyArchitecture._validate_score(
            value,
            field_name,
        )

    @staticmethod
    def _validate_direction(
        direction: str,
    ) -> str:
        if not isinstance(direction, str):
            raise TypeError(
                "direction must be a string."
            )

        direction = direction.strip().upper()

        if direction not in {"LONG", "SHORT"}:
            raise ValueError(
                "direction must be LONG or SHORT."
            )

        return direction


__all__ = [
    "StrategyCondition",
    "StrategyDecision",
    "AdvancedStrategyArchitecture",
]