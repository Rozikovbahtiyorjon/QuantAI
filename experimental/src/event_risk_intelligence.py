from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class MarketEvent:
    symbol: str
    timestamp: int
    event_type: str
    importance: float
    expected_impact: float
    hours_to_event: float
    uncertainty: float
    active: bool = True


@dataclass(frozen=True)
class EventRiskSignal:
    risk_score: float
    risk_level: str
    event_type: str
    context: str


class EventRiskIntelligence:
    def __init__(
        self,
        high_risk_threshold: float = 0.70,
        extreme_risk_threshold: float = 0.90,
    ) -> None:
        self._validate_threshold(
            high_risk_threshold,
            "high_risk_threshold",
        )
        self._validate_threshold(
            extreme_risk_threshold,
            "extreme_risk_threshold",
        )

        if high_risk_threshold >= extreme_risk_threshold:
            raise ValueError(
                "high_risk_threshold must be lower than "
                "extreme_risk_threshold."
            )

        self.high_risk_threshold = float(
            high_risk_threshold
        )
        self.extreme_risk_threshold = float(
            extreme_risk_threshold
        )
        self._previous: MarketEvent | None = None

    @property
    def previous(self) -> MarketEvent | None:
        return self._previous

    def evaluate(
        self,
        event: MarketEvent,
    ) -> EventRiskSignal:
        self._validate_event(event)

        if self._previous is not None:
            if event.symbol != self._previous.symbol:
                raise ValueError(
                    "symbol must match the previous event."
                )

            if event.timestamp < self._previous.timestamp:
                raise ValueError(
                    "timestamp must not be earlier than "
                    "the previous event."
                )

        risk_score = self._calculate_risk_score(event)

        if not event.active:
            risk_score = 0.0

        risk_level = self._risk_level(risk_score)
        context = self._context(event, risk_score)

        self._previous = event

        return EventRiskSignal(
            risk_score=risk_score,
            risk_level=risk_level,
            event_type=event.event_type,
            context=context,
        )

    def reset(self) -> None:
        self._previous = None

    def _risk_level(self, score: float) -> str:
        if score >= self.extreme_risk_threshold:
            return "EXTREME"

        if score >= self.high_risk_threshold:
            return "HIGH"

        if score > 0.0:
            return "ELEVATED"

        return "NORMAL"

    @staticmethod
    def _context(
        event: MarketEvent,
        score: float,
    ) -> str:
        if score == 0.0:
            return "NO_ACTIVE_EVENT"

        if event.hours_to_event == 0.0:
            return "EVENT_ACTIVE"

        if event.hours_to_event <= 1.0:
            return "IMMINENT_EVENT"

        if event.hours_to_event <= 24.0:
            return "NEAR_TERM_EVENT"

        return "DISTANT_EVENT"

    @staticmethod
    def _calculate_risk_score(
        event: MarketEvent,
    ) -> float:
        proximity = EventRiskIntelligence._proximity_factor(
            event.hours_to_event
        )

        score = (
            event.importance
            * event.expected_impact
            * (0.5 + 0.5 * proximity)
            * (0.5 + 0.5 * event.uncertainty)
        )

        return max(0.0, min(1.0, score))

    @staticmethod
    def _proximity_factor(
        hours_to_event: float,
    ) -> float:
        if hours_to_event <= 0.0:
            return 1.0

        if hours_to_event >= 168.0:
            return 0.0

        return 1.0 - (hours_to_event / 168.0)

    @staticmethod
    def _validate_threshold(
        value: float,
        field_name: str,
    ) -> None:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            raise TypeError(
                f"{field_name} must be a number."
            )

        if not isfinite(float(value)):
            raise ValueError(
                f"{field_name} must be finite."
            )

        if value <= 0.0 or value > 1.0:
            raise ValueError(
                f"{field_name} must be greater than zero "
                "and at most one."
            )

    @classmethod
    def _validate_event(
        cls,
        event: MarketEvent,
    ) -> None:
        if not isinstance(event, MarketEvent):
            raise TypeError(
                "event must be a MarketEvent instance."
            )

        if (
            not isinstance(event.symbol, str)
            or not event.symbol.strip()
        ):
            raise ValueError(
                "symbol must be a non-empty string."
            )

        if (
            not isinstance(event.timestamp, int)
            or isinstance(event.timestamp, bool)
            or event.timestamp < 0
        ):
            raise ValueError(
                "timestamp must be a non-negative integer."
            )

        if (
            not isinstance(event.event_type, str)
            or not event.event_type.strip()
        ):
            raise ValueError(
                "event_type must be a non-empty string."
            )

        for field_name in (
            "importance",
            "expected_impact",
            "uncertainty",
        ):
            value = getattr(event, field_name)

            cls._validate_unit_interval(
                value,
                field_name,
            )

        if (
            isinstance(event.hours_to_event, bool)
            or not isinstance(
                event.hours_to_event,
                (int, float),
            )
        ):
            raise TypeError(
                "hours_to_event must be a number."
            )

        if not isfinite(float(event.hours_to_event)):
            raise ValueError(
                "hours_to_event must be finite."
            )

        if event.hours_to_event < 0.0:
            raise ValueError(
                "hours_to_event cannot be negative."
            )

        if not isinstance(event.active, bool):
            raise TypeError(
                "active must be a boolean."
            )

    @staticmethod
    def _validate_unit_interval(
        value: float,
        field_name: str,
    ) -> None:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            raise TypeError(
                f"{field_name} must be a number."
            )

        if not isfinite(float(value)):
            raise ValueError(
                f"{field_name} must be finite."
            )

        if value < 0.0 or value > 1.0:
            raise ValueError(
                f"{field_name} must be between zero and one."
            )


__all__ = [
    "MarketEvent",
    "EventRiskSignal",
    "EventRiskIntelligence",
]