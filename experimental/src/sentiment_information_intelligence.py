from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Iterable


@dataclass(frozen=True)
class SentimentObservation:
    source: str
    timestamp: int
    sentiment: float
    attention: float
    reliability: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source must be a non-empty string.")

        if (
            not isinstance(self.timestamp, int)
            or isinstance(self.timestamp, bool)
            or self.timestamp < 0
        ):
            raise ValueError(
                "timestamp must be a non-negative integer."
            )

        self._validate_finite(
            self.sentiment,
            "sentiment",
        )
        self._validate_finite(
            self.attention,
            "attention",
        )
        self._validate_finite(
            self.reliability,
            "reliability",
        )

        if not -1.0 <= self.sentiment <= 1.0:
            raise ValueError(
                "sentiment must be between -1 and 1."
            )

        if self.attention < 0.0:
            raise ValueError(
                "attention must be non-negative."
            )

        if not 0.0 <= self.reliability <= 1.0:
            raise ValueError(
                "reliability must be between 0 and 1."
            )

    @staticmethod
    def _validate_finite(
        value: float,
        field_name: str,
    ) -> None:
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
        ):
            raise TypeError(
                f"{field_name} must be numeric."
            )

        if not isfinite(float(value)):
            raise ValueError(
                f"{field_name} must be finite."
            )


@dataclass(frozen=True)
class SentimentSnapshot:
    timestamp: int
    weighted_sentiment: float
    attention: float
    information_quality: float
    source_count: int
    context: str


class SentimentInformationEngine:
    """
    Aggregates normalized sentiment observations.

    This module does not generate trading orders or BUY/SELL signals.
    It produces a market-information context for downstream engines.
    """

    _CONTEXTS = {
        "STRONGLY_BULLISH",
        "BULLISH",
        "NEUTRAL",
        "BEARISH",
        "STRONGLY_BEARISH",
        "LOW_INFORMATION",
    }

    def __init__(
        self,
        minimum_quality: float = 0.25,
        bullish_threshold: float = 0.20,
        strong_bullish_threshold: float = 0.60,
        bearish_threshold: float = -0.20,
        strong_bearish_threshold: float = -0.60,
    ) -> None:
        self._validate_probability(
            minimum_quality,
            "minimum_quality",
        )

        self._validate_threshold(
            bullish_threshold,
            "bullish_threshold",
            lower=0.0,
            upper=1.0,
        )

        self._validate_threshold(
            strong_bullish_threshold,
            "strong_bullish_threshold",
            lower=0.0,
            upper=1.0,
        )

        self._validate_threshold(
            bearish_threshold,
            "bearish_threshold",
            lower=-1.0,
            upper=0.0,
        )

        self._validate_threshold(
            strong_bearish_threshold,
            "strong_bearish_threshold",
            lower=-1.0,
            upper=0.0,
        )

        if strong_bullish_threshold <= bullish_threshold:
            raise ValueError(
                "strong_bullish_threshold must be greater "
                "than bullish_threshold."
            )

        if strong_bearish_threshold >= bearish_threshold:
            raise ValueError(
                "strong_bearish_threshold must be lower "
                "than bearish_threshold."
            )

        self.minimum_quality = float(
            minimum_quality
        )
        self.bullish_threshold = float(
            bullish_threshold
        )
        self.strong_bullish_threshold = float(
            strong_bullish_threshold
        )
        self.bearish_threshold = float(
            bearish_threshold
        )
        self.strong_bearish_threshold = float(
            strong_bearish_threshold
        )

        self._previous: SentimentSnapshot | None = None

    @property
    def previous(self) -> SentimentSnapshot | None:
        return self._previous

    def analyze(
        self,
        observations: Iterable[SentimentObservation],
        timestamp: int | None = None,
    ) -> SentimentSnapshot:
        if observations is None:
            raise TypeError(
                "observations cannot be None."
            )

        items = tuple(observations)

        if not items:
            resolved_timestamp = (
                0
                if timestamp is None
                else self._validate_timestamp(timestamp)
            )

            snapshot = SentimentSnapshot(
                timestamp=resolved_timestamp,
                weighted_sentiment=0.0,
                attention=0.0,
                information_quality=0.0,
                source_count=0,
                context="LOW_INFORMATION",
            )

            self._previous = snapshot

            return snapshot

        for item in items:
            if not isinstance(
                item,
                SentimentObservation,
            ):
                raise TypeError(
                    "observations must contain "
                    "SentimentObservation instances."
                )

        if timestamp is None:
            resolved_timestamp = max(
                item.timestamp
                for item in items
            )
        else:
            resolved_timestamp = self._validate_timestamp(
                timestamp
            )

        total_weight = sum(
            item.attention * item.reliability
            for item in items
        )

        total_attention = sum(
            item.attention
            for item in items
        )

        if total_weight <= 0.0:
            weighted_sentiment = 0.0
            information_quality = 0.0

        else:
            weighted_sentiment = sum(
                item.sentiment
                * item.attention
                * item.reliability
                for item in items
            ) / total_weight

            information_quality = min(
                1.0,
                total_weight / max(
                    float(len(items)),
                    1.0,
                ),
            )

        context = self._classify(
            weighted_sentiment,
            information_quality,
        )

        snapshot = SentimentSnapshot(
            timestamp=resolved_timestamp,
            weighted_sentiment=weighted_sentiment,
            attention=total_attention,
            information_quality=information_quality,
            source_count=len(items),
            context=context,
        )

        self._previous = snapshot

        return snapshot

    def update(
        self,
        observation: SentimentObservation,
    ) -> SentimentSnapshot:
        if not isinstance(
            observation,
            SentimentObservation,
        ):
            raise TypeError(
                "observation must be a "
                "SentimentObservation instance."
            )

        return self.analyze(
            (observation,)
        )

    def reset(self) -> None:
        self._previous = None

    def _classify(
        self,
        sentiment: float,
        information_quality: float,
    ) -> str:
        if information_quality < self.minimum_quality:
            return "LOW_INFORMATION"

        if sentiment >= self.strong_bullish_threshold:
            return "STRONGLY_BULLISH"

        if sentiment >= self.bullish_threshold:
            return "BULLISH"

        if sentiment <= self.strong_bearish_threshold:
            return "STRONGLY_BEARISH"

        if sentiment <= self.bearish_threshold:
            return "BEARISH"

        return "NEUTRAL"

    @staticmethod
    def _validate_timestamp(timestamp: int) -> int:
        if (
            not isinstance(timestamp, int)
            or isinstance(timestamp, bool)
            or timestamp < 0
        ):
            raise ValueError(
                "timestamp must be a non-negative integer."
            )

        return timestamp

    @staticmethod
    def _validate_probability(
        value: float,
        field_name: str,
    ) -> None:
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
        ):
            raise TypeError(
                f"{field_name} must be numeric."
            )

        if not isfinite(float(value)):
            raise ValueError(
                f"{field_name} must be finite."
            )

        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(
                f"{field_name} must be between 0 and 1."
            )

    @staticmethod
    def _validate_threshold(
        value: float,
        field_name: str,
        lower: float,
        upper: float,
    ) -> None:
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
        ):
            raise TypeError(
                f"{field_name} must be numeric."
            )

        if not isfinite(float(value)):
            raise ValueError(
                f"{field_name} must be finite."
            )

        if not lower <= float(value) <= upper:
            raise ValueError(
                f"{field_name} must be between "
                f"{lower} and {upper}."
            )


__all__ = [
    "SentimentObservation",
    "SentimentSnapshot",
    "SentimentInformationEngine",
]