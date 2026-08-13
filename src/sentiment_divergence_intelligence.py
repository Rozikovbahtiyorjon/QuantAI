from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class SentimentDivergenceSnapshot:
    symbol: str
    timestamp: int
    price_change: float
    sentiment_change: float
    attention_change: float
    volume_change: float

    @property
    def price_direction(self) -> int:
        return self._direction(self.price_change)

    @property
    def sentiment_direction(self) -> int:
        return self._direction(self.sentiment_change)

    @property
    def attention_direction(self) -> int:
        return self._direction(self.attention_change)

    @property
    def volume_direction(self) -> int:
        return self._direction(self.volume_change)

    @staticmethod
    def _direction(value: float) -> int:
        if value > 0:
            return 1

        if value < 0:
            return -1

        return 0


@dataclass(frozen=True)
class SentimentDivergenceSignal:
    price_change: float
    sentiment_change: float
    attention_change: float
    volume_change: float
    divergence_score: float
    divergence: bool
    context: str


class SentimentDivergenceIntelligence:
    def __init__(
        self,
        divergence_threshold: float = 0.5,
    ) -> None:
        self._validate_threshold(
            divergence_threshold
        )

        self.divergence_threshold = float(
            divergence_threshold
        )

        self._previous: (
            SentimentDivergenceSnapshot | None
        ) = None

    @property
    def previous(
        self,
    ) -> SentimentDivergenceSnapshot | None:
        return self._previous

    def update(
        self,
        snapshot: SentimentDivergenceSnapshot,
    ) -> SentimentDivergenceSignal:
        self._validate_snapshot(snapshot)

        if self._previous is not None:
            if snapshot.symbol != self._previous.symbol:
                raise ValueError(
                    "symbol must match the previous "
                    "sentiment-divergence snapshot."
                )

            if snapshot.timestamp <= self._previous.timestamp:
                raise ValueError(
                    "timestamp must be greater than the "
                    "previous timestamp."
                )

        divergence_score = (
            self._calculate_divergence_score(
                snapshot
            )
        )

        divergence = (
            divergence_score
            >= self.divergence_threshold
        )

        context = self._classify_context(
            snapshot,
            divergence,
        )

        self._previous = snapshot

        return SentimentDivergenceSignal(
            price_change=snapshot.price_change,
            sentiment_change=snapshot.sentiment_change,
            attention_change=snapshot.attention_change,
            volume_change=snapshot.volume_change,
            divergence_score=divergence_score,
            divergence=divergence,
            context=context,
        )

    def reset(self) -> None:
        self._previous = None

    @staticmethod
    def _calculate_divergence_score(
        snapshot: SentimentDivergenceSnapshot,
    ) -> float:
        price_direction = snapshot.price_direction

        sentiment_direction = (
            snapshot.sentiment_direction
        )

        attention_direction = (
            snapshot.attention_direction
        )

        volume_direction = (
            snapshot.volume_direction
        )

        active_factors = 0
        opposing_factors = 0

        for direction in (
            sentiment_direction,
            attention_direction,
            volume_direction,
        ):
            if direction == 0 or price_direction == 0:
                continue

            active_factors += 1

            if direction != price_direction:
                opposing_factors += 1

        if active_factors == 0:
            return 0.0

        return opposing_factors / active_factors

    @staticmethod
    def _classify_context(
        snapshot: SentimentDivergenceSnapshot,
        divergence: bool,
    ) -> str:
        if not divergence:
            return "ALIGNED"

        if snapshot.price_change > 0:
            return "BEARISH_SENTIMENT_DIVERGENCE"

        if snapshot.price_change < 0:
            return "BULLISH_SENTIMENT_DIVERGENCE"

        return "NEUTRAL_PRICE_DIVERGENCE"

    @staticmethod
    def _validate_threshold(
        value: float,
    ) -> None:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            raise TypeError(
                "divergence_threshold must be a "
                "finite number."
            )

        if not isfinite(float(value)):
            raise ValueError(
                "divergence_threshold must be finite."
            )

        if value <= 0 or value > 1:
            raise ValueError(
                "divergence_threshold must be "
                "greater than zero and at most one."
            )

    @classmethod
    def _validate_snapshot(
        cls,
        snapshot: SentimentDivergenceSnapshot,
    ) -> None:
        if not isinstance(
            snapshot,
            SentimentDivergenceSnapshot,
        ):
            raise TypeError(
                "snapshot must be a "
                "SentimentDivergenceSnapshot instance."
            )

        if (
            not isinstance(snapshot.symbol, str)
            or not snapshot.symbol.strip()
        ):
            raise ValueError(
                "symbol must be a non-empty string."
            )

        if (
            not isinstance(snapshot.timestamp, int)
            or isinstance(snapshot.timestamp, bool)
            or snapshot.timestamp < 0
        ):
            raise ValueError(
                "timestamp must be a non-negative integer."
            )

        for field_name in (
            "price_change",
            "sentiment_change",
            "attention_change",
            "volume_change",
        ):
            value = getattr(
                snapshot,
                field_name,
            )

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


__all__ = [
    "SentimentDivergenceSnapshot",
    "SentimentDivergenceSignal",
    "SentimentDivergenceIntelligence",
]