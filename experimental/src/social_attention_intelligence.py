from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class SocialAttentionSnapshot:
    symbol: str
    timestamp: int
    attention: float
    social_volume: float
    engagement: float
    mentions: float
    contributors: float = 0.0

    @property
    def attention_per_mention(self) -> float:
        return self.attention / self.mentions if self.mentions else 0.0

    @property
    def engagement_per_mention(self) -> float:
        return self.engagement / self.mentions if self.mentions else 0.0


@dataclass(frozen=True)
class SocialAttentionSignal:
    attention: float
    social_volume: float
    engagement: float
    mentions: float
    attention_change: float
    social_volume_change: float
    engagement_change: float
    attention_zscore: float
    anomaly: bool
    context: str


class SocialAttentionIntelligence:
    def __init__(
        self,
        anomaly_zscore: float = 2.0,
        min_history: int = 3,
    ) -> None:
        self._validate_anomaly_zscore(anomaly_zscore)
        self._validate_min_history(min_history)

        self.anomaly_zscore = float(anomaly_zscore)
        self.min_history = min_history

        self._previous: SocialAttentionSnapshot | None = None
        self._history: list[SocialAttentionSnapshot] = []

    @property
    def previous(self) -> SocialAttentionSnapshot | None:
        return self._previous

    @property
    def history(self) -> tuple[SocialAttentionSnapshot, ...]:
        return tuple(self._history)

    def update(
        self,
        snapshot: SocialAttentionSnapshot,
    ) -> SocialAttentionSignal:
        self._validate_snapshot(snapshot)

        if self._previous is not None:
            if snapshot.symbol != self._previous.symbol:
                raise ValueError(
                    "symbol must match the previous "
                    "social-attention snapshot."
                )

            if snapshot.timestamp <= self._previous.timestamp:
                raise ValueError(
                    "timestamp must be greater than the "
                    "previous timestamp."
                )

        attention_change = self._relative_change(
            self._previous.attention
            if self._previous is not None
            else None,
            snapshot.attention,
        )

        social_volume_change = self._relative_change(
            self._previous.social_volume
            if self._previous is not None
            else None,
            snapshot.social_volume,
        )

        engagement_change = self._relative_change(
            self._previous.engagement
            if self._previous is not None
            else None,
            snapshot.engagement,
        )

        attention_zscore = self._attention_zscore(
            snapshot.attention
        )

        anomaly = (
            len(self._history) >= self.min_history
            and abs(attention_zscore) >= self.anomaly_zscore
        )

        context = self._classify_context(
            snapshot=snapshot,
            attention_change=attention_change,
            social_volume_change=social_volume_change,
            engagement_change=engagement_change,
            anomaly=anomaly,
        )

        self._history.append(snapshot)
        self._previous = snapshot

        return SocialAttentionSignal(
            attention=snapshot.attention,
            social_volume=snapshot.social_volume,
            engagement=snapshot.engagement,
            mentions=snapshot.mentions,
            attention_change=attention_change,
            social_volume_change=social_volume_change,
            engagement_change=engagement_change,
            attention_zscore=attention_zscore,
            anomaly=anomaly,
            context=context,
        )

    def reset(self) -> None:
        self._previous = None
        self._history.clear()

    def _attention_zscore(
        self,
        attention: float,
    ) -> float:
        if not self._history:
            return 0.0

        values = [
            item.attention
            for item in self._history
        ]

        mean = sum(values) / len(values)

        if len(values) < 2:
            return 0.0

        variance = sum(
            (value - mean) ** 2
            for value in values
        ) / len(values)

        if variance == 0:
            return 0.0

        return (
            attention - mean
        ) / (variance ** 0.5)

    @staticmethod
    def _relative_change(
        previous: float | None,
        current: float,
    ) -> float:
        if previous is None:
            return 0.0

        if previous == 0:
            if current == 0:
                return 0.0

            return 1.0

        return (
            current - previous
        ) / abs(previous)

    @staticmethod
    def _classify_context(
        snapshot: SocialAttentionSnapshot,
        attention_change: float,
        social_volume_change: float,
        engagement_change: float,
        anomaly: bool,
    ) -> str:
        if anomaly:
            return "ATTENTION_ANOMALY"

        if (
            attention_change > 0
            and social_volume_change > 0
            and engagement_change > 0
        ):
            return "RISING_ATTENTION"

        if (
            attention_change < 0
            and social_volume_change < 0
            and engagement_change < 0
        ):
            return "FALLING_ATTENTION"

        if (
            snapshot.attention > 0
            and snapshot.mentions > 0
            and snapshot.engagement > 0
        ):
            return "MIXED_ATTENTION"

        return "LOW_ATTENTION"

    @staticmethod
    def _validate_anomaly_zscore(
        value: float,
    ) -> None:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            raise TypeError(
                "anomaly_zscore must be a finite positive number."
            )

        if not isfinite(float(value)):
            raise ValueError(
                "anomaly_zscore must be a finite positive number."
            )

        if value <= 0:
            raise ValueError(
                "anomaly_zscore must be greater than zero."
            )

    @staticmethod
    def _validate_min_history(
        value: int,
    ) -> None:
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
        ):
            raise TypeError(
                "min_history must be an integer."
            )

        if value < 1:
            raise ValueError(
                "min_history must be greater than zero."
            )

    @classmethod
    def _validate_snapshot(
        cls,
        snapshot: SocialAttentionSnapshot,
    ) -> None:
        if not isinstance(
            snapshot,
            SocialAttentionSnapshot,
        ):
            raise TypeError(
                "snapshot must be a "
                "SocialAttentionSnapshot instance."
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
            "attention",
            "social_volume",
            "engagement",
            "mentions",
            "contributors",
        ):
            value = getattr(
                snapshot,
                field_name,
            )

            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(float(value))
            ):
                raise ValueError(
                    f"{field_name} must be a "
                    "finite non-negative number."
                )

            if value < 0:
                raise ValueError(
                    f"{field_name} must be non-negative."
                )


__all__ = [
    "SocialAttentionSnapshot",
    "SocialAttentionSignal",
    "SocialAttentionIntelligence",
]