from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any


@dataclass(frozen=True)
class DerivativesSnapshot:
    symbol: str
    timestamp: int
    price: float
    open_interest: float
    open_interest_change: float
    funding_rate: float
    futures_volume: float
    long_short_ratio: float
    liquidation_volume: float
    basis: float
    price_change: float

    @property
    def oi_change_percent(self) -> float:
        if self.open_interest == 0:
            return 0.0

        return (
            self.open_interest_change
            / self.open_interest
        ) * 100.0

    @property
    def price_oi_direction(self) -> str:
        if (
            self.price_change > 0
            and self.open_interest_change > 0
        ):
            return "PRICE_UP_OI_UP"

        if (
            self.price_change > 0
            and self.open_interest_change < 0
        ):
            return "PRICE_UP_OI_DOWN"

        if (
            self.price_change < 0
            and self.open_interest_change > 0
        ):
            return "PRICE_DOWN_OI_UP"

        if (
            self.price_change < 0
            and self.open_interest_change < 0
        ):
            return "PRICE_DOWN_OI_DOWN"

        return "NEUTRAL"


@dataclass(frozen=True)
class DerivativesSignal:
    market_state: str
    pressure: str
    price_oi_pattern: str
    funding_signal: str
    liquidation_signal: str
    basis_signal: str
    confidence: float


class FuturesDerivativesIntelligence:
    VALID_STATES = {
        "BULLISH",
        "BEARISH",
        "NEUTRAL",
        "STRESS",
    }

    def __init__(
        self,
        funding_threshold: float = 0.0005,
        liquidation_threshold: float = 0.0,
        basis_threshold: float = 0.001,
    ) -> None:
        self._validate_threshold(
            funding_threshold,
            "funding_threshold",
        )

        self._validate_non_negative_threshold(
            liquidation_threshold,
            "liquidation_threshold",
        )

        self._validate_threshold(
            basis_threshold,
            "basis_threshold",
        )

        self.funding_threshold = funding_threshold
        self.liquidation_threshold = liquidation_threshold
        self.basis_threshold = basis_threshold

        self._previous: DerivativesSnapshot | None = None

    @property
    def previous(self) -> DerivativesSnapshot | None:
        return self._previous

    def analyze(
        self,
        snapshot: DerivativesSnapshot,
    ) -> DerivativesSignal:
        self._validate_snapshot(snapshot)

        funding_signal = self._funding_signal(
            snapshot.funding_rate
        )

        liquidation_signal = self._liquidation_signal(
            snapshot.liquidation_volume
        )

        basis_signal = self._basis_signal(
            snapshot.basis
        )

        price_oi_pattern = (
            snapshot.price_oi_direction
        )

        pressure = self._derive_pressure(
            price_oi_pattern,
            funding_signal,
            basis_signal,
        )

        market_state = self._derive_market_state(
            pressure,
            liquidation_signal,
            snapshot,
        )

        confidence = self._confidence(
            price_oi_pattern,
            funding_signal,
            liquidation_signal,
            basis_signal,
        )

        self._previous = snapshot

        return DerivativesSignal(
            market_state=market_state,
            pressure=pressure,
            price_oi_pattern=price_oi_pattern,
            funding_signal=funding_signal,
            liquidation_signal=liquidation_signal,
            basis_signal=basis_signal,
            confidence=confidence,
        )

    def reset(self) -> None:
        self._previous = None

    def _derive_pressure(
        self,
        price_oi_pattern: str,
        funding_signal: str,
        basis_signal: str,
    ) -> str:
        bullish = 0
        bearish = 0

        if price_oi_pattern == "PRICE_UP_OI_UP":
            bullish += 2

        elif price_oi_pattern == "PRICE_DOWN_OI_DOWN":
            bullish += 1

        elif price_oi_pattern == "PRICE_DOWN_OI_UP":
            bearish += 2

        elif price_oi_pattern == "PRICE_UP_OI_DOWN":
            bearish += 1

        if funding_signal == "POSITIVE_EXTREME":
            bearish += 1

        elif funding_signal == "NEGATIVE_EXTREME":
            bullish += 1

        if basis_signal == "POSITIVE_PREMIUM":
            bullish += 1

        elif basis_signal == "NEGATIVE_DISCOUNT":
            bearish += 1

        if bullish > bearish:
            return "BULLISH_PRESSURE"

        if bearish > bullish:
            return "BEARISH_PRESSURE"

        return "BALANCED_PRESSURE"

    def _derive_market_state(
        self,
        pressure: str,
        liquidation_signal: str,
        snapshot: DerivativesSnapshot,
    ) -> str:
        if liquidation_signal == "HIGH_LIQUIDATION":
            return "STRESS"

        if (
            snapshot.open_interest <= 0
            or snapshot.futures_volume <= 0
        ):
            return "NEUTRAL"

        if pressure == "BULLISH_PRESSURE":
            return "BULLISH"

        if pressure == "BEARISH_PRESSURE":
            return "BEARISH"

        return "NEUTRAL"

    def _confidence(
        self,
        price_oi_pattern: str,
        funding_signal: str,
        liquidation_signal: str,
        basis_signal: str,
    ) -> float:
        score = 0.25

        if price_oi_pattern != "NEUTRAL":
            score += 0.25

        if funding_signal != "NEUTRAL":
            score += 0.15

        if liquidation_signal == "HIGH_LIQUIDATION":
            score += 0.20

        elif liquidation_signal == "ELEVATED_LIQUIDATION":
            score += 0.10

        if basis_signal != "NEUTRAL":
            score += 0.15

        return min(
            1.0,
            round(score, 6),
        )

    def _funding_signal(
        self,
        funding_rate: float,
    ) -> str:
        if funding_rate > self.funding_threshold:
            return "POSITIVE_EXTREME"

        if funding_rate < -self.funding_threshold:
            return "NEGATIVE_EXTREME"

        return "NEUTRAL"

    def _liquidation_signal(
        self,
        liquidation_volume: float,
    ) -> str:
        if (
            liquidation_volume
            >= self.liquidation_threshold
            and self.liquidation_threshold > 0
        ):
            if (
                liquidation_volume
                >= self.liquidation_threshold * 2.0
            ):
                return "HIGH_LIQUIDATION"

            return "ELEVATED_LIQUIDATION"

        return "NORMAL"

    def _basis_signal(
        self,
        basis: float,
    ) -> str:
        if basis > self.basis_threshold:
            return "POSITIVE_PREMIUM"

        if basis < -self.basis_threshold:
            return "NEGATIVE_DISCOUNT"

        return "NEUTRAL"

    @staticmethod
    def _validate_snapshot(
        snapshot: DerivativesSnapshot,
    ) -> None:
        if not isinstance(
            snapshot,
            DerivativesSnapshot,
        ):
            raise TypeError(
                "snapshot must be a "
                "DerivativesSnapshot instance."
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
                "timestamp must be a "
                "non-negative integer."
            )

        for name in (
            "price",
            "open_interest",
            "open_interest_change",
            "funding_rate",
            "futures_volume",
            "long_short_ratio",
            "liquidation_volume",
            "basis",
            "price_change",
        ):
            value = getattr(snapshot, name)

            if not isfinite(value):
                raise ValueError(
                    f"{name} must be finite."
                )

        if snapshot.price <= 0:
            raise ValueError(
                "price must be greater than zero."
            )

        if snapshot.open_interest < 0:
            raise ValueError(
                "open_interest cannot be negative."
            )

        if snapshot.futures_volume < 0:
            raise ValueError(
                "futures_volume cannot be negative."
            )

        if snapshot.long_short_ratio < 0:
            raise ValueError(
                "long_short_ratio cannot be negative."
            )

        if snapshot.liquidation_volume < 0:
            raise ValueError(
                "liquidation_volume cannot be negative."
            )

    @staticmethod
    def _validate_threshold(
        value: float,
        name: str,
    ) -> None:
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
        ):
            raise TypeError(
                f"{name} must be numeric."
            )

        if not isfinite(float(value)):
            raise ValueError(
                f"{name} must be finite."
            )

        if float(value) <= 0:
            raise ValueError(
                f"{name} must be greater than zero."
            )

    @staticmethod
    def _validate_non_negative_threshold(
        value: float,
        name: str,
    ) -> None:
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
        ):
            raise TypeError(
                f"{name} must be numeric."
            )

        if not isfinite(float(value)):
            raise ValueError(
                f"{name} must be finite."
            )

        if float(value) < 0:
            raise ValueError(
                f"{name} cannot be negative."
            )


class DerivativesSnapshotAdapter:
    """
    Normalize an external derivatives payload
    into DerivativesSnapshot.
    """

    @staticmethod
    def normalize(
        symbol: str,
        raw: Any,
    ) -> DerivativesSnapshot:
        if (
            not isinstance(symbol, str)
            or not symbol.strip()
        ):
            raise ValueError(
                "symbol must be a non-empty string."
            )

        if not isinstance(raw, dict):
            raise TypeError(
                "derivatives payload must be a dictionary."
            )

        timestamp = (
            DerivativesSnapshotAdapter._integer(
                raw.get("timestamp", 0),
                "timestamp",
            )
        )

        if timestamp < 0:
            raise ValueError(
                "timestamp cannot be negative."
            )

        snapshot = DerivativesSnapshot(
            symbol=symbol.strip(),
            timestamp=timestamp,
            price=(
                DerivativesSnapshotAdapter._number(
                    raw.get("price"),
                    "price",
                )
            ),
            open_interest=(
                DerivativesSnapshotAdapter._number(
                    raw.get("open_interest", 0.0),
                    "open_interest",
                )
            ),
            open_interest_change=(
                DerivativesSnapshotAdapter._number(
                    raw.get(
                        "open_interest_change",
                        0.0,
                    ),
                    "open_interest_change",
                )
            ),
            funding_rate=(
                DerivativesSnapshotAdapter._number(
                    raw.get("funding_rate", 0.0),
                    "funding_rate",
                )
            ),
            futures_volume=(
                DerivativesSnapshotAdapter._number(
                    raw.get("futures_volume", 0.0),
                    "futures_volume",
                )
            ),
            long_short_ratio=(
                DerivativesSnapshotAdapter._number(
                    raw.get("long_short_ratio", 0.0),
                    "long_short_ratio",
                )
            ),
            liquidation_volume=(
                DerivativesSnapshotAdapter._number(
                    raw.get(
                        "liquidation_volume",
                        0.0,
                    ),
                    "liquidation_volume",
                )
            ),
            basis=(
                DerivativesSnapshotAdapter._number(
                    raw.get("basis", 0.0),
                    "basis",
                )
            ),
            price_change=(
                DerivativesSnapshotAdapter._number(
                    raw.get("price_change", 0.0),
                    "price_change",
                )
            ),
        )

        FuturesDerivativesIntelligence._validate_snapshot(
            snapshot
        )

        return snapshot

    @staticmethod
    def _number(
        value: Any,
        name: str,
    ) -> float:
        try:
            converted = float(value)

        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{name} must be numeric."
            ) from exc

        if not isfinite(converted):
            raise ValueError(
                f"{name} must be finite."
            )

        return converted

    @staticmethod
    def _integer(
        value: Any,
        name: str,
    ) -> int:
        try:
            converted = int(value)

        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{name} must be numeric."
            ) from exc

        return converted


__all__ = [
    "DerivativesSnapshot",
    "DerivativesSignal",
    "FuturesDerivativesIntelligence",
    "DerivativesSnapshotAdapter",
]