from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any


@dataclass(frozen=True)
class PriceOIDivergenceSnapshot:
    symbol: str
    timestamp: int
    price_change: float
    open_interest_change: float
    price: float
    open_interest: float

    @property
    def pattern(self) -> str:
        if self.price_change > 0 and self.open_interest_change > 0:
            return "PRICE_UP_OI_UP"

        if self.price_change > 0 and self.open_interest_change < 0:
            return "PRICE_UP_OI_DOWN"

        if self.price_change < 0 and self.open_interest_change > 0:
            return "PRICE_DOWN_OI_UP"

        if self.price_change < 0 and self.open_interest_change < 0:
            return "PRICE_DOWN_OI_DOWN"

        return "NEUTRAL"

    @property
    def is_divergence(self) -> bool:
        return self.pattern in {
            "PRICE_UP_OI_DOWN",
            "PRICE_DOWN_OI_UP",
        }


@dataclass(frozen=True)
class PriceOIDivergenceSignal:
    pattern: str
    divergence: bool
    interpretation: str
    contribution: str
    strength: float


class PriceOIDivergenceIntelligence:
    def __init__(
        self,
        minimum_change: float = 0.0,
    ) -> None:
        if (
            not isinstance(minimum_change, (int, float))
            or isinstance(minimum_change, bool)
        ):
            raise TypeError(
                "minimum_change must be numeric."
            )

        if not isfinite(float(minimum_change)):
            raise ValueError(
                "minimum_change must be finite."
            )

        if float(minimum_change) < 0:
            raise ValueError(
                "minimum_change cannot be negative."
            )

        self.minimum_change = float(minimum_change)
        self._previous: (
            PriceOIDivergenceSnapshot | None
        ) = None

    @property
    def previous(
        self,
    ) -> PriceOIDivergenceSnapshot | None:
        return self._previous

    def analyze(
        self,
        snapshot: PriceOIDivergenceSnapshot,
    ) -> PriceOIDivergenceSignal:
        self._validate_snapshot(snapshot)

        pattern = self._pattern(snapshot)

        divergence = pattern in {
            "PRICE_UP_OI_DOWN",
            "PRICE_DOWN_OI_UP",
        }

        interpretation = self._interpretation(
            pattern
        )

        contribution = self._contribution(
            pattern
        )

        strength = self._strength(snapshot)

        self._previous = snapshot

        return PriceOIDivergenceSignal(
            pattern=pattern,
            divergence=divergence,
            interpretation=interpretation,
            contribution=contribution,
            strength=strength,
        )

    def reset(self) -> None:
        self._previous = None

    def _pattern(
        self,
        snapshot: PriceOIDivergenceSnapshot,
    ) -> str:
        price_change = snapshot.price_change
        oi_change = snapshot.open_interest_change

        if abs(price_change) < self.minimum_change:
            price_change = 0.0

        if abs(oi_change) < self.minimum_change:
            oi_change = 0.0

        if price_change > 0 and oi_change > 0:
            return "PRICE_UP_OI_UP"

        if price_change > 0 and oi_change < 0:
            return "PRICE_UP_OI_DOWN"

        if price_change < 0 and oi_change > 0:
            return "PRICE_DOWN_OI_UP"

        if price_change < 0 and oi_change < 0:
            return "PRICE_DOWN_OI_DOWN"

        return "NEUTRAL"

    @staticmethod
    def _interpretation(
        pattern: str,
    ) -> str:
        mapping = {
            "PRICE_UP_OI_UP": "NEW_LONG_BUILDUP",
            "PRICE_UP_OI_DOWN": "SHORT_COVERING",
            "PRICE_DOWN_OI_UP": "NEW_SHORT_BUILDUP",
            "PRICE_DOWN_OI_DOWN": (
                "LONG_LIQUIDATION_OR_UNWINDING"
            ),
            "NEUTRAL": "NO_CLEAR_DIVERGENCE",
        }

        return mapping[pattern]

    @staticmethod
    def _contribution(
        pattern: str,
    ) -> str:
        mapping = {
            "PRICE_UP_OI_UP": "BULLISH_CONFIRMATION",
            "PRICE_UP_OI_DOWN": "BULLISH_CAUTION",
            "PRICE_DOWN_OI_UP": "BEARISH_CONFIRMATION",
            "PRICE_DOWN_OI_DOWN": "BEARISH_CAUTION",
            "NEUTRAL": "NEUTRAL",
        }

        return mapping[pattern]

    @staticmethod
    def _strength(
        snapshot: PriceOIDivergenceSnapshot,
    ) -> float:
        magnitude = (
            abs(snapshot.price_change)
            + abs(snapshot.open_interest_change)
        ) / 2.0

        return min(
            1.0,
            round(magnitude, 6),
        )

    @staticmethod
    def _validate_snapshot(
        snapshot: PriceOIDivergenceSnapshot,
    ) -> None:
        if not isinstance(
            snapshot,
            PriceOIDivergenceSnapshot,
        ):
            raise TypeError(
                "snapshot must be a "
                "PriceOIDivergenceSnapshot instance."
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
            "price_change",
            "open_interest_change",
            "price",
            "open_interest",
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


class PriceOIDivergenceAdapter:
    @staticmethod
    def normalize(
        symbol: str,
        raw: Any,
    ) -> PriceOIDivergenceSnapshot:
        if (
            not isinstance(symbol, str)
            or not symbol.strip()
        ):
            raise ValueError(
                "symbol must be a non-empty string."
            )

        if not isinstance(raw, dict):
            raise TypeError(
                "payload must be a dictionary."
            )

        snapshot = PriceOIDivergenceSnapshot(
            symbol=symbol.strip(),
            timestamp=(
                PriceOIDivergenceAdapter._integer(
                    raw.get("timestamp", 0),
                    "timestamp",
                )
            ),
            price_change=(
                PriceOIDivergenceAdapter._number(
                    raw.get("price_change", 0.0),
                    "price_change",
                )
            ),
            open_interest_change=(
                PriceOIDivergenceAdapter._number(
                    raw.get(
                        "open_interest_change",
                        0.0,
                    ),
                    "open_interest_change",
                )
            ),
            price=(
                PriceOIDivergenceAdapter._number(
                    raw.get("price"),
                    "price",
                )
            ),
            open_interest=(
                PriceOIDivergenceAdapter._number(
                    raw.get("open_interest", 0.0),
                    "open_interest",
                )
            ),
        )

        PriceOIDivergenceIntelligence._validate_snapshot(
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
    "PriceOIDivergenceSnapshot",
    "PriceOIDivergenceSignal",
    "PriceOIDivergenceIntelligence",
    "PriceOIDivergenceAdapter",
]