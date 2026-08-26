from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable


@dataclass(frozen=True)
class LiquidityZone:
    lower_price: float
    upper_price: float
    liquidation_volume: float
    liquidation_notional: float
    event_count: int

    @property
    def center_price(self) -> float:
        return (self.lower_price + self.upper_price) / 2.0

    @property
    def width(self) -> float:
        return self.upper_price - self.lower_price

    @property
    def density(self) -> float:
        if self.width == 0:
            return self.liquidation_notional

        return self.liquidation_notional / self.width


@dataclass(frozen=True)
class LiquidityHeatmap:
    symbol: str
    timestamp: int
    zones: tuple[LiquidityZone, ...]

    @property
    def total_liquidation_volume(self) -> float:
        return sum(
            zone.liquidation_volume
            for zone in self.zones
        )

    @property
    def total_liquidation_notional(self) -> float:
        return sum(
            zone.liquidation_notional
            for zone in self.zones
        )

    @property
    def strongest_zone(self) -> LiquidityZone | None:
        if not self.zones:
            return None

        return max(
            self.zones,
            key=lambda zone: zone.liquidation_notional,
        )

    def zones_near_price(
        self,
        price: float,
        max_distance_percent: float,
    ) -> tuple[LiquidityZone, ...]:
        if not isfinite(price) or price <= 0:
            raise ValueError(
                "price must be finite and greater than zero."
            )

        if (
            not isfinite(max_distance_percent)
            or max_distance_percent < 0
        ):
            raise ValueError(
                "max_distance_percent must be finite "
                "and non-negative."
            )

        threshold = (
            price * max_distance_percent / 100.0
        )

        return tuple(
            zone
            for zone in self.zones
            if (
                zone.lower_price - threshold
                <= price
                <= zone.upper_price + threshold
            )
        )


@dataclass(frozen=True)
class LiquidityZoneSignal:
    strongest_zone_price: float | None
    strongest_zone_notional: float
    nearby_zone_count: int
    nearby_liquidation_notional: float
    concentration: float
    context: str


class LiquidityLiquidationZoneEngine:
    def __init__(
        self,
        zone_size: float,
        minimum_notional: float = 0.0,
    ) -> None:
        self._validate_positive_float(
            zone_size,
            "zone_size",
        )

        self._validate_non_negative_float(
            minimum_notional,
            "minimum_notional",
        )

        self.zone_size = float(zone_size)
        self.minimum_notional = float(
            minimum_notional
        )

        self._previous: (
            LiquidityHeatmap | None
        ) = None

    @property
    def previous(
        self,
    ) -> LiquidityHeatmap | None:
        return self._previous

    def build_heatmap(
        self,
        symbol: str,
        timestamp: int,
        events: Iterable[
            tuple[float, float, float]
        ],
    ) -> LiquidityHeatmap:
        if (
            not isinstance(symbol, str)
            or not symbol.strip()
        ):
            raise ValueError(
                "symbol must be a non-empty string."
            )

        if (
            not isinstance(timestamp, int)
            or isinstance(timestamp, bool)
            or timestamp < 0
        ):
            raise ValueError(
                "timestamp must be a non-negative integer."
            )

        buckets: dict[
            float,
            list[float],
        ] = {}

        for row in events:
            if (
                not isinstance(row, (tuple, list))
                or len(row) != 3
            ):
                raise ValueError(
                    "each event must contain "
                    "price, amount and notional."
                )

            price = self._finite_float(
                row[0],
                "price",
            )

            amount = self._finite_float(
                row[1],
                "amount",
            )

            notional = self._finite_float(
                row[2],
                "notional",
            )

            if price <= 0:
                raise ValueError(
                    "price must be greater than zero."
                )

            if amount <= 0:
                raise ValueError(
                    "amount must be greater than zero."
                )

            if notional <= 0:
                raise ValueError(
                    "notional must be greater than zero."
                )

            expected = price * amount

            tolerance = max(
                1e-9,
                abs(expected) * 1e-9,
            )

            if abs(
                notional - expected
            ) > tolerance:
                raise ValueError(
                    "notional must equal price "
                    "multiplied by amount."
                )

            if (
                notional
                < self.minimum_notional
            ):
                continue

            bucket_index = int(
                price / self.zone_size
            )

            lower_price = (
                bucket_index * self.zone_size
            )

            if price < lower_price:
                bucket_index -= 1
                lower_price = (
                    bucket_index
                    * self.zone_size
                )

            bucket = buckets.setdefault(
                lower_price,
                [0.0, 0.0, 0.0],
            )

            bucket[0] += amount
            bucket[1] += notional
            bucket[2] += 1.0

        zones = tuple(
            LiquidityZone(
                lower_price=lower_price,
                upper_price=(
                    lower_price
                    + self.zone_size
                ),
                liquidation_volume=values[0],
                liquidation_notional=values[1],
                event_count=int(values[2]),
            )
            for lower_price, values in sorted(
                buckets.items(),
                key=lambda item: item[0],
            )
        )

        heatmap = LiquidityHeatmap(
            symbol=symbol.strip(),
            timestamp=timestamp,
            zones=zones,
        )

        self._validate_heatmap(heatmap)

        self._previous = heatmap

        return heatmap

    def analyze(
        self,
        heatmap: LiquidityHeatmap,
        current_price: float,
        proximity_percent: float = 1.0,
    ) -> LiquidityZoneSignal:
        self._validate_heatmap(heatmap)

        if (
            not isfinite(current_price)
            or current_price <= 0
        ):
            raise ValueError(
                "current_price must be finite "
                "and greater than zero."
            )

        if (
            not isfinite(proximity_percent)
            or proximity_percent < 0
        ):
            raise ValueError(
                "proximity_percent must be finite "
                "and non-negative."
            )

        nearby = heatmap.zones_near_price(
            current_price,
            proximity_percent,
        )

        strongest = heatmap.strongest_zone

        nearby_notional = sum(
            zone.liquidation_notional
            for zone in nearby
        )

        total_notional = (
            heatmap.total_liquidation_notional
        )

        concentration = (
            nearby_notional / total_notional
            if total_notional > 0
            else 0.0
        )

        if not nearby:
            context = "NO_NEARBY_LIQUIDITY"

        elif concentration >= 0.5:
            context = (
                "HIGH_LIQUIDITY_CONCENTRATION"
            )

        else:
            context = "LIQUIDITY_PRESENT"

        return LiquidityZoneSignal(
            strongest_zone_price=(
                strongest.center_price
                if strongest is not None
                else None
            ),
            strongest_zone_notional=(
                strongest.liquidation_notional
                if strongest is not None
                else 0.0
            ),
            nearby_zone_count=len(nearby),
            nearby_liquidation_notional=(
                nearby_notional
            ),
            concentration=concentration,
            context=context,
        )

    def reset(self) -> None:
        self._previous = None

    @staticmethod
    def _validate_heatmap(
        heatmap: LiquidityHeatmap,
    ) -> None:
        if not isinstance(
            heatmap,
            LiquidityHeatmap,
        ):
            raise TypeError(
                "heatmap must be a "
                "LiquidityHeatmap instance."
            )

        if (
            not isinstance(
                heatmap.symbol,
                str,
            )
            or not heatmap.symbol
        ):
            raise ValueError(
                "symbol must be a non-empty string."
            )

        if (
            not isinstance(
                heatmap.timestamp,
                int,
            )
            or isinstance(
                heatmap.timestamp,
                bool,
            )
            or heatmap.timestamp < 0
        ):
            raise ValueError(
                "timestamp must be a "
                "non-negative integer."
            )

        for zone in heatmap.zones:
            if not isinstance(
                zone,
                LiquidityZone,
            ):
                raise TypeError(
                    "zones must contain "
                    "LiquidityZone instances."
                )

            if (
                not isfinite(
                    zone.lower_price
                )
                or not isfinite(
                    zone.upper_price
                )
                or zone.lower_price < 0
                or zone.upper_price
                <= zone.lower_price
            ):
                raise ValueError(
                    "zone prices must be "
                    "finite and valid."
                )

            if (
                not isfinite(
                    zone.liquidation_volume
                )
                or zone.liquidation_volume < 0
            ):
                raise ValueError(
                    "liquidation_volume must be "
                    "finite and non-negative."
                )

            if (
                not isfinite(
                    zone.liquidation_notional
                )
                or zone.liquidation_notional < 0
            ):
                raise ValueError(
                    "liquidation_notional must be "
                    "finite and non-negative."
                )

            if zone.event_count < 0:
                raise ValueError(
                    "event_count must be "
                    "non-negative."
                )

    @staticmethod
    def _validate_positive_float(
        value: float,
        field_name: str,
    ) -> None:
        if (
            not isinstance(
                value,
                (int, float),
            )
            or isinstance(
                value,
                bool,
            )
            or not isfinite(float(value))
            or value <= 0
        ):
            raise ValueError(
                f"{field_name} must be finite "
                "and greater than zero."
            )

    @staticmethod
    def _validate_non_negative_float(
        value: float,
        field_name: str,
    ) -> None:
        if (
            not isinstance(
                value,
                (int, float),
            )
            or isinstance(
                value,
                bool,
            )
            or not isfinite(float(value))
            or value < 0
        ):
            raise ValueError(
                f"{field_name} must be finite "
                "and non-negative."
            )

    @staticmethod
    def _finite_float(
        value: object,
        field_name: str,
    ) -> float:
        try:
            converted = float(value)

        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                f"{field_name} must be numeric."
            ) from exc

        if not isfinite(converted):
            raise ValueError(
                f"{field_name} must be finite."
            )

        return converted


__all__ = [
    "LiquidityZone",
    "LiquidityHeatmap",
    "LiquidityZoneSignal",
    "LiquidityLiquidationZoneEngine",
]