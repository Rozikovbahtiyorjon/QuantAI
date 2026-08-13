from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any


@dataclass(frozen=True)
class LiquidationEvent:
    symbol: str
    timestamp: int
    side: str
    price: float
    amount: float
    notional: float


@dataclass(frozen=True)
class LiquidationSnapshot:
    symbol: str
    timestamp: int
    events: tuple[LiquidationEvent, ...]

    @property
    def total_volume(self) -> float:
        return sum(event.amount for event in self.events)

    @property
    def total_notional(self) -> float:
        return sum(event.notional for event in self.events)

    @property
    def long_liquidation_volume(self) -> float:
        return sum(
            event.amount
            for event in self.events
            if event.side == "LONG"
        )

    @property
    def short_liquidation_volume(self) -> float:
        return sum(
            event.amount
            for event in self.events
            if event.side == "SHORT"
        )

    @property
    def long_liquidation_notional(self) -> float:
        return sum(
            event.notional
            for event in self.events
            if event.side == "LONG"
        )

    @property
    def short_liquidation_notional(self) -> float:
        return sum(
            event.notional
            for event in self.events
            if event.side == "SHORT"
        )

    @property
    def imbalance(self) -> float:
        total = (
            self.long_liquidation_volume
            + self.short_liquidation_volume
        )

        if total == 0:
            return 0.0

        return (
            self.short_liquidation_volume
            - self.long_liquidation_volume
        ) / total


@dataclass(frozen=True)
class LiquidationSignal:
    total_volume: float
    total_notional: float
    long_volume: float
    short_volume: float
    imbalance: float
    intensity: float
    context: str


class LiquidationIntelligenceEngine:
    VALID_SIDES = frozenset({"LONG", "SHORT"})

    def __init__(
        self,
        baseline_volume: float = 0.0,
    ) -> None:
        if (
            not isinstance(
                baseline_volume,
                (int, float),
            )
            or isinstance(
                baseline_volume,
                bool,
            )
        ):
            raise TypeError(
                "baseline_volume must be numeric."
            )

        if (
            not isfinite(
                float(baseline_volume)
            )
            or baseline_volume < 0
        ):
            raise ValueError(
                "baseline_volume must be finite "
                "and non-negative."
            )

        self.baseline_volume = float(
            baseline_volume
        )

        self._previous: (
            LiquidationSnapshot | None
        ) = None

    @property
    def previous(
        self,
    ) -> LiquidationSnapshot | None:
        return self._previous

    def update(
        self,
        snapshot: LiquidationSnapshot,
    ) -> LiquidationSignal:
        self._validate_snapshot(snapshot)

        if (
            self._previous is not None
            and snapshot.symbol
            != self._previous.symbol
        ):
            raise ValueError(
                "symbol must match the previous "
                "liquidation snapshot."
            )

        intensity = (
            snapshot.total_volume
            / self.baseline_volume
            if self.baseline_volume > 0
            else snapshot.total_volume
        )

        if snapshot.total_volume == 0:
            context = "NO_LIQUIDATIONS"

        elif snapshot.imbalance > 0:
            context = "SHORT_LIQUIDATION_DOMINANT"

        elif snapshot.imbalance < 0:
            context = "LONG_LIQUIDATION_DOMINANT"

        else:
            context = "BALANCED_LIQUIDATIONS"

        self._previous = snapshot

        return LiquidationSignal(
            total_volume=snapshot.total_volume,
            total_notional=snapshot.total_notional,
            long_volume=(
                snapshot.long_liquidation_volume
            ),
            short_volume=(
                snapshot.short_liquidation_volume
            ),
            imbalance=snapshot.imbalance,
            intensity=intensity,
            context=context,
        )

    def reset(self) -> None:
        self._previous = None

    @classmethod
    def _validate_snapshot(
        cls,
        snapshot: LiquidationSnapshot,
    ) -> None:
        if not isinstance(
            snapshot,
            LiquidationSnapshot,
        ):
            raise TypeError(
                "snapshot must be a "
                "LiquidationSnapshot instance."
            )

        if (
            not isinstance(
                snapshot.symbol,
                str,
            )
            or not snapshot.symbol
        ):
            raise ValueError(
                "symbol must be a non-empty string."
            )

        if (
            not isinstance(
                snapshot.timestamp,
                int,
            )
            or isinstance(
                snapshot.timestamp,
                bool,
            )
            or snapshot.timestamp < 0
        ):
            raise ValueError(
                "timestamp must be a "
                "non-negative integer."
            )

        if not isinstance(
            snapshot.events,
            tuple,
        ):
            raise TypeError(
                "events must be a tuple."
            )

        for event in snapshot.events:
            cls._validate_event(event)

            if event.symbol != snapshot.symbol:
                raise ValueError(
                    "event symbol must match "
                    "snapshot symbol."
                )

    @classmethod
    def _validate_event(
        cls,
        event: LiquidationEvent,
    ) -> None:
        if not isinstance(
            event,
            LiquidationEvent,
        ):
            raise TypeError(
                "events must contain "
                "LiquidationEvent instances."
            )

        if event.side not in cls.VALID_SIDES:
            raise ValueError(
                "side must be LONG or SHORT."
            )

        if (
            not isfinite(event.price)
            or event.price <= 0
        ):
            raise ValueError(
                "price must be finite "
                "and greater than zero."
            )

        if (
            not isfinite(event.amount)
            or event.amount <= 0
        ):
            raise ValueError(
                "amount must be finite "
                "and greater than zero."
            )

        if (
            not isfinite(event.notional)
            or event.notional <= 0
        ):
            raise ValueError(
                "notional must be finite "
                "and greater than zero."
            )

        expected = (
            event.price * event.amount
        )

        tolerance = max(
            1e-9,
            abs(expected) * 1e-9,
        )

        if abs(
            event.notional - expected
        ) > tolerance:
            raise ValueError(
                "notional must equal "
                "price multiplied by amount."
            )

    @classmethod
    def _normalize_events(
        cls,
        raw_events: Any,
    ) -> tuple[LiquidationEvent, ...]:
        if raw_events is None:
            return ()

        if not isinstance(
            raw_events,
            list,
        ):
            raise TypeError(
                "liquidation events must be a list."
            )

        events: list[
            LiquidationEvent
        ] = []

        for row in raw_events:
            if not isinstance(
                row,
                dict,
            ):
                raise ValueError(
                    "liquidation event must be "
                    "a dictionary."
                )

            symbol = row.get("symbol")

            if (
                not isinstance(
                    symbol,
                    str,
                )
                or not symbol.strip()
            ):
                raise ValueError(
                    "event symbol must be "
                    "a non-empty string."
                )

            symbol = symbol.strip()

            timestamp_value = row.get(
                "timestamp",
                0,
            )

            try:
                timestamp = int(
                    timestamp_value
                )
            except (
                TypeError,
                ValueError,
            ) as exc:
                raise ValueError(
                    "event timestamp must be numeric."
                ) from exc

            if timestamp < 0:
                raise ValueError(
                    "event timestamp cannot "
                    "be negative."
                )

            side = row.get("side")

            if not isinstance(
                side,
                str,
            ):
                raise TypeError(
                    "event side must be a string."
                )

            side = side.strip().upper()

            price = cls._finite_float(
                row.get("price"),
                "price",
            )

            amount = cls._finite_float(
                row.get("amount"),
                "amount",
            )

            notional_value = row.get(
                "notional"
            )

            if notional_value is None:
                notional = price * amount
            else:
                notional = cls._finite_float(
                    notional_value,
                    "notional",
                )

            event = LiquidationEvent(
                symbol=symbol,
                timestamp=timestamp,
                side=side,
                price=price,
                amount=amount,
                notional=notional,
            )

            cls._validate_event(event)

            events.append(event)

        events.sort(
            key=lambda event: (
                event.timestamp,
                event.price,
            )
        )

        return tuple(events)

    @classmethod
    def normalize(
        cls,
        symbol: str,
        timestamp: int,
        raw_events: Any,
    ) -> LiquidationSnapshot:
        if (
            not isinstance(
                symbol,
                str,
            )
            or not symbol.strip()
        ):
            raise ValueError(
                "symbol must be a non-empty string."
            )

        if (
            not isinstance(
                timestamp,
                int,
            )
            or isinstance(
                timestamp,
                bool,
            )
            or timestamp < 0
        ):
            raise ValueError(
                "timestamp must be a "
                "non-negative integer."
            )

        symbol = symbol.strip()

        events = cls._normalize_events(
            raw_events
        )

        for event in events:
            if event.symbol != symbol:
                raise ValueError(
                    "event symbol must match "
                    "snapshot symbol."
                )

        return LiquidationSnapshot(
            symbol=symbol,
            timestamp=timestamp,
            events=events,
        )

    @staticmethod
    def _finite_float(
        value: Any,
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
    "LiquidationEvent",
    "LiquidationSnapshot",
    "LiquidationSignal",
    "LiquidationIntelligenceEngine",
]