from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from src.order_book_market_data import OrderBookLevel, OrderBookSnapshot


@dataclass(frozen=True)
class OrderFlowSignal:
    spread: float | None
    spread_percent: float | None
    bid_volume: float
    ask_volume: float
    bid_notional: float
    ask_notional: float
    volume_imbalance: float
    notional_imbalance: float
    microprice: float | None
    microprice_delta: float | None
    bid_liquidity_share: float
    ask_liquidity_share: float
    pressure: float
    context: str


class OrderFlowIntelligenceEngine:
    """
    Deterministic L2 order-flow intelligence layer.

    Consumes validated OrderBookSnapshot objects and derives
    microstructure features useful for strategy filtering and
    execution decisions.

    This engine does not place orders and does not mutate the
    supplied order-book snapshot.
    """

    CONTEXT_BID_PRESSURE = "BID_PRESSURE"
    CONTEXT_ASK_PRESSURE = "ASK_PRESSURE"
    CONTEXT_BALANCED = "BALANCED"
    CONTEXT_NO_LIQUIDITY = "NO_LIQUIDITY"

    def __init__(
        self,
        depth: int | None = None,
        pressure_threshold: float = 0.15,
    ) -> None:
        if depth is not None:
            self._validate_depth(depth)

        self._validate_pressure_threshold(
            pressure_threshold
        )

        self.depth = depth
        self.pressure_threshold = float(
            pressure_threshold
        )

        self._previous: OrderBookSnapshot | None = None
        self._previous_microprice: float | None = None

    @property
    def previous(self) -> OrderBookSnapshot | None:
        return self._previous

    @property
    def previous_microprice(self) -> float | None:
        return self._previous_microprice

    def update(
        self,
        snapshot: OrderBookSnapshot,
    ) -> OrderFlowSignal:
        self._validate_snapshot(snapshot)

        if self._previous is not None:
            if snapshot.symbol != self._previous.symbol:
                raise ValueError(
                    "symbol must match the previous order-book snapshot."
                )

            if snapshot.timestamp <= self._previous.timestamp:
                raise ValueError(
                    "timestamp must be greater than the previous order-book snapshot."
                )

        bids = self._slice_levels(snapshot.bids)
        asks = self._slice_levels(snapshot.asks)

        bid_volume = sum(
            level.amount
            for level in bids
        )

        ask_volume = sum(
            level.amount
            for level in asks
        )

        bid_notional = sum(
            level.notional
            for level in bids
        )

        ask_notional = sum(
            level.notional
            for level in asks
        )

        volume_imbalance = self._imbalance(
            bid_volume,
            ask_volume,
        )

        notional_imbalance = self._imbalance(
            bid_notional,
            ask_notional,
        )

        microprice = self._microprice(snapshot)
        previous_microprice = self._previous_microprice

        if (
            microprice is None
            or previous_microprice is None
        ):
            microprice_delta = None
        else:
            microprice_delta = (
                microprice
                - previous_microprice
            )

        total_notional = (
            bid_notional
            + ask_notional
        )

        if total_notional <= 0.0:
            bid_liquidity_share = 0.0
            ask_liquidity_share = 0.0
        else:
            bid_liquidity_share = (
                bid_notional
                / total_notional
            )

            ask_liquidity_share = (
                ask_notional
                / total_notional
            )

        pressure = self._pressure(
            volume_imbalance,
            notional_imbalance,
        )

        context = self._context(
            pressure=pressure,
            bid_volume=bid_volume,
            ask_volume=ask_volume,
        )

        self._previous = snapshot
        self._previous_microprice = microprice

        return OrderFlowSignal(
            spread=snapshot.spread,
            spread_percent=snapshot.spread_percent,
            bid_volume=bid_volume,
            ask_volume=ask_volume,
            bid_notional=bid_notional,
            ask_notional=ask_notional,
            volume_imbalance=volume_imbalance,
            notional_imbalance=notional_imbalance,
            microprice=microprice,
            microprice_delta=microprice_delta,
            bid_liquidity_share=bid_liquidity_share,
            ask_liquidity_share=ask_liquidity_share,
            pressure=pressure,
            context=context,
        )

    def reset(self) -> None:
        self._previous = None
        self._previous_microprice = None

    def _slice_levels(
        self,
        levels: tuple[OrderBookLevel, ...],
    ) -> tuple[OrderBookLevel, ...]:
        if self.depth is None:
            return levels

        return levels[: self.depth]

    @staticmethod
    def _imbalance(
        bid_value: float,
        ask_value: float,
    ) -> float:
        total = bid_value + ask_value

        if total <= 0.0:
            return 0.0

        return (
            bid_value - ask_value
        ) / total

    @staticmethod
    def _microprice(
        snapshot: OrderBookSnapshot,
    ) -> float | None:
        best_bid = snapshot.best_bid
        best_ask = snapshot.best_ask

        if best_bid is None or best_ask is None:
            return None

        bid_level = snapshot.bids[0]
        ask_level = snapshot.asks[0]

        total = (
            bid_level.amount
            + ask_level.amount
        )

        if total <= 0.0:
            return None

        return (
            best_ask * bid_level.amount
            + best_bid * ask_level.amount
        ) / total

    @staticmethod
    def _pressure(
        volume_imbalance: float,
        notional_imbalance: float,
    ) -> float:
        return max(
            -1.0,
            min(
                1.0,
                0.5 * volume_imbalance
                + 0.5 * notional_imbalance,
            ),
        )

    def _context(
        self,
        pressure: float,
        bid_volume: float,
        ask_volume: float,
    ) -> str:
        if (
            bid_volume <= 0.0
            and ask_volume <= 0.0
        ):
            return self.CONTEXT_NO_LIQUIDITY

        if pressure >= self.pressure_threshold:
            return self.CONTEXT_BID_PRESSURE

        if pressure <= -self.pressure_threshold:
            return self.CONTEXT_ASK_PRESSURE

        return self.CONTEXT_BALANCED

    @staticmethod
    def _validate_depth(
        depth: int,
    ) -> None:
        if (
            isinstance(depth, bool)
            or not isinstance(depth, int)
        ):
            raise TypeError(
                "depth must be an integer or None."
            )

        if depth <= 0:
            raise ValueError(
                "depth must be greater than zero."
            )

    @staticmethod
    def _validate_pressure_threshold(
        value: float,
    ) -> None:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            raise TypeError(
                "pressure_threshold must be a finite number."
            )

        if not isfinite(float(value)):
            raise ValueError(
                "pressure_threshold must be a finite number."
            )

        if not 0.0 < float(value) <= 1.0:
            raise ValueError(
                "pressure_threshold must be greater than 0 and at most 1.0."
            )

    @classmethod
    def _validate_snapshot(
        cls,
        snapshot: OrderBookSnapshot,
    ) -> None:
        if not isinstance(
            snapshot,
            OrderBookSnapshot,
        ):
            raise TypeError(
                "snapshot must be an OrderBookSnapshot instance."
            )

        if (
            not isinstance(snapshot.symbol, str)
            or not snapshot.symbol.strip()
        ):
            raise ValueError(
                "symbol must be a non-empty string."
            )

        if (
            isinstance(snapshot.timestamp, bool)
            or not isinstance(snapshot.timestamp, int)
            or snapshot.timestamp < 0
        ):
            raise ValueError(
                "timestamp must be a non-negative integer."
            )

        for level in (
            snapshot.bids
            + snapshot.asks
        ):
            if not isinstance(
                level,
                OrderBookLevel,
            ):
                raise TypeError(
                    "order-book levels must be OrderBookLevel instances."
                )

            if (
                not isfinite(level.price)
                or not isfinite(level.amount)
            ):
                raise ValueError(
                    "order-book level values must be finite."
                )

            if level.price <= 0.0:
                raise ValueError(
                    "order-book level price must be greater than zero."
                )

            if level.amount <= 0.0:
                raise ValueError(
                    "order-book level amount must be greater than zero."
                )

        for index in range(
            1,
            len(snapshot.bids),
        ):
            if (
                snapshot.bids[index].price
                > snapshot.bids[index - 1].price
            ):
                raise ValueError(
                    "bids must be sorted by descending price."
                )

        for index in range(
            1,
            len(snapshot.asks),
        ):
            if (
                snapshot.asks[index].price
                < snapshot.asks[index - 1].price
            ):
                raise ValueError(
                    "asks must be sorted by ascending price."
                )

        if (
            snapshot.bids
            and snapshot.asks
            and snapshot.best_bid >= snapshot.best_ask
        ):
            raise ValueError(
                "best bid must be lower than best ask."
            )


__all__ = [
    "OrderFlowSignal",
    "OrderFlowIntelligenceEngine",
]