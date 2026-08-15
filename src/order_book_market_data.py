from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any


@dataclass(frozen=True)
class OrderBookLevel:
    price: float
    amount: float

    @property
    def notional(self) -> float:
        return self.price * self.amount


@dataclass(frozen=True)
class OrderBookSnapshot:
    symbol: str
    timestamp: int
    bids: tuple[OrderBookLevel, ...]
    asks: tuple[OrderBookLevel, ...]

    @property
    def best_bid(self) -> float | None:
        if not self.bids:
            return None

        return self.bids[0].price

    @property
    def best_ask(self) -> float | None:
        if not self.asks:
            return None

        return self.asks[0].price

    @property
    def spread(self) -> float | None:
        if (
            self.best_bid is None
            or self.best_ask is None
        ):
            return None

        return self.best_ask - self.best_bid

    @property
    def mid_price(self) -> float | None:
        if (
            self.best_bid is None
            or self.best_ask is None
        ):
            return None

        return (
            self.best_bid + self.best_ask
        ) / 2.0

    @property
    def spread_percent(self) -> float | None:
        mid_price = self.mid_price

        if (
            mid_price is None
            or mid_price == 0
        ):
            return None

        return (
            self.spread / mid_price
        ) * 100.0

    def bid_volume(
        self,
        depth: int | None = None,
    ) -> float:
        return sum(
            level.amount
            for level in self._levels(
                self.bids,
                depth,
            )
        )

    def ask_volume(
        self,
        depth: int | None = None,
    ) -> float:
        return sum(
            level.amount
            for level in self._levels(
                self.asks,
                depth,
            )
        )

    def bid_notional(
        self,
        depth: int | None = None,
    ) -> float:
        return sum(
            level.notional
            for level in self._levels(
                self.bids,
                depth,
            )
        )

    def ask_notional(
        self,
        depth: int | None = None,
    ) -> float:
        return sum(
            level.notional
            for level in self._levels(
                self.asks,
                depth,
            )
        )

    def imbalance(
        self,
        depth: int | None = None,
    ) -> float:
        bid_volume = self.bid_volume(depth)
        ask_volume = self.ask_volume(depth)

        total = bid_volume + ask_volume

        if total == 0:
            return 0.0

        return (
            bid_volume - ask_volume
        ) / total

    @staticmethod
    def _levels(
        levels: tuple[OrderBookLevel, ...],
        depth: int | None,
    ) -> tuple[OrderBookLevel, ...]:
        if depth is None:
            return levels

        if (
            not isinstance(depth, int)
            or isinstance(depth, bool)
        ):
            raise TypeError(
                "depth must be an integer or None."
            )

        if depth <= 0:
            raise ValueError(
                "depth must be greater than zero."
            )

        return levels[:depth]


@dataclass(frozen=True)
class OrderBookSignal:
    spread: float | None
    spread_percent: float | None
    imbalance: float
    bid_volume: float
    ask_volume: float
    context: str


class OrderBookMarketDataEngine:
    def __init__(
        self,
        depth: int | None = None,
    ) -> None:
        if depth is not None:
            self._validate_depth(depth)

        self.depth = depth
        self._previous: OrderBookSnapshot | None = None

    @property
    def previous(
        self,
    ) -> OrderBookSnapshot | None:
        return self._previous

    def update(
        self,
        data: OrderBookSnapshot,
    ) -> OrderBookSignal:
        self._validate_snapshot(data)

        if self._previous is not None:
            if (
                data.symbol
                != self._previous.symbol
            ):
                raise ValueError(
                    "symbol must match the previous "
                    "order-book snapshot."
                )

        bid_volume = data.bid_volume(
            self.depth
        )

        ask_volume = data.ask_volume(
            self.depth
        )

        imbalance = data.imbalance(
            self.depth
        )

        if imbalance > 0:
            context = "BID_DOMINANT"

        elif imbalance < 0:
            context = "ASK_DOMINANT"

        else:
            context = "BALANCED"

        self._previous = data

        return OrderBookSignal(
            spread=data.spread,
            spread_percent=data.spread_percent,
            imbalance=imbalance,
            bid_volume=bid_volume,
            ask_volume=ask_volume,
            context=context,
        )

    def reset(self) -> None:
        self._previous = None

    @staticmethod
    def _validate_depth(
        depth: int,
    ) -> None:
        if (
            not isinstance(depth, int)
            or isinstance(depth, bool)
        ):
            raise TypeError(
                "depth must be an integer or None."
            )

        if depth <= 0:
            raise ValueError(
                "depth must be greater than zero."
            )

    @classmethod
    def _validate_snapshot(
        cls,
        data: OrderBookSnapshot,
    ) -> None:
        if not isinstance(
            data,
            OrderBookSnapshot,
        ):
            raise TypeError(
                "data must be an OrderBookSnapshot instance."
            )

        if (
            not isinstance(data.symbol, str)
            or not data.symbol
        ):
            raise ValueError(
                "symbol must be a non-empty string."
            )

        if (
            not isinstance(data.timestamp, int)
            or isinstance(data.timestamp, bool)
            or data.timestamp < 0
        ):
            raise ValueError(
                "timestamp must be a non-negative integer."
            )

        for level in (
            data.bids + data.asks
        ):
            cls._validate_level(level)

        if data.bids:
            for index in range(
                1,
                len(data.bids),
            ):
                if (
                    data.bids[index].price
                    > data.bids[index - 1].price
                ):
                    raise ValueError(
                        "bids must be sorted "
                        "by descending price."
                    )

        if data.asks:
            for index in range(
                1,
                len(data.asks),
            ):
                if (
                    data.asks[index].price
                    < data.asks[index - 1].price
                ):
                    raise ValueError(
                        "asks must be sorted "
                        "by ascending price."
                    )

        if (
            data.bids
            and data.asks
            and data.best_bid >= data.best_ask
        ):
            raise ValueError(
                "best bid must be lower than best ask."
            )

    @staticmethod
    def _validate_level(
        level: OrderBookLevel,
    ) -> None:
        if not isinstance(
            level,
            OrderBookLevel,
        ):
            raise TypeError(
                "order-book levels must be "
                "OrderBookLevel instances."
            )

        if (
            not isfinite(level.price)
            or not isfinite(level.amount)
        ):
            raise ValueError(
                "order-book level values must be finite."
            )

        if level.price <= 0:
            raise ValueError(
                "order-book level price "
                "must be greater than zero."
            )

        if level.amount <= 0:
            raise ValueError(
                "order-book level amount "
                "must be greater than zero."
            )


class ExchangeOrderBookMarketData:
    """
    Read-only CCXT order-book adapter.

    This class only reads public market data
    and never places orders.
    """

    def __init__(
        self,
        market_data: Any,
    ) -> None:
        if market_data is None:
            raise TypeError(
                "market_data cannot be None."
            )

        if not hasattr(
            market_data,
            "exchange",
        ):
            raise TypeError(
                "market_data must provide "
                "a CCXT exchange."
            )

        self.market_data = market_data
        self.exchange = market_data.exchange

    def fetch(
        self,
        symbol: str,
        limit: int | None = None,
    ) -> OrderBookSnapshot:
        if not isinstance(
            symbol,
            str,
        ):
            raise TypeError(
                "symbol must be a string."
            )

        symbol = symbol.strip()

        if not symbol:
            raise ValueError(
                "symbol cannot be empty."
            )

        method = getattr(
            self.exchange,
            "fetch_order_book",
            None,
        )

        if not callable(method):
            raise RuntimeError(
                "Exchange does not support "
                "fetch_order_book."
            )

        if limit is not None:
            if (
                not isinstance(limit, int)
                or isinstance(limit, bool)
            ):
                raise TypeError(
                    "limit must be an integer or None."
                )

            if limit <= 0:
                raise ValueError(
                    "limit must be greater than zero."
                )

            raw = method(
                symbol,
                limit=limit,
            )

        else:
            raw = method(symbol)

        return self._normalize(
            symbol,
            raw,
        )

    @classmethod
    def _normalize(
        cls,
        symbol: str,
        raw: Any,
    ) -> OrderBookSnapshot:
        if not isinstance(
            raw,
            dict,
        ):
            raise TypeError(
                "Exchange order-book response "
                "must be a dictionary."
            )

        timestamp_value = raw.get(
            "timestamp"
        )

        if timestamp_value is None:
            timestamp = 0

        else:
            try:
                timestamp = int(
                    timestamp_value
                )

            except (
                TypeError,
                ValueError,
            ) as exc:
                raise ValueError(
                    "Order-book timestamp "
                    "must be numeric."
                ) from exc

            if timestamp < 0:
                raise ValueError(
                    "Order-book timestamp "
                    "cannot be negative."
                )

        bids = cls._normalize_levels(
            raw.get("bids", []),
            descending=True,
        )

        asks = cls._normalize_levels(
            raw.get("asks", []),
            descending=False,
        )

        snapshot = OrderBookSnapshot(
            symbol=symbol,
            timestamp=timestamp,
            bids=bids,
            asks=asks,
        )

        OrderBookMarketDataEngine._validate_snapshot(
            snapshot
        )

        return snapshot

    @staticmethod
    def _normalize_levels(
        raw_levels: Any,
        descending: bool,
    ) -> tuple[OrderBookLevel, ...]:
        if raw_levels is None:
            return ()

        if not isinstance(
            raw_levels,
            list,
        ):
            raise TypeError(
                "Order-book levels must be a list."
            )

        levels: list[
            OrderBookLevel
        ] = []

        for row in raw_levels:
            if not isinstance(
                row,
                (list, tuple),
            ):
                raise ValueError(
                    "Order-book level must be "
                    "a list or tuple."
                )

            if len(row) < 2:
                raise ValueError(
                    "Order-book level must contain "
                    "price and amount."
                )

            price = (
                ExchangeOrderBookMarketData
                ._finite_float(
                    row[0],
                    "price",
                )
            )

            amount = (
                ExchangeOrderBookMarketData
                ._finite_float(
                    row[1],
                    "amount",
                )
            )

            if price <= 0:
                raise ValueError(
                    "order-book level price "
                    "must be greater than zero."
                )

            if amount <= 0:
                raise ValueError(
                    "order-book level amount "
                    "must be greater than zero."
                )

            levels.append(
                OrderBookLevel(
                    price=price,
                    amount=amount,
                )
            )

        levels.sort(
            key=lambda level: level.price,
            reverse=descending,
        )

        return tuple(levels)

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
    "OrderBookLevel",
    "OrderBookSnapshot",
    "OrderBookSignal",
    "OrderBookMarketDataEngine",
    "ExchangeOrderBookMarketData",
]