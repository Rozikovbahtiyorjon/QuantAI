from __future__ import annotations

from unittest.mock import Mock

import pytest

from src.order_book_market_data import (
    ExchangeOrderBookMarketData,
    OrderBookLevel,
    OrderBookMarketDataEngine,
    OrderBookSnapshot,
)


def make_snapshot() -> OrderBookSnapshot:
    return OrderBookSnapshot(
        symbol="BTC/USDT:USDT",
        timestamp=1000,
        bids=(
            OrderBookLevel(
                price=100.0,
                amount=5.0,
            ),
            OrderBookLevel(
                price=99.0,
                amount=3.0,
            ),
            OrderBookLevel(
                price=98.0,
                amount=2.0,
            ),
        ),
        asks=(
            OrderBookLevel(
                price=101.0,
                amount=2.0,
            ),
            OrderBookLevel(
                price=102.0,
                amount=3.0,
            ),
            OrderBookLevel(
                price=103.0,
                amount=5.0,
            ),
        ),
    )


def test_snapshot_calculates_market_metrics() -> None:
    snapshot = make_snapshot()

    assert snapshot.best_bid == 100.0
    assert snapshot.best_ask == 101.0

    assert snapshot.spread == 1.0
    assert snapshot.mid_price == 100.5

    assert snapshot.spread_percent == pytest.approx(
        0.9950248756
    )

    assert snapshot.bid_volume() == 10.0
    assert snapshot.ask_volume() == 10.0

    assert snapshot.bid_notional() == 993.0
    assert snapshot.ask_notional() == 1023.0

    assert snapshot.imbalance() == 0.0


def test_depth_limits_volume_calculation() -> None:
    snapshot = make_snapshot()

    assert snapshot.bid_volume(
        depth=1
    ) == 5.0

    assert snapshot.ask_volume(
        depth=1
    ) == 2.0

    assert snapshot.bid_volume(
        depth=2
    ) == 8.0

    assert snapshot.ask_volume(
        depth=2
    ) == 5.0

    assert snapshot.imbalance(
        depth=1
    ) == pytest.approx(
        3.0 / 7.0
    )


def test_engine_detects_bid_dominance() -> None:
    engine = OrderBookMarketDataEngine()

    result = engine.update(
        make_snapshot()
    )

    assert result.spread == 1.0
    assert result.spread_percent == pytest.approx(
        0.9950248756
    )

    assert result.imbalance == 0.0
    assert result.context == "BALANCED"


def test_engine_detects_bid_dominance_with_depth() -> None:
    engine = OrderBookMarketDataEngine(
        depth=1
    )

    result = engine.update(
        make_snapshot()
    )

    assert result.bid_volume == 5.0
    assert result.ask_volume == 2.0

    assert result.imbalance == pytest.approx(
        3.0 / 7.0
    )

    assert result.context == "BID_DOMINANT"


def test_engine_detects_symbol_change() -> None:
    engine = OrderBookMarketDataEngine()

    engine.update(
        make_snapshot()
    )

    changed = OrderBookSnapshot(
        symbol="ETH/USDT:USDT",
        timestamp=2000,
        bids=(
            OrderBookLevel(
                price=2000.0,
                amount=1.0,
            ),
        ),
        asks=(
            OrderBookLevel(
                price=2001.0,
                amount=1.0,
            ),
        ),
    )

    with pytest.raises(ValueError):
        engine.update(changed)


def test_engine_reset() -> None:
    engine = OrderBookMarketDataEngine()

    engine.update(
        make_snapshot()
    )

    assert engine.previous is not None

    engine.reset()

    assert engine.previous is None


def test_exchange_adapter_normalizes_order_book() -> None:
    exchange = Mock()

    exchange.fetch_order_book.return_value = {
        "timestamp": 12345,
        "bids": [
            [99.0, 2.0],
            [100.0, 3.0],
            [98.0, 1.0],
        ],
        "asks": [
            [103.0, 1.0],
            [101.0, 2.0],
            [102.0, 4.0],
        ],
    }

    market_data = Mock()
    market_data.exchange = exchange

    provider = ExchangeOrderBookMarketData(
        market_data
    )

    result = provider.fetch(
        symbol="BTC/USDT:USDT",
        limit=50,
    )

    assert result.timestamp == 12345

    assert result.bids[0].price == 100.0
    assert result.bids[1].price == 99.0

    assert result.asks[0].price == 101.0
    assert result.asks[1].price == 102.0

    exchange.fetch_order_book.assert_called_once_with(
        "BTC/USDT:USDT",
        limit=50,
    )


def test_exchange_adapter_without_limit() -> None:
    exchange = Mock()

    exchange.fetch_order_book.return_value = {
        "timestamp": 1000,
        "bids": [
            [100.0, 1.0],
        ],
        "asks": [
            [101.0, 1.0],
        ],
    }

    market_data = Mock()
    market_data.exchange = exchange

    provider = ExchangeOrderBookMarketData(
        market_data
    )

    result = provider.fetch(
        "BTC/USDT:USDT"
    )

    assert result.best_bid == 100.0
    assert result.best_ask == 101.0

    exchange.fetch_order_book.assert_called_once_with(
        "BTC/USDT:USDT"
    )


def test_invalid_exchange_adapter() -> None:
    with pytest.raises(TypeError):
        ExchangeOrderBookMarketData(None)

    with pytest.raises(TypeError):
        ExchangeOrderBookMarketData(
            object()
        )


def test_invalid_fetch_parameters() -> None:
    exchange = Mock()

    market_data = Mock()
    market_data.exchange = exchange

    provider = ExchangeOrderBookMarketData(
        market_data
    )

    with pytest.raises(TypeError):
        provider.fetch(
            123
        )

    with pytest.raises(ValueError):
        provider.fetch(
            ""
        )

    with pytest.raises(TypeError):
        provider.fetch(
            "BTC/USDT:USDT",
            limit="10",
        )

    with pytest.raises(ValueError):
        provider.fetch(
            "BTC/USDT:USDT",
            limit=0,
        )


def test_invalid_order_book_structure() -> None:
    with pytest.raises(ValueError):
        ExchangeOrderBookMarketData._normalize_levels(
            [[100.0]],
            descending=True,
        )

    with pytest.raises(TypeError):
        ExchangeOrderBookMarketData._normalize_levels(
            "invalid",
            descending=True,
        )


def test_invalid_snapshot_crossed_book() -> None:
    snapshot = OrderBookSnapshot(
        symbol="BTC/USDT:USDT",
        timestamp=1000,
        bids=(
            OrderBookLevel(
                price=101.0,
                amount=1.0,
            ),
        ),
        asks=(
            OrderBookLevel(
                price=100.0,
                amount=1.0,
            ),
        ),
    )

    engine = OrderBookMarketDataEngine()

    with pytest.raises(ValueError):
        engine.update(snapshot)
