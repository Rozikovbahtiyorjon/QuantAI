from __future__ import annotations

from unittest.mock import Mock

import pytest

from experimental.src.derivatives_market_data import (
    DerivativesMarketData,
    ExchangeDerivativesMarketData,
)
from experimental.src.exchange_market_data import ExchangeMarketData


def make_exchange_mock() -> Mock:
    exchange = Mock()

    exchange.fetch_ticker.side_effect = [
        {
            "timestamp": 1000,
            "last": 105000.0,
            "close": 105000.0,
        },
        {
            "timestamp": 1000,
            "last": 104500.0,
            "close": 104500.0,
        },
    ]

    exchange.fetch_open_interest.return_value = {
        "openInterestValue": 2500000000.0,
    }

    exchange.fetch_funding_rate.return_value = {
        "fundingRate": 0.0001,
    }

    exchange.fetch_liquidations.return_value = [
        {
            "amount": 2.0,
            "price": 100000.0,
        },
        {
            "amount": 1.0,
            "price": 50000.0,
        },
    ]

    exchange.fetch_long_short_ratio.return_value = {
        "longShortRatio": 1.25,
    }

    return exchange


def make_market_provider(
    exchange: Mock,
) -> Mock:
    provider = Mock(
        spec=ExchangeMarketData
    )

    provider.exchange = exchange

    return provider


def test_fetch_builds_derivatives_market_data() -> None:
    exchange = make_exchange_mock()

    provider = ExchangeDerivativesMarketData(
        make_market_provider(exchange)
    )

    result = provider.fetch(
        symbol="BTC/USDT:USDT",
        spot_symbol="BTC/USDT",
    )

    assert isinstance(
        result,
        DerivativesMarketData,
    )

    assert result.symbol == "BTC/USDT:USDT"
    assert result.timestamp == 1000
    assert result.price == 105000.0

    assert result.open_interest == 2500000000.0
    assert result.funding_rate == 0.0001

    assert result.liquidation_volume == 250000.0

    assert result.long_short_ratio == 1.25
    assert result.spot_price == 104500.0

    assert result.basis == 500.0


def test_fetch_uses_exchange_methods() -> None:
    exchange = make_exchange_mock()

    provider = ExchangeDerivativesMarketData(
        make_market_provider(exchange)
    )

    provider.fetch(
        symbol="BTC/USDT:USDT",
        spot_symbol="BTC/USDT",
    )

    exchange.fetch_ticker.assert_any_call(
        "BTC/USDT:USDT"
    )

    exchange.fetch_open_interest.assert_called_once_with(
        "BTC/USDT:USDT"
    )

    exchange.fetch_funding_rate.assert_called_once_with(
        "BTC/USDT:USDT"
    )

    exchange.fetch_liquidations.assert_called_once_with(
        "BTC/USDT:USDT",
        limit=100,
    )

    exchange.fetch_long_short_ratio.assert_called_once_with(
        "BTC/USDT:USDT"
    )


def test_missing_liquidation_endpoint_returns_zero() -> None:
    exchange = make_exchange_mock()

    exchange.fetch_liquidations = None

    provider = ExchangeDerivativesMarketData(
        make_market_provider(exchange)
    )

    result = provider.fetch(
        symbol="BTC/USDT:USDT",
        spot_symbol="BTC/USDT",
    )

    assert result.liquidation_volume == 0.0


def test_missing_long_short_endpoint_returns_neutral_ratio() -> None:
    exchange = make_exchange_mock()

    exchange.fetch_long_short_ratio = None

    provider = ExchangeDerivativesMarketData(
        make_market_provider(exchange)
    )

    result = provider.fetch(
        symbol="BTC/USDT:USDT",
        spot_symbol="BTC/USDT",
    )

    assert result.long_short_ratio == 1.0


def test_missing_open_interest_endpoint_raises() -> None:
    exchange = make_exchange_mock()

    exchange.fetch_open_interest = None

    provider = ExchangeDerivativesMarketData(
        make_market_provider(exchange)
    )

    with pytest.raises(RuntimeError):
        provider.fetch(
            symbol="BTC/USDT:USDT",
            spot_symbol="BTC/USDT",
        )


def test_missing_funding_endpoint_raises() -> None:
    exchange = make_exchange_mock()

    exchange.fetch_funding_rate = None

    provider = ExchangeDerivativesMarketData(
        make_market_provider(exchange)
    )

    with pytest.raises(RuntimeError):
        provider.fetch(
            symbol="BTC/USDT:USDT",
            spot_symbol="BTC/USDT",
        )


def test_invalid_market_provider() -> None:
    with pytest.raises(TypeError):
        ExchangeDerivativesMarketData(None)


def test_invalid_symbol() -> None:
    exchange = make_exchange_mock()

    provider = ExchangeDerivativesMarketData(
        make_market_provider(exchange)
    )

    with pytest.raises(ValueError):
        provider.fetch(
            symbol="",
            spot_symbol="BTC/USDT",
        )