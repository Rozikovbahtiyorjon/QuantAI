import pandas as pd
import pytest

from src.exchange_market_data import (
    DEFAULT_EXCHANGE,
    DEFAULT_LIMIT,
    DEFAULT_TIMEFRAME,
    OHLCV_COLUMNS,
    ExchangeMarketData,
)


class FakeExchange:
    def __init__(self, config):
        self.config = config
        self.has = {
            "fetchOHLCV": True,
        }

    def load_markets(self):
        return {
            "BTC/USDT": {
                "symbol": "BTC/USDT",
            }
        }

    def fetch_ohlcv(
        self,
        symbol,
        timeframe="15m",
        limit=None,
        since=None,
        params=None,
    ):
        return [
            [
                1000,
                100.0,
                105.0,
                99.0,
                103.0,
                10.0,
            ],
            [
                2000,
                103.0,
                108.0,
                102.0,
                107.0,
                12.0,
            ],
        ]

    def close(self):
        return None


@pytest.fixture
def fake_provider(monkeypatch):
    monkeypatch.setattr(
        "src.exchange_market_data.ccxt.binance",
        FakeExchange,
    )

    return ExchangeMarketData(
        exchange_id="binance"
    )


def test_constants():
    assert DEFAULT_EXCHANGE == "binance"
    assert DEFAULT_TIMEFRAME == "15m"
    assert DEFAULT_LIMIT > 0

    assert OHLCV_COLUMNS == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]


def test_invalid_exchange():
    with pytest.raises(ValueError):
        ExchangeMarketData(
            exchange_id="definitely_invalid_exchange"
        )


def test_empty_exchange_id():
    with pytest.raises(ValueError):
        ExchangeMarketData(
            exchange_id=""
        )


def test_invalid_timeout():
    with pytest.raises(ValueError):
        ExchangeMarketData(
            exchange_id="binance",
            timeout=0,
        )


def test_has_fetch_ohlcv(fake_provider):
    assert fake_provider.has_fetch_ohlcv is True


def test_load_markets(fake_provider):
    markets = fake_provider.load_markets()

    assert "BTC/USDT" in markets


def test_fetch_ohlcv(fake_provider):
    df = fake_provider.fetch_ohlcv(
        symbol="BTC/USDT",
        timeframe="15m",
        limit=2,
    )

    assert isinstance(
        df,
        pd.DataFrame,
    )

    assert list(df.columns) == OHLCV_COLUMNS

    assert len(df) == 2

    assert pd.api.types.is_datetime64_any_dtype(
        df["timestamp"]
    )


def test_fetch_ohlcv_values(fake_provider):
    df = fake_provider.fetch_ohlcv(
        symbol="BTC/USDT"
    )

    assert df.iloc[0]["open"] == 100.0
    assert df.iloc[0]["high"] == 105.0
    assert df.iloc[0]["low"] == 99.0
    assert df.iloc[0]["close"] == 103.0
    assert df.iloc[0]["volume"] == 10.0


def test_fetch_ohlcv_sorted(fake_provider):
    df = fake_provider.fetch_ohlcv(
        symbol="BTC/USDT"
    )

    assert df["timestamp"].is_monotonic_increasing


def test_empty_symbol(fake_provider):
    with pytest.raises(ValueError):
        fake_provider.fetch_ohlcv(
            symbol=""
        )


def test_invalid_symbol_type(fake_provider):
    with pytest.raises(TypeError):
        fake_provider.fetch_ohlcv(
            symbol=123
        )


def test_empty_timeframe(fake_provider):
    with pytest.raises(ValueError):
        fake_provider.fetch_ohlcv(
            symbol="BTC/USDT",
            timeframe="",
        )


def test_invalid_timeframe_type(fake_provider):
    with pytest.raises(TypeError):
        fake_provider.fetch_ohlcv(
            symbol="BTC/USDT",
            timeframe=15,
        )


def test_invalid_limit(fake_provider):
    with pytest.raises(ValueError):
        fake_provider.fetch_ohlcv(
            symbol="BTC/USDT",
            limit=0,
        )


def test_invalid_since(fake_provider):
    with pytest.raises(ValueError):
        fake_provider.fetch_ohlcv(
            symbol="BTC/USDT",
            since=-1,
        )


def test_fetch_latest(fake_provider):
    candle = fake_provider.fetch_latest(
        symbol="BTC/USDT"
    )

    assert isinstance(
        candle,
        pd.Series,
    )

    assert candle["close"] == 107.0


def test_normalize_empty_data():
    df = ExchangeMarketData._normalize_ohlcv(
        []
    )

    assert isinstance(
        df,
        pd.DataFrame,
    )

    assert df.empty

    assert list(df.columns) == OHLCV_COLUMNS


def test_normalize_none():
    df = ExchangeMarketData._normalize_ohlcv(
        None
    )

    assert isinstance(
        df,
        pd.DataFrame,
    )

    assert df.empty


def test_invalid_ohlcv_row():
    with pytest.raises(ValueError):
        ExchangeMarketData._normalize_ohlcv(
            [
                [
                    1000,
                    100.0,
                    105.0,
                ]
            ]
        )


def test_invalid_ohlcv_response():
    with pytest.raises(TypeError):
        ExchangeMarketData._normalize_ohlcv(
            "invalid"
        )


def test_close(fake_provider):
    result = fake_provider.close()

    assert result is None