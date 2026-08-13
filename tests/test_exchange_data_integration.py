import pandas as pd

from src.exchange_data_validator import (
    ExchangeDataValidator,
)
from src.exchange_market_data import (
    ExchangeMarketData,
)


def test_market_data_normalization_to_validator():
    raw_data = [
        [
            1000,
            100.0,
            105.0,
            99.0,
            103.0,
            1000.0,
        ],
        [
            901000,
            103.0,
            108.0,
            102.0,
            107.0,
            1200.0,
        ],
        [
            1801000,
            107.0,
            110.0,
            106.0,
            109.0,
            1500.0,
        ],
    ]

    df = ExchangeMarketData._normalize_ohlcv(
        raw_data
    )

    assert isinstance(
        df,
        pd.DataFrame,
    )

    assert len(df) == 3

    validator = ExchangeDataValidator(
        timeframe_minutes=15
    )

    result = validator.validate(
        df
    )

    assert result.valid is True
    assert result.errors == []
    assert result.warnings == []
    assert result.rows_checked == 3


def test_market_data_columns_match_validator():
    raw_data = [
        [
            1000,
            100.0,
            105.0,
            99.0,
            103.0,
            1000.0,
        ]
    ]

    df = ExchangeMarketData._normalize_ohlcv(
        raw_data
    )

    result = ExchangeDataValidator().validate(
        df
    )

    assert result.valid is True

    assert set(
        df.columns
    ) == {
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }