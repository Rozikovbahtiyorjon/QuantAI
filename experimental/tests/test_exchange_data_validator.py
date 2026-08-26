import pandas as pd
import pytest

from experimental.src.exchange_data_validator import (
    REQUIRED_COLUMNS,
    ExchangeDataValidationResult,
    ExchangeDataValidator,
    validate_exchange_data,
)


def make_valid_dataframe():
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-08-12 00:00:00",
                    "2026-08-12 00:15:00",
                    "2026-08-12 00:30:00",
                ],
                utc=True,
            ),
            "open": [100.0, 101.0, 102.0],
            "high": [105.0, 106.0, 107.0],
            "low": [99.0, 100.0, 101.0],
            "close": [103.0, 104.0, 106.0],
            "volume": [1000.0, 1200.0, 1500.0],
        }
    )


def test_required_columns():
    assert REQUIRED_COLUMNS == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]


def test_validator_initialization():
    validator = ExchangeDataValidator(timeframe_minutes=15)

    assert validator.timeframe_minutes == 15


def test_invalid_timeframe():
    with pytest.raises(ValueError):
        ExchangeDataValidator(timeframe_minutes=0)


def test_valid_dataframe():
    df = make_valid_dataframe()

    result = ExchangeDataValidator(timeframe_minutes=15).validate(df)

    assert isinstance(result, ExchangeDataValidationResult)
    assert result.valid is True
    assert result.errors == []
    assert result.warnings == []
    assert result.rows_checked == 3
    assert result.duplicate_timestamps == 0
    assert result.missing_timestamps == 0


def test_empty_dataframe():
    df = pd.DataFrame()

    result = ExchangeDataValidator().validate(df)

    assert result.valid is False
    assert "DataFrame cannot be empty." in result.errors


def test_invalid_dataframe_type():
    result = ExchangeDataValidator().validate([1, 2, 3])

    assert result.valid is False
    assert "df must be a pandas DataFrame." in result.errors


def test_missing_columns():
    df = make_valid_dataframe()

    df = df.drop(columns=["volume"])

    result = ExchangeDataValidator().validate(df)

    assert result.valid is False
    assert any("volume" in error for error in result.errors)


def test_invalid_timestamp():
    df = make_valid_dataframe()

    df["timestamp"] = df["timestamp"].astype(object)
    df.loc[1, "timestamp"] = "invalid"

    result = ExchangeDataValidator().validate(df)

    assert result.valid is False
    assert any("Invalid timestamps" in error for error in result.errors)


def test_duplicate_timestamp():
    df = make_valid_dataframe()

    df.loc[2, "timestamp"] = df.loc[1, "timestamp"]

    result = ExchangeDataValidator().validate(df)

    assert result.valid is False
    assert result.duplicate_timestamps == 2
    assert any("Duplicate timestamps" in error for error in result.errors)


def test_invalid_numeric_value():
    df = make_valid_dataframe()

    df["close"] = df["close"].astype(object)
    df.loc[1, "close"] = "invalid"

    result = ExchangeDataValidator().validate(df)

    assert result.valid is False
    assert any("close" in error for error in result.errors)


def test_non_positive_price():
    df = make_valid_dataframe()

    df.loc[1, "close"] = 0.0

    result = ExchangeDataValidator().validate(df)

    assert result.valid is False
    assert any("Non-positive OHLC prices" in error for error in result.errors)


def test_high_lower_than_low():
    df = make_valid_dataframe()

    df.loc[1, "high"] = 90.0

    result = ExchangeDataValidator().validate(df)

    assert result.valid is False
    assert any(
        "High price is lower than low price" in error
        for error in result.errors
    )


def test_open_outside_range():
    df = make_valid_dataframe()

    df.loc[1, "open"] = 200.0

    result = ExchangeDataValidator().validate(df)

    assert result.valid is False
    assert any("Open price is outside" in error for error in result.errors)


def test_close_outside_range():
    df = make_valid_dataframe()

    df.loc[1, "close"] = 200.0

    result = ExchangeDataValidator().validate(df)

    assert result.valid is False
    assert any("Close price is outside" in error for error in result.errors)


def test_negative_volume():
    df = make_valid_dataframe()

    df.loc[1, "volume"] = -10.0

    result = ExchangeDataValidator().validate(df)

    assert result.valid is False
    assert any("Negative volume" in error for error in result.errors)


def test_non_chronological_timestamps():
    df = make_valid_dataframe()

    timestamp = df.loc[0, "timestamp"]

    df.loc[0, "timestamp"] = df.loc[2, "timestamp"]
    df.loc[2, "timestamp"] = timestamp

    result = ExchangeDataValidator().validate(df)

    assert result.valid is False
    assert any("chronological order" in error for error in result.errors)


def test_missing_candle_interval():
    df = make_valid_dataframe()

    df.loc[2, "timestamp"] = pd.Timestamp(
        "2026-08-12 01:00:00",
        tz="UTC",
    )

    result = ExchangeDataValidator(timeframe_minutes=15).validate(df)

    assert result.valid is True
    assert result.missing_timestamps == 1
    assert len(result.warnings) == 1
    assert "Missing candle intervals" in result.warnings[0]


def test_convenience_function():
    df = make_valid_dataframe()

    result = validate_exchange_data(
        df,
        timeframe_minutes=15,
    )

    assert result.valid is True
    assert result.rows_checked == 3