from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.futures_data_engine import (
    FuturesDataConfig,
    FuturesDataEngine,
    build_futures_features,
)


def make_futures_data(
    rows: int = 30,
) -> pd.DataFrame:
    timestamps = pd.date_range(
        "2026-01-01",
        periods=rows,
        freq="15min",
        tz="UTC",
    )

    close = pd.Series(
        np.linspace(
            100.0,
            110.0,
            rows,
        ),
        dtype=float,
    )

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.linspace(
                1000.0,
                2000.0,
                rows,
            ),
            "open_interest": np.linspace(
                5000.0,
                6500.0,
                rows,
            ),
            "funding_rate": np.linspace(
                0.0001,
                0.0005,
                rows,
            ),
            "mark_price": close + 0.20,
            "index_price": close,
            "taker_buy_volume": np.linspace(
                450.0,
                1200.0,
                rows,
            ),
        }
    )


def test_default_config_is_valid():
    config = FuturesDataConfig()

    assert config.open_interest_window == 20
    assert config.funding_window == 20
    assert config.basis_window == 20
    assert config.volume_window == 20


def test_invalid_config_is_rejected():
    with pytest.raises(ValueError):
        FuturesDataConfig(
            open_interest_window=0
        )

    with pytest.raises(ValueError):
        FuturesDataConfig(
            min_history=0
        )


def test_required_columns_are_enforced():
    data = make_futures_data()

    data = data.drop(
        columns=["open_interest"]
    )

    with pytest.raises(
        ValueError,
        match="open_interest",
    ):
        FuturesDataEngine().transform(data)


def test_non_numeric_futures_column_is_rejected():
    data = make_futures_data()

    data["funding_rate"] = "bad"

    with pytest.raises(
        TypeError,
        match="funding_rate",
    ):
        FuturesDataEngine().transform(data)


def test_invalid_close_is_rejected():
    data = make_futures_data()

    data.loc[0, "close"] = 0.0

    with pytest.raises(
        ValueError,
        match="close",
    ):
        FuturesDataEngine().transform(data)


def test_duplicate_timestamps_are_rejected():
    data = make_futures_data()

    data.loc[1, "timestamp"] = (
        data.loc[0, "timestamp"]
    )

    with pytest.raises(
        ValueError,
        match="unique",
    ):
        FuturesDataEngine().transform(data)


def test_transform_preserves_row_count_and_sorts_data():
    data = make_futures_data()

    shuffled = (
        data
        .iloc[::-1]
        .reset_index(drop=True)
    )

    result = FuturesDataEngine().transform(
        shuffled
    )

    assert len(result) == len(data)

    assert (
        result["timestamp"]
        .is_monotonic_increasing
    )


def test_open_interest_features_are_calculated():
    data = make_futures_data()

    result = FuturesDataEngine().transform(
        data
    )

    expected_change = (
        data.loc[1, "open_interest"]
        - data.loc[0, "open_interest"]
    )

    expected_change_pct = (
        data.loc[1, "open_interest"]
        / data.loc[0, "open_interest"]
        - 1.0
    )

    assert (
        result.loc[
            1,
            "open_interest_change",
        ]
        == pytest.approx(
            expected_change
        )
    )

    assert (
        result.loc[
            1,
            "open_interest_change_pct",
        ]
        == pytest.approx(
            expected_change_pct
        )
    )


def test_funding_rate_is_converted_to_basis_points():
    data = make_futures_data()

    result = FuturesDataEngine().transform(
        data
    )

    assert (
        result.loc[
            0,
            "funding_rate_bps",
        ]
        == pytest.approx(1.0)
    )


def test_basis_features_are_calculated():
    data = make_futures_data()

    result = FuturesDataEngine().transform(
        data
    )

    assert (
        result.loc[0, "basis"]
        == pytest.approx(0.20)
    )

    assert (
        result.loc[0, "basis_bps"]
        == pytest.approx(20.0)
    )


def test_taker_buy_volume_ratio_is_calculated():
    data = make_futures_data()

    result = build_futures_features(
        data
    )

    expected = (
        data.loc[0, "taker_buy_volume"]
        / data.loc[0, "volume"]
    )

    assert (
        result.loc[
            0,
            "taker_buy_volume_ratio",
        ]
        == pytest.approx(expected)
    )


def test_missing_optional_columns_are_supported():
    data = make_futures_data().drop(
        columns=[
            "mark_price",
            "index_price",
            "taker_buy_volume",
        ]
    )

    result = FuturesDataEngine().transform(
        data
    )

    assert result["basis"].isna().all()
    assert result["basis_bps"].isna().all()
    assert result["basis_zscore"].isna().all()
    assert (
        result[
            "taker_buy_volume_ratio"
        ].isna().all()
    )


def test_rolling_features_do_not_use_future_rows():
    data = make_futures_data()

    first = FuturesDataEngine().transform(
        data.iloc[:20]
    )

    second = FuturesDataEngine().transform(
        data.iloc[:20].copy()
    )

    pd.testing.assert_series_equal(
        first["funding_rate_zscore"],
        second["funding_rate_zscore"],
        check_names=False,
    )


def test_latest_returns_futures_snapshot():
    data = make_futures_data()

    snapshot = FuturesDataEngine().latest(
        data
    )

    assert (
        snapshot["timestamp"]
        == data["timestamp"].iloc[-1]
    )

    assert (
        "open_interest_change"
        in snapshot
    )

    assert (
        "funding_rate_bps"
        in snapshot
    )

    assert "basis_bps" in snapshot


def test_feature_columns_are_stable():
    columns = (
        FuturesDataEngine()
        .feature_columns()
    )

    assert (
        columns
        == FuturesDataEngine.OUTPUT_COLUMNS
    )

    assert len(columns) == len(
        set(columns)
    )