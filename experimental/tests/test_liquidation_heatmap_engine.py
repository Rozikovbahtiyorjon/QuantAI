from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experimental.src.liquidation_heatmap_engine import (
    LiquidationHeatmapConfig,
    LiquidationHeatmapEngine,
    build_liquidation_heatmap,
)


def make_liquidations(
    rows: int = 20,
) -> pd.DataFrame:
    timestamps = pd.date_range(
        "2026-01-01",
        periods=rows,
        freq="15min",
        tz="UTC",
    )

    prices = np.linspace(
        100.0,
        119.0,
        rows,
    )

    sides = np.where(
        np.arange(rows) % 2 == 0,
        "long",
        "short",
    )

    quantities = np.linspace(
        1.0,
        2.0,
        rows,
    )

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "price": prices,
            "side": sides,
            "quantity": quantities,
            "notional": prices * quantities,
            "symbol": "BTC/USDT",
        }
    )


def test_default_config_is_valid():
    config = LiquidationHeatmapConfig()

    assert config.price_bins == 50
    assert config.lookback == 100
    assert (
        config.level_quantile
        == pytest.approx(0.75)
    )


def test_invalid_config_is_rejected():
    with pytest.raises(ValueError):
        LiquidationHeatmapConfig(
            price_bins=1
        )

    with pytest.raises(ValueError):
        LiquidationHeatmapConfig(
            lookback=0
        )

    with pytest.raises(ValueError):
        LiquidationHeatmapConfig(
            level_quantile=0
        )

    with pytest.raises(ValueError):
        LiquidationHeatmapConfig(
            min_event_count=0
        )


def test_required_columns_are_enforced():
    data = make_liquidations().drop(
        columns=["side"]
    )

    with pytest.raises(
        ValueError,
        match="side",
    ):
        LiquidationHeatmapEngine().build_heatmap(
            data
        )


def test_empty_dataframe_is_rejected():
    data = make_liquidations().iloc[0:0]

    with pytest.raises(ValueError):
        LiquidationHeatmapEngine().build_heatmap(
            data
        )


def test_invalid_side_is_rejected():
    data = make_liquidations()

    data.loc[0, "side"] = "unknown"

    with pytest.raises(
        ValueError,
        match="long",
    ):
        LiquidationHeatmapEngine().build_heatmap(
            data
        )


def test_non_positive_price_is_rejected():
    data = make_liquidations()

    data.loc[0, "price"] = 0.0

    with pytest.raises(
        ValueError,
        match="price",
    ):
        LiquidationHeatmapEngine().build_heatmap(
            data
        )


def test_non_positive_quantity_is_rejected():
    data = make_liquidations()

    data.loc[0, "quantity"] = 0.0

    with pytest.raises(
        ValueError,
        match="quantity",
    ):
        LiquidationHeatmapEngine().build_heatmap(
            data
        )


def test_heatmap_contains_expected_columns():
    data = make_liquidations()

    result = (
        LiquidationHeatmapEngine()
        .build_heatmap(data)
    )

    assert list(result.columns) == list(
        LiquidationHeatmapEngine.OUTPUT_COLUMNS
    )


def test_heatmap_is_sorted_by_price():
    data = (
        make_liquidations()
        .iloc[::-1]
        .reset_index(drop=True)
    )

    result = (
        LiquidationHeatmapEngine()
        .build_heatmap(data)
    )

    assert result[
        "price"
    ].is_monotonic_increasing


def test_long_and_short_volume_are_separated():
    data = make_liquidations()

    config = LiquidationHeatmapConfig(
        price_bins=4,
        lookback=20,
        level_quantile=0.5,
    )

    result = (
        LiquidationHeatmapEngine(config)
        .build_heatmap(data)
    )

    expected_long = data.loc[
        data["side"] == "long",
        "notional",
    ].sum()

    expected_short = data.loc[
        data["side"] == "short",
        "notional",
    ].sum()

    assert (
        result[
            "long_liquidation_volume"
        ].sum()
        == pytest.approx(expected_long)
    )

    assert (
        result[
            "short_liquidation_volume"
        ].sum()
        == pytest.approx(expected_short)
    )


def test_total_volume_matches_long_plus_short():
    data = make_liquidations()

    result = (
        LiquidationHeatmapEngine()
        .build_heatmap(data)
    )

    expected = (
        result[
            "long_liquidation_volume"
        ]
        + result[
            "short_liquidation_volume"
        ]
    )

    pd.testing.assert_series_equal(
        result[
            "total_liquidation_volume"
        ],
        expected,
        check_names=False,
    )


def test_relative_intensity_is_bounded():
    data = make_liquidations()

    result = (
        LiquidationHeatmapEngine()
        .build_heatmap(data)
    )

    assert (
        result[
            "relative_intensity"
        ]
        .between(0.0, 1.0)
        .all()
    )


def test_level_strength_is_bounded():
    data = make_liquidations()

    result = (
        LiquidationHeatmapEngine()
        .build_heatmap(data)
    )

    assert (
        result[
            "level_strength"
        ]
        .between(0.0, 1.0)
        .all()
    )


def test_levels_return_only_strong_levels():
    data = make_liquidations()

    config = LiquidationHeatmapConfig(
        price_bins=5,
        lookback=20,
        level_quantile=0.5,
    )

    engine = LiquidationHeatmapEngine(
        config
    )

    levels = engine.levels(data)

    assert not levels.empty

    assert (
        levels["level_strength"] > 0.0
    ).all()


def test_nearest_levels_are_identified():
    data = make_liquidations()

    config = LiquidationHeatmapConfig(
        price_bins=4,
        lookback=20,
        level_quantile=0.5,
    )

    engine = LiquidationHeatmapEngine(
        config
    )

    result = engine.nearest_levels(
        data,
        current_price=110.0,
    )

    assert (
        result["nearest_support"] is not None
        or result["nearest_resistance"]
        is not None
    )


def test_latest_returns_strongest_level():
    data = make_liquidations()

    config = LiquidationHeatmapConfig(
        price_bins=4,
        lookback=20,
        level_quantile=0.5,
    )

    engine = LiquidationHeatmapEngine(
        config
    )

    result = engine.latest(
        data,
        current_price=110.0,
    )

    assert (
        "strongest_level"
        in result
    )

    assert (
        "strongest_level_strength"
        in result
    )


def test_convenience_function_matches_engine():
    data = make_liquidations()

    config = LiquidationHeatmapConfig(
        price_bins=5,
        lookback=20,
        level_quantile=0.5,
    )

    direct = (
        LiquidationHeatmapEngine(config)
        .build_heatmap(data)
    )

    convenience = (
        build_liquidation_heatmap(
            data,
            config,
        )
    )

    pd.testing.assert_frame_equal(
        direct,
        convenience,
    )