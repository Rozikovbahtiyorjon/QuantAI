from __future__ import annotations

import pytest

from src.liquidity_liquidation_zones import (
    LiquidityHeatmap,
    LiquidityLiquidationZoneEngine,
    LiquidityZone,
)


def test_build_heatmap_groups_events_into_price_zones() -> None:
    engine = LiquidityLiquidationZoneEngine(
        zone_size=10.0
    )

    heatmap = engine.build_heatmap(
        symbol="BTC/USDT:USDT",
        timestamp=1000,
        events=[
            (101.0, 2.0, 202.0),
            (105.0, 3.0, 315.0),
            (119.0, 1.0, 119.0),
        ],
    )

    assert len(heatmap.zones) == 2

    first = heatmap.zones[0]
    second = heatmap.zones[1]

    assert first.lower_price == 100.0
    assert first.upper_price == 110.0
    assert first.liquidation_volume == 5.0
    assert first.liquidation_notional == 517.0
    assert first.event_count == 2

    assert second.lower_price == 110.0
    assert second.upper_price == 120.0
    assert second.liquidation_volume == 1.0
    assert second.liquidation_notional == 119.0
    assert second.event_count == 1


def test_zone_metrics() -> None:
    zone = LiquidityZone(
        lower_price=100.0,
        upper_price=110.0,
        liquidation_volume=5.0,
        liquidation_notional=500.0,
        event_count=2,
    )

    assert zone.center_price == 105.0
    assert zone.width == 10.0
    assert zone.density == 50.0


def test_heatmap_totals_and_strongest_zone() -> None:
    engine = LiquidityLiquidationZoneEngine(
        zone_size=10.0
    )

    heatmap = engine.build_heatmap(
        "BTC/USDT:USDT",
        1000,
        [
            (101.0, 1.0, 101.0),
            (105.0, 2.0, 210.0),
            (125.0, 5.0, 625.0),
        ],
    )

    assert heatmap.total_liquidation_volume == 8.0
    assert heatmap.total_liquidation_notional == 936.0

    assert heatmap.strongest_zone is not None
    assert heatmap.strongest_zone.center_price == 125.0
    assert (
        heatmap.strongest_zone.liquidation_notional
        == 625.0
    )


def test_zones_near_price() -> None:
    engine = LiquidityLiquidationZoneEngine(
        zone_size=10.0
    )

    heatmap = engine.build_heatmap(
        "BTC/USDT:USDT",
        1000,
        [
            (101.0, 1.0, 101.0),
            (121.0, 1.0, 121.0),
            (151.0, 1.0, 151.0),
        ],
    )

    nearby = heatmap.zones_near_price(
        price=105.0,
        max_distance_percent=1.0,
    )

    assert len(nearby) == 1
    assert nearby[0].lower_price == 100.0


def test_analyze_high_concentration() -> None:
    engine = LiquidityLiquidationZoneEngine(
        zone_size=10.0
    )

    heatmap = engine.build_heatmap(
        "BTC/USDT:USDT",
        1000,
        [
            (101.0, 5.0, 505.0),
            (105.0, 5.0, 525.0),
            (151.0, 1.0, 151.0),
        ],
    )

    signal = engine.analyze(
        heatmap,
        current_price=105.0,
        proximity_percent=1.0,
    )

    assert signal.strongest_zone_price == 105.0
    assert signal.strongest_zone_notional == 1030.0
    assert signal.nearby_zone_count == 1
    assert (
        signal.nearby_liquidation_notional
        == 1030.0
    )

    assert signal.concentration == pytest.approx(
        1030.0 / 1181.0
    )

    assert (
        signal.context
        == "HIGH_LIQUIDITY_CONCENTRATION"
    )


def test_analyze_without_nearby_zones() -> None:
    engine = LiquidityLiquidationZoneEngine(
        zone_size=10.0
    )

    heatmap = engine.build_heatmap(
        "BTC/USDT:USDT",
        1000,
        [
            (101.0, 1.0, 101.0),
        ],
    )

    signal = engine.analyze(
        heatmap,
        current_price=200.0,
        proximity_percent=1.0,
    )

    assert signal.nearby_zone_count == 0
    assert (
        signal.nearby_liquidation_notional
        == 0.0
    )
    assert signal.concentration == 0.0
    assert (
        signal.context
        == "NO_NEARBY_LIQUIDITY"
    )


def test_minimum_notional_filters_events() -> None:
    engine = LiquidityLiquidationZoneEngine(
        zone_size=10.0,
        minimum_notional=100.0,
    )

    heatmap = engine.build_heatmap(
        "BTC/USDT:USDT",
        1000,
        [
            (101.0, 0.5, 50.5),
            (105.0, 2.0, 210.0),
        ],
    )

    assert len(heatmap.zones) == 1
    assert (
        heatmap.total_liquidation_notional
        == 210.0
    )


def test_empty_heatmap() -> None:
    engine = LiquidityLiquidationZoneEngine(
        zone_size=10.0
    )

    heatmap = engine.build_heatmap(
        "BTC/USDT:USDT",
        1000,
        [],
    )

    assert heatmap.zones == ()
    assert heatmap.total_liquidation_volume == 0.0
    assert heatmap.total_liquidation_notional == 0.0
    assert heatmap.strongest_zone is None

    signal = engine.analyze(
        heatmap,
        current_price=100.0,
    )

    assert signal.strongest_zone_price is None
    assert signal.strongest_zone_notional == 0.0
    assert signal.nearby_zone_count == 0
    assert signal.concentration == 0.0
    assert (
        signal.context
        == "NO_NEARBY_LIQUIDITY"
    )


def test_reset() -> None:
    engine = LiquidityLiquidationZoneEngine(
        zone_size=10.0
    )

    engine.build_heatmap(
        "BTC/USDT:USDT",
        1000,
        [
            (101.0, 1.0, 101.0),
        ],
    )

    assert engine.previous is not None

    engine.reset()

    assert engine.previous is None


def test_invalid_event_structure() -> None:
    engine = LiquidityLiquidationZoneEngine(
        zone_size=10.0
    )

    with pytest.raises(ValueError):
        engine.build_heatmap(
            "BTC/USDT:USDT",
            1000,
            [[100.0, 1.0]],
        )

    with pytest.raises(ValueError):
        engine.build_heatmap(
            "BTC/USDT:USDT",
            1000,
            [
                (
                    100.0,
                    1.0,
                    101.0,
                )
            ],
        )


def test_invalid_engine_configuration() -> None:
    with pytest.raises(ValueError):
        LiquidityLiquidationZoneEngine(
            zone_size=0
        )

    with pytest.raises(ValueError):
        LiquidityLiquidationZoneEngine(
            zone_size=10.0,
            minimum_notional=-1.0,
        )


def test_invalid_heatmap_type() -> None:
    with pytest.raises(TypeError):
        LiquidityLiquidationZoneEngine._validate_heatmap(
            object()
        )


def test_invalid_current_price() -> None:
    engine = LiquidityLiquidationZoneEngine(
        zone_size=10.0
    )

    heatmap = LiquidityHeatmap(
        symbol="BTC/USDT:USDT",
        timestamp=1000,
        zones=(),
    )

    with pytest.raises(ValueError):
        heatmap.zones_near_price(
            price=0.0,
            max_distance_percent=1.0,
        )

    with pytest.raises(ValueError):
        engine.analyze(
            heatmap,
            current_price=0.0,
        )