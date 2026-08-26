from __future__ import annotations

import pytest

from experimental.src.futures_derivatives_intelligence import (
    DerivativesSignal,
    DerivativesSnapshot,
    DerivativesSnapshotAdapter,
    FuturesDerivativesIntelligence,
)


def make_snapshot(**overrides: float) -> DerivativesSnapshot:
    data = {
        "symbol": "BTC/USDT:USDT",
        "timestamp": 1000,
        "price": 100000.0,
        "open_interest": 1000.0,
        "open_interest_change": 50.0,
        "funding_rate": 0.0001,
        "futures_volume": 5000.0,
        "long_short_ratio": 1.1,
        "liquidation_volume": 10.0,
        "basis": 0.002,
        "price_change": 0.01,
    }

    data.update(overrides)

    return DerivativesSnapshot(**data)


def test_snapshot_calculates_oi_change_percent_and_pattern() -> None:
    snapshot = make_snapshot()

    assert snapshot.oi_change_percent == pytest.approx(5.0)
    assert (
        snapshot.price_oi_direction
        == "PRICE_UP_OI_UP"
    )


def test_price_oi_patterns() -> None:
    assert (
        make_snapshot(
            price_change=0.01,
            open_interest_change=-10.0,
        ).price_oi_direction
        == "PRICE_UP_OI_DOWN"
    )

    assert (
        make_snapshot(
            price_change=-0.01,
            open_interest_change=10.0,
        ).price_oi_direction
        == "PRICE_DOWN_OI_UP"
    )

    assert (
        make_snapshot(
            price_change=-0.01,
            open_interest_change=-10.0,
        ).price_oi_direction
        == "PRICE_DOWN_OI_DOWN"
    )

    assert (
        make_snapshot(
            price_change=0.0,
            open_interest_change=0.0,
        ).price_oi_direction
        == "NEUTRAL"
    )


def test_bullish_derivatives_state() -> None:
    engine = FuturesDerivativesIntelligence()

    result = engine.analyze(
        make_snapshot()
    )

    assert isinstance(
        result,
        DerivativesSignal,
    )

    assert result.market_state == "BULLISH"
    assert result.pressure == "BULLISH_PRESSURE"
    assert (
        result.price_oi_pattern
        == "PRICE_UP_OI_UP"
    )
    assert result.basis_signal == "POSITIVE_PREMIUM"
    assert result.confidence > 0.5


def test_bearish_derivatives_state() -> None:
    engine = FuturesDerivativesIntelligence()

    result = engine.analyze(
        make_snapshot(
            price_change=-0.02,
            open_interest_change=60.0,
            funding_rate=0.001,
            basis=-0.002,
        )
    )

    assert result.market_state == "BEARISH"
    assert result.pressure == "BEARISH_PRESSURE"
    assert (
        result.price_oi_pattern
        == "PRICE_DOWN_OI_UP"
    )
    assert (
        result.funding_signal
        == "POSITIVE_EXTREME"
    )
    assert (
        result.basis_signal
        == "NEGATIVE_DISCOUNT"
    )


def test_stress_state_on_high_liquidations() -> None:
    engine = FuturesDerivativesIntelligence(
        liquidation_threshold=100.0
    )

    result = engine.analyze(
        make_snapshot(
            liquidation_volume=250.0
        )
    )

    assert result.market_state == "STRESS"
    assert (
        result.liquidation_signal
        == "HIGH_LIQUIDATION"
    )


def test_elevated_liquidation_signal() -> None:
    engine = FuturesDerivativesIntelligence(
        liquidation_threshold=100.0
    )

    result = engine.analyze(
        make_snapshot(
            liquidation_volume=100.0
        )
    )

    assert (
        result.liquidation_signal
        == "ELEVATED_LIQUIDATION"
    )

    assert result.market_state != "STRESS"


def test_funding_signals() -> None:
    engine = FuturesDerivativesIntelligence(
        funding_threshold=0.0005
    )

    positive = engine.analyze(
        make_snapshot(
            funding_rate=0.001
        )
    )

    negative = engine.analyze(
        make_snapshot(
            funding_rate=-0.001
        )
    )

    neutral = engine.analyze(
        make_snapshot(
            funding_rate=0.0001
        )
    )

    assert (
        positive.funding_signal
        == "POSITIVE_EXTREME"
    )

    assert (
        negative.funding_signal
        == "NEGATIVE_EXTREME"
    )

    assert (
        neutral.funding_signal
        == "NEUTRAL"
    )


def test_basis_signals() -> None:
    engine = FuturesDerivativesIntelligence(
        basis_threshold=0.001
    )

    positive = engine.analyze(
        make_snapshot(
            basis=0.002
        )
    )

    negative = engine.analyze(
        make_snapshot(
            basis=-0.002
        )
    )

    neutral = engine.analyze(
        make_snapshot(
            basis=0.0005
        )
    )

    assert (
        positive.basis_signal
        == "POSITIVE_PREMIUM"
    )

    assert (
        negative.basis_signal
        == "NEGATIVE_DISCOUNT"
    )

    assert (
        neutral.basis_signal
        == "NEUTRAL"
    )


def test_neutral_market_state() -> None:
    engine = FuturesDerivativesIntelligence()

    result = engine.analyze(
        make_snapshot(
            open_interest_change=0.0,
            price_change=0.0,
            funding_rate=0.0,
            basis=0.0,
            liquidation_volume=0.0,
        )
    )

    assert result.market_state == "NEUTRAL"
    assert (
        result.pressure
        == "BALANCED_PRESSURE"
    )

    assert result.confidence == pytest.approx(
        0.25
    )


def test_previous_snapshot_and_reset() -> None:
    engine = FuturesDerivativesIntelligence()
    snapshot = make_snapshot()

    assert engine.previous is None

    engine.analyze(snapshot)

    assert engine.previous == snapshot

    engine.reset()

    assert engine.previous is None


def test_invalid_snapshot_type() -> None:
    engine = FuturesDerivativesIntelligence()

    with pytest.raises(TypeError):
        engine.analyze({})


def test_invalid_snapshot_values() -> None:
    engine = FuturesDerivativesIntelligence()

    with pytest.raises(ValueError):
        engine.analyze(
            make_snapshot(
                price=0.0
            )
        )

    with pytest.raises(ValueError):
        engine.analyze(
            make_snapshot(
                open_interest=-1.0
            )
        )

    with pytest.raises(ValueError):
        engine.analyze(
            make_snapshot(
                liquidation_volume=-1.0
            )
        )


def test_invalid_engine_thresholds() -> None:
    with pytest.raises(TypeError):
        FuturesDerivativesIntelligence(
            funding_threshold="0.1"
        )

    with pytest.raises(ValueError):
        FuturesDerivativesIntelligence(
            funding_threshold=0.0
        )

    with pytest.raises(ValueError):
        FuturesDerivativesIntelligence(
            liquidation_threshold=-1.0
        )

    with pytest.raises(ValueError):
        FuturesDerivativesIntelligence(
            basis_threshold=0.0
        )


def test_snapshot_adapter_normalizes_payload() -> None:
    raw = {
        "timestamp": "12345",
        "price": "100000",
        "open_interest": "2500",
        "open_interest_change": "100",
        "funding_rate": "0.0002",
        "futures_volume": "50000",
        "long_short_ratio": "1.2",
        "liquidation_volume": "25",
        "basis": "0.0015",
        "price_change": "0.02",
    }

    snapshot = DerivativesSnapshotAdapter.normalize(
        " BTC/USDT:USDT ",
        raw,
    )

    assert snapshot.symbol == "BTC/USDT:USDT"
    assert snapshot.timestamp == 12345
    assert snapshot.price == 100000.0
    assert snapshot.open_interest == 2500.0
    assert snapshot.funding_rate == 0.0002


def test_snapshot_adapter_rejects_invalid_payload() -> None:
    with pytest.raises(TypeError):
        DerivativesSnapshotAdapter.normalize(
            "BTC/USDT:USDT",
            [],
        )

    with pytest.raises(ValueError):
        DerivativesSnapshotAdapter.normalize(
            "BTC/USDT:USDT",
            {
                "price": "invalid"
            },
        )

    with pytest.raises(ValueError):
        DerivativesSnapshotAdapter.normalize(
            "",
            {
                "price": 100.0
            },
        )