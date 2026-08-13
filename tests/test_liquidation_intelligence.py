from __future__ import annotations

import pytest

from src.liquidation_intelligence import (
    LiquidationEvent,
    LiquidationIntelligenceEngine,
    LiquidationSnapshot,
)


def make_snapshot() -> LiquidationSnapshot:
    return LiquidationSnapshot(
        symbol="BTC/USDT:USDT",
        timestamp=1000,
        events=(
            LiquidationEvent(
                symbol="BTC/USDT:USDT",
                timestamp=1000,
                side="LONG",
                price=100.0,
                amount=2.0,
                notional=200.0,
            ),
            LiquidationEvent(
                symbol="BTC/USDT:USDT",
                timestamp=1001,
                side="SHORT",
                price=101.0,
                amount=3.0,
                notional=303.0,
            ),
        ),
    )


def test_snapshot_calculates_liquidation_metrics() -> None:
    snapshot = make_snapshot()

    assert snapshot.total_volume == 5.0
    assert snapshot.total_notional == 503.0

    assert (
        snapshot.long_liquidation_volume
        == 2.0
    )

    assert (
        snapshot.short_liquidation_volume
        == 3.0
    )

    assert (
        snapshot.long_liquidation_notional
        == 200.0
    )

    assert (
        snapshot.short_liquidation_notional
        == 303.0
    )

    assert snapshot.imbalance == pytest.approx(
        0.2
    )


def test_engine_builds_signal() -> None:
    engine = LiquidationIntelligenceEngine()

    signal = engine.update(
        make_snapshot()
    )

    assert signal.total_volume == 5.0
    assert signal.total_notional == 503.0
    assert signal.long_volume == 2.0
    assert signal.short_volume == 3.0

    assert signal.imbalance == pytest.approx(
        0.2
    )

    assert signal.intensity == 5.0

    assert (
        signal.context
        == "SHORT_LIQUIDATION_DOMINANT"
    )


def test_engine_baseline_intensity() -> None:
    engine = LiquidationIntelligenceEngine(
        baseline_volume=10.0
    )

    signal = engine.update(
        make_snapshot()
    )

    assert signal.intensity == pytest.approx(
        0.5
    )


def test_normalize_events() -> None:
    snapshot = (
        LiquidationIntelligenceEngine.normalize(
            symbol="BTC/USDT:USDT",
            timestamp=2000,
            raw_events=[
                {
                    "symbol": "BTC/USDT:USDT",
                    "timestamp": 2002,
                    "side": "short",
                    "price": 101.0,
                    "amount": 2.0,
                },
                {
                    "symbol": "BTC/USDT:USDT",
                    "timestamp": 2001,
                    "side": "long",
                    "price": 100.0,
                    "amount": 3.0,
                },
            ],
        )
    )

    assert len(snapshot.events) == 2

    assert (
        snapshot.events[0].timestamp
        == 2001
    )

    assert (
        snapshot.events[0].side
        == "LONG"
    )

    assert (
        snapshot.events[0].notional
        == 300.0
    )

    assert (
        snapshot.events[1].side
        == "SHORT"
    )


def test_empty_snapshot() -> None:
    snapshot = LiquidationSnapshot(
        symbol="BTC/USDT:USDT",
        timestamp=1000,
        events=(),
    )

    engine = LiquidationIntelligenceEngine()

    signal = engine.update(snapshot)

    assert signal.total_volume == 0.0
    assert signal.total_notional == 0.0
    assert signal.imbalance == 0.0
    assert signal.intensity == 0.0

    assert (
        signal.context
        == "NO_LIQUIDATIONS"
    )


def test_engine_rejects_symbol_change() -> None:
    engine = LiquidationIntelligenceEngine()

    engine.update(make_snapshot())

    changed = LiquidationSnapshot(
        symbol="ETH/USDT:USDT",
        timestamp=2000,
        events=(),
    )

    with pytest.raises(ValueError):
        engine.update(changed)


@pytest.mark.parametrize(
    "side",
    [
        "",
        "BUY",
        "SELL",
        "UNKNOWN",
    ],
)
def test_invalid_side(
    side: str,
) -> None:
    event = LiquidationEvent(
        symbol="BTC/USDT:USDT",
        timestamp=1000,
        side=side,
        price=100.0,
        amount=1.0,
        notional=100.0,
    )

    with pytest.raises(ValueError):
        LiquidationIntelligenceEngine._validate_event(
            event
        )


def test_invalid_notional() -> None:
    event = LiquidationEvent(
        symbol="BTC/USDT:USDT",
        timestamp=1000,
        side="LONG",
        price=100.0,
        amount=2.0,
        notional=250.0,
    )

    with pytest.raises(ValueError):
        LiquidationIntelligenceEngine._validate_event(
            event
        )


def test_invalid_raw_structure() -> None:
    with pytest.raises(TypeError):
        LiquidationIntelligenceEngine._normalize_events(
            "invalid"
        )

    with pytest.raises(ValueError):
        LiquidationIntelligenceEngine._normalize_events(
            [
                {
                    "price": 100.0,
                    "amount": 1.0,
                }
            ]
        )

    with pytest.raises(ValueError):
        LiquidationIntelligenceEngine._normalize_events(
            [
                {
                    "symbol": "BTC/USDT:USDT",
                    "timestamp": 1000,
                    "side": "LONG",
                    "price": 100.0,
                    "amount": 1.0,
                    "notional": 101.0,
                }
            ]
        )


def test_reset() -> None:
    engine = LiquidationIntelligenceEngine()

    engine.update(make_snapshot())

    assert engine.previous is not None

    engine.reset()

    assert engine.previous is None


def test_snapshot_rejects_mismatched_event_symbol() -> None:
    snapshot = LiquidationSnapshot(
        symbol="BTC/USDT:USDT",
        timestamp=1000,
        events=(
            LiquidationEvent(
                symbol="ETH/USDT:USDT",
                timestamp=1000,
                side="LONG",
                price=100.0,
                amount=1.0,
                notional=100.0,
            ),
        ),
    )

    with pytest.raises(ValueError):
        LiquidationIntelligenceEngine._validate_snapshot(
            snapshot
        )