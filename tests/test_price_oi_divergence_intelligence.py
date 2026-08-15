from __future__ import annotations

import pytest

from src.price_oi_divergence_intelligence import (
    PriceOIDivergenceAdapter,
    PriceOIDivergenceIntelligence,
    PriceOIDivergenceSignal,
    PriceOIDivergenceSnapshot,
)


def make_snapshot(
    **overrides: float,
) -> PriceOIDivergenceSnapshot:
    data = {
        "symbol": "BTC/USDT:USDT",
        "timestamp": 1000,
        "price_change": 0.02,
        "open_interest_change": 0.03,
        "price": 100000.0,
        "open_interest": 1000.0,
    }

    data.update(overrides)

    return PriceOIDivergenceSnapshot(**data)


def test_snapshot_detects_basic_pattern() -> None:
    snapshot = make_snapshot()

    assert snapshot.pattern == "PRICE_UP_OI_UP"
    assert snapshot.is_divergence is False


def test_snapshot_detects_divergence() -> None:
    assert (
        make_snapshot(
            price_change=0.02,
            open_interest_change=-0.03,
        ).is_divergence
        is True
    )

    assert (
        make_snapshot(
            price_change=-0.02,
            open_interest_change=0.03,
        ).is_divergence
        is True
    )


def test_all_snapshot_patterns() -> None:
    assert (
        make_snapshot(
            price_change=0.01,
            open_interest_change=-0.01,
        ).pattern
        == "PRICE_UP_OI_DOWN"
    )

    assert (
        make_snapshot(
            price_change=-0.01,
            open_interest_change=0.01,
        ).pattern
        == "PRICE_DOWN_OI_UP"
    )

    assert (
        make_snapshot(
            price_change=-0.01,
            open_interest_change=-0.01,
        ).pattern
        == "PRICE_DOWN_OI_DOWN"
    )

    assert (
        make_snapshot(
            price_change=0.0,
            open_interest_change=0.0,
        ).pattern
        == "NEUTRAL"
    )


def test_short_covering_signal() -> None:
    engine = PriceOIDivergenceIntelligence()

    result = engine.analyze(
        make_snapshot(
            price_change=0.02,
            open_interest_change=-0.03,
        )
    )

    assert isinstance(
        result,
        PriceOIDivergenceSignal,
    )

    assert result.pattern == "PRICE_UP_OI_DOWN"
    assert result.divergence is True
    assert result.interpretation == "SHORT_COVERING"
    assert result.contribution == "BULLISH_CAUTION"


def test_new_short_build_up_signal() -> None:
    engine = PriceOIDivergenceIntelligence()

    result = engine.analyze(
        make_snapshot(
            price_change=-0.02,
            open_interest_change=0.03,
        )
    )

    assert result.pattern == "PRICE_DOWN_OI_UP"
    assert result.divergence is True
    assert result.interpretation == "NEW_SHORT_BUILDUP"
    assert result.contribution == "BEARISH_CONFIRMATION"


def test_new_long_build_up_signal() -> None:
    engine = PriceOIDivergenceIntelligence()

    result = engine.analyze(
        make_snapshot()
    )

    assert result.interpretation == "NEW_LONG_BUILDUP"
    assert result.contribution == "BULLISH_CONFIRMATION"


def test_long_unwinding_signal() -> None:
    engine = PriceOIDivergenceIntelligence()

    result = engine.analyze(
        make_snapshot(
            price_change=-0.02,
            open_interest_change=-0.03,
        )
    )

    assert (
        result.interpretation
        == "LONG_LIQUIDATION_OR_UNWINDING"
    )

    assert result.contribution == "BEARISH_CAUTION"


def test_neutral_signal() -> None:
    engine = PriceOIDivergenceIntelligence()

    result = engine.analyze(
        make_snapshot(
            price_change=0.0,
            open_interest_change=0.0,
        )
    )

    assert result.pattern == "NEUTRAL"
    assert result.divergence is False
    assert (
        result.interpretation
        == "NO_CLEAR_DIVERGENCE"
    )
    assert result.contribution == "NEUTRAL"
    assert result.strength == 0.0


def test_minimum_change_filter() -> None:
    engine = PriceOIDivergenceIntelligence(
        minimum_change=0.01
    )

    result = engine.analyze(
        make_snapshot(
            price_change=0.005,
            open_interest_change=-0.02,
        )
    )

    assert result.pattern == "NEUTRAL"
    assert result.divergence is False


def test_strength_is_bounded() -> None:
    engine = PriceOIDivergenceIntelligence()

    result = engine.analyze(
        make_snapshot(
            price_change=2.0,
            open_interest_change=3.0,
        )
    )

    assert result.strength == 1.0


def test_previous_and_reset() -> None:
    engine = PriceOIDivergenceIntelligence()
    snapshot = make_snapshot()

    assert engine.previous is None

    engine.analyze(snapshot)

    assert engine.previous == snapshot

    engine.reset()

    assert engine.previous is None


def test_invalid_engine_threshold() -> None:
    with pytest.raises(TypeError):
        PriceOIDivergenceIntelligence(
            minimum_change="0.1"
        )

    with pytest.raises(ValueError):
        PriceOIDivergenceIntelligence(
            minimum_change=-0.1
        )


def test_invalid_snapshot() -> None:
    engine = PriceOIDivergenceIntelligence()

    with pytest.raises(TypeError):
        engine.analyze({})

    with pytest.raises(ValueError):
        engine.analyze(
            make_snapshot(price=0.0)
        )

    with pytest.raises(ValueError):
        engine.analyze(
            make_snapshot(
                open_interest=-1.0
            )
        )


def test_adapter_normalizes_payload() -> None:
    raw = {
        "timestamp": "12345",
        "price_change": "0.02",
        "open_interest_change": "-0.03",
        "price": "100000",
        "open_interest": "2500",
    }

    snapshot = PriceOIDivergenceAdapter.normalize(
        " BTC/USDT:USDT ",
        raw,
    )

    assert snapshot.symbol == "BTC/USDT:USDT"
    assert snapshot.timestamp == 12345
    assert snapshot.price_change == 0.02
    assert snapshot.open_interest_change == -0.03


def test_adapter_rejects_invalid_payload() -> None:
    with pytest.raises(TypeError):
        PriceOIDivergenceAdapter.normalize(
            "BTC/USDT:USDT",
            [],
        )

    with pytest.raises(ValueError):
        PriceOIDivergenceAdapter.normalize(
            "BTC/USDT:USDT",
            {"price": "invalid"},
        )

    with pytest.raises(ValueError):
        PriceOIDivergenceAdapter.normalize(
            "",
            {"price": 100.0},
        )