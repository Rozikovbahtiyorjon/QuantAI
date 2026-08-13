from __future__ import annotations

import pytest

from src.unified_market_intelligence import (
    UnifiedMarketIntelligenceLayer,
)


def test_default_weights_are_normalized() -> None:
    engine = UnifiedMarketIntelligenceLayer()

    assert sum(engine.weights.values()) == pytest.approx(1.0)


def test_single_component_is_preserved() -> None:
    engine = UnifiedMarketIntelligenceLayer()

    result = engine.evaluate(
        {"ml": 0.8},
        regime="TREND_UP",
    )

    assert result.score == pytest.approx(0.8)
    assert result.direction == "BULLISH"
    assert result.regime == "TREND_UP"
    assert result.components_used == 1
    assert result.components_total == 10


def test_weighted_score_uses_available_components() -> None:
    engine = UnifiedMarketIntelligenceLayer(
        weights={
            "technical": 1.0,
            "ml": 3.0,
        }
    )

    result = engine.evaluate(
        {
            "technical": 0.2,
            "ml": 0.8,
        },
        regime="RANGE",
    )

    assert result.score == pytest.approx(0.65)
    assert result.direction == "BULLISH"


def test_missing_components_are_ignored() -> None:
    engine = UnifiedMarketIntelligenceLayer(
        weights={
            "technical": 1.0,
            "ml": 1.0,
        }
    )

    result = engine.evaluate(
        {"technical": 0.2},
        regime="RANGE",
    )

    assert result.score == pytest.approx(0.2)
    assert result.components_used == 1


def test_bearish_direction() -> None:
    engine = UnifiedMarketIntelligenceLayer()

    result = engine.evaluate(
        {"ml": 0.1},
        regime="TREND_DOWN",
    )

    assert result.direction == "BEARISH"
    assert result.confidence == pytest.approx(0.8)


def test_neutral_direction() -> None:
    engine = UnifiedMarketIntelligenceLayer()

    result = engine.evaluate(
        {"ml": 0.5},
        regime="RANGE",
    )

    assert result.direction == "NEUTRAL"
    assert result.confidence == pytest.approx(0.0)


def test_event_risk_levels() -> None:
    engine = UnifiedMarketIntelligenceLayer()

    normal = engine.evaluate(
        {"ml": 0.6},
        regime="RANGE",
        event_risk=0.1,
    )

    extreme = engine.evaluate(
        {"ml": 0.6},
        regime="RANGE",
        event_risk=0.8,
    )

    assert normal.risk_level == "NORMAL"
    assert extreme.risk_level == "EXTREME"


def test_constructor_validation() -> None:
    with pytest.raises(TypeError):
        UnifiedMarketIntelligenceLayer(
            weights=None,
            neutral_score="0.5",
        )

    with pytest.raises(ValueError):
        UnifiedMarketIntelligenceLayer(
            weights={"ml": 0.0}
        )

    with pytest.raises(TypeError):
        UnifiedMarketIntelligenceLayer(
            weights={"ml": True}
        )

    with pytest.raises(ValueError):
        UnifiedMarketIntelligenceLayer(
            neutral_score=1.1
        )


def test_evaluate_validation() -> None:
    engine = UnifiedMarketIntelligenceLayer()

    with pytest.raises(TypeError):
        engine.evaluate(
            "invalid",
            regime="RANGE",
        )

    with pytest.raises(TypeError):
        engine.evaluate(
            {"ml": "0.8"},
            regime="RANGE",
        )

    with pytest.raises(ValueError):
        engine.evaluate(
            {"ml": 1.1},
            regime="RANGE",
        )

    with pytest.raises(ValueError):
        engine.evaluate(
            {},
            regime="RANGE",
        )

    with pytest.raises(TypeError):
        engine.evaluate(
            {"ml": 0.8},
            regime=123,
        )

    with pytest.raises(ValueError):
        engine.evaluate(
            {"ml": 0.8},
            regime="",
        )

    with pytest.raises(ValueError):
        engine.evaluate(
            {"ml": 0.8},
            regime="RANGE",
            event_risk=-0.1,
        )


def test_all_components_are_supported() -> None:
    engine = UnifiedMarketIntelligenceLayer()

    result = engine.evaluate(
        {
            "technical": 0.6,
            "ml": 0.7,
            "order_flow": 0.65,
            "derivatives": 0.55,
            "liquidation": 0.45,
            "regime": 0.7,
            "sentiment": 0.6,
            "social_attention": 0.5,
            "sentiment_divergence": 0.4,
            "event_risk": 0.3,
        },
        regime="TREND_UP",
        event_risk=0.4,
    )

    assert result.components_used == 10
    assert 0.0 <= result.score <= 1.0
    assert 0.0 <= result.confidence <= 1.0


def test_confidence_is_bounded() -> None:
    engine = UnifiedMarketIntelligenceLayer()

    assert engine.evaluate(
        {"ml": 0.0},
        regime="RANGE",
    ).confidence == pytest.approx(1.0)

    assert engine.evaluate(
        {"ml": 1.0},
        regime="RANGE",
    ).confidence == pytest.approx(1.0)


def test_reset_independence() -> None:
    engine = UnifiedMarketIntelligenceLayer()

    first = engine.evaluate(
        {"ml": 0.2},
        regime="RANGE",
    )

    second = engine.evaluate(
        {"ml": 0.8},
        regime="RANGE",
    )

    assert first.score == pytest.approx(0.2)
    assert second.score == pytest.approx(0.8)


def test_custom_weights() -> None:
    engine = UnifiedMarketIntelligenceLayer(
        weights={
            "order_flow": 2.0,
            "derivatives": 1.0,
        }
    )

    result = engine.evaluate(
        {
            "order_flow": 1.0,
            "derivatives": 0.0,
        },
        regime="HIGH_VOLATILITY",
    )

    assert result.score == pytest.approx(2.0 / 3.0)
    assert result.components_used == 2


def test_event_risk_does_not_change_market_score() -> None:
    engine = UnifiedMarketIntelligenceLayer()

    result_a = engine.evaluate(
        {"ml": 0.7},
        regime="TREND_UP",
        event_risk=0.1,
    )

    result_b = engine.evaluate(
        {"ml": 0.7},
        regime="TREND_UP",
        event_risk=0.9,
    )

    assert result_a.score == pytest.approx(
        result_b.score
    )

    assert result_a.direction == result_b.direction
    assert result_a.risk_level != result_b.risk_level