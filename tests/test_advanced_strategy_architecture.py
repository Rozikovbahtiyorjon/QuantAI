from __future__ import annotations

import pytest

from src.advanced_strategy_architecture import (
    AdvancedStrategyArchitecture,
    StrategyDecision,
)


def make_inputs() -> dict[str, object]:
    return {
        "direction": "LONG",
        "filters": {
            "regime": True,
            "liquidity": True,
        },
        "entry_conditions": {
            "technical": 0.8,
            "market_intelligence": 0.7,
        },
        "confirmation": {
            "ml": 0.8,
            "order_flow": 0.7,
        },
        "confidence": 0.9,
        "risk_allowed": True,
        "regime": "TREND_UP",
    }


def test_strong_long_produces_buy() -> None:
    engine = AdvancedStrategyArchitecture()

    result = engine.evaluate(**make_inputs())

    assert isinstance(result, StrategyDecision)
    assert result.action == "BUY"
    assert result.filters_passed is True
    assert result.risk_allowed is True
    assert result.entry_score == pytest.approx(0.75)
    assert result.confirmation_score == pytest.approx(0.75)
    assert result.confidence == pytest.approx(0.9)
    assert result.score == pytest.approx(0.8)
    assert result.regime == "TREND_UP"


def test_strong_short_produces_sell() -> None:
    engine = AdvancedStrategyArchitecture()

    data = make_inputs()
    data["direction"] = "SHORT"
    data["regime"] = "TREND_DOWN"

    result = engine.evaluate(**data)

    assert result.action == "SELL"
    assert result.regime == "TREND_DOWN"


def test_failed_filter_blocks_strategy() -> None:
    engine = AdvancedStrategyArchitecture()

    data = make_inputs()
    data["filters"] = {
        "regime": True,
        "liquidity": False,
    }

    result = engine.evaluate(**data)

    assert result.action == "BLOCK"
    assert result.filters_passed is False


def test_risk_blocks_strategy() -> None:
    engine = AdvancedStrategyArchitecture()

    data = make_inputs()
    data["risk_allowed"] = False

    result = engine.evaluate(**data)

    assert result.action == "BLOCK"
    assert result.risk_allowed is False


def test_low_entry_score_holds() -> None:
    engine = AdvancedStrategyArchitecture()

    data = make_inputs()
    data["entry_conditions"] = {
        "technical": 0.4,
        "market_intelligence": 0.5,
    }

    result = engine.evaluate(**data)

    assert result.action == "HOLD"
    assert result.entry_score == pytest.approx(0.45)


def test_low_confirmation_score_holds() -> None:
    engine = AdvancedStrategyArchitecture()

    data = make_inputs()
    data["confirmation"] = {
        "ml": 0.4,
        "order_flow": 0.5,
    }

    result = engine.evaluate(**data)

    assert result.action == "HOLD"
    assert result.confirmation_score == pytest.approx(0.45)


def test_low_confidence_holds() -> None:
    engine = AdvancedStrategyArchitecture()

    data = make_inputs()
    data["confidence"] = 0.5

    result = engine.evaluate(**data)

    assert result.action == "HOLD"


def test_threshold_boundary_allows_trade() -> None:
    engine = AdvancedStrategyArchitecture(
        entry_threshold=0.6,
        confirmation_threshold=0.6,
        confidence_threshold=0.6,
    )

    data = make_inputs()
    data["entry_conditions"] = {
        "technical": 0.6,
        "market_intelligence": 0.6,
    }
    data["confirmation"] = {
        "ml": 0.6,
        "order_flow": 0.6,
    }
    data["confidence"] = 0.6

    result = engine.evaluate(**data)

    assert result.action == "BUY"


def test_custom_thresholds() -> None:
    engine = AdvancedStrategyArchitecture(
        entry_threshold=0.8,
        confirmation_threshold=0.8,
        confidence_threshold=0.8,
    )

    result = engine.evaluate(**make_inputs())

    assert result.action == "HOLD"


def test_empty_filters_are_rejected() -> None:
    engine = AdvancedStrategyArchitecture()

    data = make_inputs()
    data["filters"] = {}

    with pytest.raises(ValueError):
        engine.evaluate(**data)


def test_empty_entry_conditions_are_rejected() -> None:
    engine = AdvancedStrategyArchitecture()

    data = make_inputs()
    data["entry_conditions"] = {}

    with pytest.raises(ValueError):
        engine.evaluate(**data)


def test_invalid_scores_are_rejected() -> None:
    engine = AdvancedStrategyArchitecture()

    data = make_inputs()
    data["confirmation"] = {
        "ml": 1.1,
    }

    with pytest.raises(ValueError):
        engine.evaluate(**data)


def test_constructor_validation() -> None:
    with pytest.raises(TypeError):
        AdvancedStrategyArchitecture(
            entry_threshold="0.6",
        )

    with pytest.raises(ValueError):
        AdvancedStrategyArchitecture(
            confirmation_threshold=-0.1,
        )

    with pytest.raises(ValueError):
        AdvancedStrategyArchitecture(
            confidence_threshold=1.1,
        )


def test_evaluate_argument_validation() -> None:
    engine = AdvancedStrategyArchitecture()

    data = make_inputs()
    data["direction"] = "BUY"

    with pytest.raises(ValueError):
        engine.evaluate(**data)

    data = make_inputs()
    data["risk_allowed"] = 1

    with pytest.raises(TypeError):
        engine.evaluate(**data)

    data = make_inputs()
    data["regime"] = ""

    with pytest.raises(ValueError):
        engine.evaluate(**data)