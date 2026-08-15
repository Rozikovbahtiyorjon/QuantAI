from __future__ import annotations

import pytest

from src.trading_activity_optimizer import (
    ActivityAction,
    ActivityOptimization,
)
from src.trading_activity_policy import (
    ActivityPolicyDecision,
    TradingActivityPolicy,
)


def make_optimization(
    action=ActivityAction.INCREASE,
    confidence=0.9,
    adjustments=None,
):
    if adjustments is None:
        adjustments = {
            "entry_threshold": -0.01,
            "confidence_threshold": -0.01,
        }

    return ActivityOptimization(
        action=action,
        trade_target=5,
        confidence=confidence,
        reason="test",
        adjustments=adjustments,
    )


def test_policy_configuration() -> None:
    policy = TradingActivityPolicy(
        minimum_confidence=0.75,
        maximum_adjustment=0.03,
    )

    assert policy.minimum_confidence == 0.75
    assert policy.maximum_adjustment == 0.03


def test_invalid_confidence() -> None:
    with pytest.raises(ValueError):
        TradingActivityPolicy(
            minimum_confidence=1.1
        )


def test_invalid_adjustment_limit() -> None:
    with pytest.raises(ValueError):
        TradingActivityPolicy(
            maximum_adjustment=0.0
        )


def test_hold_is_not_approved() -> None:
    policy = TradingActivityPolicy()

    result = policy.evaluate(
        make_optimization(
            action=ActivityAction.HOLD
        )
    )

    assert isinstance(
        result,
        ActivityPolicyDecision,
    )
    assert result.approved is False
    assert result.action is ActivityAction.HOLD


def test_high_confidence_increase_is_approved() -> None:
    policy = TradingActivityPolicy()

    result = policy.evaluate(
        make_optimization(
            action=ActivityAction.INCREASE,
            confidence=0.9,
        )
    )

    assert result.approved is True
    assert result.action is ActivityAction.INCREASE
    assert (
        result.adjustments["entry_threshold"]
        == -0.01
    )


def test_high_confidence_decrease_is_approved() -> None:
    policy = TradingActivityPolicy()

    result = policy.evaluate(
        make_optimization(
            action=ActivityAction.DECREASE,
            confidence=0.9,
            adjustments={
                "entry_threshold": 0.01,
                "confidence_threshold": 0.01,
            },
        )
    )

    assert result.approved is True
    assert result.action is ActivityAction.DECREASE


def test_low_confidence_is_rejected() -> None:
    policy = TradingActivityPolicy(
        minimum_confidence=0.8
    )

    result = policy.evaluate(
        make_optimization(
            confidence=0.7
        )
    )

    assert result.approved is False
    assert result.action is ActivityAction.HOLD


def test_adjustment_is_capped() -> None:
    policy = TradingActivityPolicy(
        maximum_adjustment=0.02
    )

    result = policy.evaluate(
        make_optimization(
            adjustments={
                "entry_threshold": -0.10,
                "confidence_threshold": 0.10,
            }
        )
    )

    assert (
        result.adjustments["entry_threshold"]
        == -0.02
    )
    assert (
        result.adjustments["confidence_threshold"]
        == 0.02
    )


def test_mapping_evaluation() -> None:
    policy = TradingActivityPolicy()

    result = policy.evaluate_from_mapping(
        {
            "action": "INCREASE",
            "trade_target": 6,
            "confidence": 0.9,
            "reason": "low activity",
            "adjustments": {
                "entry_threshold": -0.01,
            },
        }
    )

    assert result.approved is True
    assert result.action is ActivityAction.INCREASE


def test_mapping_hold() -> None:
    policy = TradingActivityPolicy()

    result = policy.evaluate_from_mapping(
        {
            "action": "HOLD",
            "trade_target": 5,
            "confidence": 0.9,
            "reason": "stable",
            "adjustments": {},
        }
    )

    assert result.approved is False
    assert result.action is ActivityAction.HOLD


def test_mapping_requires_fields() -> None:
    policy = TradingActivityPolicy()

    with pytest.raises(ValueError):
        policy.evaluate_from_mapping(
            {
                "action": "HOLD",
                "confidence": 0.9,
            }
        )


def test_mapping_rejects_invalid_adjustments() -> None:
    policy = TradingActivityPolicy()

    with pytest.raises(TypeError):
        policy.evaluate_from_mapping(
            {
                "action": "HOLD",
                "confidence": 0.9,
                "reason": "test",
                "adjustments": 1,
            }
        )


def test_evaluate_rejects_invalid_object() -> None:
    policy = TradingActivityPolicy()

    with pytest.raises(TypeError):
        policy.evaluate("invalid")


def test_policy_preserves_confidence() -> None:
    policy = TradingActivityPolicy()

    result = policy.evaluate(
        make_optimization(
            confidence=0.91
        )
    )

    assert result.confidence == 0.91


def test_policy_returns_mapping_adjustments() -> None:
    policy = TradingActivityPolicy()

    result = policy.evaluate(
        make_optimization()
    )

    assert isinstance(
        result.adjustments,
        dict,
    )