from __future__ import annotations

import pytest

from src.order_flow_intelligence import OrderFlowSignal
from src.order_flow_strategy_integration import (
    OrderFlowStrategyDecision,
    OrderFlowStrategyIntegration,
)
from src.strategy import SignalResult


def make_strategy_result(
    signal: str = "BUY",
    approved: bool = True,
) -> SignalResult:
    return SignalResult(
        signal=signal,
        trade_approved=approved,
    )


def make_order_flow_signal(
    pressure: float = 0.30,
    context: str = "BID_PRESSURE",
) -> OrderFlowSignal:
    return OrderFlowSignal(
        spread=1.0,
        spread_percent=1.0,
        bid_volume=13.0,
        ask_volume=7.0,
        bid_notional=1300.0,
        ask_notional=707.0,
        volume_imbalance=0.30,
        notional_imbalance=0.30,
        microprice=100.5,
        microprice_delta=None,
        bid_liquidity_share=0.65,
        ask_liquidity_share=0.35,
        pressure=pressure,
        context=context,
    )


def test_buy_is_confirmed_by_positive_order_flow() -> None:
    integration = OrderFlowStrategyIntegration()

    result = integration.evaluate(
        make_strategy_result("BUY"),
        make_order_flow_signal(
            pressure=0.30,
            context="BID_PRESSURE",
        ),
    )

    assert isinstance(
        result,
        OrderFlowStrategyDecision,
    )

    assert result.signal == "BUY"
    assert result.approved is True

    assert result.order_flow_score == pytest.approx(
        0.65
    )

    assert "confirms" in result.reason


def test_buy_is_blocked_by_strong_ask_pressure() -> None:
    integration = OrderFlowStrategyIntegration()

    result = integration.evaluate(
        make_strategy_result("BUY"),
        make_order_flow_signal(
            pressure=-0.30,
            context="ASK_PRESSURE",
        ),
    )

    assert result.signal == "HOLD"
    assert result.approved is False

    assert (
        "conflicts with BUY"
        in result.reason
    )


def test_sell_is_confirmed_by_negative_order_flow() -> None:
    integration = OrderFlowStrategyIntegration()

    result = integration.evaluate(
        make_strategy_result("SELL"),
        make_order_flow_signal(
            pressure=-0.30,
            context="ASK_PRESSURE",
        ),
    )

    assert result.signal == "SELL"
    assert result.approved is True

    assert result.order_flow_score == pytest.approx(
        0.35
    )

    assert "confirms" in result.reason


def test_sell_is_blocked_by_strong_bid_pressure() -> None:
    integration = OrderFlowStrategyIntegration()

    result = integration.evaluate(
        make_strategy_result("SELL"),
        make_order_flow_signal(
            pressure=0.30,
            context="BID_PRESSURE",
        ),
    )

    assert result.signal == "HOLD"
    assert result.approved is False

    assert (
        "conflicts with SELL"
        in result.reason
    )


def test_strategy_hold_cannot_become_trade() -> None:
    integration = OrderFlowStrategyIntegration()

    result = integration.evaluate(
        make_strategy_result(
            "HOLD",
            approved=False,
        ),
        make_order_flow_signal(
            pressure=1.0,
            context="BID_PRESSURE",
        ),
    )

    assert result.signal == "HOLD"
    assert result.approved is False
    assert result.strategy_signal == "HOLD"


def test_unapproved_strategy_cannot_become_trade() -> None:
    integration = OrderFlowStrategyIntegration()

    result = integration.evaluate(
        make_strategy_result(
            "BUY",
            approved=False,
        ),
        make_order_flow_signal(
            pressure=1.0,
            context="BID_PRESSURE",
        ),
    )

    assert result.signal == "HOLD"
    assert result.approved is False
    assert result.strategy_signal == "BUY"
    assert result.strategy_approved is False


def test_balanced_order_flow_does_not_block_direction() -> None:
    integration = OrderFlowStrategyIntegration()

    result = integration.evaluate(
        make_strategy_result("BUY"),
        make_order_flow_signal(
            pressure=-0.0024875621890547263,
            context="BALANCED",
        ),
    )

    assert result.signal == "BUY"
    assert result.approved is True

    assert result.order_flow_score == pytest.approx(
        0.49875621890547265
    )


def test_pressure_at_threshold_is_treated_as_conflict() -> None:
    integration = OrderFlowStrategyIntegration(
        conflict_threshold=0.20,
    )

    result = integration.evaluate(
        make_strategy_result("BUY"),
        make_order_flow_signal(
            pressure=-0.20,
            context="ASK_PRESSURE",
        ),
    )

    assert result.signal == "HOLD"
    assert result.approved is False


def test_invalid_strategy_result_is_rejected() -> None:
    integration = OrderFlowStrategyIntegration()

    with pytest.raises(
        TypeError,
        match="SignalResult",
    ):
        integration.evaluate(
            "invalid",  # type: ignore[arg-type]
            make_order_flow_signal(),
        )


def test_invalid_order_flow_signal_is_rejected() -> None:
    integration = OrderFlowStrategyIntegration()

    with pytest.raises(
        TypeError,
        match="OrderFlowSignal",
    ):
        integration.evaluate(
            make_strategy_result(),
            "invalid",  # type: ignore[arg-type]
        )


def test_constructor_validation() -> None:
    with pytest.raises(
        TypeError,
        match="finite number",
    ):
        OrderFlowStrategyIntegration(
            conflict_threshold="0.15",  # type: ignore[arg-type]
        )

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        OrderFlowStrategyIntegration(
            conflict_threshold=0.0,
        )

    with pytest.raises(
        ValueError,
        match="at most 1.0",
    ):
        OrderFlowStrategyIntegration(
            conflict_threshold=1.1,
        )