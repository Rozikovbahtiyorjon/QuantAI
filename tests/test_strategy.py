from __future__ import annotations

from dataclasses import replace

import pytest

from src.order_book_market_data import (
    OrderBookLevel,
    OrderBookSnapshot,
)
from src.order_flow_intelligence import (
    OrderFlowIntelligenceEngine,
)
from src.strategy import (
    ORDER_FLOW_CONFLICT_THRESHOLD,
    SignalResult,
    apply_order_flow_gate,
)


def make_strategy_result(
    signal: str = "BUY",
    approved: bool = True,
) -> SignalResult:
    return SignalResult(
        signal=signal,
        score=4.0,
        confidence=80.0,
        entry=100.0,
        stop_loss=98.0,
        take_profit=104.0,
        reasons=["Strategy approved"],
        ai_signal=signal,
        ai_confidence=80.0,
        ml_signal=signal,
        ml_probability=80.0,
        fusion_signal=signal,
        combined_confidence=80.0,
        trade_approved=approved,
    )


def make_order_flow_signal(
    bid_amount: float = 10.0,
    ask_amount: float = 5.0,
):
    engine = OrderFlowIntelligenceEngine()

    snapshot = OrderBookSnapshot(
        symbol="BTC/USDT",
        timestamp=1000,
        bids=(
            OrderBookLevel(
                100.0,
                bid_amount,
            ),
        ),
        asks=(
            OrderBookLevel(
                101.0,
                ask_amount,
            ),
        ),
    )

    return engine.update(
        snapshot
    )


def test_strategy_result_defaults_to_hold() -> None:
    result = SignalResult()

    assert result.signal == "HOLD"
    assert result.trade_approved is False
    assert result.order_flow_enabled is False
    assert result.order_flow_signal == "HOLD"
    assert result.order_flow_approved is False


def test_strategy_without_order_flow_remains_compatible() -> None:
    result = make_strategy_result(
        signal="BUY",
        approved=True,
    )

    output = apply_order_flow_gate(
        result,
        None,
    )

    assert output is result
    assert output.signal == "BUY"
    assert output.trade_approved is True
    assert output.order_flow_enabled is False
    assert output.order_flow_reason == ""


def test_buy_with_bid_pressure_is_allowed() -> None:
    result = make_strategy_result(
        signal="BUY",
        approved=True,
    )

    order_flow = make_order_flow_signal(
        bid_amount=10.0,
        ask_amount=5.0,
    )

    output = apply_order_flow_gate(
        result,
        order_flow,
    )

    assert output.signal == "BUY"
    assert output.trade_approved is True
    assert output.order_flow_enabled is True
    assert output.order_flow_approved is True
    assert output.order_flow_context == "BID_PRESSURE"
    assert output.order_flow_pressure > 0.0
    assert output.order_flow_score > 0.5


def test_sell_with_ask_pressure_is_allowed() -> None:
    result = make_strategy_result(
        signal="SELL",
        approved=True,
    )

    order_flow = make_order_flow_signal(
        bid_amount=5.0,
        ask_amount=10.0,
    )

    output = apply_order_flow_gate(
        result,
        order_flow,
    )

    assert output.signal == "SELL"
    assert output.trade_approved is True
    assert output.order_flow_enabled is True
    assert output.order_flow_approved is True
    assert output.order_flow_context == "ASK_PRESSURE"
    assert output.order_flow_pressure < 0.0
    assert output.order_flow_score < 0.5


def test_buy_is_blocked_by_strong_ask_pressure() -> None:
    result = make_strategy_result(
        signal="BUY",
        approved=True,
    )

    order_flow = make_order_flow_signal(
        bid_amount=2.0,
        ask_amount=20.0,
    )

    assert (
        order_flow.pressure
        <= -ORDER_FLOW_CONFLICT_THRESHOLD
    )

    output = apply_order_flow_gate(
        result,
        order_flow,
    )

    assert output.signal == "HOLD"
    assert output.trade_approved is False
    assert output.order_flow_enabled is True
    assert output.order_flow_approved is False
    assert output.order_flow_context == "ASK_PRESSURE"
    assert (
        "conflicts with BUY"
        in output.order_flow_reason
    )
    assert output.stop_loss == pytest.approx(
        output.entry
    )
    assert output.take_profit == pytest.approx(
        output.entry
    )


def test_sell_is_blocked_by_strong_bid_pressure() -> None:
    result = make_strategy_result(
        signal="SELL",
        approved=True,
    )

    order_flow = make_order_flow_signal(
        bid_amount=20.0,
        ask_amount=2.0,
    )

    assert (
        order_flow.pressure
        >= ORDER_FLOW_CONFLICT_THRESHOLD
    )

    output = apply_order_flow_gate(
        result,
        order_flow,
    )

    assert output.signal == "HOLD"
    assert output.trade_approved is False
    assert output.order_flow_enabled is True
    assert output.order_flow_approved is False
    assert output.order_flow_context == "BID_PRESSURE"
    assert (
        "conflicts with SELL"
        in output.order_flow_reason
    )


def test_strategy_hold_cannot_become_buy() -> None:
    result = make_strategy_result(
        signal="HOLD",
        approved=False,
    )

    order_flow = make_order_flow_signal(
        bid_amount=100.0,
        ask_amount=1.0,
    )

    output = apply_order_flow_gate(
        result,
        order_flow,
    )

    assert output.signal == "HOLD"
    assert output.trade_approved is False
    assert output.order_flow_enabled is True
    assert output.order_flow_approved is False


def test_strategy_not_approved_cannot_become_trade() -> None:
    result = make_strategy_result(
        signal="BUY",
        approved=False,
    )

    order_flow = make_order_flow_signal(
        bid_amount=100.0,
        ask_amount=1.0,
    )

    output = apply_order_flow_gate(
        result,
        order_flow,
    )

    assert output.signal == "HOLD"
    assert output.trade_approved is False
    assert output.order_flow_approved is False


def test_order_flow_neutral_preserves_buy() -> None:
    result = make_strategy_result(
        signal="BUY",
        approved=True,
    )

    order_flow = make_order_flow_signal(
        bid_amount=10.0,
        ask_amount=10.0,
    )

    output = apply_order_flow_gate(
        result,
        order_flow,
    )

    assert output.signal == "BUY"
    assert output.trade_approved is True
    assert output.order_flow_approved is True
    assert output.order_flow_context == "BALANCED"
    assert output.order_flow_score == pytest.approx(
        0.5
    )


def test_order_flow_neutral_preserves_sell() -> None:
    result = make_strategy_result(
        signal="SELL",
        approved=True,
    )

    order_flow = make_order_flow_signal(
        bid_amount=10.0,
        ask_amount=10.0,
    )

    output = apply_order_flow_gate(
        result,
        order_flow,
    )

    assert output.signal == "SELL"
    assert output.trade_approved is True
    assert output.order_flow_approved is True
    assert output.order_flow_score == pytest.approx(
        0.5
    )


def test_order_flow_score_is_bounded() -> None:
    buy_signal = make_order_flow_signal(
        bid_amount=100.0,
        ask_amount=1.0,
    )

    sell_signal = make_order_flow_signal(
        bid_amount=1.0,
        ask_amount=100.0,
    )

    buy_result = apply_order_flow_gate(
        make_strategy_result(),
        buy_signal,
    )

    sell_result = apply_order_flow_gate(
        make_strategy_result(
            signal="SELL"
        ),
        sell_signal,
    )

    assert 0.0 <= buy_result.order_flow_score <= 1.0
    assert 0.0 <= sell_result.order_flow_score <= 1.0


def test_invalid_order_flow_signal_is_rejected() -> None:
    result = make_strategy_result()

    with pytest.raises(
        TypeError,
        match="OrderFlowSignal",
    ):
        apply_order_flow_gate(
            result,
            "invalid",  # type: ignore[arg-type]
        )


def test_invalid_strategy_result_is_rejected() -> None:
    order_flow = make_order_flow_signal()

    with pytest.raises(
        TypeError,
    ):
        from src.order_flow_strategy_integration import (
            OrderFlowStrategyIntegration,
        )

        OrderFlowStrategyIntegration().evaluate(
            "invalid",  # type: ignore[arg-type]
            order_flow,
        )


def test_order_flow_reason_is_added_to_strategy_reasons() -> None:
    result = make_strategy_result()

    order_flow = make_order_flow_signal(
        bid_amount=10.0,
        ask_amount=5.0,
    )

    original_count = len(
        result.reasons
    )

    output = apply_order_flow_gate(
        result,
        order_flow,
    )

    assert len(output.reasons) == (
        original_count + 1
    )

    assert any(
        reason.startswith("OrderFlow:")
        for reason in output.reasons
    )


def test_blocked_order_flow_resets_levels_to_entry() -> None:
    result = make_strategy_result(
        signal="BUY",
        approved=True,
    )

    order_flow = make_order_flow_signal(
        bid_amount=1.0,
        ask_amount=100.0,
    )

    output = apply_order_flow_gate(
        result,
        order_flow,
    )

    assert output.signal == "HOLD"
    assert output.trade_approved is False
    assert output.stop_loss == pytest.approx(
        output.entry
    )
    assert output.take_profit == pytest.approx(
        output.entry
    )


def test_apply_gate_preserves_fusion_diagnostics() -> None:
    result = make_strategy_result(
        signal="BUY",
        approved=True,
    )

    original = replace(result)

    order_flow = make_order_flow_signal(
        bid_amount=1.0,
        ask_amount=100.0,
    )

    output = apply_order_flow_gate(
        result,
        order_flow,
    )

    assert output.ai_signal == original.ai_signal
    assert output.ai_confidence == original.ai_confidence
    assert output.ml_signal == original.ml_signal
    assert output.ml_probability == original.ml_probability
    assert output.fusion_signal == original.fusion_signal
    assert (
        output.combined_confidence
        == original.combined_confidence
    )


def test_order_flow_context_is_exposed() -> None:
    result = make_strategy_result()

    order_flow = make_order_flow_signal(
        bid_amount=5.0,
        ask_amount=20.0,
    )

    output = apply_order_flow_gate(
        result,
        order_flow,
    )

    assert output.order_flow_context == (
        order_flow.context
    )

    assert output.order_flow_pressure == pytest.approx(
        order_flow.pressure
    )