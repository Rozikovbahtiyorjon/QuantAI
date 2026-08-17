from __future__ import annotations

import pytest

from src.order_book_market_data import (
    OrderBookLevel,
    OrderBookSnapshot,
)
from src.order_flow_intelligence import (
    OrderFlowIntelligenceEngine,
)
from src.order_flow_unified_integration import (
    OrderFlowIntegrationResult,
    OrderFlowUnifiedMarketIntegration,
)
from src.unified_market_intelligence import (
    UnifiedMarketIntelligenceLayer,
)


def make_snapshot(
    timestamp: int = 1000,
    bid_amount: float = 10.0,
    ask_amount: float = 5.0,
) -> OrderBookSnapshot:
    return OrderBookSnapshot(
        symbol="BTC/USDT",
        timestamp=timestamp,
        bids=(
            OrderBookLevel(100.0, bid_amount),
        ),
        asks=(
            OrderBookLevel(101.0, ask_amount),
        ),
    )


def make_signal(
    bid_amount: float = 10.0,
    ask_amount: float = 5.0,
):
    engine = OrderFlowIntelligenceEngine()

    return engine.update(
        make_snapshot(
            bid_amount=bid_amount,
            ask_amount=ask_amount,
        )
    )


def test_positive_pressure_maps_above_neutral() -> None:
    signal = make_signal(
        bid_amount=10.0,
        ask_amount=5.0,
    )

    adapter = OrderFlowUnifiedMarketIntegration()

    score = adapter.order_flow_score(signal)

    assert signal.pressure > 0.0
    assert score > 0.5
    assert 0.0 <= score <= 1.0


def test_negative_pressure_maps_below_neutral() -> None:
    signal = make_signal(
        bid_amount=5.0,
        ask_amount=10.0,
    )

    adapter = OrderFlowUnifiedMarketIntegration()

    score = adapter.order_flow_score(signal)

    assert signal.pressure < 0.0
    assert score < 0.5
    assert 0.0 <= score <= 1.0


def test_balanced_pressure_maps_to_neutral() -> None:
    signal = make_signal(
        bid_amount=10.0,
        ask_amount=10.0,
    )

    adapter = OrderFlowUnifiedMarketIntegration()

    assert signal.context == "BALANCED"

    assert signal.pressure == pytest.approx(
        -0.0024875621890547263
    )

    assert adapter.order_flow_score(signal) == pytest.approx(
        0.5
    )


def test_evaluate_injects_order_flow_component() -> None:
    signal = make_signal()

    adapter = OrderFlowUnifiedMarketIntegration()

    result = adapter.evaluate(
        signal,
        components={"ml": 0.8},
        regime="TREND_UP",
    )

    assert isinstance(
        result,
        OrderFlowIntegrationResult,
    )

    assert result.unified_signal.components_used == 2

    assert result.order_flow_score == pytest.approx(
        0.66555924695
    )


def test_order_flow_is_not_lost_when_other_components_are_present() -> None:
    signal = make_signal()

    adapter = OrderFlowUnifiedMarketIntegration()

    result = adapter.evaluate(
        signal,
        components={
            "technical": 0.6,
            "ml": 0.8,
            "derivatives": 0.7,
        },
        regime="TREND_UP",
    )

    assert result.unified_signal.components_used == 4
    assert result.unified_signal.components_total == 10
    assert result.unified_signal.score > 0.6


def test_event_risk_is_forwarded() -> None:
    signal = make_signal()

    adapter = OrderFlowUnifiedMarketIntegration()

    result = adapter.evaluate(
        signal,
        components={"ml": 0.7},
        regime="RANGE",
        event_risk=0.9,
    )

    assert result.unified_signal.risk_level == "EXTREME"


def test_custom_unified_layer_is_used() -> None:
    layer = UnifiedMarketIntelligenceLayer(
        weights={
            "order_flow": 3.0,
            "ml": 1.0,
        }
    )

    adapter = OrderFlowUnifiedMarketIntegration(
        unified_layer=layer
    )

    signal = make_signal()

    result = adapter.evaluate(
        signal,
        components={"ml": 0.0},
        regime="RANGE",
    )

    assert result.unified_signal.components_used == 2

    assert result.unified_signal.score == pytest.approx(
        (
            result.order_flow_score * 3.0
        ) / 4.0
    )


def test_invalid_signal_is_rejected() -> None:
    adapter = OrderFlowUnifiedMarketIntegration()

    with pytest.raises(
        TypeError,
        match="OrderFlowSignal",
    ):
        adapter.order_flow_score(
            "invalid"  # type: ignore[arg-type]
        )


def test_invalid_components_are_rejected() -> None:
    adapter = OrderFlowUnifiedMarketIntegration()

    signal = make_signal()

    with pytest.raises(
        TypeError,
        match="mapping",
    ):
        adapter.evaluate(
            signal,
            components=[],  # type: ignore[arg-type]
        )


def test_order_flow_score_is_bounded() -> None:
    signal = make_signal()

    adapter = OrderFlowUnifiedMarketIntegration()

    assert 0.0 <= adapter.order_flow_score(signal) <= 1.0