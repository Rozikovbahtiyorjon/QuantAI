from __future__ import annotations

import pytest

from src.order_book_market_data import (
    OrderBookLevel,
    OrderBookSnapshot,
)
from src.order_flow_intelligence import (
    OrderFlowIntelligenceEngine,
    OrderFlowSignal,
)


def make_snapshot(
    timestamp: int = 1000,
    bids: tuple[OrderBookLevel, ...] | None = None,
    asks: tuple[OrderBookLevel, ...] | None = None,
) -> OrderBookSnapshot:
    return OrderBookSnapshot(
        symbol="BTC/USDT",
        timestamp=timestamp,
        bids=(
            (
                OrderBookLevel(100.0, 10.0),
                OrderBookLevel(99.0, 5.0),
            )
            if bids is None
            else bids
        ),
        asks=(
            (
                OrderBookLevel(101.0, 5.0),
                OrderBookLevel(102.0, 5.0),
            )
            if asks is None
            else asks
        ),
    )


def test_basic_signal_and_microprice() -> None:
    engine = OrderFlowIntelligenceEngine(
        depth=2
    )

    result = engine.update(
        make_snapshot()
    )

    assert isinstance(
        result,
        OrderFlowSignal,
    )

    assert result.bid_volume == pytest.approx(
        15.0
    )

    assert result.ask_volume == pytest.approx(
        10.0
    )

    assert result.bid_notional == pytest.approx(
        1495.0
    )

    assert result.ask_notional == pytest.approx(
        1015.0
    )

    assert result.volume_imbalance == pytest.approx(
        0.2
    )

    assert result.notional_imbalance == pytest.approx(
        480.0 / 2510.0
    )

    assert result.pressure > 0.0

    assert result.context == "BID_PRESSURE"

    assert result.microprice == pytest.approx(
        100.6666666667
    )

    assert result.microprice_delta is None


def test_depth_limits_levels() -> None:
    engine = OrderFlowIntelligenceEngine(
        depth=1
    )

    result = engine.update(
        make_snapshot()
    )

    assert result.bid_volume == pytest.approx(
        10.0
    )

    assert result.ask_volume == pytest.approx(
        5.0
    )

    assert result.bid_notional == pytest.approx(
        1000.0
    )

    assert result.ask_notional == pytest.approx(
        505.0
    )


def test_balanced_book() -> None:
    engine = OrderFlowIntelligenceEngine()

    result = engine.update(
        make_snapshot(
            bids=(
                OrderBookLevel(
                    100.0,
                    10.0,
                ),
            ),
            asks=(
                OrderBookLevel(
                    101.0,
                    10.0,
                ),
            ),
        )
    )

    assert result.volume_imbalance == pytest.approx(
        0.0
    )

    assert result.notional_imbalance == pytest.approx(
        -1.0 / 201.0
    )

    assert result.context == "BALANCED"


def test_ask_pressure() -> None:
    engine = OrderFlowIntelligenceEngine()

    result = engine.update(
        make_snapshot(
            bids=(
                OrderBookLevel(
                    100.0,
                    2.0,
                ),
            ),
            asks=(
                OrderBookLevel(
                    101.0,
                    20.0,
                ),
            ),
        )
    )

    assert result.pressure < -0.15
    assert result.context == "ASK_PRESSURE"


def test_no_liquidity_context() -> None:
    engine = OrderFlowIntelligenceEngine()

    result = engine.update(
        make_snapshot(
            bids=(),
            asks=(),
        )
    )

    assert result.context == "NO_LIQUIDITY"
    assert result.pressure == pytest.approx(0.0)
    assert result.microprice is None


def test_microprice_delta() -> None:
    engine = OrderFlowIntelligenceEngine()

    engine.update(
        make_snapshot()
    )

    result = engine.update(
        make_snapshot(
            timestamp=2000,
            bids=(
                OrderBookLevel(
                    100.0,
                    20.0,
                ),
            ),
            asks=(
                OrderBookLevel(
                    101.0,
                    5.0,
                ),
            ),
        )
    )

    assert result.microprice == pytest.approx(
        100.8
    )

    assert result.microprice_delta == pytest.approx(
        100.8
        - (
            100.0 * 5.0
            + 101.0 * 10.0
        ) / 15.0
    )


def test_liquidity_shares_sum_to_one() -> None:
    engine = OrderFlowIntelligenceEngine()

    result = engine.update(
        make_snapshot()
    )

    assert (
        result.bid_liquidity_share
        + result.ask_liquidity_share
    ) == pytest.approx(1.0)


def test_previous_snapshot_is_tracked() -> None:
    engine = OrderFlowIntelligenceEngine()

    snapshot = make_snapshot()

    engine.update(
        snapshot
    )

    assert engine.previous == snapshot
    assert engine.previous_microprice is not None


def test_reset_clears_state() -> None:
    engine = OrderFlowIntelligenceEngine()

    engine.update(
        make_snapshot()
    )

    engine.reset()

    assert engine.previous is None
    assert engine.previous_microprice is None


def test_symbol_change_is_rejected() -> None:
    engine = OrderFlowIntelligenceEngine()

    engine.update(
        make_snapshot()
    )

    invalid = OrderBookSnapshot(
        symbol="ETH/USDT",
        timestamp=2000,
        bids=(
            OrderBookLevel(
                100.0,
                1.0,
            ),
        ),
        asks=(
            OrderBookLevel(
                101.0,
                1.0,
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match="symbol must match",
    ):
        engine.update(
            invalid
        )


def test_timestamp_regression_is_rejected() -> None:
    engine = OrderFlowIntelligenceEngine()

    engine.update(
        make_snapshot(
            timestamp=2000
        )
    )

    with pytest.raises(
        ValueError,
        match="timestamp must be greater",
    ):
        engine.update(
            make_snapshot(
                timestamp=2000
            )
        )


def test_constructor_validation() -> None:
    with pytest.raises(TypeError):
        OrderFlowIntelligenceEngine(
            depth=True
        )

    with pytest.raises(ValueError):
        OrderFlowIntelligenceEngine(
            depth=0
        )

    with pytest.raises(TypeError):
        OrderFlowIntelligenceEngine(
            pressure_threshold="0.2",  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError):
        OrderFlowIntelligenceEngine(
            pressure_threshold=0.0
        )

    with pytest.raises(ValueError):
        OrderFlowIntelligenceEngine(
            pressure_threshold=1.1
        )


def test_invalid_snapshot_type() -> None:
    engine = OrderFlowIntelligenceEngine()

    with pytest.raises(
        TypeError,
        match="OrderBookSnapshot",
    ):
        engine.update(
            "invalid"  # type: ignore[arg-type]
        )


def test_invalid_snapshot_level_is_rejected() -> None:
    engine = OrderFlowIntelligenceEngine()

    invalid = OrderBookSnapshot(
        symbol="BTC/USDT",
        timestamp=1000,
        bids=(
            OrderBookLevel(
                100.0,
                -1.0,
            ),
        ),
        asks=(
            OrderBookLevel(
                101.0,
                1.0,
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match="amount",
    ):
        engine.update(
            invalid
        )