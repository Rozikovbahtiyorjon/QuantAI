from __future__ import annotations

import pandas as pd
import pytest

import src.paper_trading_runner as paper_trading_runner
from src.confidence_engine import ConfidenceEngine
from src.order_book_market_data import (
    OrderBookLevel,
    OrderBookSnapshot,
)
from src.order_flow_intelligence import (
    OrderFlowIntelligenceEngine,
)
from src.paper_trading_pipeline import (
    PaperTradingPipeline,
)
from src.paper_trading_runner import (
    PaperTradingRunner,
)
from src.strategy import (
    SignalResult,
    generate_signal_result,
)


def make_market_data(
    rows: int = 4,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [
                100.0 + index
                for index in range(rows)
            ],
            "high": [
                101.0 + index
                for index in range(rows)
            ],
            "low": [
                99.0 + index
                for index in range(rows)
            ],
            "close": [
                100.0 + index
                for index in range(rows)
            ],
            "atr": [
                1.0
                for _ in range(rows)
            ],
            "volume": [
                1000.0
                for _ in range(rows)
            ],
        }
    )


def make_confidence_result(
    decision: str = "BUY",
    score: float = 2.0,
):
    engine = ConfidenceEngine()

    engine.add_component(
        "trend",
        score,
    )

    engine.add_component(
        "momentum",
        score,
    )

    engine.add_component(
        "volume",
        score,
    )

    engine.add_component(
        "volatility",
        score,
    )

    engine.add_component(
        "structure",
        score,
    )

    result = engine.evaluate()

    assert result.decision == decision

    return result


def make_order_flow_snapshot(
    bid_amount: float,
    ask_amount: float,
    timestamp: int = 1,
) -> OrderBookSnapshot:
    return OrderBookSnapshot(
        symbol="BTC/USDT",
        timestamp=timestamp,
        bids=(
            OrderBookLevel(
                price=100.0,
                amount=bid_amount,
            ),
        ),
        asks=(
            OrderBookLevel(
                price=101.0,
                amount=ask_amount,
            ),
        ),
    )


def test_confidence_stage_produces_trade_direction() -> None:
    result = make_confidence_result(
        decision="BUY",
        score=2.0,
    )

    assert result.decision == "BUY"

    assert result.total_score == pytest.approx(
        2.0
    )

    assert result.confidence == pytest.approx(
        70.0
    )

    assert result.probability == pytest.approx(
        70.0
    )

    assert len(
        result.components
    ) == 5


def test_order_flow_stage_produces_directional_pressure() -> None:
    engine = OrderFlowIntelligenceEngine(
        pressure_threshold=0.15,
    )

    snapshot = make_order_flow_snapshot(
        bid_amount=20.0,
        ask_amount=2.0,
    )

    signal = engine.update(
        snapshot
    )

    assert signal.context == (
        OrderFlowIntelligenceEngine.CONTEXT_BID_PRESSURE
    )

    assert signal.pressure > 0.15

    assert signal.bid_volume == pytest.approx(
        20.0
    )

    assert signal.ask_volume == pytest.approx(
        2.0
    )

    assert signal.bid_notional == pytest.approx(
        2000.0
    )

    assert signal.ask_notional == pytest.approx(
        202.0
    )


def test_trade_boundary_executes_approved_strategy_signal() -> None:
    runner = PaperTradingRunner(
        enable_risk_controls=False,
        initial_balance=1000.0,
        commission=0.0004,
        quantity=1.0,
    )

    signal = SignalResult(
        signal="BUY",
        score=2.0,
        confidence=78.0,
        entry=100.0,
        stop_loss=98.0,
        take_profit=104.0,
        reasons=[
            "BUY approved.",
            "OrderFlow: confirmed.",
        ],
        ai_signal="BUY",
        ai_confidence=70.0,
        ml_signal="BUY",
        ml_probability=90.0,
        fusion_signal="BUY",
        combined_confidence=78.0,
        trade_approved=True,
        fusion_reason="ML confirms BUY",
        order_flow_signal="BUY",
        order_flow_enabled=True,
        order_flow_approved=True,
        order_flow_context="BID_PRESSURE",
        order_flow_score=0.8,
        order_flow_pressure=0.6,
        order_flow_reason="OrderFlow confirms BUY.",
    )

    step = runner.process_signal(
        signal
    )

    assert step.position_opened is True

    assert step.position_closed is False

    assert runner.has_position is True

    assert runner.engine.position is not None

    assert runner.engine.position.side == (
        "LONG"
    )

    assert runner.engine.position.entry_price == pytest.approx(
        100.0
    )

    assert runner.balance == pytest.approx(
        999.96
    )


def test_blocked_signal_never_reaches_trade_boundary() -> None:
    runner = PaperTradingRunner(
        enable_risk_controls=False,
        initial_balance=1000.0,
        commission=0.0004,
        quantity=1.0,
    )

    signal = SignalResult(
        signal="HOLD",
        score=0.0,
        confidence=55.0,
        entry=100.0,
        stop_loss=100.0,
        take_profit=100.0,
        trade_approved=False,
        order_flow_enabled=True,
        order_flow_approved=False,
        order_flow_context="ASK_PRESSURE",
        order_flow_score=0.2,
        order_flow_pressure=-0.6,
        order_flow_reason=(
            "OrderFlow conflicts with BUY."
        ),
    )

    step = runner.process_signal(
        signal
    )

    assert step.position_opened is False

    assert step.position_closed is False

    assert step.trade is None

    assert runner.has_position is False

    assert runner.balance == pytest.approx(
        1000.0
    )

    assert runner.realized_profit == pytest.approx(
        0.0
    )


def test_complete_virtual_trade_preserves_account_state() -> None:
    runner = PaperTradingRunner(
        enable_risk_controls=False,
        initial_balance=1000.0,
        commission=0.0004,
        quantity=1.0,
    )

    signal = SignalResult(
        signal="BUY",
        score=2.0,
        confidence=78.0,
        entry=100.0,
        stop_loss=98.0,
        take_profit=104.0,
        trade_approved=True,
        order_flow_enabled=True,
        order_flow_approved=True,
        order_flow_context="BID_PRESSURE",
        order_flow_score=0.8,
        order_flow_pressure=0.6,
    )

    runner.process_signal(
        signal
    )

    assert runner.balance == pytest.approx(
        999.96
    )

    trade = runner.close_position(
        price=102.0,
        signal=signal,
    ).trade

    assert trade is not None

    assert trade.side == "LONG"

    assert trade.entry_price == pytest.approx(
        100.0
    )

    assert trade.exit_price == pytest.approx(
        102.0
    )

    assert trade.gross_profit == pytest.approx(
        2.0
    )

    assert trade.fees == pytest.approx(
        0.0808
    )

    assert trade.net_profit == pytest.approx(
        1.9192
    )

    assert runner.has_position is False

    assert runner.balance == pytest.approx(
        1001.9192
    )

    assert runner.realized_profit == pytest.approx(
        1.9192
    )

    assert len(
        runner.engine.trade_history
    ) == 1


def test_pipeline_preserves_state_across_sequential_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    market_data = make_market_data(
        rows=3
    )

    calls = {
        "count": 0
    }

    def deterministic_signal(
        df: pd.DataFrame,
    ) -> SignalResult:
        calls["count"] += 1

        if calls["count"] == 1:
            return SignalResult(
                signal="BUY",
                entry=100.0,
                stop_loss=98.0,
                take_profit=104.0,
                confidence=78.0,
                trade_approved=True,
                order_flow_enabled=True,
                order_flow_approved=True,
                order_flow_context="BID_PRESSURE",
                order_flow_score=0.8,
                order_flow_pressure=0.6,
            )

        return SignalResult(
            signal="HOLD",
            entry=float(
                df["close"].iloc[-1]
            ),
            stop_loss=float(
                df["close"].iloc[-1]
            ),
            take_profit=float(
                df["close"].iloc[-1]
            ),
            trade_approved=False,
            order_flow_enabled=True,
            order_flow_approved=False,
            order_flow_context="BALANCED",
            order_flow_score=0.5,
            order_flow_pressure=0.0,
        )

    monkeypatch.setattr(
        paper_trading_runner,
        "generate_signal_result",
        deterministic_signal,
    )

    pipeline = PaperTradingPipeline(
        enable_risk_controls=False,
        initial_balance=1000.0,
        commission=0.0004,
        quantity=1.0,
    )

    result = pipeline.run(
        market_data
    )

    assert result.total_steps == 3

    assert result.opened_positions == 1

    assert result.closed_positions == 0

    assert pipeline.has_position is True

    assert pipeline.session.runner.engine.position is not None

    assert pipeline.session.runner.engine.position.side == (
        "LONG"
    )

    assert pipeline.balance == pytest.approx(
        999.96
    )

    assert pipeline.realized_profit == pytest.approx(
        0.0
    )

    pipeline.session.runner.close_position(
        price=102.0
    )

    assert pipeline.has_position is False

    assert pipeline.realized_profit == pytest.approx(
        1.9192
    )

    assert pipeline.balance == pytest.approx(
        1001.9192
    )

    final_result = pipeline.session.result

    assert final_result.opened_positions == 1

    assert final_result.closed_positions == 1

    assert final_result.final_balance == pytest.approx(
        1001.9192
    )

    assert final_result.realized_profit == pytest.approx(
        1.9192
    )


def test_pipeline_reset_restores_initial_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    market_data = make_market_data(
        rows=1
    )

    monkeypatch.setattr(
        paper_trading_runner,
        "generate_signal_result",
        lambda df: SignalResult(
            signal="BUY",
            entry=100.0,
            stop_loss=98.0,
            take_profit=104.0,
            confidence=78.0,
            trade_approved=True,
            order_flow_enabled=True,
            order_flow_approved=True,
            order_flow_context="BID_PRESSURE",
            order_flow_score=0.8,
            order_flow_pressure=0.6,
        ),
    )

    pipeline = PaperTradingPipeline(
        enable_risk_controls=False,
        initial_balance=1000.0,
        commission=0.0004,
        quantity=1.0,
    )

    pipeline.run(
        market_data
    )

    assert pipeline.has_position is True

    assert pipeline.balance == pytest.approx(
        999.96
    )

    pipeline.reset()

    assert pipeline.result is None

    assert pipeline.has_position is False

    assert pipeline.balance == pytest.approx(
        1000.0
    )

    assert pipeline.realized_profit == pytest.approx(
        0.0
    )

    assert pipeline.steps == []