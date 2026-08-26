from __future__ import annotations

import math

import pandas as pd
import pytest

from src.paper_trading_engine import PaperTradingEngine
from src.paper_trading_pipeline import PaperTradingPipeline
from src.paper_trading_runner import PaperTradingRunner
from src.paper_trading_session import PaperTradingSession
from src.strategy import SignalResult


def make_signal(
    signal: str = "HOLD",
    entry: float = 100.0,
    approved: bool = True,
) -> SignalResult:
    return SignalResult(
        signal=signal,
        score=4.0 if signal != "HOLD" else 0.0,
        confidence=80.0 if signal != "HOLD" else 0.0,
        entry=entry,
        stop_loss=(
            98.0
            if signal == "BUY"
            else 102.0
            if signal == "SELL"
            else entry
        ),
        take_profit=(
            104.0
            if signal == "BUY"
            else 96.0
            if signal == "SELL"
            else entry
        ),
        trade_approved=approved,
        fusion_signal=signal,
        order_flow_signal=(
            signal
            if signal != "HOLD"
            else "HOLD"
        ),
        order_flow_enabled=(
            signal != "HOLD"
        ),
        order_flow_approved=(
            approved
            and signal != "HOLD"
        ),
        order_flow_context="BALANCED",
        order_flow_score=0.5,
        order_flow_pressure=0.0,
        order_flow_reason="",
    )


def make_ohlcv(
    rows: int = 4,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [
                100.0 + i
                for i in range(rows)
            ],
            "high": [
                101.0 + i
                for i in range(rows)
            ],
            "low": [
                99.0 + i
                for i in range(rows)
            ],
            "close": [
                100.5 + i
                for i in range(rows)
            ],
            "volume": [
                1000.0
                for _ in range(rows)
            ],
        }
    )


def test_paper_engine_long_accounting_is_deterministic() -> None:
    engine = PaperTradingEngine(
        initial_balance=1000.0,
        commission=0.0004,
    )

    position = engine.open_position(
        side="LONG",
        price=100.0,
        quantity=1.0,
    )

    assert position.side == "LONG"
    assert engine.has_position is True
    assert engine.balance == pytest.approx(
        999.96
    )

    trade = engine.close_position(
        102.0
    )

    expected_exit_fee = (
        102.0 * 0.0004
    )

    expected_net = (
        2.0
        - 0.04
        - expected_exit_fee
    )

    assert trade.side == "LONG"

    assert trade.gross_profit == pytest.approx(
        2.0
    )

    assert trade.fees == pytest.approx(
        0.04 + expected_exit_fee
    )

    assert trade.net_profit == pytest.approx(
        expected_net
    )

    assert engine.has_position is False

    assert engine.balance == pytest.approx(
        1000.0 + expected_net
    )

    assert engine.realized_profit == pytest.approx(
        expected_net
    )

    assert len(
        engine.trade_history
    ) == 1


def test_paper_engine_short_accounting_is_deterministic() -> None:
    engine = PaperTradingEngine(
        initial_balance=1000.0,
        commission=0.0004,
    )

    engine.open_position(
        side="SHORT",
        price=100.0,
        quantity=1.0,
    )

    trade = engine.close_position(
        98.0
    )

    expected_exit_fee = (
        98.0 * 0.0004
    )

    expected_net = (
        2.0
        - 0.04
        - expected_exit_fee
    )

    assert trade.side == "SHORT"

    assert trade.gross_profit == pytest.approx(
        2.0
    )

    assert trade.net_profit == pytest.approx(
        expected_net
    )

    assert engine.balance == pytest.approx(
        1000.0 + expected_net
    )

    assert engine.realized_profit == pytest.approx(
        expected_net
    )


def test_paper_engine_rejects_invalid_position_transitions() -> None:
    engine = PaperTradingEngine()

    with pytest.raises(
        RuntimeError
    ):
        engine.close_position(
            100.0
        )

    engine.open_position(
        side="LONG",
        price=100.0,
        quantity=1.0,
    )

    with pytest.raises(
        RuntimeError
    ):
        engine.open_position(
            side="SHORT",
            price=100.0,
            quantity=1.0,
        )


def test_runner_hold_does_not_change_account_state() -> None:
    runner = PaperTradingRunner(
        initial_balance=1000.0,
        commission=0.0004,
        quantity=1.0,
    )

    result = runner.process_signal(
        make_signal(
            signal="HOLD",
            approved=False,
        )
    )

    assert result.position_opened is False
    assert result.position_closed is False
    assert result.trade is None

    assert runner.balance == pytest.approx(
        1000.0
    )

    assert runner.has_position is False

    assert runner.realized_profit == pytest.approx(
        0.0
    )


def test_runner_buy_opens_long_once() -> None:
    runner = PaperTradingRunner(
        initial_balance=1000.0,
        commission=0.0004,
        quantity=1.0,
    )

    first = runner.process_signal(
        make_signal(
            signal="BUY",
            entry=100.0,
            approved=True,
        )
    )

    second = runner.process_signal(
        make_signal(
            signal="BUY",
            entry=101.0,
            approved=True,
        )
    )

    assert first.position_opened is True
    assert first.position_closed is False

    assert second.position_opened is False

    assert runner.has_position is True

    assert runner.engine.position is not None

    assert runner.engine.position.side == "LONG"

    assert runner.engine.position.entry_price == pytest.approx(
        100.0
    )


def test_runner_sell_opens_short_once() -> None:
    runner = PaperTradingRunner(
        initial_balance=1000.0,
        commission=0.0004,
        quantity=1.0,
    )

    result = runner.process_signal(
        make_signal(
            signal="SELL",
            entry=100.0,
            approved=True,
        )
    )

    assert result.position_opened is True

    assert runner.has_position is True

    assert runner.engine.position is not None

    assert runner.engine.position.side == "SHORT"

    assert runner.engine.position.entry_price == pytest.approx(
        100.0
    )


def test_runner_close_position_records_trade() -> None:
    runner = PaperTradingRunner(
        initial_balance=1000.0,
        commission=0.0004,
        quantity=1.0,
    )

    runner.process_signal(
        make_signal(
            signal="BUY",
            entry=100.0,
            approved=True,
        )
    )

    result = runner.close_position(
        price=102.0,
        signal=make_signal(
            signal="HOLD",
            entry=102.0,
            approved=False,
        ),
    )

    assert result.position_closed is True

    assert result.trade is not None

    assert result.trade.side == "LONG"

    assert runner.has_position is False

    assert runner.realized_profit == pytest.approx(
        result.trade.net_profit
    )


def test_session_processes_every_market_row(
) -> None:
    df = make_ohlcv(
        5
    )

    session = PaperTradingSession(
        initial_balance=1000.0,
        commission=0.0004,
        quantity=1.0,
    )

    module = __import__(
        "src.paper_trading_session",
        fromlist=[
            "generate_signal_result"
        ],
    )

    original = (
        module.generate_signal_result
    )

    try:
        module.generate_signal_result = (
            lambda window: make_signal(
                signal="HOLD",
                entry=float(
                    window["close"].iloc[-1]
                ),
                approved=False,
            )
        )

        result = session.run(
            df
        )

    finally:
        module.generate_signal_result = (
            original
        )

    assert result.total_steps == 5

    assert len(
        result.steps
    ) == 5

    assert result.opened_positions == 0

    assert result.closed_positions == 0

    assert result.initial_balance == pytest.approx(
        1000.0
    )

    assert result.final_balance == pytest.approx(
        1000.0
    )

    assert result.realized_profit == pytest.approx(
        0.0
    )


def test_pipeline_aggregates_session_state_and_reset(
) -> None:
    df = make_ohlcv(
        3
    )

    pipeline = PaperTradingPipeline(
        initial_balance=1000.0,
        commission=0.0004,
        quantity=1.0,
    )

    module = __import__(
        "src.paper_trading_session",
        fromlist=[
            "generate_signal_result"
        ],
    )

    original = (
        module.generate_signal_result
    )

    try:
        module.generate_signal_result = (
            lambda window: make_signal(
                signal="HOLD",
                entry=float(
                    window["close"].iloc[-1]
                ),
                approved=False,
            )
        )

        result = pipeline.run(
            df
        )

    finally:
        module.generate_signal_result = (
            original
        )

    assert result.total_steps == 3

    assert result.opened_positions == 0

    assert result.closed_positions == 0

    assert result.initial_balance == pytest.approx(
        1000.0
    )

    assert result.final_balance == pytest.approx(
        1000.0
    )

    assert result.realized_profit == pytest.approx(
        0.0
    )

    assert result.return_percent == pytest.approx(
        0.0
    )

    assert pipeline.result is result

    assert pipeline.balance == pytest.approx(
        1000.0
    )

    assert pipeline.has_position is False

    pipeline.reset()

    assert pipeline.result is None

    assert pipeline.balance == pytest.approx(
        1000.0
    )

    assert pipeline.has_position is False

    assert pipeline.realized_profit == pytest.approx(
        0.0
    )

    assert pipeline.steps == []


def test_strategy_result_contains_full_paper_execution_contract(
) -> None:
    signal = make_signal(
        signal="BUY",
        entry=100.0,
        approved=True,
    )

    required = (
        "signal",
        "trade_approved",
        "entry",
        "stop_loss",
        "take_profit",
        "order_flow_enabled",
        "order_flow_approved",
        "order_flow_context",
        "order_flow_score",
        "order_flow_pressure",
    )

    for name in required:
        assert hasattr(
            signal,
            name
        )

    assert signal.signal == "BUY"

    assert signal.trade_approved is True

    assert signal.entry > 0

    assert signal.stop_loss > 0

    assert signal.take_profit > 0

    assert math.isfinite(
        signal.order_flow_score
    )

    assert math.isfinite(
        signal.order_flow_pressure
    )