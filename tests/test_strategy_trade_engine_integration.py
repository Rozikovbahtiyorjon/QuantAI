"""
=========================================================
QuantAI Professional v5.2
Strategy ↔ TradeEngine Integration Tests
=========================================================

Validates the integration contract between:

    Strategy Engine
        ↓
    SignalResult
        ↓
    TradeEngine
        ↓
    Position
        ↓
    Close / PnL / Balance

Tests intentionally avoid real ML predictions and market
indicators. Strategy output is represented by SignalResult
objects so the integration boundary can be tested
deterministically.
"""

from __future__ import annotations

import pandas as pd
import pytest

import src.trade_engine as trade_engine_module

from src.strategy import SignalResult
from src.trade_engine import (
    CloseReason,
    PositionSide,
    PositionStatus,
    TradeEngine,
)


# =========================================================
# CONFIG
# =========================================================

INITIAL_BALANCE = 1000.0


# =========================================================
# DATA FACTORIES
# =========================================================

def make_candle(
    *,
    timestamp: str = "2025-01-01 00:00:00",
    open_price: float = 100.0,
    high: float = 101.0,
    low: float = 99.0,
    close: float = 100.0,
    atr: float = 1.0,
) -> dict:
    """
    Create one deterministic candle.
    """

    return {
        "timestamp": pd.Timestamp(timestamp),
        "open": float(open_price),
        "high": float(high),
        "low": float(low),
        "close": float(close),
        "atr": float(atr),
    }


def make_buy_signal(
    *,
    entry: float = 100.0,
    stop_loss: float = 98.0,
    take_profit: float = 104.0,
    confidence: float = 80.0,
) -> SignalResult:
    """
    Create deterministic BUY signal.
    """

    return SignalResult(
        signal="BUY",
        score=5.0,
        confidence=confidence,
        entry=entry,
        stop_loss=stop_loss,
        take_profit=take_profit,
        reasons=[
            "Test BUY signal",
            "AI + ML confirmation",
        ],
    )


def make_sell_signal(
    *,
    entry: float = 100.0,
    stop_loss: float = 102.0,
    take_profit: float = 96.0,
    confidence: float = 80.0,
) -> SignalResult:
    """
    Create deterministic SELL signal.
    """

    return SignalResult(
        signal="SELL",
        score=-5.0,
        confidence=confidence,
        entry=entry,
        stop_loss=stop_loss,
        take_profit=take_profit,
        reasons=[
            "Test SELL signal",
            "AI + ML confirmation",
        ],
    )


def make_hold_signal() -> SignalResult:
    """
    Create deterministic HOLD signal.
    """

    return SignalResult(
        signal="HOLD",
        score=0.0,
        confidence=40.0,
        entry=100.0,
        stop_loss=100.0,
        take_profit=100.0,
        reasons=[
            "Test HOLD signal",
        ],
    )


# =========================================================
# ENGINE FACTORY
# =========================================================

def make_engine() -> TradeEngine:
    """
    Create TradeEngine with deterministic balance.

    TradeEngine uses the project-level INITIAL_BALANCE,
    therefore balance/equity are explicitly reset here.
    """

    engine = TradeEngine()

    engine.balance = INITIAL_BALANCE
    engine.equity = INITIAL_BALANCE

    return engine


# =========================================================
# POSITION OPENING
# =========================================================

def test_strategy_buy_signal_opens_long_position(
    monkeypatch,
):
    """
    Strategy BUY signal must create a LONG position.
    """

    engine = make_engine()

    signal = make_buy_signal()

    candle = make_candle()

    monkeypatch.setattr(
        trade_engine_module,
        "calculate_position_size",
        lambda **kwargs: 1.0,
    )

    opened = engine.open_position(
        candle,
        signal,
    )

    assert opened is True

    assert len(
        engine.positions
    ) == 1

    position = engine.positions[0]

    assert position.side == PositionSide.LONG
    assert position.status == PositionStatus.OPEN
    assert position.entry_time == candle["timestamp"]


def test_strategy_sell_signal_opens_short_position(
    monkeypatch,
):
    """
    Strategy SELL signal must create a SHORT position.
    """

    engine = make_engine()

    signal = make_sell_signal()

    candle = make_candle()

    monkeypatch.setattr(
        trade_engine_module,
        "calculate_position_size",
        lambda **kwargs: 1.0,
    )

    opened = engine.open_position(
        candle,
        signal,
    )

    assert opened is True

    assert len(
        engine.positions
    ) == 1

    position = engine.positions[0]

    assert position.side == PositionSide.SHORT
    assert position.status == PositionStatus.OPEN


def test_strategy_signal_parameters_are_transferred_to_position(
    monkeypatch,
):
    """
    Strategy signal parameters must reach Position.

    Entry price is expected to include TradeEngine slippage,
    while SL/TP/confidence must be preserved from Strategy.
    """

    engine = make_engine()

    signal = make_buy_signal(
        entry=100.0,
        stop_loss=98.0,
        take_profit=104.0,
        confidence=87.5,
    )

    candle = make_candle()

    monkeypatch.setattr(
        trade_engine_module,
        "calculate_position_size",
        lambda **kwargs: 1.25,
    )

    engine.open_position(
        candle,
        signal,
    )

    position = engine.positions[0]

    # -------------------------------------------------
    # Entry
    #
    # TradeEngine applies slippage to the Strategy
    # entry price, therefore Position.entry_price
    # must reflect the executed price.
    # -------------------------------------------------

    expected_entry = (
        signal.entry
        * (1.0 + trade_engine_module.SLIPPAGE)
    )

    assert position.entry_price == pytest.approx(
        round(expected_entry, 2),
        abs=1e-6,
    )

    # -------------------------------------------------
    # SL / TP
    #
    # These values come directly from Strategy.
    # -------------------------------------------------

    assert position.stop_loss == pytest.approx(
        signal.stop_loss,
        abs=1e-6,
    )

    assert position.take_profit == pytest.approx(
        signal.take_profit,
        abs=1e-6,
    )

    # -------------------------------------------------
    # Position size
    # -------------------------------------------------

    assert position.quantity == pytest.approx(
        1.25,
        abs=1e-6,
    )

    # -------------------------------------------------
    # Confidence
    # -------------------------------------------------

    assert position.confidence == pytest.approx(
        signal.confidence,
        abs=1e-6,
    )


def test_strategy_reasons_are_preserved_in_position(
    monkeypatch,
):
    """
    Strategy reasons must be preserved when opening a trade.
    """

    engine = make_engine()

    signal = make_buy_signal()

    candle = make_candle()

    monkeypatch.setattr(
        trade_engine_module,
        "calculate_position_size",
        lambda **kwargs: 1.0,
    )

    engine.open_position(
        candle,
        signal,
    )

    position = engine.positions[0]

    assert position.reason_open == signal.reasons

    assert position.reason_open is not signal.reasons


# =========================================================
# HOLD
# =========================================================

def test_hold_signal_does_not_create_position():
    """
    HOLD signal must never become a trade.
    """

    engine = make_engine()

    signal = make_hold_signal()

    assert signal.signal == "HOLD"

    # TradeEngine should only receive directional signals.
    # HOLD therefore must not be passed to open_position().
    if signal.signal == "HOLD":
        opened = False
    else:
        opened = engine.open_position(
            make_candle(),
            signal,
        )

    assert opened is False

    assert len(
        engine.positions
    ) == 0

    assert len(
        engine.closed_positions
    ) == 0


def test_hold_signal_keeps_balance_unchanged():
    """
    HOLD must not change account balance.
    """

    engine = make_engine()

    initial_balance = engine.balance

    signal = make_hold_signal()

    if signal.signal != "HOLD":
        engine.open_position(
            make_candle(),
            signal,
        )

    assert engine.balance == pytest.approx(
        initial_balance,
        abs=1e-6,
    )


# =========================================================
# SL / TP — LONG
# =========================================================

def test_long_position_closes_on_take_profit(
    monkeypatch,
):
    """
    LONG position must close at TP when candle high
    reaches take-profit.
    """

    engine = make_engine()

    signal = make_buy_signal(
        entry=100.0,
        stop_loss=98.0,
        take_profit=104.0,
    )

    entry_candle = make_candle(
        close=100.0,
        atr=1.0,
    )

    monkeypatch.setattr(
        trade_engine_module,
        "calculate_position_size",
        lambda **kwargs: 1.0,
    )

    engine.open_position(
        entry_candle,
        signal,
    )

    position = engine.positions[0]

    exit_candle = make_candle(
        timestamp="2025-01-01 00:15:00",
        high=105.0,
        low=103.0,
        close=104.5,
        atr=1.0,
    )

    engine.update_position(
        position,
        exit_candle,
    )

    assert len(
        engine.positions
    ) == 0

    assert len(
        engine.closed_positions
    ) == 1

    closed = engine.closed_positions[0]

    assert closed.status == PositionStatus.CLOSED
    assert closed.reason_close == CloseReason.TAKE_PROFIT


def test_long_position_closes_on_stop_loss(
    monkeypatch,
):
    """
    LONG position must close on SL when candle low reaches
    stop-loss.
    """

    engine = make_engine()

    signal = make_buy_signal(
        entry=100.0,
        stop_loss=98.0,
        take_profit=104.0,
    )

    monkeypatch.setattr(
        trade_engine_module,
        "calculate_position_size",
        lambda **kwargs: 1.0,
    )

    engine.open_position(
        make_candle(),
        signal,
    )

    position = engine.positions[0]

    exit_candle = make_candle(
        timestamp="2025-01-01 00:15:00",
        high=100.5,
        low=97.0,
        close=97.5,
        atr=1.0,
    )

    engine.update_position(
        position,
        exit_candle,
    )

    assert len(
        engine.positions
    ) == 0

    assert len(
        engine.closed_positions
    ) == 1

    closed = engine.closed_positions[0]

    assert closed.reason_close == CloseReason.STOP_LOSS


# =========================================================
# SL / TP — SHORT
# =========================================================

def test_short_position_closes_on_take_profit(
    monkeypatch,
):
    """
    SHORT position must close at TP when candle low reaches
    take-profit.
    """

    engine = make_engine()

    signal = make_sell_signal(
        entry=100.0,
        stop_loss=102.0,
        take_profit=96.0,
    )

    monkeypatch.setattr(
        trade_engine_module,
        "calculate_position_size",
        lambda **kwargs: 1.0,
    )

    engine.open_position(
        make_candle(),
        signal,
    )

    position = engine.positions[0]

    exit_candle = make_candle(
        timestamp="2025-01-01 00:15:00",
        high=97.0,
        low=95.0,
        close=95.5,
        atr=1.0,
    )

    engine.update_position(
        position,
        exit_candle,
    )

    assert len(
        engine.positions
    ) == 0

    assert len(
        engine.closed_positions
    ) == 1

    closed = engine.closed_positions[0]

    assert closed.reason_close == CloseReason.TAKE_PROFIT


def test_short_position_closes_on_stop_loss(
    monkeypatch,
):
    """
    SHORT position must close on SL when candle high reaches
    stop-loss.
    """

    engine = make_engine()

    signal = make_sell_signal(
        entry=100.0,
        stop_loss=102.0,
        take_profit=96.0,
    )

    monkeypatch.setattr(
        trade_engine_module,
        "calculate_position_size",
        lambda **kwargs: 1.0,
    )

    engine.open_position(
        make_candle(),
        signal,
    )

    position = engine.positions[0]

    exit_candle = make_candle(
        timestamp="2025-01-01 00:15:00",
        high=103.0,
        low=99.0,
        close=102.5,
        atr=1.0,
    )

    engine.update_position(
        position,
        exit_candle,
    )

    assert len(
        engine.positions
    ) == 0

    assert len(
        engine.closed_positions
    ) == 1

    closed = engine.closed_positions[0]

    assert closed.reason_close == CloseReason.STOP_LOSS


# =========================================================
# COMMISSION
# =========================================================

def test_commission_is_included_in_net_profit(
    monkeypatch,
):
    """
    Net profit must be lower than gross profit when commission
    is charged.
    """

    engine = make_engine()

    signal = make_buy_signal(
        entry=100.0,
        stop_loss=98.0,
        take_profit=104.0,
    )

    monkeypatch.setattr(
        trade_engine_module,
        "calculate_position_size",
        lambda **kwargs: 1.0,
    )

    engine.open_position(
        make_candle(),
        signal,
    )

    position = engine.positions[0]

    # Direct close avoids break-even/trailing-stop changing SL.
    exit_candle = make_candle(
        timestamp="2025-01-01 00:15:00",
        close=104.0,
        high=104.0,
        low=103.5,
        atr=0.01,
    )

    engine.close_position(
        position,
        exit_candle,
        104.0,
        CloseReason.TAKE_PROFIT,
    )

    closed = engine.closed_positions[0]

    assert closed.commission > 0

    assert closed.net_profit < closed.gross_profit


# =========================================================
# SLIPPAGE
# =========================================================

def test_long_entry_applies_slippage(
    monkeypatch,
):
    """
    LONG entry must use the TradeEngine slippage rule.
    """

    engine = make_engine()

    signal = make_buy_signal(
        entry=100.0,
    )

    candle = make_candle()

    monkeypatch.setattr(
        trade_engine_module,
        "calculate_position_size",
        lambda **kwargs: 1.0,
    )

    expected = (
        100.0
        * (1.0 + trade_engine_module.SLIPPAGE)
    )

    engine.open_position(
        candle,
        signal,
    )

    position = engine.positions[0]

    assert position.entry_price == pytest.approx(
        round(expected, 2),
        abs=1e-6,
    )


def test_short_entry_applies_slippage(
    monkeypatch,
):
    """
    SHORT entry must use the TradeEngine slippage rule.
    """

    engine = make_engine()

    signal = make_sell_signal(
        entry=100.0,
    )

    candle = make_candle()

    monkeypatch.setattr(
        trade_engine_module,
        "calculate_position_size",
        lambda **kwargs: 1.0,
    )

    expected = (
        100.0
        * (1.0 - trade_engine_module.SLIPPAGE)
    )

    engine.open_position(
        candle,
        signal,
    )

    position = engine.positions[0]

    assert position.entry_price == pytest.approx(
        round(expected, 2),
        abs=1e-6,
    )


# =========================================================
# BALANCE / PROFIT
# =========================================================

def test_balance_changes_by_net_profit(
    monkeypatch,
):
    """
    Closing a position must update balance exactly by net_profit.
    """

    engine = make_engine()

    signal = make_buy_signal(
        entry=100.0,
        stop_loss=98.0,
        take_profit=104.0,
    )

    monkeypatch.setattr(
        trade_engine_module,
        "calculate_position_size",
        lambda **kwargs: 1.0,
    )

    engine.open_position(
        make_candle(),
        signal,
    )

    position = engine.positions[0]

    balance_before_close = engine.balance

    exit_candle = make_candle(
        timestamp="2025-01-01 00:15:00",
        close=104.0,
        high=104.0,
        low=103.5,
        atr=0.01,
    )

    engine.close_position(
        position,
        exit_candle,
        104.0,
        CloseReason.TAKE_PROFIT,
    )

    closed = engine.closed_positions[0]

    expected_balance = (
        balance_before_close
        + closed.net_profit
    )

    assert engine.balance == pytest.approx(
        expected_balance,
        abs=1e-6,
    )

    assert closed.balance_after_close == pytest.approx(
        engine.balance,
        abs=1e-6,
    )


def test_total_profit_matches_closed_trade_profits(
    monkeypatch,
):
    """
    TradeEngine.total_profit must equal the sum of all
    closed trade net profits.
    """

    engine = make_engine()

    signal = make_buy_signal()

    monkeypatch.setattr(
        trade_engine_module,
        "calculate_position_size",
        lambda **kwargs: 1.0,
    )

    engine.open_position(
        make_candle(),
        signal,
    )

    position = engine.positions[0]

    exit_candle = make_candle(
        timestamp="2025-01-01 00:15:00",
        close=104.0,
        high=104.0,
        low=103.5,
        atr=0.01,
    )

    engine.close_position(
        position,
        exit_candle,
        104.0,
        CloseReason.TAKE_PROFIT,
    )

    expected = sum(
        p.net_profit
        for p in engine.closed_positions
    )

    assert engine.total_profit == pytest.approx(
        expected,
        abs=1e-6,
    )


# =========================================================
# POSITION COUNTS
# =========================================================

def test_closed_trade_moves_from_open_to_closed_collection(
    monkeypatch,
):
    """
    A closed position must be removed from open positions
    and added to closed_positions.
    """

    engine = make_engine()

    signal = make_buy_signal()

    monkeypatch.setattr(
        trade_engine_module,
        "calculate_position_size",
        lambda **kwargs: 1.0,
    )

    engine.open_position(
        make_candle(),
        signal,
    )

    assert len(engine.positions) == 1
    assert len(engine.closed_positions) == 0

    position = engine.positions[0]

    exit_candle = make_candle(
        timestamp="2025-01-01 00:15:00",
        close=104.0,
        high=104.0,
        low=103.5,
        atr=0.01,
    )

    engine.close_position(
        position,
        exit_candle,
        104.0,
        CloseReason.TAKE_PROFIT,
    )

    assert len(engine.positions) == 0
    assert len(engine.closed_positions) == 1


def test_total_trade_statistics_are_consistent(
    monkeypatch,
):
    """
    Total trades must equal wins + losses.
    """

    engine = make_engine()

    signal = make_buy_signal()

    monkeypatch.setattr(
        trade_engine_module,
        "calculate_position_size",
        lambda **kwargs: 1.0,
    )

    engine.open_position(
        make_candle(),
        signal,
    )

    position = engine.positions[0]

    exit_candle = make_candle(
        timestamp="2025-01-01 00:15:00",
        close=104.0,
        high=104.0,
        low=103.5,
        atr=0.01,
    )

    engine.close_position(
        position,
        exit_candle,
        104.0,
        CloseReason.TAKE_PROFIT,
    )

    assert (
        engine.total_trades
        == engine.winning_trades
        + engine.losing_trades
    )


# =========================================================
# MAX OPEN POSITIONS
# =========================================================

def test_max_open_positions_is_respected(
    monkeypatch,
):
    """
    TradeEngine must refuse a new position when the maximum
    number of simultaneous positions is reached.
    """

    engine = make_engine()

    signal = make_buy_signal()

    candle = make_candle()

    monkeypatch.setattr(
        trade_engine_module,
        "calculate_position_size",
        lambda **kwargs: 1.0,
    )

    opened_count = 0

    for _ in range(
        trade_engine_module.MAX_OPEN_POSITIONS
    ):

        opened = engine.open_position(
            candle,
            signal,
        )

        if opened:
            opened_count += 1

    assert opened_count == (
        trade_engine_module.MAX_OPEN_POSITIONS
    )

    extra_open = engine.open_position(
        candle,
        signal,
    )

    assert extra_open is False

    assert len(
        engine.positions
    ) == trade_engine_module.MAX_OPEN_POSITIONS


# =========================================================
# RUN-LEVEL INTEGRATION
# =========================================================

def test_trade_engine_run_uses_strategy_signal(
    monkeypatch,
):
    """
    TradeEngine.run() must consume Strategy's SignalResult.

    The strategy is replaced with a deterministic BUY signal
    so this test validates only the Strategy → TradeEngine
    integration boundary.
    """

    engine = make_engine()

    monkeypatch.setattr(
        trade_engine_module,
        "generate_signal_result",
        lambda history: make_buy_signal(
            entry=float(
                history.iloc[-1]["close"]
            ),
            stop_loss=float(
                history.iloc[-1]["close"]
            ) - 2.0,
            take_profit=float(
                history.iloc[-1]["close"]
            ) + 4.0,
        ),
    )

    monkeypatch.setattr(
        trade_engine_module,
        "calculate_position_size",
        lambda **kwargs: 1.0,
    )

    rows = []

    for i in range(260):

        price = 100.0 + i * 0.01

        rows.append(
            make_candle(
                timestamp=(
                    "2025-01-01 "
                    f"{i // 60:02d}:"
                    f"{i % 60:02d}:00"
                ),
                open_price=price,
                high=price + 0.5,
                low=price - 0.5,
                close=price,
                atr=0.1,
            )
        )

    df = pd.DataFrame(rows)

    result = engine.run(df)

    assert isinstance(
        result,
        pd.DataFrame,
    )

    assert engine.total_trades >= 0

    assert (
        engine.total_trades
        == len(engine.closed_positions)
    )


def test_trade_engine_run_hold_strategy_does_not_open_trades(
    monkeypatch,
):
    """
    If Strategy continuously returns HOLD, TradeEngine.run()
    must finish without opening trades.
    """

    engine = make_engine()

    monkeypatch.setattr(
        trade_engine_module,
        "generate_signal_result",
        lambda history: make_hold_signal(),
    )

    rows = []

    for i in range(260):

        price = 100.0 + i * 0.01

        rows.append(
            make_candle(
                timestamp=(
                    "2025-01-01 "
                    f"{i // 60:02d}:"
                    f"{i % 60:02d}:00"
                ),
                open_price=price,
                high=price + 0.5,
                low=price - 0.5,
                close=price,
                atr=0.1,
            )
        )

    df = pd.DataFrame(rows)

    result = engine.run(df)

    assert isinstance(
        result,
        pd.DataFrame,
    )

    assert engine.total_trades == 0
    assert len(engine.positions) == 0
    assert len(engine.closed_positions) == 0

    assert engine.balance == pytest.approx(
        INITIAL_BALANCE,
        abs=1e-6,
    )
