"""
====================================================
QuantAI Professional v5
Trade Engine Position Lifecycle Tests
====================================================

Tests:
    - opening LONG position
    - opening SHORT position
    - position properties
    - updating position
    - TAKE_PROFIT
    - STOP_LOSS
    - closing position
    - balance update
    - trade archiving
    - dataframe export
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.trade_engine import (
    TradeEngine,
    Position,
    PositionSide,
    PositionStatus,
    CloseReason,
)
from src.strategy import SignalResult


# ============================================================
# HELPERS
# ============================================================

def make_candle(
    timestamp="2026-01-01 00:00:00",
    open_price=100.0,
    high=101.0,
    low=99.0,
    close=100.0,
    atr=1.0,
):
    """Create a minimal valid candle."""

    return {
        "timestamp": pd.Timestamp(timestamp),
        "open": float(open_price),
        "high": float(high),
        "low": float(low),
        "close": float(close),
        "volume": 1000.0,
        "atr": float(atr),
    }


def make_long_signal(
    entry=100.0,
    stop_loss=98.0,
    take_profit=104.0,
    confidence=70.0,
):
    """Create a BUY signal."""

    return SignalResult(
        signal="BUY",
        confidence=confidence,
        score=2.0,
        entry=entry,
        stop_loss=stop_loss,
        take_profit=take_profit,
        reasons=["Test LONG signal"],
    )


def make_short_signal(
    entry=100.0,
    stop_loss=102.0,
    take_profit=96.0,
    confidence=70.0,
):
    """Create a SELL signal."""

    return SignalResult(
        signal="SELL",
        confidence=confidence,
        score=-2.0,
        entry=entry,
        stop_loss=stop_loss,
        take_profit=take_profit,
        reasons=["Test SHORT signal"],
    )


# ============================================================
# POSITION ID
# ============================================================

def test_position_ids_are_incremental():
    """Position IDs must increase sequentially."""

    engine = TradeEngine()

    first_id = engine.next_position_id()
    second_id = engine.next_position_id()
    third_id = engine.next_position_id()

    assert first_id == 1
    assert second_id == 2
    assert third_id == 3


# ============================================================
# OPEN LONG
# ============================================================

def test_open_long_position():
    """TradeEngine must correctly open a LONG position."""

    engine = TradeEngine()

    candle = make_candle()

    signal = make_long_signal()

    result = engine.open_position(
        candle,
        signal,
    )

    assert result is True

    assert len(engine.positions) == 1

    position = engine.positions[0]

    assert isinstance(position, Position)

    assert position.side == PositionSide.LONG

    assert position.status == PositionStatus.OPEN

    assert position.entry_time == candle["timestamp"]

    assert position.entry_price > 0

    assert position.stop_loss == 98.0

    assert position.take_profit == 104.0

    assert position.quantity > 0

    assert position.confidence == 70.0

    assert position.reason_open == [
        "Test LONG signal"
    ]


# ============================================================
# OPEN SHORT
# ============================================================

def test_open_short_position():
    """TradeEngine must correctly open a SHORT position."""

    engine = TradeEngine()

    candle = make_candle()

    signal = make_short_signal()

    result = engine.open_position(
        candle,
        signal,
    )

    assert result is True

    assert len(engine.positions) == 1

    position = engine.positions[0]

    assert position.side == PositionSide.SHORT

    assert position.status == PositionStatus.OPEN

    assert position.entry_time == candle["timestamp"]

    assert position.entry_price > 0

    assert position.stop_loss == 102.0

    assert position.take_profit == 96.0

    assert position.quantity > 0


# ============================================================
# OPEN POSITION LIMIT
# ============================================================

def test_engine_respects_max_open_positions():
    """
    TradeEngine must stop opening positions after
    MAX_OPEN_POSITIONS is reached.
    """

    from config.settings import MAX_OPEN_POSITIONS

    engine = TradeEngine()

    candle = make_candle()

    signal = make_long_signal()

    opened = 0

    for _ in range(MAX_OPEN_POSITIONS):
        if engine.open_position(
            candle,
            signal,
        ):
            opened += 1

    assert opened == MAX_OPEN_POSITIONS

    assert (
        len(engine.get_open_positions())
        == MAX_OPEN_POSITIONS
    )

    result = engine.open_position(
        candle,
        signal,
    )

    assert result is False

    assert (
        len(engine.get_open_positions())
        == MAX_OPEN_POSITIONS
    )


# ============================================================
# GET OPEN POSITIONS
# ============================================================

def test_get_open_positions_returns_only_open_positions():
    """get_open_positions() must exclude closed positions."""

    engine = TradeEngine()

    candle = make_candle()

    signal = make_long_signal()

    engine.open_position(
        candle,
        signal,
    )

    assert len(
        engine.get_open_positions()
    ) == 1

    position = engine.positions[0]

    position.status = PositionStatus.CLOSED

    assert (
        len(engine.get_open_positions())
        == 0
    )


# ============================================================
# CLOSE POSITION MANUALLY
# ============================================================

def test_close_position_updates_balance():
    """Closing a profitable position must update balance."""

    engine = TradeEngine()

    initial_balance = engine.balance

    candle_entry = make_candle(
        timestamp="2026-01-01 00:00:00"
    )

    signal = make_long_signal()

    engine.open_position(
        candle_entry,
        signal,
    )

    position = engine.positions[0]

    candle_exit = make_candle(
        timestamp="2026-01-01 00:15:00",
        close=104.0,
        high=104.0,
        low=103.0,
    )

    engine.close_position(
        position,
        candle_exit,
        104.0,
        CloseReason.TAKE_PROFIT,
    )

    assert position.status == PositionStatus.CLOSED

    assert position.exit_time == candle_exit["timestamp"]

    assert position.exit_price > 0

    assert position.reason_close == CloseReason.TAKE_PROFIT

    assert position.net_profit > 0

    assert engine.balance > initial_balance

    assert len(engine.positions) == 0

    assert len(engine.closed_positions) == 1


# ============================================================
# CLOSE LOSING POSITION
# ============================================================

def test_close_position_updates_balance_on_loss():
    """Closing a losing position must reduce balance."""

    engine = TradeEngine()

    initial_balance = engine.balance

    candle_entry = make_candle()

    signal = make_long_signal(
        entry=100.0,
        stop_loss=98.0,
        take_profit=104.0,
    )

    engine.open_position(
        candle_entry,
        signal,
    )

    position = engine.positions[0]

    candle_exit = make_candle(
        timestamp="2026-01-01 00:15:00",
        close=98.0,
        high=98.0,
        low=97.5,
    )

    engine.close_position(
        position,
        candle_exit,
        98.0,
        CloseReason.STOP_LOSS,
    )

    assert position.status == PositionStatus.CLOSED

    assert position.reason_close == CloseReason.STOP_LOSS

    assert position.net_profit < 0

    assert engine.balance < initial_balance

    assert len(engine.positions) == 0

    assert len(engine.closed_positions) == 1


# ============================================================
# TAKE PROFIT LONG
# ============================================================

def test_long_position_hits_take_profit():
    """LONG position must close when candle reaches take profit."""

    engine = TradeEngine()

    candle_entry = make_candle()

    signal = make_long_signal(
        entry=100.0,
        stop_loss=98.0,
        take_profit=104.0,
    )

    engine.open_position(
        candle_entry,
        signal,
    )

    position = engine.positions[0]

    candle = make_candle(
        timestamp="2026-01-01 00:15:00",
        high=105.0,
        low=103.0,
        close=104.5,
        atr=1.0,
    )

    engine.update_position(
        position,
        candle,
    )

    assert position.status == PositionStatus.CLOSED

    assert position.reason_close == CloseReason.TAKE_PROFIT

    assert len(engine.positions) == 0

    assert len(engine.closed_positions) == 1


# ============================================================
# STOP LOSS LONG
# ============================================================

def test_long_position_hits_stop_loss():
    """LONG position must close when candle reaches stop loss."""

    engine = TradeEngine()

    candle_entry = make_candle()

    signal = make_long_signal(
        entry=100.0,
        stop_loss=98.0,
        take_profit=104.0,
    )

    engine.open_position(
        candle_entry,
        signal,
    )

    position = engine.positions[0]

    candle = make_candle(
        timestamp="2026-01-01 00:15:00",
        high=100.5,
        low=97.0,
        close=97.5,
        atr=1.0,
    )

    engine.update_position(
        position,
        candle,
    )

    assert position.status == PositionStatus.CLOSED

    assert position.reason_close == CloseReason.STOP_LOSS

    assert len(engine.positions) == 0

    assert len(engine.closed_positions) == 1


# ============================================================
# TAKE PROFIT SHORT
# ============================================================

def test_short_position_hits_take_profit():
    """SHORT position must close when candle reaches take profit."""

    engine = TradeEngine()

    candle_entry = make_candle()

    signal = make_short_signal(
        entry=100.0,
        stop_loss=102.0,
        take_profit=96.0,
    )

    engine.open_position(
        candle_entry,
        signal,
    )

    position = engine.positions[0]

    candle = make_candle(
        timestamp="2026-01-01 00:15:00",
        high=97.0,
        low=95.0,
        close=95.5,
        atr=1.0,
    )

    engine.update_position(
        position,
        candle,
    )

    assert position.status == PositionStatus.CLOSED

    assert position.reason_close == CloseReason.TAKE_PROFIT

    assert len(engine.positions) == 0

    assert len(engine.closed_positions) == 1


# ============================================================
# STOP LOSS SHORT
# ============================================================

def test_short_position_hits_stop_loss():
    """SHORT position must close when candle reaches stop loss."""

    engine = TradeEngine()

    candle_entry = make_candle()

    signal = make_short_signal(
        entry=100.0,
        stop_loss=102.0,
        take_profit=96.0,
    )

    engine.open_position(
        candle_entry,
        signal,
    )

    position = engine.positions[0]

    candle = make_candle(
        timestamp="2026-01-01 00:15:00",
        high=103.0,
        low=99.5,
        close=102.5,
        atr=1.0,
    )

    engine.update_position(
        position,
        candle,
    )

    assert position.status == PositionStatus.CLOSED

    assert position.reason_close == CloseReason.STOP_LOSS

    assert len(engine.positions) == 0

    assert len(engine.closed_positions) == 1


# ============================================================
# POSITION BARS
# ============================================================

def test_position_bars_open_increase():
    """bars_open must increase after position updates."""

    engine = TradeEngine()

    candle_entry = make_candle()

    signal = make_long_signal()

    engine.open_position(
        candle_entry,
        signal,
    )

    position = engine.positions[0]

    candle = make_candle(
        timestamp="2026-01-01 00:15:00",
        high=100.5,
        low=99.5,
        close=100.0,
        atr=1.0,
    )

    engine.update_position(
        position,
        candle,
    )

    if position.status == PositionStatus.OPEN:
        assert position.bars_open == 1


# ============================================================
# TRADE DATAFRAME
# ============================================================

def test_closed_position_appears_in_dataframe():
    """Closed positions must appear in to_dataframe()."""

    engine = TradeEngine()

    candle_entry = make_candle()

    signal = make_long_signal()

    engine.open_position(
        candle_entry,
        signal,
    )

    position = engine.positions[0]

    candle_exit = make_candle(
        timestamp="2026-01-01 00:15:00",
        high=105.0,
        low=103.0,
        close=104.0,
        atr=1.0,
    )

    engine.close_position(
        position,
        candle_exit,
        104.0,
        CloseReason.TAKE_PROFIT,
    )

    trades = engine.to_dataframe()

    assert isinstance(
        trades,
        pd.DataFrame,
    )

    assert len(trades) == 1

    expected_columns = [
        "id",
        "side",
        "entry_time",
        "exit_time",
        "entry",
        "exit",
        "stop_loss",
        "take_profit",
        "quantity",
        "confidence",
        "bars",
        "gross_profit",
        "commission",
        "net_profit",
        "balance",
        "close_reason",
    ]

    for column in expected_columns:
        assert column in trades.columns

    row = trades.iloc[0]

    assert row["side"] == "BUY"

    assert row["close_reason"] == "TAKE_PROFIT"

    assert row["net_profit"] > 0


# ============================================================
# STATISTICS
# ============================================================

def test_trade_statistics_after_profitable_trade():
    """Trade statistics must update after closing a profitable trade."""

    engine = TradeEngine()

    candle_entry = make_candle()

    signal = make_long_signal()

    engine.open_position(
        candle_entry,
        signal,
    )

    position = engine.positions[0]

    candle_exit = make_candle(
        timestamp="2026-01-01 00:15:00",
        high=105.0,
        low=103.0,
        close=104.0,
    )

    engine.close_position(
        position,
        candle_exit,
        104.0,
        CloseReason.TAKE_PROFIT,
    )

    assert engine.total_trades == 1

    assert engine.winning_trades == 1

    assert engine.losing_trades == 0

    assert engine.win_rate == 100.0

    assert engine.total_profit > 0


# ============================================================
# STATISTICS AFTER LOSS
# ============================================================

def test_trade_statistics_after_losing_trade():
    """Trade statistics must update after a losing trade."""

    engine = TradeEngine()

    candle_entry = make_candle()

    signal = make_long_signal()

    engine.open_position(
        candle_entry,
        signal,
    )

    position = engine.positions[0]

    candle_exit = make_candle(
        timestamp="2026-01-01 00:15:00",
        high=98.0,
        low=97.0,
        close=98.0,
    )

    engine.close_position(
        position,
        candle_exit,
        98.0,
        CloseReason.STOP_LOSS,
    )

    assert engine.total_trades == 1

    assert engine.winning_trades == 0

    assert engine.losing_trades == 1

    assert engine.win_rate == 0.0

    assert engine.total_profit < 0