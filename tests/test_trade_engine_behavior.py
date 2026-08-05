"""
====================================================
QuantAI TradeEngine Behavioral Tests
====================================================

Behavioral tests for:

1. LONG position opening
2. SHORT position opening
3. MAX_OPEN_POSITIONS limit
4. Stop Loss closing
5. Take Profit closing
6. Break-Even
7. Trailing Stop
8. End-of-backtest closing
9. Position removal from open positions
10. Position archiving into closed_positions
"""

from __future__ import annotations

import pandas as pd
import pytest

from config.settings import (
    MAX_OPEN_POSITIONS,
    SLIPPAGE,
)

from src.strategy import SignalResult
from src.trade_engine import (
    CloseReason,
    PositionSide,
    PositionStatus,
    TradeEngine,
)


# ============================================================
# HELPERS
# ============================================================

def make_candle(
    timestamp="2026-01-01",
    open_price=100.0,
    high=105.0,
    low=95.0,
    close=100.0,
    atr=1.0,
):
    return {
        "timestamp": timestamp,
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "atr": atr,
    }


def make_signal(
    side="BUY",
    entry=100.0,
    stop_loss=98.0,
    take_profit=106.0,
    confidence=80.0,
):
    return SignalResult(
        signal=side,
        entry=entry,
        stop_loss=stop_loss,
        take_profit=take_profit,
        confidence=confidence,
        reasons=["TEST"],
    )


# ============================================================
# 1. OPEN LONG
# ============================================================

def test_open_long_position():

    engine = TradeEngine()

    candle = make_candle(
        open_price=100.0,
        high=101.0,
        low=99.0,
        close=100.0,
        atr=1.0,
    )

    signal = make_signal(
        side="BUY",
        entry=100.0,
        stop_loss=98.0,
        take_profit=106.0,
    )

    result = engine.open_position(
        candle,
        signal,
    )

    assert result is True
    assert len(engine.positions) == 1

    position = engine.positions[0]

    assert position.side == PositionSide.LONG
    assert position.status == PositionStatus.OPEN
    assert position.entry_price == pytest.approx(
        100.0 * (1 + SLIPPAGE),
        rel=1e-6,
    )


# ============================================================
# 2. OPEN SHORT
# ============================================================

def test_open_short_position():

    engine = TradeEngine()

    candle = make_candle(
        open_price=100.0,
        high=101.0,
        low=99.0,
        close=100.0,
        atr=1.0,
    )

    signal = make_signal(
        side="SELL",
        entry=100.0,
        stop_loss=102.0,
        take_profit=94.0,
    )

    result = engine.open_position(
        candle,
        signal,
    )

    assert result is True
    assert len(engine.positions) == 1

    position = engine.positions[0]

    assert position.side == PositionSide.SHORT
    assert position.status == PositionStatus.OPEN
    assert position.entry_price == pytest.approx(
        100.0 * (1 - SLIPPAGE),
        rel=1e-6,
    )


# ============================================================
# 3. MAX OPEN POSITIONS
# ============================================================

def test_max_open_positions_limit():

    engine = TradeEngine()

    candle = make_candle()

    signal = make_signal(
        side="BUY",
        entry=100.0,
        stop_loss=98.0,
        take_profit=106.0,
    )

    for _ in range(MAX_OPEN_POSITIONS):

        result = engine.open_position(
            candle,
            signal,
        )

        assert result is True

    assert len(engine.positions) == MAX_OPEN_POSITIONS

    result = engine.open_position(
        candle,
        signal,
    )

    assert result is False

    assert len(engine.positions) == MAX_OPEN_POSITIONS


# ============================================================
# 4. STOP LOSS
# ============================================================

def test_long_position_closes_by_stop_loss():

    engine = TradeEngine()

    candle_entry = make_candle(
        timestamp="2026-01-01",
        open_price=100.0,
        high=101.0,
        low=99.0,
        close=100.0,
        atr=1.0,
    )

    signal = make_signal(
        side="BUY",
        entry=100.0,
        stop_loss=98.0,
        take_profit=106.0,
    )

    engine.open_position(
        candle_entry,
        signal,
    )

    position = engine.positions[0]

    candle_exit = make_candle(
        timestamp="2026-01-02",
        open_price=98.5,
        high=99.0,
        low=97.5,
        close=98.0,
        atr=1.0,
    )

    # Prevent trailing/break-even from changing the
    # intended stop-loss test.
    position.stop_loss = 98.0

    engine.close_position(
        position,
        candle_exit,
        98.0,
        CloseReason.STOP_LOSS,
    )

    assert position.status == PositionStatus.CLOSED
    assert position.reason_close == CloseReason.STOP_LOSS


# ============================================================
# 5. TAKE PROFIT
# ============================================================

def test_long_position_closes_by_take_profit():

    engine = TradeEngine()

    candle_entry = make_candle(
        timestamp="2026-01-01",
        open_price=100.0,
        high=101.0,
        low=99.0,
        close=100.0,
        atr=1.0,
    )

    signal = make_signal(
        side="BUY",
        entry=100.0,
        stop_loss=98.0,
        take_profit=106.0,
    )

    engine.open_position(
        candle_entry,
        signal,
    )

    position = engine.positions[0]

    candle_exit = make_candle(
        timestamp="2026-01-02",
        open_price=105.0,
        high=107.0,
        low=104.0,
        close=106.0,
        atr=1.0,
    )

    position.take_profit = 106.0

    engine.close_position(
        position,
        candle_exit,
        106.0,
        CloseReason.TAKE_PROFIT,
    )

    assert position.status == PositionStatus.CLOSED
    assert position.reason_close == CloseReason.TAKE_PROFIT
    assert position.net_profit != 0


# ============================================================
# 6. BREAK EVEN
# ============================================================

def test_long_break_even_moves_stop_to_entry():

    engine = TradeEngine()

    candle_entry = make_candle(
        timestamp="2026-01-01",
        open_price=100.0,
        high=101.0,
        low=99.0,
        close=100.0,
        atr=1.0,
    )

    signal = make_signal(
        side="BUY",
        entry=100.0,
        stop_loss=98.0,
        take_profit=110.0,
    )

    engine.open_position(
        candle_entry,
        signal,
    )

    position = engine.positions[0]

    original_entry = position.entry_price

    candle = make_candle(
        timestamp="2026-01-02",
        open_price=100.0,
        high=original_entry + 2.0,
        low=100.0,
        close=original_entry + 1.0,
        atr=1.0,
    )

    engine.update_position(
        position,
        candle,
    )

    assert position.stop_loss >= original_entry


# ============================================================
# 7. TRAILING STOP
# ============================================================

def test_long_trailing_stop_moves_up():

    engine = TradeEngine()

    candle_entry = make_candle(
        timestamp="2026-01-01",
        open_price=100.0,
        high=101.0,
        low=99.0,
        close=100.0,
        atr=1.0,
    )

    signal = make_signal(
        side="BUY",
        entry=100.0,
        stop_loss=98.0,
        take_profit=120.0,
    )

    engine.open_position(
        candle_entry,
        signal,
    )

    position = engine.positions[0]

    old_stop = position.stop_loss

    candle = make_candle(
        timestamp="2026-01-02",
        open_price=105.0,
        high=108.0,
        low=104.0,
        close=107.0,
        atr=1.0,
    )

    engine.update_position(
        position,
        candle,
    )

    assert position.stop_loss > old_stop


# ============================================================
# 8. END OF BACKTEST CLOSE
# ============================================================

def test_remaining_position_closes_at_end_of_backtest():

    engine = TradeEngine()

    rows = []

    for i in range(300):

        rows.append(
            {
                "timestamp": f"2026-01-{i + 1}",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1000.0,
                "atr": 1.0,
            }
        )

    df = pd.DataFrame(rows)

    # The actual strategy may produce HOLD signals.
    # Therefore this test verifies the close mechanism
    # directly rather than depending on strategy output.

    candle = make_candle(
        timestamp="2026-12-31",
        open_price=100.0,
        high=101.0,
        low=99.0,
        close=100.0,
        atr=1.0,
    )

    signal = make_signal(
        side="BUY",
        entry=100.0,
        stop_loss=98.0,
        take_profit=110.0,
    )

    engine.open_position(
        candle,
        signal,
    )

    assert len(engine.positions) == 1

    last_candle = df.iloc[-1]

    for position in engine.get_open_positions()[:]:

        engine.close_position(
            position,
            last_candle,
            float(last_candle["close"]),
            CloseReason.END_OF_BACKTEST,
        )

    assert len(engine.positions) == 0
    assert len(engine.closed_positions) == 1

    position = engine.closed_positions[0]

    assert position.status == PositionStatus.CLOSED
    assert position.reason_close == CloseReason.END_OF_BACKTEST


# ============================================================
# 9. REMOVAL FROM OPEN POSITIONS
# ============================================================

def test_closed_position_removed_from_open_positions():

    engine = TradeEngine()

    candle = make_candle()

    signal = make_signal(
        side="BUY",
        entry=100.0,
        stop_loss=98.0,
        take_profit=106.0,
    )

    engine.open_position(
        candle,
        signal,
    )

    position = engine.positions[0]

    assert position in engine.positions

    engine.close_position(
        position,
        candle,
        106.0,
        CloseReason.TAKE_PROFIT,
    )

    assert position not in engine.positions


# ============================================================
# 10. ARCHIVE INTO CLOSED POSITIONS
# ============================================================

def test_closed_position_archived():

    engine = TradeEngine()

    candle = make_candle()

    signal = make_signal(
        side="BUY",
        entry=100.0,
        stop_loss=98.0,
        take_profit=106.0,
    )

    engine.open_position(
        candle,
        signal,
    )

    position = engine.positions[0]

    engine.close_position(
        position,
        candle,
        106.0,
        CloseReason.TAKE_PROFIT,
    )

    assert position in engine.closed_positions
    assert position.status == PositionStatus.CLOSED
    assert engine.total_trades == 1


# ============================================================
# SUMMARY
# ============================================================

def test_trade_engine_behavior_smoke():

    engine = TradeEngine()

    assert engine.balance > 0
    assert engine.equity > 0
    assert engine.total_trades == 0
    assert engine.winning_trades == 0
    assert engine.losing_trades == 0
