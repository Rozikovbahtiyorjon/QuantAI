"""
====================================================
QuantAI Professional v5
Trade Engine Unit Tests
====================================================
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.trade_engine import (
    CloseReason,
    PositionSide,
    PositionStatus,
    TradeEngine,
)


# ====================================================
# HELPERS
# ====================================================

def make_candle(
    timestamp="2026-01-01 00:00:00",
    open_price=100.0,
    high=101.0,
    low=99.0,
    close=100.0,
    volume=1000.0,
    atr=1.0,
):
    return {
        "timestamp": pd.Timestamp(timestamp),
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "atr": atr,
    }


# ====================================================
# INITIALIZATION
# ====================================================

def test_trade_engine_initialization():
    """TradeEngine must initialize with a clean state."""

    engine = TradeEngine()

    assert engine.balance > 0
    assert engine.equity == engine.balance

    assert engine.positions == []
    assert engine.closed_positions == []

    assert engine.position_counter == 0


# ====================================================
# POSITION ID
# ====================================================

def test_next_position_id_increments():
    """Position IDs must increment sequentially."""

    engine = TradeEngine()

    first = engine.next_position_id()
    second = engine.next_position_id()
    third = engine.next_position_id()

    assert first == 1
    assert second == 2
    assert third == 3


# ====================================================
# OPEN POSITION LIMIT
# ====================================================

def test_can_open_position_initially():
    """Fresh engine must allow opening a position."""

    engine = TradeEngine()

    assert engine.can_open_position() is True


# ====================================================
# OPEN POSITIONS
# ====================================================

def test_get_open_positions_returns_only_open_positions():
    """get_open_positions must return only OPEN positions."""

    engine = TradeEngine()

    candle = make_candle()

    position = engine.positions

    assert position == []

    assert engine.get_open_positions() == []


# ====================================================
# COMMISSION
# ====================================================

def test_calculate_commission():
    """Commission must equal quantity * price * commission rate."""

    engine = TradeEngine()

    quantity = 2.0
    price = 100.0

    commission = engine.calculate_commission(
        quantity,
        price,
    )

    expected = (
        quantity
        * price
        * __import__(
            "config.settings",
            fromlist=["COMMISSION"],
        ).COMMISSION
    )

    assert commission == pytest.approx(
        expected
    )


# ====================================================
# SLIPPAGE
# ====================================================

def test_apply_slippage_long():
    """Long entry must apply positive slippage."""

    engine = TradeEngine()

    price = 100.0

    result = engine.apply_slippage(
        PositionSide.LONG,
        price,
    )

    assert result > price


def test_apply_slippage_short():
    """Short entry must apply negative slippage."""

    engine = TradeEngine()

    price = 100.0

    result = engine.apply_slippage(
        PositionSide.SHORT,
        price,
    )

    assert result < price


# ====================================================
# ENUMS
# ====================================================

def test_position_side_values():
    """Position side enum must match strategy signals."""

    assert PositionSide.LONG.value == "BUY"
    assert PositionSide.SHORT.value == "SELL"


def test_position_status_values():
    """Position status enum must be correct."""

    assert PositionStatus.OPEN.value == "OPEN"
    assert PositionStatus.CLOSED.value == "CLOSED"


def test_close_reason_values():
    """Close reasons must contain required engine states."""

    assert CloseReason.TAKE_PROFIT.value == "TAKE_PROFIT"
    assert CloseReason.STOP_LOSS.value == "STOP_LOSS"
    assert CloseReason.END_OF_BACKTEST.value == "END_OF_BACKTEST"


# ====================================================
# DATAFRAME OUTPUT
# ====================================================

def test_to_dataframe_empty_engine():
    """Fresh engine must return an empty trade DataFrame."""

    engine = TradeEngine()

    trades = engine.to_dataframe()

    assert isinstance(
        trades,
        pd.DataFrame,
    )

    assert len(trades) == 0

    required_columns = [
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

    for column in required_columns:
        assert column in trades.columns


# ====================================================
# STATISTICS
# ====================================================

def test_statistics_on_empty_engine():
    """Fresh engine must have zero trade statistics."""

    engine = TradeEngine()

    assert engine.total_trades == 0
    assert engine.winning_trades == 0
    assert engine.losing_trades == 0
    assert engine.total_profit == 0.0
    assert engine.win_rate == 0.0


# ====================================================
# RUN EMPTY DATAFRAME
# ====================================================

def test_run_empty_dataframe():
    """
    Current TradeEngine implementation should return
    an empty trade DataFrame when no candles exist.
    """

    engine = TradeEngine()

    df = pd.DataFrame(
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "atr",
        ]
    )

    result = engine.run(df)

    assert isinstance(
        result,
        pd.DataFrame,
    )

    assert len(result) == 0


# ====================================================
# BASIC CANDLE DATA
# ====================================================

def test_make_candle_helper():
    """Diagnostic helper must create valid candle data."""

    candle = make_candle()

    assert candle["open"] == 100.0
    assert candle["high"] == 101.0
    assert candle["low"] == 99.0
    assert candle["close"] == 100.0
    assert candle["atr"] == 1.0
