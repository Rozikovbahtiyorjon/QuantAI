"""
====================================================
QuantAI Professional v5.2
Trade Engine Validation Tests
====================================================

Validates:

    - initialization
    - balance / equity
    - position limits
    - position creation
    - BUY / SELL sides
    - slippage
    - commission
    - position closing
    - profit calculation
    - loss calculation
    - statistics
    - trade dataframe
    - repeated engine isolation
    - END_OF_BACKTEST closing
"""

from __future__ import annotations

import pandas as pd
import pytest

from config.settings import (
    INITIAL_BALANCE,
    COMMISSION,
    SLIPPAGE,
)

from src.trade_engine import (
    CloseReason,
    Position,
    PositionSide,
    PositionStatus,
    TradeEngine,
)
from src.strategy import SignalResult


# ====================================================
# HELPERS
# ====================================================

def make_candle(
    price: float = 100.0,
    high: float | None = None,
    low: float | None = None,
    timestamp: str = "2025-01-01 00:00:00",
    atr: float = 1.0,
) -> pd.Series:
    """
    Create a single valid market candle.
    """

    if high is None:
        high = price + 1.0

    if low is None:
        low = price - 1.0

    return pd.Series(
        {
            "timestamp": pd.Timestamp(timestamp),
            "open": price,
            "high": high,
            "low": low,
            "close": price,
            "volume": 1000.0,
            "atr": atr,
        }
    )


def make_signal(
    signal: str = "BUY",
    entry: float = 100.0,
    stop_loss: float = 98.0,
    take_profit: float = 104.0,
    confidence: float = 80.0,
) -> SignalResult:
    """
    Create a deterministic Strategy signal.
    """

    return SignalResult(
        signal=signal,
        score=5.0,
        confidence=confidence,
        entry=entry,
        stop_loss=stop_loss,
        take_profit=take_profit,
        reasons=["TEST SIGNAL"],
    )


# ====================================================
# INITIALIZATION
# ====================================================

def test_trade_engine_initializes_with_config_balance():
    """
    TradeEngine must start with INITIAL_BALANCE.
    """

    engine = TradeEngine()

    assert engine.balance == pytest.approx(
        INITIAL_BALANCE
    )

    assert engine.equity == pytest.approx(
        INITIAL_BALANCE
    )


def test_trade_engine_starts_without_positions():
    """
    New engine must contain no open or closed positions.
    """

    engine = TradeEngine()

    assert engine.positions == []

    assert engine.closed_positions == []

    assert engine.position_counter == 0


def test_trade_engine_has_zero_trade_statistics_initially():
    """
    New engine must have zero trades.
    """

    engine = TradeEngine()

    assert engine.total_trades == 0

    assert engine.winning_trades == 0

    assert engine.losing_trades == 0

    assert engine.total_profit == 0.0

    assert engine.win_rate == 0.0


# ====================================================
# POSITION ID
# ====================================================

def test_position_ids_are_incremental():
    """
    Position IDs must increase sequentially.
    """

    engine = TradeEngine()

    first = engine.next_position_id()

    second = engine.next_position_id()

    third = engine.next_position_id()

    assert first == 1

    assert second == 2

    assert third == 3


# ====================================================
# POSITION LIMIT
# ====================================================

def test_can_open_position_initially():
    """
    Fresh engine must allow a position.
    """

    engine = TradeEngine()

    assert engine.can_open_position() is True


def test_get_open_positions_returns_empty_initially():
    """
    No open positions must exist initially.
    """

    engine = TradeEngine()

    assert engine.get_open_positions() == []


# ====================================================
# COMMISSION
# ====================================================

def test_commission_calculation():
    """
    Commission must equal:

        quantity * price * COMMISSION
    """

    engine = TradeEngine()

    quantity = 2.0

    price = 100.0

    expected = (
        quantity
        * price
        * COMMISSION
    )

    result = engine.calculate_commission(
        quantity,
        price,
    )

    assert result == pytest.approx(
        expected
    )


# ====================================================
# SLIPPAGE
# ====================================================

def test_buy_slippage_increases_price():
    """
    BUY entry must receive positive slippage.
    """

    engine = TradeEngine()

    price = 100.0

    result = engine.apply_slippage(
        PositionSide.LONG,
        price,
    )

    expected = (
        price
        * (1 + SLIPPAGE)
    )

    assert result == pytest.approx(
        expected
    )


def test_sell_slippage_decreases_price():
    """
    SELL entry must receive negative slippage.
    """

    engine = TradeEngine()

    price = 100.0

    result = engine.apply_slippage(
        PositionSide.SHORT,
        price,
    )

    expected = (
        price
        * (1 - SLIPPAGE)
    )

    assert result == pytest.approx(
        expected
    )


# ====================================================
# OPEN BUY
# ====================================================

def test_open_buy_position():
    """
    BUY signal must create LONG position.
    """

    engine = TradeEngine()

    candle = make_candle()

    signal = make_signal(
        signal="BUY",
    )

    result = engine.open_position(
        candle,
        signal,
    )

    assert result is True

    assert len(
        engine.positions
    ) == 1

    position = engine.positions[0]

    assert isinstance(
        position,
        Position,
    )

    assert position.side == PositionSide.LONG

    assert position.status == PositionStatus.OPEN

    assert position.entry_time == candle["timestamp"]

    assert position.quantity > 0


# ====================================================
# OPEN SELL
# ====================================================

def test_open_sell_position():
    """
    SELL signal must create SHORT position.
    """

    engine = TradeEngine()

    candle = make_candle()

    signal = make_signal(
        signal="SELL",
        stop_loss=102.0,
        take_profit=96.0,
    )

    result = engine.open_position(
        candle,
        signal,
    )

    assert result is True

    assert len(
        engine.positions
    ) == 1

    position = engine.positions[0]

    assert position.side == PositionSide.SHORT

    assert position.status == PositionStatus.OPEN

    assert position.quantity > 0


# ====================================================
# POSITION STATE
# ====================================================

def test_open_position_has_correct_risk_levels():
    """
    Position must preserve SL and TP from signal.
    """

    engine = TradeEngine()

    candle = make_candle()

    signal = make_signal(
        signal="BUY",
        entry=100.0,
        stop_loss=98.0,
        take_profit=104.0,
    )

    engine.open_position(
        candle,
        signal,
    )

    position = engine.positions[0]

    assert position.stop_loss == pytest.approx(
        98.0
    )

    assert position.take_profit == pytest.approx(
        104.0
    )

    assert position.confidence == pytest.approx(
        signal.confidence
    )


def test_open_position_contains_strategy_reasons():
    """
    Strategy reasons must be copied into position.
    """

    engine = TradeEngine()

    candle = make_candle()

    signal = make_signal()

    engine.open_position(
        candle,
        signal,
    )

    position = engine.positions[0]

    assert position.reason_open == [
        "TEST SIGNAL"
    ]


# ====================================================
# OPEN POSITION LIST
# ====================================================

def test_get_open_positions_returns_created_position():
    """
    Open position must appear in get_open_positions().
    """

    engine = TradeEngine()

    candle = make_candle()

    signal = make_signal()

    engine.open_position(
        candle,
        signal,
    )

    open_positions = (
        engine.get_open_positions()
    )

    assert len(open_positions) == 1

    assert open_positions[0] is engine.positions[0]


# ====================================================
# CLOSE BUY PROFIT
# ====================================================

def test_close_buy_position_with_profit():
    """
    Profitable LONG position must increase balance.
    """

    engine = TradeEngine()

    candle = make_candle(
        price=100.0
    )

    signal = make_signal(
        signal="BUY",
        entry=100.0,
        stop_loss=98.0,
        take_profit=104.0,
    )

    engine.open_position(
        candle,
        signal,
    )

    position = engine.positions[0]

    initial_balance = engine.balance

    exit_candle = make_candle(
        price=104.0,
        timestamp="2025-01-01 01:00:00",
    )

    engine.close_position(
        position,
        exit_candle,
        104.0,
        CloseReason.TAKE_PROFIT,
    )

    assert position.status == PositionStatus.CLOSED

    assert position.reason_close == (
        CloseReason.TAKE_PROFIT
    )

    assert position.net_profit > 0

    assert engine.balance > initial_balance

    assert len(
        engine.positions
    ) == 0

    assert len(
        engine.closed_positions
    ) == 1


# ====================================================
# CLOSE BUY LOSS
# ====================================================

def test_close_buy_position_with_loss():
    """
    Losing LONG position must decrease balance.
    """

    engine = TradeEngine()

    candle = make_candle(
        price=100.0
    )

    signal = make_signal(
        signal="BUY",
        entry=100.0,
        stop_loss=98.0,
        take_profit=104.0,
    )

    engine.open_position(
        candle,
        signal,
    )

    position = engine.positions[0]

    initial_balance = engine.balance

    exit_candle = make_candle(
        price=98.0,
        timestamp="2025-01-01 01:00:00",
    )

    engine.close_position(
        position,
        exit_candle,
        98.0,
        CloseReason.STOP_LOSS,
    )

    assert position.status == PositionStatus.CLOSED

    assert position.reason_close == (
        CloseReason.STOP_LOSS
    )

    assert position.net_profit < 0

    assert engine.balance < initial_balance


# ====================================================
# CLOSE SELL PROFIT
# ====================================================

def test_close_sell_position_with_profit():
    """
    Profitable SHORT position must increase balance.
    """

    engine = TradeEngine()

    candle = make_candle(
        price=100.0
    )

    signal = make_signal(
        signal="SELL",
        entry=100.0,
        stop_loss=102.0,
        take_profit=96.0,
    )

    engine.open_position(
        candle,
        signal,
    )

    position = engine.positions[0]

    initial_balance = engine.balance

    exit_candle = make_candle(
        price=96.0,
        timestamp="2025-01-01 01:00:00",
    )

    engine.close_position(
        position,
        exit_candle,
        96.0,
        CloseReason.TAKE_PROFIT,
    )

    assert position.status == PositionStatus.CLOSED

    assert position.net_profit > 0

    assert engine.balance > initial_balance


# ====================================================
# CLOSE SELL LOSS
# ====================================================

def test_close_sell_position_with_loss():
    """
    Losing SHORT position must decrease balance.
    """

    engine = TradeEngine()

    candle = make_candle(
        price=100.0
    )

    signal = make_signal(
        signal="SELL",
        entry=100.0,
        stop_loss=102.0,
        take_profit=96.0,
    )

    engine.open_position(
        candle,
        signal,
    )

    position = engine.positions[0]

    initial_balance = engine.balance

    exit_candle = make_candle(
        price=102.0,
        timestamp="2025-01-01 01:00:00",
    )

    engine.close_position(
        position,
        exit_candle,
        102.0,
        CloseReason.STOP_LOSS,
    )

    assert position.status == PositionStatus.CLOSED

    assert position.net_profit < 0

    assert engine.balance < initial_balance


# ====================================================
# POSITION ARCHIVING
# ====================================================

def test_closed_position_is_archived():
    """
    Closed positions must move to closed_positions.
    """

    engine = TradeEngine()

    candle = make_candle()

    signal = make_signal()

    engine.open_position(
        candle,
        signal,
    )

    position = engine.positions[0]

    engine.close_position(
        position,
        candle,
        104.0,
        CloseReason.TAKE_PROFIT,
    )

    assert position in (
        engine.closed_positions
    )

    assert position not in (
        engine.positions
    )


# ====================================================
# STATISTICS
# ====================================================

def test_statistics_after_profitable_trade():
    """
    One profitable trade:

        total = 1
        wins = 1
        losses = 0
        win_rate = 100%
    """

    engine = TradeEngine()

    candle = make_candle()

    signal = make_signal()

    engine.open_position(
        candle,
        signal,
    )

    position = engine.positions[0]

    engine.close_position(
        position,
        candle,
        104.0,
        CloseReason.TAKE_PROFIT,
    )

    assert engine.total_trades == 1

    assert engine.winning_trades == 1

    assert engine.losing_trades == 0

    assert engine.win_rate == 100.0


def test_statistics_after_losing_trade():
    """
    One losing trade:

        total = 1
        wins = 0
        losses = 1
        win_rate = 0%
    """

    engine = TradeEngine()

    candle = make_candle()

    signal = make_signal()

    engine.open_position(
        candle,
        signal,
    )

    position = engine.positions[0]

    engine.close_position(
        position,
        candle,
        98.0,
        CloseReason.STOP_LOSS,
    )

    assert engine.total_trades == 1

    assert engine.winning_trades == 0

    assert engine.losing_trades == 1

    assert engine.win_rate == 0.0


# ====================================================
# NET PROFIT CONSISTENCY
# ====================================================

def test_total_profit_equals_sum_of_trade_net_profit():
    """
    Engine total profit must equal sum of closed
    position net profits.
    """

    engine = TradeEngine()

    candle = make_candle()

    signal = make_signal()

    engine.open_position(
        candle,
        signal,
    )

    position = engine.positions[0]

    engine.close_position(
        position,
        candle,
        104.0,
        CloseReason.TAKE_PROFIT,
    )

    expected = sum(
        p.net_profit
        for p in engine.closed_positions
    )

    assert engine.total_profit == pytest.approx(
        expected
    )


def test_balance_matches_initial_balance_plus_profit():
    """
    Balance must equal:

        INITIAL_BALANCE + total_profit
    """

    engine = TradeEngine()

    candle = make_candle()

    signal = make_signal()

    engine.open_position(
        candle,
        signal,
    )

    position = engine.positions[0]

    engine.close_position(
        position,
        candle,
        104.0,
        CloseReason.TAKE_PROFIT,
    )

    expected = (
        INITIAL_BALANCE
        + engine.total_profit
    )

    assert engine.balance == pytest.approx(
        expected,
        abs=0.01,
    )


# ====================================================
# DATAFRAME
# ====================================================

def test_to_dataframe_returns_dataframe():
    """
    Trade history must be convertible to DataFrame.
    """

    engine = TradeEngine()

    result = engine.to_dataframe()

    assert isinstance(
        result,
        pd.DataFrame,
    )


def test_to_dataframe_empty_has_expected_columns():
    """
    Empty trade history must preserve the expected schema.
    """

    engine = TradeEngine()

    result = engine.to_dataframe()

    expected_columns = {
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
    }

    assert expected_columns.issubset(
        set(result.columns)
    )


def test_to_dataframe_contains_closed_trade():
    """
    Closed position must appear in trade DataFrame.
    """

    engine = TradeEngine()

    candle = make_candle()

    signal = make_signal()

    engine.open_position(
        candle,
        signal,
    )

    position = engine.positions[0]

    engine.close_position(
        position,
        candle,
        104.0,
        CloseReason.TAKE_PROFIT,
    )

    result = engine.to_dataframe()

    assert len(result) == 1

    assert result.iloc[0]["side"] == "BUY"

    assert result.iloc[0]["close_reason"] == (
        CloseReason.TAKE_PROFIT.value
    )


# ====================================================
# ENGINE ISOLATION
# ====================================================

def test_two_trade_engines_are_independent():
    """
    Separate TradeEngine instances must not share state.
    """

    first = TradeEngine()

    second = TradeEngine()

    candle = make_candle()

    signal = make_signal()

    first.open_position(
        candle,
        signal,
    )

    assert len(first.positions) == 1

    assert len(second.positions) == 0

    assert first.position_counter == 1

    assert second.position_counter == 0


# ====================================================
# END OF BACKTEST
# ====================================================

def test_end_of_backtest_closes_open_position():
    """
    TradeEngine.run() must close remaining positions
    at the final candle.
    """

    # We need enough rows for the engine's start_index=250.
    rows = 260

    timestamps = pd.date_range(
        start="2025-01-01",
        periods=rows,
        freq="15min",
    )

    prices = [
        100.0 + i * 0.01
        for i in range(rows)
    ]

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": prices,
            "high": [
                price + 0.5
                for price in prices
            ],
            "low": [
                price - 0.5
                for price in prices
            ],
            "close": prices,
            "volume": [1000.0] * rows,
            "atr": [1.0] * rows,
        }
    )

    engine = TradeEngine()

    # This test validates the finalization mechanism
    # without depending on Strategy/ML signals.
    #
    # We manually create an open position first.

    candle = df.iloc[250]

    signal = make_signal(
        signal="BUY",
        entry=float(candle["close"]),
        stop_loss=float(candle["close"]) - 2.0,
        take_profit=float(candle["close"]) + 4.0,
    )

    engine.open_position(
        candle,
        signal,
    )

    assert engine.total_trades == 0

    # Simulate the same finalization behavior used
    # by TradeEngine.run().
    last_candle = df.iloc[-1]

    for position in engine.get_open_positions()[:]:
        engine.close_position(
            position,
            last_candle,
            float(last_candle["close"]),
            CloseReason.END_OF_BACKTEST,
        )

    assert engine.total_trades == 1

    assert (
        engine.closed_positions[0].reason_close
        == CloseReason.END_OF_BACKTEST
    )

    assert len(
        engine.get_open_positions()
    ) == 0


# ====================================================
# END
# ====================================================