"""
QuantAI Professional v5
Backtest + Trade Engine Integration Tests

Проверяет интеграцию:

    BacktestEngine
        ↓
    TradeEngine
        ↓
    Strategy signal
        ↓
    Position
        ↓
    Close
        ↓
    BacktestResult
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.backtest_engine import (
    BacktestEngine,
    BacktestResult,
    MINIMUM_ROWS,
)

from src.strategy import SignalResult


# ============================================================
# TEST DATA
# ============================================================

INITIAL_BALANCE = 1000.0


def make_ohlcv(rows: int = MINIMUM_ROWS + 20) -> pd.DataFrame:
    """
    Create deterministic prepared OHLCV data.

    The DataFrame already contains ATR because
    BacktestEngine expects prepared data.
    """

    data = []

    for i in range(rows):

        close = 100.0

        high = 100.5
        low = 99.5

        # ----------------------------------------------------
        # Candle after entry.
        #
        # Position should reach TAKE_PROFIT.
        # ----------------------------------------------------

        if i == 251:

            close = 101.0
            high = 101.2
            low = 100.9

        data.append(
            {
                "timestamp": pd.Timestamp(
                    "2026-01-01"
                ) + pd.Timedelta(
                    minutes=15 * i
                ),
                "open": close,
                "high": high,
                "low": low,
                "close": close,
                "volume": 1000.0,
                "atr": 0.1,
            }
        )

    return pd.DataFrame(data)


# ============================================================
# DETERMINISTIC STRATEGY
# ============================================================

def make_buy_signal() -> SignalResult:
    """
    Deterministic BUY signal used only for integration testing.
    """

    result = SignalResult()

    result.signal = "BUY"
    result.confidence = 90.0
    result.score = 5.0

    result.entry = 100.0
    result.stop_loss = 99.0
    result.take_profit = 101.0

    result.reasons = [
        "Integration test BUY signal"
    ]

    return result


def make_hold_signal() -> SignalResult:
    """
    Deterministic HOLD signal.
    """

    result = SignalResult()

    result.signal = "HOLD"
    result.confidence = 0.0
    result.score = 0.0

    result.entry = 100.0
    result.stop_loss = 100.0
    result.take_profit = 100.0

    result.reasons = [
        "Integration test HOLD signal"
    ]

    return result


# ============================================================
# TEST 1
# ============================================================

def test_backtest_engine_returns_backtest_result(
    monkeypatch,
):
    """
    BacktestEngine.run() must return BacktestResult.
    """

    def fake_strategy(df):

        if len(df) == 251:

            return make_buy_signal()

        return make_hold_signal()

    monkeypatch.setattr(
        "src.trade_engine.generate_signal_result",
        fake_strategy,
    )

    df = make_ohlcv()

    engine = BacktestEngine(
        initial_balance=INITIAL_BALANCE,
    )

    result = engine.run(df)

    assert isinstance(
        result,
        BacktestResult,
    )


# ============================================================
# TEST 2
# ============================================================

def test_backtest_initial_balance_is_preserved(
    monkeypatch,
):
    """
    Backtest must start with the configured balance.
    """

    def fake_strategy(df):
        return make_hold_signal()

    monkeypatch.setattr(
        "src.trade_engine.generate_signal_result",
        fake_strategy,
    )

    df = make_ohlcv()

    engine = BacktestEngine(
        initial_balance=INITIAL_BALANCE,
    )

    result = engine.run(df)

    assert result.initial_balance == pytest.approx(
        INITIAL_BALANCE
    )


# ============================================================
# TEST 3
# ============================================================

def test_backtest_can_execute_trade(
    monkeypatch,
):
    """
    A BUY signal must result in at least one closed trade.
    """

    def fake_strategy(df):

        if len(df) == 251:

            return make_buy_signal()

        return make_hold_signal()

    monkeypatch.setattr(
        "src.trade_engine.generate_signal_result",
        fake_strategy,
    )

    df = make_ohlcv()

    engine = BacktestEngine(
        initial_balance=INITIAL_BALANCE,
    )

    result = engine.run(df)

    assert result.total_trades >= 1


# ============================================================
# TEST 4
# ============================================================

def test_backtest_trade_is_closed(
    monkeypatch,
):
    """
    The executed position must eventually become
    a closed trade.
    """

    def fake_strategy(df):

        if len(df) == 251:

            return make_buy_signal()

        return make_hold_signal()

    monkeypatch.setattr(
        "src.trade_engine.generate_signal_result",
        fake_strategy,
    )

    df = make_ohlcv()

    engine = BacktestEngine(
        initial_balance=INITIAL_BALANCE,
    )

    result = engine.run(df)

    assert result.total_trades == (
        result.winning_trades
        + result.losing_trades
    )


# ============================================================
# TEST 5
# ============================================================

def test_backtest_statistics_are_consistent(
    monkeypatch,
):
    """
    Trade statistics must be internally consistent.
    """

    def fake_strategy(df):

        if len(df) == 251:

            return make_buy_signal()

        return make_hold_signal()

    monkeypatch.setattr(
        "src.trade_engine.generate_signal_result",
        fake_strategy,
    )

    df = make_ohlcv()

    engine = BacktestEngine(
        initial_balance=INITIAL_BALANCE,
    )

    result = engine.run(df)

    assert result.total_trades >= 1

    assert (
        result.winning_trades
        + result.losing_trades
        == result.total_trades
    )

    assert 0.0 <= result.win_rate <= 100.0


# ============================================================
# TEST 6
# ============================================================

def test_backtest_profit_matches_balance_difference(
    monkeypatch,
):
    """
    Final balance minus initial balance must match
    the resulting net profit.
    """

    def fake_strategy(df):

        if len(df) == 251:

            return make_buy_signal()

        return make_hold_signal()

    monkeypatch.setattr(
        "src.trade_engine.generate_signal_result",
        fake_strategy,
    )

    df = make_ohlcv()

    engine = BacktestEngine(
        initial_balance=INITIAL_BALANCE,
    )

    result = engine.run(df)

    expected_profit = (
        result.final_balance
        - result.initial_balance
    )

    assert result.net_profit == pytest.approx(
        expected_profit,
        abs=0.01,
    )


# ============================================================
# TEST 7
# ============================================================

def test_backtest_trades_are_returned(
    monkeypatch,
):
    """
    BacktestResult.trades must contain executed
    trade history.
    """

    def fake_strategy(df):

        if len(df) == 251:

            return make_buy_signal()

        return make_hold_signal()

    monkeypatch.setattr(
        "src.trade_engine.generate_signal_result",
        fake_strategy,
    )

    df = make_ohlcv()

    engine = BacktestEngine(
        initial_balance=INITIAL_BALANCE,
    )

    result = engine.run(df)

    assert result.trades is not None

    assert len(result.trades) == result.total_trades


# ============================================================
# TEST 8
# ============================================================

def test_trade_contains_required_fields(
    monkeypatch,
):
    """
    Every completed trade must contain the fields
    produced by TradeEngine.to_dataframe().
    """

    def fake_strategy(df):

        if len(df) == 251:

            return make_buy_signal()

        return make_hold_signal()

    monkeypatch.setattr(
        "src.trade_engine.generate_signal_result",
        fake_strategy,
    )

    df = make_ohlcv()

    engine = BacktestEngine(
        initial_balance=INITIAL_BALANCE,
    )

    result = engine.run(df)

    assert len(result.trades) >= 1

    trade = result.trades[0]

    required_fields = {
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

    assert required_fields.issubset(
        trade.keys()
    )


# ============================================================
# TEST 9
# ============================================================

def test_trade_side_is_buy(
    monkeypatch,
):
    """
    Deterministic BUY strategy must produce a BUY trade.
    """

    def fake_strategy(df):

        if len(df) == 251:

            return make_buy_signal()

        return make_hold_signal()

    monkeypatch.setattr(
        "src.trade_engine.generate_signal_result",
        fake_strategy,
    )

    df = make_ohlcv()

    engine = BacktestEngine(
        initial_balance=INITIAL_BALANCE,
    )

    result = engine.run(df)

    assert len(result.trades) >= 1

    assert result.trades[0]["side"] == "BUY"


# ============================================================
# TEST 10
# ============================================================

def test_backtest_engine_creates_fresh_trade_engine(
    monkeypatch,
):
    """
    Every BacktestEngine.run() call must start with
    a clean TradeEngine state.
    """

    def fake_strategy(df):

        if len(df) == 251:

            return make_buy_signal()

        return make_hold_signal()

    monkeypatch.setattr(
        "src.trade_engine.generate_signal_result",
        fake_strategy,
    )

    df = make_ohlcv()

    engine = BacktestEngine(
        initial_balance=INITIAL_BALANCE,
    )

    first_result = engine.run(df)

    first_trade_count = (
        first_result.total_trades
    )

    first_balance = (
        first_result.final_balance
    )

    second_result = engine.run(df)

    assert second_result.total_trades == (
        first_trade_count
    )

    assert second_result.initial_balance == pytest.approx(
        INITIAL_BALANCE
    )

    assert second_result.final_balance == pytest.approx(
        first_balance,
        abs=0.01,
    )


# ============================================================
# TEST 11
# ============================================================

def test_no_signal_produces_no_trades(
    monkeypatch,
):
    """
    HOLD-only strategy must not create trades.
    """

    def fake_strategy(df):
        return make_hold_signal()

    monkeypatch.setattr(
        "src.trade_engine.generate_signal_result",
        fake_strategy,
    )

    df = make_ohlcv()

    engine = BacktestEngine(
        initial_balance=INITIAL_BALANCE,
    )

    result = engine.run(df)

    assert result.total_trades == 0
    assert result.winning_trades == 0
    assert result.losing_trades == 0
    assert result.win_rate == 0.0

    assert result.final_balance == pytest.approx(
        INITIAL_BALANCE
    )

    assert result.net_profit == pytest.approx(
        0.0
    )


# ============================================================
# TEST 12
# ============================================================

def test_backtest_result_property_is_updated(
    monkeypatch,
):
    """
    BacktestEngine.result must contain the latest
    BacktestResult after run().
    """

    def fake_strategy(df):
        return make_hold_signal()

    monkeypatch.setattr(
        "src.trade_engine.generate_signal_result",
        fake_strategy,
    )

    df = make_ohlcv()

    engine = BacktestEngine(
        initial_balance=INITIAL_BALANCE,
    )

    assert engine.result is None

    result = engine.run(df)

    assert engine.result is result
