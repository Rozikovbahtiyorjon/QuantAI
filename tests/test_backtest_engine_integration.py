"""
=========================================================
QuantAI Professional v5
Backtest Engine Integration Tests
=========================================================

Validates the real integration:

    Prepared Data
        ↓
    BacktestEngine
        ↓
    TradeEngine
        ↓
    Strategy
        ↓
    Positions
        ↓
    Closed Trades
        ↓
    Balance
        ↓
    BacktestResult

These tests intentionally use the real BacktestEngine
and the real TradeEngine.

No exchange API.
No CCXT.
No live trading.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.backtest_engine import (
    MINIMUM_ROWS,
    BacktestEngine,
    BacktestResult,
)


# =========================================================
# CONFIG
# =========================================================

INITIAL_BALANCE = 1000.0


# =========================================================
# DATA FACTORY
# =========================================================

def make_prepared_dataframe(
    rows: int = MINIMUM_ROWS,
) -> pd.DataFrame:
    """
    Create a prepared historical DataFrame compatible with
    BacktestEngine.

    Strategy-specific columns are included so the real
    Strategy Engine can operate on the data.
    """

    timestamps = pd.date_range(
        start="2025-01-01",
        periods=rows,
        freq="15min",
    )

    close = [
        100.0 + (i * 0.05)
        for i in range(rows)
    ]

    data = pd.DataFrame(
        {
            "timestamp": timestamps,

            "open": close,

            "high": [
                price + 1.0
                for price in close
            ],

            "low": [
                price - 1.0
                for price in close
            ],

            "close": close,

            "volume": [
                1000.0
            ] * rows,

            "atr": [
                1.0
            ] * rows,

            # =============================================
            # Strategy columns
            # =============================================

            "ema_fast": close,

            "ema_slow": [
                price - 0.10
                for price in close
            ],

            "ema_trend": [
                price - 0.20
                for price in close
            ],

            "adx": [
                30.0
            ] * rows,

            "rsi": [
                60.0
            ] * rows,

            "macd": [
                1.0
            ] * rows,

            "macd_signal": [
                0.5
            ] * rows,

            "macd_hist": [
                0.5
            ] * rows,

            "volume_sma20": [
                800.0
            ] * rows,
        }
    )

    return data


# =========================================================
# HELPERS
# =========================================================

def make_engine() -> BacktestEngine:
    """
    Create BacktestEngine with deterministic initial balance.
    """

    return BacktestEngine(
        initial_balance=INITIAL_BALANCE,
    )


# =========================================================
# BASIC INTEGRATION
# =========================================================

def test_backtest_engine_runs_real_trade_engine():
    """
    BacktestEngine.run() must execute through the real
    TradeEngine.
    """

    engine = make_engine()

    df = make_prepared_dataframe()

    result = engine.run(df)

    assert isinstance(
        result,
        BacktestResult,
    )

    assert engine.trade_engine is not None


def test_backtest_result_contains_real_balance():
    """
    BacktestResult must contain a real final balance produced
    by TradeEngine.
    """

    engine = make_engine()

    df = make_prepared_dataframe()

    result = engine.run(df)

    assert result.initial_balance == pytest.approx(
        INITIAL_BALANCE
    )

    assert isinstance(
        result.final_balance,
        float,
    )


def test_backtest_result_balance_is_positive():
    """
    TradeEngine must not produce an invalid negative balance
    during this integration scenario.
    """

    engine = make_engine()

    df = make_prepared_dataframe()

    result = engine.run(df)

    assert result.final_balance > 0


# =========================================================
# TRADE ENGINE STATE
# =========================================================

def test_trade_engine_state_matches_backtest_result():
    """
    BacktestResult must reflect the final TradeEngine state.
    """

    engine = make_engine()

    df = make_prepared_dataframe()

    result = engine.run(df)

    assert result.final_balance == pytest.approx(
        engine.trade_engine.balance,
        abs=1e-6,
    )


def test_trade_count_matches_trade_engine():
    """
    BacktestResult.total_trades must match the real
    TradeEngine closed-position count.
    """

    engine = make_engine()

    df = make_prepared_dataframe()

    result = engine.run(df)

    assert result.total_trades == (
        engine.trade_engine.total_trades
    )


def test_winning_trades_match_trade_engine():
    """
    Winning trade statistics must come from the real
    TradeEngine.
    """

    engine = make_engine()

    df = make_prepared_dataframe()

    result = engine.run(df)

    assert result.winning_trades == (
        engine.trade_engine.winning_trades
    )


def test_losing_trades_match_trade_engine():
    """
    Losing trade statistics must come from the real
    TradeEngine.
    """

    engine = make_engine()

    df = make_prepared_dataframe()

    result = engine.run(df)

    assert result.losing_trades == (
        engine.trade_engine.losing_trades
    )


def test_win_rate_matches_trade_engine():
    """
    Win rate must match TradeEngine.
    """

    engine = make_engine()

    df = make_prepared_dataframe()

    result = engine.run(df)

    assert result.win_rate == pytest.approx(
        engine.trade_engine.win_rate,
        abs=1e-6,
    )


# =========================================================
# PROFIT INTEGRATION
# =========================================================

def test_net_profit_matches_trade_engine_total_profit():
    """
    BacktestResult.net_profit must match the real
    TradeEngine total_profit.
    """

    engine = make_engine()

    df = make_prepared_dataframe()

    result = engine.run(df)

    assert result.net_profit == pytest.approx(
        engine.trade_engine.total_profit,
        abs=1e-6,
    )


def test_net_profit_matches_balance_difference():
    """
    Final balance must equal:

        initial balance + net profit
    """

    engine = make_engine()

    df = make_prepared_dataframe()

    result = engine.run(df)

    expected = (
        result.initial_balance
        + result.net_profit
    )

    assert result.final_balance == pytest.approx(
        expected,
        abs=1e-6,
    )


# =========================================================
# TRADE HISTORY
# =========================================================

def test_backtest_trades_match_trade_engine_history():
    """
    BacktestResult.trades must represent the same completed
    trades produced by TradeEngine.
    """

    engine = make_engine()

    df = make_prepared_dataframe()

    result = engine.run(df)

    trades = result.trades

    assert trades is not None

    assert len(trades) == (
        engine.trade_engine.total_trades
    )


def test_trade_history_is_dataframe_compatible():
    """
    Trade history returned by BacktestEngine must be usable
    as a list of trade dictionaries.
    """

    engine = make_engine()

    df = make_prepared_dataframe()

    result = engine.run(df)

    trades = result.trades

    assert isinstance(
        trades,
        list,
    )

    if trades:

        assert isinstance(
            trades[0],
            dict,
        )


# =========================================================
# REQUIRED TRADE FIELDS
# =========================================================

def test_completed_trade_contains_core_fields():
    """
    Every completed trade returned by the integration layer
    must contain the core execution fields.
    """

    engine = make_engine()

    df = make_prepared_dataframe()

    result = engine.run(df)

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

    for trade in result.trades:

        assert required_fields.issubset(
            trade.keys()
        )


# =========================================================
# SOURCE DATA PROTECTION
# =========================================================

def test_backtest_does_not_modify_dataframe_index():
    """
    BacktestEngine must not modify the caller's DataFrame index.
    """

    engine = make_engine()

    df = make_prepared_dataframe()

    original_index = df.index.copy()

    engine.run(df)

    assert df.index.equals(
        original_index
    )


def test_backtest_does_not_modify_dataframe_columns():
    """
    BacktestEngine must not remove or rename caller columns.
    """

    engine = make_engine()

    df = make_prepared_dataframe()

    original_columns = list(
        df.columns
    )

    engine.run(df)

    assert list(
        df.columns
    ) == original_columns


def test_backtest_does_not_modify_dataframe_values():
    """
    BacktestEngine must operate on a copy of the caller's
    DataFrame.
    """

    engine = make_engine()

    df = make_prepared_dataframe()

    original = df.copy(deep=True)

    engine.run(df)

    pd.testing.assert_frame_equal(
        df,
        original,
    )


# =========================================================
# STATE ISOLATION
# =========================================================

def test_repeated_backtests_create_independent_trade_engines():
    """
    Two consecutive backtests must use different TradeEngine
    instances.
    """

    engine = make_engine()

    df = make_prepared_dataframe()

    engine.run(df)

    first_trade_engine = (
        engine.trade_engine
    )

    engine.run(df)

    second_trade_engine = (
        engine.trade_engine
    )

    assert first_trade_engine is not second_trade_engine


def test_repeated_backtests_do_not_accumulate_trade_history():
    """
    Trade history from the first backtest must not leak into
    the second backtest.
    """

    engine = make_engine()

    df = make_prepared_dataframe()

    first_result = engine.run(df)

    second_result = engine.run(df)

    assert second_result.total_trades == (
        first_result.total_trades
    )

    assert len(
        second_result.trades
    ) == len(
        first_result.trades
    )


def test_repeated_backtests_start_from_same_balance():
    """
    Every backtest must start from the configured balance.
    """

    engine = make_engine()

    df = make_prepared_dataframe()

    first_result = engine.run(df)

    second_result = engine.run(df)

    assert first_result.initial_balance == pytest.approx(
        INITIAL_BALANCE
    )

    assert second_result.initial_balance == pytest.approx(
        INITIAL_BALANCE
    )


def test_position_counter_is_reset_between_runs():
    """
    A fresh TradeEngine must reset position IDs.

    If trades exist, the first trade ID of every independent
    run must start from 1.
    """

    engine = make_engine()

    df = make_prepared_dataframe()

    first_result = engine.run(df)

    second_result = engine.run(df)

    if first_result.trades:

        assert first_result.trades[0]["id"] == 1

    if second_result.trades:

        assert second_result.trades[0]["id"] == 1


# =========================================================
# RESULT PERSISTENCE
# =========================================================

def test_latest_result_is_persisted():
    """
    engine.result must point to the latest BacktestResult.
    """

    engine = make_engine()

    df = make_prepared_dataframe()

    result = engine.run(df)

    assert engine.result is result


def test_result_is_replaced_after_second_run():
    """
    Second run must replace the previous result.
    """

    engine = make_engine()

    df = make_prepared_dataframe()

    first_result = engine.run(df)

    second_result = engine.run(df)

    assert engine.result is second_result

    assert engine.result is not first_result


# =========================================================
# CONSISTENCY
# =========================================================

def test_trade_statistics_are_consistent():
    """
    Winning + losing trades must equal total trades.
    """

    engine = make_engine()

    df = make_prepared_dataframe()

    result = engine.run(df)

    assert (
        result.winning_trades
        + result.losing_trades
        == result.total_trades
    )


def test_trade_balance_is_last_balance():
    """
    If completed trades exist, the balance recorded on the
    last trade must equal BacktestResult.final_balance.
    """

    engine = make_engine()

    df = make_prepared_dataframe()

    result = engine.run(df)

    if result.trades:

        last_trade = result.trades[-1]

        assert last_trade["balance"] == pytest.approx(
            result.final_balance,
            abs=1e-6,
        )


# =========================================================
# REPORT INTEGRATION
# =========================================================

def test_report_accepts_real_integration_result(
    capsys,
):
    """
    print_report() must accept the real result generated by
    BacktestEngine.run().
    """

    engine = make_engine()

    df = make_prepared_dataframe()

    result = engine.run(df)

    engine.print_report(
        result
    )

    output = capsys.readouterr().out

    assert "QUANTAI BACKTEST REPORT" in output
    assert "Initial Balance" in output
    assert "Final Balance" in output
    assert "Net Profit" in output
    assert "Total Trades" in output