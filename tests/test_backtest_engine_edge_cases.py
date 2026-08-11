"""
=========================================================
QuantAI Professional v5
Backtest Engine Edge Case Tests
=========================================================

Validates unusual and boundary scenarios:

    - exact minimum dataset
    - below minimum dataset
    - HOLD-only scenario
    - zero trades
    - single trade
    - repeated runs
    - state isolation
    - balance consistency
    - profit consistency
    - win-rate consistency
    - END_OF_BACKTEST handling
    - invalid numeric data
    - NaN data
    - source DataFrame protection
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

def make_dataframe(
    rows: int = MINIMUM_ROWS,
) -> pd.DataFrame:
    """
    Create a prepared DataFrame compatible with the complete
    BacktestEngine -> TradeEngine -> Strategy pipeline.
    """

    timestamps = pd.date_range(
        start="2025-01-01",
        periods=rows,
        freq="15min",
    )

    close = [
        100.0 + i * 0.05
        for i in range(rows)
    ]

    return pd.DataFrame(
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

            # =================================================
            # Strategy columns
            # =================================================

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
                50.0
            ] * rows,

            "macd": [
                0.0
            ] * rows,

            "macd_signal": [
                0.0
            ] * rows,

            "macd_hist": [
                0.0
            ] * rows,

            "volume_sma20": [
                1000.0
            ] * rows,
        }
    )


def make_engine() -> BacktestEngine:
    """
    Create deterministic BacktestEngine.
    """

    return BacktestEngine(
        initial_balance=INITIAL_BALANCE,
    )


# =========================================================
# MINIMUM DATASET
# =========================================================

def test_exact_minimum_dataset_is_accepted():
    """
    Exactly MINIMUM_ROWS must be accepted.
    """

    engine = make_engine()

    df = make_dataframe(
        MINIMUM_ROWS
    )

    result = engine.run(df)

    assert isinstance(
        result,
        BacktestResult,
    )


def test_dataset_one_row_below_minimum_is_rejected():
    """
    MINIMUM_ROWS - 1 must be rejected.
    """

    engine = make_engine()

    df = make_dataframe(
        MINIMUM_ROWS - 1
    )

    with pytest.raises(ValueError):

        engine.run(df)


# =========================================================
# ZERO TRADE / HOLD SCENARIO
# =========================================================

def test_hold_only_scenario_is_safe():
    """
    A market that produces no approved trades must still
    generate a valid BacktestResult.
    """

    engine = make_engine()

    df = make_dataframe()

    result = engine.run(df)

    assert isinstance(
        result,
        BacktestResult,
    )

    assert result.total_trades >= 0


def test_zero_trade_result_is_internally_consistent():
    """
    If the strategy produces zero trades, statistics must
    remain valid.
    """

    engine = make_engine()

    df = make_dataframe()

    result = engine.run(df)

    if result.total_trades == 0:

        assert result.winning_trades == 0

        assert result.losing_trades == 0

        assert result.win_rate == 0.0

        assert result.trades == []


# =========================================================
# RESULT BALANCE
# =========================================================

def test_no_trade_backtest_preserves_initial_balance():
    """
    With zero trades, balance must remain unchanged.
    """

    engine = make_engine()

    df = make_dataframe()

    result = engine.run(df)

    if result.total_trades == 0:

        assert result.final_balance == pytest.approx(
            INITIAL_BALANCE,
            abs=1e-6,
        )

        assert result.net_profit == pytest.approx(
            0.0,
            abs=1e-6,
        )


# =========================================================
# PROFIT CONSISTENCY
# =========================================================

def test_profit_never_disagrees_with_balance_difference():
    """
    Net profit must always equal:

        final_balance - initial_balance
    """

    engine = make_engine()

    df = make_dataframe()

    result = engine.run(df)

    expected = (
        result.final_balance
        - result.initial_balance
    )

    assert result.net_profit == pytest.approx(
        expected,
        abs=1e-6,
    )


# =========================================================
# WIN RATE
# =========================================================

def test_win_rate_is_zero_when_no_trades():
    """
    Zero trades must produce 0% win rate.
    """

    engine = make_engine()

    df = make_dataframe()

    result = engine.run(df)

    if result.total_trades == 0:

        assert result.win_rate == 0.0


def test_win_rate_is_between_zero_and_hundred():
    """
    Win rate must always remain inside [0, 100].
    """

    engine = make_engine()

    df = make_dataframe()

    result = engine.run(df)

    assert 0.0 <= result.win_rate <= 100.0


# =========================================================
# TRADE STATISTICS
# =========================================================

def test_winning_and_losing_trades_never_exceed_total():
    """
    Statistics must remain mathematically valid.
    """

    engine = make_engine()

    df = make_dataframe()

    result = engine.run(df)

    assert (
        result.winning_trades
        <= result.total_trades
    )

    assert (
        result.losing_trades
        <= result.total_trades
    )


def test_trade_statistics_sum_to_total():
    """
    Winning + losing trades must equal total trades.
    """

    engine = make_engine()

    df = make_dataframe()

    result = engine.run(df)

    assert (
        result.winning_trades
        + result.losing_trades
        == result.total_trades
    )


# =========================================================
# TRADE HISTORY
# =========================================================

def test_trade_history_is_empty_when_no_trades():
    """
    Zero-trade backtest must expose an empty trade collection.
    """

    engine = make_engine()

    df = make_dataframe()

    result = engine.run(df)

    if result.total_trades == 0:

        assert isinstance(
            result.trades,
            list,
        )

        assert result.trades == []


def test_trade_history_length_matches_total_trades():
    """
    Trade history length must match total trade count.
    """

    engine = make_engine()

    df = make_dataframe()

    result = engine.run(df)

    assert len(
        result.trades
    ) == result.total_trades


# =========================================================
# END OF BACKTEST
# =========================================================

def test_open_positions_are_not_left_after_backtest():
    """
    Backtest must finish with no OPEN positions.
    Remaining positions must be closed at the end.
    """

    engine = make_engine()

    df = make_dataframe()

    result = engine.run(df)

    open_positions = (
        engine.trade_engine.get_open_positions()
    )

    assert open_positions == []


def test_all_returned_trades_are_closed():
    """
    Every trade returned by BacktestEngine must represent a
    completed position.
    """

    engine = make_engine()

    df = make_dataframe()

    result = engine.run(df)

    for trade in result.trades:

        assert trade["exit_time"] is not None

        assert trade["close_reason"] is not None


# =========================================================
# REPEATED RUNS
# =========================================================

def test_multiple_runs_have_identical_results():
    """
    Running the same deterministic dataset twice must produce
    equivalent statistics.
    """

    engine = make_engine()

    df = make_dataframe()

    first = engine.run(df)

    second = engine.run(df)

    assert second.initial_balance == pytest.approx(
        first.initial_balance,
        abs=1e-6,
    )

    assert second.final_balance == pytest.approx(
        first.final_balance,
        abs=1e-6,
    )

    assert second.net_profit == pytest.approx(
        first.net_profit,
        abs=1e-6,
    )

    assert second.total_trades == (
        first.total_trades
    )

    assert second.winning_trades == (
        first.winning_trades
    )

    assert second.losing_trades == (
        first.losing_trades
    )

    assert second.win_rate == pytest.approx(
        first.win_rate,
        abs=1e-6,
    )


def test_repeated_runs_do_not_change_initial_balance():
    """
    Every run must start from INITIAL_BALANCE.
    """

    engine = make_engine()

    df = make_dataframe()

    for _ in range(3):

        result = engine.run(df)

        assert result.initial_balance == pytest.approx(
            INITIAL_BALANCE,
            abs=1e-6,
        )


def test_repeated_runs_do_not_accumulate_positions():
    """
    Previous positions must never leak into a later run.
    """

    engine = make_engine()

    df = make_dataframe()

    engine.run(df)

    first_trade_engine = (
        engine.trade_engine
    )

    engine.run(df)

    second_trade_engine = (
        engine.trade_engine
    )

    assert first_trade_engine is not second_trade_engine

    assert (
        second_trade_engine.position_counter
        >= 0
    )


# =========================================================
# POSITION ID ISOLATION
# =========================================================

def test_position_ids_restart_between_independent_runs():
    """
    Fresh TradeEngine must restart position numbering.
    """

    engine = make_engine()

    df = make_dataframe()

    first = engine.run(df)

    second = engine.run(df)

    if first.trades:

        assert first.trades[0]["id"] == 1

    if second.trades:

        assert second.trades[0]["id"] == 1


# =========================================================
# DATAFRAME PROTECTION
# =========================================================

def test_original_dataframe_index_is_unchanged():
    """
    The original DataFrame index must survive the backtest.
    """

    engine = make_engine()

    df = make_dataframe()

    original_index = df.index.copy()

    engine.run(df)

    assert df.index.equals(
        original_index
    )


def test_original_dataframe_columns_are_unchanged():
    """
    The original DataFrame columns must remain unchanged.
    """

    engine = make_engine()

    df = make_dataframe()

    original_columns = list(
        df.columns
    )

    engine.run(df)

    assert list(
        df.columns
    ) == original_columns


def test_original_dataframe_values_are_unchanged():
    """
    Backtest must not mutate original market data.
    """

    engine = make_engine()

    df = make_dataframe()

    original = df.copy(
        deep=True
    )

    engine.run(df)

    pd.testing.assert_frame_equal(
        df,
        original,
    )


# =========================================================
# INVALID NUMERIC DATA
# =========================================================

@pytest.mark.parametrize(
    "column",
    [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "atr",
    ],
)
def test_invalid_numeric_column_is_rejected(
    column,
):
    """
    Invalid required numeric data must be rejected.
    """

    engine = make_engine()

    df = make_dataframe()

    df[column] = "INVALID"

    with pytest.raises(TypeError):

        engine.run(df)


# =========================================================
# NaN DATA
# =========================================================

@pytest.mark.parametrize(
    "column",
    [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "atr",
    ],
)
def test_nan_required_numeric_column_is_rejected(
    column,
):
    """
    NaN in required market data must be rejected.
    """

    engine = make_engine()

    df = make_dataframe()

    df.loc[
        MINIMUM_ROWS // 2,
        column,
    ] = float("nan")

    with pytest.raises(ValueError):

        engine.run(df)


# =========================================================
# BALANCE AFTER TRADES
# =========================================================

def test_last_trade_balance_matches_result_balance():
    """
    When trades exist, the last recorded trade balance must
    equal the final BacktestResult balance.
    """

    engine = make_engine()

    df = make_dataframe()

    result = engine.run(df)

    if result.trades:

        last_trade = result.trades[-1]

        assert last_trade["balance"] == pytest.approx(
            result.final_balance,
            abs=1e-6,
        )


def test_trade_profit_sum_matches_engine_profit():
    """
    Sum of individual net profits must equal TradeEngine
    total profit.
    """

    engine = make_engine()

    df = make_dataframe()

    result = engine.run(df)

    trade_profit = sum(
        float(
            trade["net_profit"]
        )
        for trade in result.trades
    )

    assert trade_profit == pytest.approx(
        result.net_profit,
        abs=0.02,
    )


# =========================================================
# RESULT OBJECT
# =========================================================

def test_result_is_persisted_after_edge_case_run():
    """
    Even an edge-case backtest must persist its result.
    """

    engine = make_engine()

    df = make_dataframe()

    result = engine.run(df)

    assert engine.result is result


def test_result_is_not_none_after_successful_run():
    """
    Successful execution must always create a result.
    """

    engine = make_engine()

    df = make_dataframe()

    engine.run(df)

    assert engine.result is not None


# =========================================================
# REPORT
# =========================================================

def test_report_handles_edge_case_result(
    capsys,
):
    """
    Report must work even when no trades occurred.
    """

    engine = make_engine()

    df = make_dataframe()

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


# =========================================================
# END
# =========================================================