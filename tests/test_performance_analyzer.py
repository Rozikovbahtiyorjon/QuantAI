"""
=========================================================
QuantAI Professional v5
Performance Analyzer pytest tests.
=========================================================

Validates:

    - valid trade history
    - zero trades
    - winning / losing trades
    - win rate
    - net profit
    - average profit
    - profit factor
    - maximum drawdown
    - balance consistency
    - input validation
    - missing columns
    - NaN values
    - repeated analysis
    - source DataFrame protection
    - result structure
    - report generation
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.performance_analyzer import (
    PerformanceAnalyzer,
    PerformanceResult,
)


# =========================================================
# CONFIG
# =========================================================

INITIAL_BALANCE = 1000.0


# =========================================================
# DATA FACTORY
# =========================================================

def make_trades() -> pd.DataFrame:
    """
    Create deterministic closed-trade history.
    """

    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],

            "side": [
                "BUY",
                "SELL",
                "BUY",
                "SELL",
                "BUY",
            ],

            "entry_time": pd.date_range(
                "2026-01-01",
                periods=5,
                freq="15min",
            ),

            "exit_time": pd.date_range(
                "2026-01-01 01:00",
                periods=5,
                freq="15min",
            ),

            "entry": [
                100.0,
                105.0,
                110.0,
                115.0,
                120.0,
            ],

            "exit": [
                102.0,
                103.0,
                114.0,
                113.0,
                121.0,
            ],

            "stop_loss": [
                99.0,
                106.0,
                108.0,
                116.0,
                119.0,
            ],

            "take_profit": [
                102.0,
                103.0,
                114.0,
                113.0,
                121.0,
            ],

            "quantity": [
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],

            "confidence": [
                80.0,
                75.0,
                90.0,
                70.0,
                85.0,
            ],

            "bars": [
                5,
                4,
                7,
                3,
                6,
            ],

            "gross_profit": [
                2.0,
                2.0,
                4.0,
                2.0,
                1.0,
            ],

            "commission": [
                0.10,
                0.10,
                0.10,
                0.10,
                0.10,
            ],

            "net_profit": [
                1.90,
                -2.10,
                3.90,
                -2.10,
                0.90,
            ],

            "balance": [
                1001.90,
                999.80,
                1003.70,
                1001.60,
                1002.50,
            ],

            "close_reason": [
                "TAKE_PROFIT",
                "STOP_LOSS",
                "TAKE_PROFIT",
                "STOP_LOSS",
                "TAKE_PROFIT",
            ],
        }
    )


def make_empty_trades() -> pd.DataFrame:
    """
    Create an empty but structurally valid trade DataFrame.
    """

    return make_trades().iloc[0:0].copy()


def make_winning_trades() -> pd.DataFrame:
    """
    Create trades where every trade is profitable.
    """

    df = make_trades()

    df["net_profit"] = [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
    ]

    df["gross_profit"] = df["net_profit"]

    df["commission"] = 0.0

    return df


def make_losing_trades() -> pd.DataFrame:
    """
    Create trades where every trade is losing.
    """

    df = make_trades()

    df["net_profit"] = [
        -1.0,
        -2.0,
        -3.0,
        -4.0,
        -5.0,
    ]

    df["gross_profit"] = df["net_profit"]

    df["commission"] = 0.0

    return df


def make_engine() -> PerformanceAnalyzer:
    """
    Create deterministic analyzer.
    """

    return PerformanceAnalyzer(
        initial_balance=INITIAL_BALANCE,
    )


# =========================================================
# VALIDATION
# =========================================================

def test_valid_trade_dataframe_is_accepted():
    """
    Valid trade history must pass validation.
    """

    analyzer = make_engine()

    df = make_trades()

    analyzer.validate_data(df)


def test_non_dataframe_is_rejected():
    """
    PerformanceAnalyzer must require pandas DataFrame.
    """

    analyzer = make_engine()

    with pytest.raises(TypeError):

        analyzer.validate_data([])


def test_empty_dataframe_is_accepted_as_zero_trade_case():
    """
    Empty trade history should be treated as a valid
    zero-trade performance case.
    """

    analyzer = make_engine()

    df = make_empty_trades()

    analyzer.validate_data(df)


def test_missing_required_column_is_rejected():
    """
    Required performance columns must be present.
    """

    analyzer = make_engine()

    df = make_trades()

    df = df.drop(
        columns=["net_profit"]
    )

    with pytest.raises(
        ValueError,
        match="Missing required columns",
    ):

        analyzer.validate_data(df)


def test_invalid_net_profit_column_is_rejected():
    """
    net_profit must be numeric.
    """

    analyzer = make_engine()

    df = make_trades()

    df["net_profit"] = "INVALID"

    with pytest.raises(TypeError):

        analyzer.validate_data(df)


def test_nan_net_profit_is_rejected():
    """
    NaN net_profit must be rejected.
    """

    analyzer = make_engine()

    df = make_trades()

    df.loc[
        0,
        "net_profit",
    ] = float("nan")

    with pytest.raises(ValueError):

        analyzer.validate_data(df)


# =========================================================
# INITIAL BALANCE
# =========================================================

def test_initial_balance_is_applied():
    """
    Custom initial balance must be stored.
    """

    analyzer = PerformanceAnalyzer(
        initial_balance=5000.0,
    )

    assert analyzer.initial_balance == 5000.0


def test_invalid_initial_balance_is_rejected():
    """
    Initial balance must be greater than zero.
    """

    with pytest.raises(ValueError):

        PerformanceAnalyzer(
            initial_balance=0.0,
        )


def test_negative_initial_balance_is_rejected():
    """
    Negative balance must be rejected.
    """

    with pytest.raises(ValueError):

        PerformanceAnalyzer(
            initial_balance=-100.0,
        )


# =========================================================
# BASIC RESULT
# =========================================================

def test_analysis_returns_performance_result():
    """
    analyze() must return PerformanceResult.
    """

    analyzer = make_engine()

    result = analyzer.analyze(
        make_trades()
    )

    assert isinstance(
        result,
        PerformanceResult,
    )


def test_result_is_persisted():
    """
    Latest result must be available through analyzer.result.
    """

    analyzer = make_engine()

    result = analyzer.analyze(
        make_trades()
    )

    assert analyzer.result is result


def test_result_is_not_none_after_analysis():
    """
    Successful analysis must create a result.
    """

    analyzer = make_engine()

    analyzer.analyze(
        make_trades()
    )

    assert analyzer.result is not None


# =========================================================
# TRADE COUNT
# =========================================================

def test_total_trades_is_correct():
    """
    Total trades must match DataFrame length.
    """

    analyzer = make_engine()

    result = analyzer.analyze(
        make_trades()
    )

    assert result.total_trades == 5


def test_zero_trade_count_is_zero():
    """
    Empty trade history must produce zero trades.
    """

    analyzer = make_engine()

    result = analyzer.analyze(
        make_empty_trades()
    )

    assert result.total_trades == 0


# =========================================================
# WINNING / LOSING TRADES
# =========================================================

def test_winning_trade_count_is_correct():
    """
    Positive net_profit values must be counted as wins.
    """

    analyzer = make_engine()

    result = analyzer.analyze(
        make_trades()
    )

    assert result.winning_trades == 3


def test_losing_trade_count_is_correct():
    """
    Non-positive net_profit values must be counted as losses.
    """

    analyzer = make_engine()

    result = analyzer.analyze(
        make_trades()
    )

    assert result.losing_trades == 2


def test_winning_and_losing_trades_sum_to_total():
    """
    Wins + losses must equal total trades.
    """

    analyzer = make_engine()

    result = analyzer.analyze(
        make_trades()
    )

    assert (
        result.winning_trades
        + result.losing_trades
        == result.total_trades
    )


# =========================================================
# WIN RATE
# =========================================================

def test_win_rate_is_correct():
    """
    3 wins out of 5 trades = 60%.
    """

    analyzer = make_engine()

    result = analyzer.analyze(
        make_trades()
    )

    assert result.win_rate == pytest.approx(
        60.0,
        abs=1e-6,
    )


def test_zero_trade_win_rate_is_zero():
    """
    Zero trades must produce 0% win rate.
    """

    analyzer = make_engine()

    result = analyzer.analyze(
        make_empty_trades()
    )

    assert result.win_rate == 0.0


def test_all_winning_trades_have_100_percent_win_rate():
    """
    All winning trades must produce 100% win rate.
    """

    analyzer = make_engine()

    result = analyzer.analyze(
        make_winning_trades()
    )

    assert result.win_rate == pytest.approx(
        100.0,
        abs=1e-6,
    )


def test_all_losing_trades_have_zero_percent_win_rate():
    """
    All losing trades must produce 0% win rate.
    """

    analyzer = make_engine()

    result = analyzer.analyze(
        make_losing_trades()
    )

    assert result.win_rate == pytest.approx(
        0.0,
        abs=1e-6,
    )


# =========================================================
# NET PROFIT
# =========================================================

def test_net_profit_is_sum_of_trade_profits():
    """
    Net profit must equal sum of net_profit column.
    """

    analyzer = make_engine()

    df = make_trades()

    result = analyzer.analyze(df)

    expected = df["net_profit"].sum()

    assert result.net_profit == pytest.approx(
        expected,
        abs=1e-6,
    )


def test_final_balance_matches_initial_plus_profit():
    """
    Final balance must equal:

        initial_balance + net_profit
    """

    analyzer = make_engine()

    result = analyzer.analyze(
        make_trades()
    )

    expected = (
        result.initial_balance
        + result.net_profit
    )

    assert result.final_balance == pytest.approx(
        expected,
        abs=1e-6,
    )


def test_zero_trade_profit_is_zero():
    """
    Zero trades must produce zero net profit.
    """

    analyzer = make_engine()

    result = analyzer.analyze(
        make_empty_trades()
    )

    assert result.net_profit == pytest.approx(
        0.0,
        abs=1e-6,
    )


# =========================================================
# AVERAGE PROFIT
# =========================================================

def test_average_profit_is_correct():
    """
    Average profit must equal net profit / trade count.
    """

    analyzer = make_engine()

    result = analyzer.analyze(
        make_trades()
    )

    expected = (
        result.net_profit
        / result.total_trades
    )

    assert result.average_profit == pytest.approx(
        expected,
        abs=1e-6,
    )


def test_average_profit_is_zero_without_trades():
    """
    No trades must produce zero average profit.
    """

    analyzer = make_engine()

    result = analyzer.analyze(
        make_empty_trades()
    )

    assert result.average_profit == 0.0


# =========================================================
# PROFIT FACTOR
# =========================================================

def test_profit_factor_is_correct():
    """
    Profit factor:

        gross profits / absolute gross losses
    """

    analyzer = make_engine()

    result = analyzer.analyze(
        make_trades()
    )

    gross_wins = (
        1.90
        + 3.90
        + 0.90
    )

    gross_losses = (
        2.10
        + 2.10
    )

    expected = (
        gross_wins
        / gross_losses
    )

    assert result.profit_factor == pytest.approx(
        expected,
        abs=1e-6,
    )


def test_profit_factor_is_zero_without_trades():
    """
    Zero trades must not produce division by zero.
    """

    analyzer = make_engine()

    result = analyzer.analyze(
        make_empty_trades()
    )

    assert result.profit_factor == 0.0


def test_profit_factor_for_all_winning_trades_is_infinite():
    """
    No losses means infinite profit factor.
    """

    analyzer = make_engine()

    result = analyzer.analyze(
        make_winning_trades()
    )

    assert result.profit_factor == float(
        "inf"
    )


def test_profit_factor_for_all_losing_trades_is_zero():
    """
    No winning trades means zero profit factor.
    """

    analyzer = make_engine()

    result = analyzer.analyze(
        make_losing_trades()
    )

    assert result.profit_factor == 0.0


# =========================================================
# MAXIMUM DRAWDOWN
# =========================================================

def test_max_drawdown_is_non_negative():
    """
    Maximum drawdown must never be negative.
    """

    analyzer = make_engine()

    result = analyzer.analyze(
        make_trades()
    )

    assert result.max_drawdown >= 0.0


def test_max_drawdown_is_zero_without_trades():
    """
    Zero trades must produce zero drawdown.
    """

    analyzer = make_engine()

    result = analyzer.analyze(
        make_empty_trades()
    )

    assert result.max_drawdown == 0.0


# =========================================================
# SOURCE DATAFRAME PROTECTION
# =========================================================

def test_analysis_does_not_modify_source_dataframe():
    """
    Analyzer must not mutate the input DataFrame.
    """

    analyzer = make_engine()

    df = make_trades()

    original = df.copy(
        deep=True
    )

    analyzer.analyze(df)

    pd.testing.assert_frame_equal(
        df,
        original,
    )


def test_source_dataframe_index_is_unchanged():
    """
    Original DataFrame index must remain unchanged.
    """

    analyzer = make_engine()

    df = make_trades()

    original_index = df.index.copy()

    analyzer.analyze(df)

    assert df.index.equals(
        original_index
    )


def test_source_dataframe_columns_are_unchanged():
    """
    Original DataFrame columns must remain unchanged.
    """

    analyzer = make_engine()

    df = make_trades()

    original_columns = list(
        df.columns
    )

    analyzer.analyze(df)

    assert list(
        df.columns
    ) == original_columns


# =========================================================
# REPEATED ANALYSIS
# =========================================================

def test_repeated_analysis_is_deterministic():
    """
    Analyzing the same trades repeatedly must return
    identical statistics.
    """

    analyzer = make_engine()

    df = make_trades()

    first = analyzer.analyze(df)

    second = analyzer.analyze(df)

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


# =========================================================
# RESULT BOUNDARIES
# =========================================================

def test_win_rate_is_inside_valid_range():
    """
    Win rate must always be [0, 100].
    """

    analyzer = make_engine()

    result = analyzer.analyze(
        make_trades()
    )

    assert 0.0 <= result.win_rate <= 100.0


def test_trade_statistics_are_consistent():
    """
    Performance statistics must remain mathematically valid.
    """

    analyzer = make_engine()

    result = analyzer.analyze(
        make_trades()
    )

    assert result.total_trades >= 0

    assert result.winning_trades >= 0

    assert result.losing_trades >= 0

    assert (
        result.winning_trades
        <= result.total_trades
    )

    assert (
        result.losing_trades
        <= result.total_trades
    )


# =========================================================
# REPORT
# =========================================================

def test_print_report_produces_expected_output(
    capsys,
):
    """
    Report must contain the major performance fields.
    """

    analyzer = make_engine()

    result = analyzer.analyze(
        make_trades()
    )

    analyzer.print_report(
        result
    )

    output = capsys.readouterr().out

    assert "QUANTAI PERFORMANCE REPORT" in output

    assert "Initial Balance" in output

    assert "Final Balance" in output

    assert "Net Profit" in output

    assert "Total Trades" in output

    assert "Winning Trades" in output

    assert "Losing Trades" in output

    assert "Win Rate" in output

    assert "Profit Factor" in output

    assert "Maximum Drawdown" in output


def test_print_report_handles_zero_trade_result(
    capsys,
):
    """
    Report must also work for zero-trade analysis.
    """

    analyzer = make_engine()

    result = analyzer.analyze(
        make_empty_trades()
    )

    analyzer.print_report(
        result
    )

    output = capsys.readouterr().out

    assert "QUANTAI PERFORMANCE REPORT" in output

    assert "Total Trades" in output

    assert "0" in output


# =========================================================
# RESULT STRUCTURE
# =========================================================

def test_performance_result_contains_expected_fields():
    """
    PerformanceResult must expose the expected statistics.
    """

    analyzer = make_engine()

    result = analyzer.analyze(
        make_trades()
    )

    assert hasattr(
        result,
        "initial_balance",
    )

    assert hasattr(
        result,
        "final_balance",
    )

    assert hasattr(
        result,
        "net_profit",
    )

    assert hasattr(
        result,
        "total_trades",
    )

    assert hasattr(
        result,
        "winning_trades",
    )

    assert hasattr(
        result,
        "losing_trades",
    )

    assert hasattr(
        result,
        "win_rate",
    )

    assert hasattr(
        result,
        "average_profit",
    )

    assert hasattr(
        result,
        "profit_factor",
    )

    assert hasattr(
        result,
        "max_drawdown",
    )


# =========================================================
# END
# =========================================================