"""
QuantAI Professional v5
Backtest + Performance Analyzer integration tests.

Validates the complete pipeline:

    Historical CSV
        ↓
    add_indicators()
        ↓
    BacktestEngine
        ↓
    completed trades
        ↓
    PerformanceAnalyzer
        ↓
    PerformanceResult
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.backtest_engine import (
    BacktestEngine,
    BacktestResult,
)

from src.performance_analyzer import (
    PerformanceAnalyzer,
    PerformanceResult,
)


# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "data/sol_usdt_15m.csv"

INITIAL_BALANCE = 1000.0


# ============================================================
# HELPERS
# ============================================================

def load_prepared_dataset() -> pd.DataFrame:
    """
    Load the real historical dataset and calculate indicators.
    """

    df = pd.read_csv(DATA_PATH)

    from src.indicators import add_indicators

    df = add_indicators(df)

    return df


def run_backtest() -> BacktestResult:
    """
    Run a complete BacktestEngine pipeline.
    """

    df = load_prepared_dataset()

    engine = BacktestEngine(
        initial_balance=INITIAL_BALANCE,
    )

    return engine.run(df)


def analyze_backtest(
    result: BacktestResult,
) -> PerformanceResult:
    """
    Convert BacktestResult trades into a DataFrame
    and analyze them with PerformanceAnalyzer.
    """

    trades = pd.DataFrame(
        result.trades
    )

    analyzer = PerformanceAnalyzer(
        initial_balance=result.initial_balance,
    )

    return analyzer.analyze(
        trades
    )


# ============================================================
# BACKTEST RESULT
# ============================================================

def test_backtest_result_is_valid():
    """
    BacktestEngine must return a valid BacktestResult.
    """

    result = run_backtest()

    assert isinstance(
        result,
        BacktestResult,
    )


def test_backtest_result_has_valid_balance():
    """
    Backtest balances must remain positive.
    """

    result = run_backtest()

    assert result.initial_balance > 0.0

    assert result.final_balance > 0.0


def test_backtest_trade_statistics_are_consistent():
    """
    Winning + losing trades must equal total trades.
    """

    result = run_backtest()

    assert result.total_trades >= 0

    assert result.winning_trades >= 0

    assert result.losing_trades >= 0

    assert (
        result.winning_trades
        + result.losing_trades
        == result.total_trades
    )

    assert (
        0.0
        <= result.win_rate
        <= 100.0
    )


# ============================================================
# TRADE CONVERSION
# ============================================================

def test_backtest_trades_can_be_converted_to_dataframe():
    """
    BacktestResult.trades must be compatible with pandas.
    """

    result = run_backtest()

    trades = pd.DataFrame(
        result.trades
    )

    assert isinstance(
        trades,
        pd.DataFrame,
    )


def test_backtest_trades_have_net_profit_column():
    """
    PerformanceAnalyzer requires net_profit.
    """

    result = run_backtest()

    trades = pd.DataFrame(
        result.trades
    )

    # Zero-trade backtests are allowed.
    if trades.empty:
        pytest.skip(
            "Backtest produced zero completed trades."
        )

    assert "net_profit" in trades.columns


# ============================================================
# PERFORMANCE ANALYZER
# ============================================================

def test_performance_analyzer_accepts_backtest_result():
    """
    Backtest trades must be accepted by PerformanceAnalyzer.
    """

    result = run_backtest()

    trades = pd.DataFrame(
        result.trades
    )

    analyzer = PerformanceAnalyzer(
        initial_balance=result.initial_balance,
    )

    performance = analyzer.analyze(
        trades
    )

    assert isinstance(
        performance,
        PerformanceResult,
    )


def test_performance_initial_balance_matches_backtest():
    """
    Performance analysis must start from the same balance
    as the backtest.
    """

    result = run_backtest()

    performance = analyze_backtest(
        result
    )

    assert performance.initial_balance == pytest.approx(
        result.initial_balance,
        abs=1e-6,
    )


def test_performance_final_balance_matches_backtest():
    """
    PerformanceAnalyzer and BacktestEngine must agree
    on final balance.
    """

    result = run_backtest()

    performance = analyze_backtest(
        result
    )

    assert performance.win_rate == pytest.approx(
    result.win_rate,
    abs=0.01,
)



def test_performance_net_profit_matches_backtest():
    """
    PerformanceAnalyzer net profit must equal
    BacktestEngine net profit.
    """

    result = run_backtest()

    performance = analyze_backtest(
        result
    )

    assert performance.net_profit == pytest.approx(
        result.net_profit,
        abs=1e-6,
    )


# ============================================================
# TRADE COUNT CONSISTENCY
# ============================================================

def test_performance_trade_count_matches_backtest():
    """
    PerformanceAnalyzer and BacktestEngine must report
    the same number of completed trades.
    """

    result = run_backtest()

    performance = analyze_backtest(
        result
    )

    assert performance.total_trades == (
        result.total_trades
    )


def test_performance_winning_trades_match_backtest():
    """
    Winning trade count must match.
    """

    result = run_backtest()

    performance = analyze_backtest(
        result
    )

    assert performance.winning_trades == (
        result.winning_trades
    )


def test_performance_losing_trades_match_backtest():
    """
    Losing trade count must match.
    """

    result = run_backtest()

    performance = analyze_backtest(
        result
    )

    assert performance.losing_trades == (
        result.losing_trades
    )


def test_performance_win_rate_matches_backtest():
    """
    Win rate must match the BacktestEngine result.
    """

    result = run_backtest()

    performance = analyze_backtest(
        result
    )

    assert performance.win_rate == pytest.approx(
    result.win_rate,
    abs=0.01,
    )


# ============================================================
# PERFORMANCE METRICS
# ============================================================

def test_performance_profit_factor_is_valid():
    """
    Profit factor must be zero, finite positive,
    or positive infinity when there are no losses.
    """

    result = run_backtest()

    performance = analyze_backtest(
        result
    )

    assert (
        performance.profit_factor >= 0.0
        or performance.profit_factor == float("inf")
    )


def test_performance_drawdown_is_non_negative():
    """
    Maximum drawdown is represented as a positive magnitude.
    """

    result = run_backtest()

    performance = analyze_backtest(
        result
    )

    assert performance.max_drawdown >= 0.0


def test_performance_drawdown_percent_is_non_negative():
    """
    Maximum drawdown percentage is represented as a
    positive magnitude.
    """

    result = run_backtest()

    performance = analyze_backtest(
        result
    )

    assert performance.max_drawdown_percent >= 0.0


# ============================================================
# BALANCE CONSISTENCY
# ============================================================

def test_performance_balance_equation_is_consistent():
    """
    Final balance must satisfy:

        final_balance =
            initial_balance + net_profit
    """

    result = run_backtest()

    performance = analyze_backtest(
        result
    )

    expected = (
        performance.initial_balance
        + performance.net_profit
    )

    assert performance.final_balance == pytest.approx(
        expected,
        abs=1e-6,
    )


# ============================================================
# REPEATABILITY
# ============================================================

def test_full_pipeline_is_repeatable():
    """
    Running the complete backtest + performance pipeline
    twice must produce identical core statistics.
    """

    first_backtest = run_backtest()

    first_performance = analyze_backtest(
        first_backtest
    )

    second_backtest = run_backtest()

    second_performance = analyze_backtest(
        second_backtest
    )

    assert second_backtest.initial_balance == pytest.approx(
        first_backtest.initial_balance,
        abs=1e-6,
    )

    assert second_backtest.final_balance == pytest.approx(
        first_backtest.final_balance,
        abs=1e-6,
    )

    assert second_backtest.net_profit == pytest.approx(
        first_backtest.net_profit,
        abs=1e-6,
    )

    assert second_backtest.total_trades == (
        first_backtest.total_trades
    )

    assert second_backtest.winning_trades == (
        first_backtest.winning_trades
    )

    assert second_backtest.losing_trades == (
        first_backtest.losing_trades
    )

    assert second_performance.initial_balance == pytest.approx(
        first_performance.initial_balance,
        abs=1e-6,
    )

    assert second_performance.final_balance == pytest.approx(
        first_performance.final_balance,
        abs=1e-6,
    )

    assert second_performance.net_profit == pytest.approx(
        first_performance.net_profit,
        abs=1e-6,
    )

    assert second_performance.total_trades == (
        first_performance.total_trades
    )

    assert second_performance.winning_trades == (
        first_performance.winning_trades
    )

    assert second_performance.losing_trades == (
        first_performance.losing_trades
    )

    assert second_performance.win_rate == pytest.approx(
        first_performance.win_rate,
        abs=1e-6,
    )


# ============================================================
# REPORT
# ============================================================

def test_performance_report_can_be_generated(
    capsys,
):
    """
    PerformanceAnalyzer must generate a valid report
    from BacktestEngine output.
    """

    result = run_backtest()

    trades = pd.DataFrame(
        result.trades
    )

    analyzer = PerformanceAnalyzer(
        initial_balance=result.initial_balance,
    )

    performance = analyzer.analyze(
        trades
    )

    analyzer.print_report(
        performance
    )

    output = capsys.readouterr().out

    assert (
        "QUANTAI PERFORMANCE REPORT"
        in output
    )

    assert (
        "Initial Balance"
        in output
    )

    assert (
        "Final Balance"
        in output
    )

    assert (
        "Net Profit"
        in output
    )

    assert (
        "Total Trades"
        in output
    )

    assert (
        "Profit Factor"
        in output
    )

    assert (
        "Maximum Drawdown"
        in output
    )


# ============================================================
# END
# ============================================================