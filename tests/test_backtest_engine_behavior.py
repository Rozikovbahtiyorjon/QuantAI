"""
QuantAI Professional v5
Backtest Engine behavior tests.
"""

import pandas as pd
import pytest

from src.indicators import add_indicators

from src.backtest_engine import (
    BacktestEngine,
    BacktestResult,
    MINIMUM_ROWS,
)


def make_dataframe(rows: int = MINIMUM_ROWS) -> pd.DataFrame:
    """Create prepared OHLCV data for BacktestEngine."""

    timestamps = pd.date_range(
        "2026-01-01",
        periods=rows,
        freq="15min",
    )

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100.0] * rows,
            "high": [101.0] * rows,
            "low": [99.0] * rows,
            "close": [100.0] * rows,
            "volume": [1000.0] * rows,
        }
    )

    df = add_indicators(df)

    return df


def test_backtest_engine_returns_result():
    engine = BacktestEngine()
    df = make_dataframe()

    result = engine.run(df)

    assert isinstance(result, BacktestResult)


def test_backtest_result_has_positive_balance():
    engine = BacktestEngine()
    df = make_dataframe()

    result = engine.run(df)

    assert result.initial_balance > 0
    assert result.final_balance > 0


def test_backtest_trade_statistics_are_consistent():
    engine = BacktestEngine()
    df = make_dataframe()

    result = engine.run(df)

    assert result.total_trades >= 0
    assert result.winning_trades >= 0
    assert result.losing_trades >= 0

    assert (
        result.winning_trades
        + result.losing_trades
        == result.total_trades
    )


def test_backtest_win_rate_is_valid():
    engine = BacktestEngine()
    df = make_dataframe()

    result = engine.run(df)

    assert 0.0 <= result.win_rate <= 100.0


def test_backtest_real_dataset_returns_result():
    df = pd.read_csv(
        "data/sol_usdt_15m.csv"
    )

    from src.indicators import add_indicators

    df = add_indicators(df)

    engine = BacktestEngine()
    result = engine.run(df)

    assert isinstance(result, BacktestResult)
    assert result.initial_balance > 0
    assert result.final_balance > 0


def test_backtest_real_dataset_statistics_are_consistent():
    df = pd.read_csv(
        "data/sol_usdt_15m.csv"
    )

    from src.indicators import add_indicators

    df = add_indicators(df)

    engine = BacktestEngine()
    result = engine.run(df)

    assert (
        result.winning_trades
        + result.losing_trades
        == result.total_trades
    )

    assert 0.0 <= result.win_rate <= 100.0


def test_backtest_net_profit_matches_balances():
    engine = BacktestEngine(
        initial_balance=1000.0
    )

    df = make_dataframe()
    result = engine.run(df)

    expected_profit = (
        result.final_balance
        - result.initial_balance
    )

    assert result.net_profit == pytest.approx(
        expected_profit
    )


def test_backtest_does_not_produce_negative_balance():
    engine = BacktestEngine(
        initial_balance=1000.0
    )

    df = make_dataframe()
    result = engine.run(df)

    assert result.final_balance >= 0


def test_backtest_is_repeatable():
    df = pd.read_csv(
        "data/sol_usdt_15m.csv"
    )

    from src.indicators import add_indicators

    df = add_indicators(df)

    engine1 = BacktestEngine()
    result1 = engine1.run(df)

    engine2 = BacktestEngine()
    result2 = engine2.run(df)

    assert result1.total_trades == result2.total_trades
    assert result1.winning_trades == result2.winning_trades
    assert result1.losing_trades == result2.losing_trades

    assert result1.final_balance == pytest.approx(
        result2.final_balance
    )