"""
=========================================================
QuantAI Professional v5
Backtest Engine Validation Tests
=========================================================

Validates:

    - prepared data validation
    - minimum dataset size
    - required columns
    - numeric columns
    - NaN protection
    - initial balance
    - final balance
    - net profit
    - trade statistics
    - result persistence
    - fresh TradeEngine per run
    - repeated backtest isolation
    - result object integrity
    - report validation
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.backtest_engine import (
    MINIMUM_ROWS,
    REQUIRED_COLUMNS,
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

def make_valid_dataframe(
    rows: int = MINIMUM_ROWS,
) -> pd.DataFrame:
    """
    Create a valid prepared OHLCV DataFrame compatible with
    BacktestEngine + TradeEngine + Strategy.

    The factory provides all columns directly required by
    Strategy while keeping the dataset deterministic.
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

    high = [
        price + 1.0
        for price in close
    ]

    low = [
        price - 1.0
        for price in close
    ]

    volume = [
        1000.0
        for _ in range(rows)
    ]

    return pd.DataFrame(
        {
            "timestamp": timestamps,

            "open": close,

            "high": high,

            "low": low,

            "close": close,

            "volume": volume,

            "atr": [1.0] * rows,

            # Strategy trend indicators
            "ema_fast": [
                price + 0.20
                for price in close
            ],

            "ema_slow": [
                price + 0.10
                for price in close
            ],

            "ema_trend": close,

            "adx": [25.0] * rows,

            # Strategy momentum indicators
            "rsi": [55.0] * rows,

            "macd": [1.0] * rows,

            "macd_signal": [0.5] * rows,

            "macd_hist": [0.5] * rows,

            # Strategy volume indicator
            "volume_sma20": [1000.0] * rows,
        }
    )
    """
    Create a valid prepared OHLCV DataFrame.

    The data intentionally contains no strategy-specific
    indicators beyond the columns required by BacktestEngine.
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
            "volume": [1000.0] * rows,
            "atr": [1.0] * rows,
        }
    )


# =========================================================
# VALIDATION
# =========================================================

def test_valid_dataframe_passes_validation():
    """Valid prepared data must pass validation."""

    engine = BacktestEngine(
        initial_balance=INITIAL_BALANCE,
    )

    df = make_valid_dataframe()

    engine.validate_data(df)


def test_non_dataframe_is_rejected():
    """BacktestEngine must reject non-DataFrame input."""

    engine = BacktestEngine()

    with pytest.raises(TypeError):
        engine.validate_data(
            [1, 2, 3]
        )


def test_empty_dataframe_is_rejected():
    """Empty DataFrame must be rejected."""

    engine = BacktestEngine()

    df = pd.DataFrame()

    with pytest.raises(ValueError):
        engine.validate_data(df)


def test_missing_required_columns_are_rejected():
    """Missing required columns must raise ValueError."""

    engine = BacktestEngine()

    df = make_valid_dataframe()

    df = df.drop(
        columns=["atr"]
    )

    with pytest.raises(ValueError):
        engine.validate_data(df)


def test_all_required_columns_are_defined():
    """Required column configuration must contain core OHLCV data."""

    expected = {
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "atr",
    }

    assert expected.issubset(
        REQUIRED_COLUMNS
    )


def test_dataset_below_minimum_rows_is_rejected():
    """Backtest must reject datasets below minimum size."""

    engine = BacktestEngine()

    df = make_valid_dataframe(
        MINIMUM_ROWS - 1
    )

    with pytest.raises(ValueError):
        engine.validate_data(df)


def test_minimum_dataset_size_is_accepted():
    """Dataset with exactly minimum rows must pass."""

    engine = BacktestEngine()

    df = make_valid_dataframe(
        MINIMUM_ROWS
    )

    engine.validate_data(df)


# =========================================================
# NUMERIC VALIDATION
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
def test_required_numeric_column_must_be_numeric(
    column,
):
    """Required market columns must contain numeric values."""

    engine = BacktestEngine()

    df = make_valid_dataframe()

    df[column] = "invalid"

    with pytest.raises(TypeError):
        engine.validate_data(df)


# =========================================================
# NaN VALIDATION
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
def test_nan_in_required_numeric_column_is_rejected(
    column,
):
    """NaN values in required numeric columns must be rejected."""

    engine = BacktestEngine()

    df = make_valid_dataframe()

    df.loc[
        MINIMUM_ROWS - 1,
        column,
    ] = float("nan")

    with pytest.raises(ValueError):
        engine.validate_data(df)


# =========================================================
# INITIAL BALANCE
# =========================================================

def test_initial_balance_is_stored():
    """Configured initial balance must be stored."""

    engine = BacktestEngine(
        initial_balance=INITIAL_BALANCE,
    )

    assert engine.initial_balance == pytest.approx(
        INITIAL_BALANCE
    )


def test_non_positive_initial_balance_is_rejected():
    """Initial balance must be positive."""

    with pytest.raises(ValueError):
        BacktestEngine(
            initial_balance=0
        )

    with pytest.raises(ValueError):
        BacktestEngine(
            initial_balance=-100
        )


def test_default_initial_balance_is_supported():
    """BacktestEngine must support configuration without explicit balance."""

    engine = BacktestEngine()

    assert engine.initial_balance is None


# =========================================================
# RESULT STATE
# =========================================================

def test_result_is_none_before_first_run():
    """No result must exist before a backtest is executed."""

    engine = BacktestEngine()

    assert engine.result is None


# =========================================================
# RUN
# =========================================================

def test_run_returns_backtest_result():
    """run() must return BacktestResult."""

    engine = BacktestEngine(
        initial_balance=INITIAL_BALANCE,
    )

    df = make_valid_dataframe()

    result = engine.run(df)

    assert isinstance(
        result,
        BacktestResult,
    )


def test_run_result_initial_balance_is_correct():
    """Backtest result must preserve configured initial balance."""

    engine = BacktestEngine(
        initial_balance=INITIAL_BALANCE,
    )

    df = make_valid_dataframe()

    result = engine.run(df)

    assert result.initial_balance == pytest.approx(
        INITIAL_BALANCE
    )


def test_run_result_is_persisted():
    """Latest result must be available through engine.result."""

    engine = BacktestEngine(
        initial_balance=INITIAL_BALANCE,
    )

    df = make_valid_dataframe()

    result = engine.run(df)

    assert engine.result is result


# =========================================================
# RESULT MATHEMATICS
# =========================================================

def test_net_profit_matches_balance_difference():
    """
    Net profit must equal:

        final_balance - initial_balance

    when TradeEngine does not provide another authoritative
    net-profit value.
    """

    engine = BacktestEngine(
        initial_balance=INITIAL_BALANCE,
    )

    df = make_valid_dataframe()

    result = engine.run(df)

    expected = (
        result.final_balance
        - result.initial_balance
    )

    assert result.net_profit == pytest.approx(
        expected,
        abs=1e-6,
    )


def test_trade_statistics_are_internally_consistent():
    """
    Winning + losing trades must equal total trades.
    """

    engine = BacktestEngine(
        initial_balance=INITIAL_BALANCE,
    )

    df = make_valid_dataframe()

    result = engine.run(df)

    assert (
        result.winning_trades
        + result.losing_trades
        == result.total_trades
    )


def test_win_rate_matches_trade_statistics():
    """Win rate must match winning/total trade count."""

    engine = BacktestEngine(
        initial_balance=INITIAL_BALANCE,
    )

    df = make_valid_dataframe()

    result = engine.run(df)

    if result.total_trades == 0:

        assert result.win_rate == 0.0

    else:

        expected = round(
            (
                result.winning_trades
                / result.total_trades
            )
            * 100.0,
            2,
        )

        assert result.win_rate == pytest.approx(
            expected
        )


# =========================================================
# TRADES
# =========================================================

def test_trades_collection_is_available():
    """Backtest result must expose trades."""

    engine = BacktestEngine(
        initial_balance=INITIAL_BALANCE,
    )

    df = make_valid_dataframe()

    result = engine.run(df)

    assert result.trades is not None


def test_trade_count_matches_trades_collection_when_possible():
    """
    If trades are returned as a list, its length should match
    total trade count.
    """

    engine = BacktestEngine(
        initial_balance=INITIAL_BALANCE,
    )

    df = make_valid_dataframe()

    result = engine.run(df)

    if isinstance(
        result.trades,
        list,
    ):

        assert len(
            result.trades
        ) == result.total_trades


# =========================================================
# STATE ISOLATION
# =========================================================

def test_each_run_creates_fresh_trade_engine():
    """
    Every BacktestEngine.run() must create a fresh TradeEngine.
    """

    engine = BacktestEngine(
        initial_balance=INITIAL_BALANCE,
    )

    df = make_valid_dataframe()

    first_trade_engine = engine.trade_engine

    engine.run(df)

    second_trade_engine = engine.trade_engine

    assert first_trade_engine is not second_trade_engine


def test_repeated_runs_do_not_accumulate_trades():
    """
    Running the same dataset twice must not duplicate historical
    trades from the previous run.
    """

    engine = BacktestEngine(
        initial_balance=INITIAL_BALANCE,
    )

    df = make_valid_dataframe()

    first_result = engine.run(df)

    first_trade_count = (
        first_result.total_trades
    )

    second_result = engine.run(df)

    second_trade_count = (
        second_result.total_trades
    )

    assert second_trade_count == first_trade_count


def test_repeated_runs_start_from_same_initial_balance():
    """
    A second run must not inherit the balance from the first run.
    """

    engine = BacktestEngine(
        initial_balance=INITIAL_BALANCE,
    )

    df = make_valid_dataframe()

    first_result = engine.run(df)

    second_result = engine.run(df)

    assert (
        first_result.initial_balance
        == second_result.initial_balance
        == INITIAL_BALANCE
    )


# =========================================================
# DATAFRAME PROTECTION
# =========================================================

def test_run_does_not_modify_original_dataframe_index():
    """
    BacktestEngine should operate on a copied/reset DataFrame
    rather than mutating the caller's index.
    """

    engine = BacktestEngine(
        initial_balance=INITIAL_BALANCE,
    )

    df = make_valid_dataframe()

    original_index = df.index.copy()

    engine.run(df)

    assert df.index.equals(
        original_index
    )


def test_run_does_not_remove_original_columns():
    """Backtest must not remove caller DataFrame columns."""

    engine = BacktestEngine(
        initial_balance=INITIAL_BALANCE,
    )

    df = make_valid_dataframe()

    original_columns = list(
        df.columns
    )

    engine.run(df)

    assert list(
        df.columns
    ) == original_columns


# =========================================================
# REPORT
# =========================================================

def test_print_report_accepts_valid_result(
    capsys,
):
    """print_report() must accept a valid BacktestResult."""

    engine = BacktestEngine(
        initial_balance=INITIAL_BALANCE,
    )

    df = make_valid_dataframe()

    result = engine.run(df)

    engine.print_report(
        result
    )

    output = capsys.readouterr().out

    assert (
        "QUANTAI BACKTEST REPORT"
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


def test_print_report_rejects_invalid_result():
    """print_report() must reject objects that are not BacktestResult."""

    with pytest.raises(TypeError):
        BacktestEngine.print_report(
            None
        )


# =========================================================
# RESULT OBJECT
# =========================================================

def test_backtest_result_contains_expected_fields():
    """BacktestResult must contain all required statistics."""

    result = BacktestResult(
        initial_balance=1000.0,
        final_balance=1010.0,
        net_profit=10.0,
        total_trades=2,
        winning_trades=1,
        losing_trades=1,
        win_rate=50.0,
        trades=[],
    )

    assert result.initial_balance == 1000.0
    assert result.final_balance == 1010.0
    assert result.net_profit == 10.0
    assert result.total_trades == 2
    assert result.winning_trades == 1
    assert result.losing_trades == 1
    assert result.win_rate == 50.0
    assert result.trades == []


# =========================================================
# END
# =========================================================