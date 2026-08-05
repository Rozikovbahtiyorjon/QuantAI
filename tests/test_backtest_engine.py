"""
QuantAI Professional v5
Backtest Engine pytest tests.
"""

import pandas as pd
import pytest

from src.backtest_engine import (
    BacktestEngine,
    BacktestResult,
    MINIMUM_ROWS,
)


# ============================================================
# HELPERS
# ============================================================

def make_valid_dataframe(rows: int = MINIMUM_ROWS) -> pd.DataFrame:
    """Create minimal valid prepared OHLCV data."""

    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01",
                periods=rows,
                freq="15min",
            ),
            "open": [100.0] * rows,
            "high": [101.0] * rows,
            "low": [99.0] * rows,
            "close": [100.0] * rows,
            "volume": [1000.0] * rows,
            "atr": [1.0] * rows,
        }
    )


# ============================================================
# VALIDATION TESTS
# ============================================================

def test_validate_data_accepts_valid_dataframe():
    """Valid prepared data must pass validation."""

    engine = BacktestEngine()

    df = make_valid_dataframe()

    engine.validate_data(df)


def test_validate_data_rejects_non_dataframe():
    """BacktestEngine must require pandas DataFrame."""

    engine = BacktestEngine()

    with pytest.raises(TypeError):
        engine.validate_data([])


def test_validate_data_rejects_empty_dataframe():
    """Empty DataFrame must be rejected."""

    engine = BacktestEngine()

    df = pd.DataFrame()

    with pytest.raises(ValueError, match="empty"):
        engine.validate_data(df)


def test_validate_data_rejects_missing_columns():
    """Missing required columns must be detected."""

    engine = BacktestEngine()

    df = make_valid_dataframe()

    df = df.drop(columns=["atr"])

    with pytest.raises(ValueError, match="Missing required columns"):
        engine.validate_data(df)


def test_validate_data_rejects_too_few_rows():
    """Backtest must require at least MINIMUM_ROWS."""

    engine = BacktestEngine()

    df = make_valid_dataframe(MINIMUM_ROWS - 1)

    with pytest.raises(ValueError, match="at least"):
        engine.validate_data(df)


def test_validate_data_rejects_non_numeric_column():
    """Required market columns must be numeric."""

    engine = BacktestEngine()

    df = make_valid_dataframe()

    df["close"] = "invalid"

    with pytest.raises(TypeError, match="close"):
        engine.validate_data(df)


def test_validate_data_rejects_nan_values():
    """NaN values in required numeric columns must be rejected."""

    engine = BacktestEngine()

    df = make_valid_dataframe()

    df.loc[0, "close"] = float("nan")

    with pytest.raises(
        ValueError,
        match="NaN",
    ):
        engine.validate_data(df)


# ============================================================
# BALANCE TEST
# ============================================================

def test_initial_balance_is_applied():
    """BacktestEngine must apply custom initial balance."""

    engine = BacktestEngine(
        initial_balance=5000.0
    )

    assert engine.trade_engine.balance == 5000.0
    assert engine.trade_engine.equity == 5000.0


# ============================================================
# RESULT TEST
# ============================================================

def test_backtest_result_structure():
    """BacktestResult must contain expected statistics."""

    result = BacktestResult(
        initial_balance=1000.0,
        final_balance=1024.52,
        net_profit=24.52,
        total_trades=20,
        winning_trades=6,
        losing_trades=14,
        win_rate=30.0,
        trades=[],
    )

    assert result.initial_balance == 1000.0
    assert result.final_balance == 1024.52
    assert result.net_profit == 24.52
    assert result.total_trades == 20
    assert result.winning_trades == 6
    assert result.losing_trades == 14
    assert result.win_rate == 30.0
    assert result.trades == []


# ============================================================
# REAL INTEGRATION TEST
# ============================================================

def test_backtest_engine_real_dataset():
    """
    Run BacktestEngine against the real SOL/USDT historical dataset.
    """

    df = pd.read_csv(
        "data/sol_usdt_15m.csv"
    )

    from src.indicators import add_indicators

    df = add_indicators(df)

    engine = BacktestEngine()

    result = engine.run(df)

    assert isinstance(
        result,
        BacktestResult,
    )

    assert result.initial_balance > 0

    assert result.final_balance > 0

    assert result.total_trades >= 0

    assert result.winning_trades >= 0

    assert result.losing_trades >= 0

    assert (
        result.winning_trades
        + result.losing_trades
        == result.total_trades
    )

    assert 0.0 <= result.win_rate <= 100.0