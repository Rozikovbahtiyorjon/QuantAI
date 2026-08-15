"""
====================================================
QuantAI Professional v3.0
Backtest Engine Determinism Tests
====================================================

Purpose
-------
Verify that BacktestEngine produces deterministic results
when executed multiple times on the same deterministic
input dataset.

Covered
-------
1. Same input -> same result
2. Trade history is deterministic
3. Trade order is deterministic
4. Entry / exit prices are deterministic
5. PnL is deterministic
6. Close reasons are deterministic
7. BacktestResult scalar fields are deterministic
8. Input DataFrame is not modified
9. Independent engine instances produce identical results
10. Repeated runs remain stable

Important
---------
The Strategy layer requires prepared indicator columns.
The deterministic fixture therefore contains the strategy
columns expected by the current QuantAI pipeline.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from config.settings import INITIAL_BALANCE

from src.backtest_engine import (
    BacktestEngine,
    BacktestResult,
)


# ============================================================
# Deterministic Dataset
# ============================================================


def make_deterministic_dataframe(
    rows: int = 360,
) -> pd.DataFrame:
    """
    Build a deterministic OHLCV DataFrame containing all
    columns required by the current Strategy pipeline.

    No random values are used.
    """

    if rows < 300:
        rows = 300

    index = np.arange(rows, dtype=float)

    # --------------------------------------------------------
    # Deterministic price model
    # --------------------------------------------------------

    close = (
        100.0
        + index * 0.02
        + np.sin(index / 9.0) * 0.5
    )

    open_price = close - 0.05

    high = close + 0.20

    low = close - 0.20

    volume = (
        1000.0
        + (index % 20) * 10.0
    )

    # --------------------------------------------------------
    # Deterministic indicators
    # --------------------------------------------------------

    ema_fast = close + 0.10

    ema_slow = close - 0.10

    ema_trend = close

    adx = np.full(
        rows,
        30.0,
        dtype=float,
    )

    rsi = np.full(
        rows,
        55.0,
        dtype=float,
    )

    macd = np.full(
        rows,
        0.50,
        dtype=float,
    )

    macd_signal = np.full(
        rows,
        0.20,
        dtype=float,
    )

    macd_hist = (
        macd
        - macd_signal
    )

    volume_sma20 = np.full(
        rows,
        1000.0,
        dtype=float,
    )

    atr = np.full(
        rows,
        0.50,
        dtype=float,
    )

    # --------------------------------------------------------
    # Additional common strategy fields
    # --------------------------------------------------------

    returns = pd.Series(
        close
    ).pct_change().fillna(0.0).to_numpy()

    volatility = np.full(
        rows,
        0.01,
        dtype=float,
    )

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    timestamp = pd.date_range(
        start="2025-01-01",
        periods=rows,
        freq="15min",
    )

    # --------------------------------------------------------
    # DataFrame
    # --------------------------------------------------------

    df = pd.DataFrame(
        {
            "timestamp": timestamp,

            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,

            "returns": returns,

            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "ema_trend": ema_trend,

            "adx": adx,
            "rsi": rsi,

            "macd": macd,
            "macd_signal": macd_signal,
            "macd_hist": macd_hist,

            "volume_sma20": volume_sma20,

            "atr": atr,

            "volatility": volatility,
        }
    )

    return df


# ============================================================
# Helpers
# ============================================================


def run_silently(
    engine: BacktestEngine,
    df: pd.DataFrame,
):
    """
    Execute BacktestEngine while suppressing console output.
    """

    return engine.run(df)


def normalize_trades(
    trades,
):
    """
    Convert trades into a stable comparable representation.

    Supports both DataFrame and list-like trade containers.
    """

    if trades is None:
        return []

    if isinstance(trades, pd.DataFrame):

        result = trades.copy()

        if result.empty:
            return []

        # Normalize datetime values.
        for column in result.columns:

            if (
                "time" in str(column).lower()
                and pd.api.types.is_datetime64_any_dtype(
                    result[column]
                )
            ):
                result[column] = (
                    result[column]
                    .astype("int64")
                )

        # Normalize ordering.
        sort_columns = [
            column
            for column in (
                "id",
                "entry_time",
                "exit_time",
            )
            if column in result.columns
        ]

        if sort_columns:
            result = result.sort_values(
                sort_columns
            ).reset_index(drop=True)

        return result.to_dict(
            orient="records"
        )

    if isinstance(trades, (list, tuple)):

        normalized = []

        for trade in trades:

            if hasattr(
                trade,
                "__dict__",
            ):
                data = dict(
                    trade.__dict__
                )

                for key, value in list(
                    data.items()
                ):

                    if hasattr(
                        value,
                        "value",
                    ):
                        data[key] = value.value

                    elif isinstance(
                        value,
                        pd.Timestamp,
                    ):
                        data[key] = (
                            value.isoformat()
                        )

                normalized.append(data)

            elif isinstance(
                trade,
                dict,
            ):
                normalized.append(
                    copy.deepcopy(trade)
                )

            else:
                normalized.append(
                    repr(trade)
                )

        return normalized

    return repr(trades)


def result_signature(
    result,
):
    """
    Extract the deterministic public BacktestResult
    representation.
    """

    assert isinstance(
        result,
        BacktestResult,
    )

    signature = {}

    fields = (
        "initial_balance",
        "final_balance",
        "net_profit",
        "total_trades",
        "winning_trades",
        "losing_trades",
        "win_rate",
    )

    for field_name in fields:

        assert hasattr(
            result,
            field_name,
        )

        value = getattr(
            result,
            field_name,
        )

        if isinstance(
            value,
            float,
        ):
            value = round(
                value,
                10,
            )

        signature[field_name] = value

    signature["trades"] = normalize_trades(
        result.trades
    )

    return signature


# ============================================================
# Test 1
# ============================================================


def test_same_input_produces_same_result():
    """
    The same deterministic input must produce the same
    BacktestResult when executed by independent engines.
    """

    df = make_deterministic_dataframe()

    first_engine = BacktestEngine(
        initial_balance=INITIAL_BALANCE,
    )

    second_engine = BacktestEngine(
        initial_balance=INITIAL_BALANCE,
    )

    first_result = run_silently(
        first_engine,
        df,
    )

    second_result = run_silently(
        second_engine,
        df,
    )

    assert result_signature(
        first_result
    ) == result_signature(
        second_result
    )


# ============================================================
# Test 2
# ============================================================


def test_trade_history_is_deterministic():
    """
    Trade history must be identical between repeated
    independent backtest runs.
    """

    df = make_deterministic_dataframe()

    first_engine = BacktestEngine(
        initial_balance=INITIAL_BALANCE,
    )

    second_engine = BacktestEngine(
        initial_balance=INITIAL_BALANCE,
    )

    first_result = run_silently(
        first_engine,
        df,
    )

    second_result = run_silently(
        second_engine,
        df,
    )

    first_trades = normalize_trades(
        first_result.trades
    )

    second_trades = normalize_trades(
        second_result.trades
    )

    assert first_trades == second_trades


# ============================================================
# Test 3
# ============================================================


def test_trade_order_is_deterministic():
    """
    Trade order must be identical between independent
    backtest runs.
    """

    df = make_deterministic_dataframe()

    first_engine = BacktestEngine(
        initial_balance=INITIAL_BALANCE,
    )

    second_engine = BacktestEngine(
        initial_balance=INITIAL_BALANCE,
    )

    first_result = run_silently(
        first_engine,
        df,
    )

    second_result = run_silently(
        second_engine,
        df,
    )

    first_trades = normalize_trades(
        first_result.trades
    )

    second_trades = normalize_trades(
        second_result.trades
    )

    assert len(first_trades) == len(
        second_trades
    )

    for first_trade, second_trade in zip(
        first_trades,
        second_trades,
    ):
        assert first_trade == second_trade


# ============================================================
# Test 4
# ============================================================


def test_trade_prices_are_deterministic():
    """
    Entry and exit prices must be identical between
    repeated independent runs.
    """

    df = make_deterministic_dataframe()

    first_engine = BacktestEngine(
        initial_balance=INITIAL_BALANCE,
    )

    second_engine = BacktestEngine(
        initial_balance=INITIAL_BALANCE,
    )

    first_result = run_silently(
        first_engine,
        df,
    )

    second_result = run_silently(
        second_engine,
        df,
    )

    first_trades = normalize_trades(
        first_result.trades
    )

    second_trades = normalize_trades(
        second_result.trades
    )

    assert len(first_trades) == len(
        second_trades
    )

    price_fields = (
        "entry",
        "exit",
        "entry_price",
        "exit_price",
        "stop_loss",
        "take_profit",
        "quantity",
    )

    for first_trade, second_trade in zip(
        first_trades,
        second_trades,
    ):

        for field_name in price_fields:

            if (
                field_name in first_trade
                and field_name in second_trade
            ):

                assert first_trade[
                    field_name
                ] == pytest.approx(
                    second_trade[
                        field_name
                    ],
                    abs=1e-10,
                )


# ============================================================
# Test 5
# ============================================================


def test_trade_pnl_is_deterministic():
    """
    Gross profit, commission and net profit must be
    identical between independent runs.
    """

    df = make_deterministic_dataframe()

    first_engine = BacktestEngine(
        initial_balance=INITIAL_BALANCE,
    )

    second_engine = BacktestEngine(
        initial_balance=INITIAL_BALANCE,
    )

    first_result = run_silently(
        first_engine,
        df,
    )

    second_result = run_silently(
        second_engine,
        df,
    )

    first_trades = normalize_trades(
        first_result.trades
    )

    second_trades = normalize_trades(
        second_result.trades
    )

    pnl_fields = (
        "gross_profit",
        "commission",
        "net_profit",
        "balance",
        "balance_after_close",
    )

    for first_trade, second_trade in zip(
        first_trades,
        second_trades,
    ):

        for field_name in pnl_fields:

            if (
                field_name in first_trade
                and field_name in second_trade
            ):

                assert first_trade[
                    field_name
                ] == pytest.approx(
                    second_trade[
                        field_name
                    ],
                    abs=1e-10,
                )


# ============================================================
# Test 6
# ============================================================


def test_trade_close_reasons_are_deterministic():
    """
    Every trade must have the same close reason
    in repeated independent runs.
    """

    df = make_deterministic_dataframe()

    first_engine = BacktestEngine(
        initial_balance=INITIAL_BALANCE,
    )

    second_engine = BacktestEngine(
        initial_balance=INITIAL_BALANCE,
    )

    first_result = run_silently(
        first_engine,
        df,
    )

    second_result = run_silently(
        second_engine,
        df,
    )

    first_trades = normalize_trades(
        first_result.trades
    )

    second_trades = normalize_trades(
        second_result.trades
    )

    first_reasons = [
        trade.get(
            "close_reason",
            trade.get(
                "reason_close"
            ),
        )
        for trade in first_trades
    ]

    second_reasons = [
        trade.get(
            "close_reason",
            trade.get(
                "reason_close"
            ),
        )
        for trade in second_trades
    ]

    assert first_reasons == second_reasons


# ============================================================
# Test 7
# ============================================================


def test_result_object_fields_are_deterministic():
    """
    All public scalar BacktestResult fields must remain
    deterministic between repeated runs.
    """

    df = make_deterministic_dataframe()

    first_engine = BacktestEngine(
        initial_balance=INITIAL_BALANCE,
    )

    second_engine = BacktestEngine(
        initial_balance=INITIAL_BALANCE,
    )

    first_result = run_silently(
        first_engine,
        df,
    )

    second_result = run_silently(
        second_engine,
        df,
    )

    fields = (
        "initial_balance",
        "final_balance",
        "net_profit",
        "total_trades",
        "winning_trades",
        "losing_trades",
        "win_rate",
    )

    for field_name in fields:

        first_value = getattr(
            first_result,
            field_name,
        )

        second_value = getattr(
            second_result,
            field_name,
        )

        if isinstance(
            first_value,
            float,
        ):

            assert first_value == pytest.approx(
                second_value,
                abs=1e-10,
            )

        else:

            assert first_value == second_value


# ============================================================
# Test 8
# ============================================================


def test_independent_engine_instances_produce_same_result():
    """
    Two completely independent BacktestEngine instances
    must not influence each other.
    """

    df = make_deterministic_dataframe()

    engine_a = BacktestEngine(
        initial_balance=INITIAL_BALANCE,
    )

    result_a = run_silently(
        engine_a,
        df,
    )

    engine_b = BacktestEngine(
        initial_balance=INITIAL_BALANCE,
    )

    result_b = run_silently(
        engine_b,
        df,
    )

    assert result_signature(
        result_a
    ) == result_signature(
        result_b
    )


# ============================================================
# Test 9
# ============================================================


def test_repeated_runs_remain_stable():
    """
    Execute the same deterministic dataset several times.
    Every result must match the first result.
    """

    df = make_deterministic_dataframe()

    baseline_engine = BacktestEngine(
        initial_balance=INITIAL_BALANCE,
    )

    baseline_result = run_silently(
        baseline_engine,
        df,
    )

    baseline_signature = result_signature(
        baseline_result
    )

    for _ in range(3):

        engine = BacktestEngine(
            initial_balance=INITIAL_BALANCE,
        )

        result = run_silently(
            engine,
            df,
        )

        assert result_signature(
            result
        ) == baseline_signature


# ============================================================
# Test 10
# ============================================================


def test_determinism_test_does_not_modify_input_dataframe():
    """
    Running the backtest must not mutate the caller's
    DataFrame.
    """

    df = make_deterministic_dataframe()

    original = df.copy(
        deep=True
    )

    engine = BacktestEngine(
        initial_balance=INITIAL_BALANCE,
    )

    run_silently(
        engine,
        df,
    )

    pd.testing.assert_frame_equal(
        df,
        original,
        check_dtype=True,
        check_index_type=True,
        check_column_type=True,
    )


# ============================================================
# Test 11
# ============================================================


def test_input_dataframe_shape_is_preserved():
    """
    Backtest execution must not change the shape of the
    caller's DataFrame.
    """

    df = make_deterministic_dataframe()

    original_shape = df.shape

    engine = BacktestEngine(
        initial_balance=INITIAL_BALANCE,
    )

    run_silently(
        engine,
        df,
    )

    assert df.shape == original_shape


# ============================================================
# Test 12
# ============================================================


def test_input_dataframe_columns_are_preserved():
    """
    Backtest execution must not add, remove or reorder
    caller DataFrame columns.
    """

    df = make_deterministic_dataframe()

    original_columns = list(
        df.columns
    )

    engine = BacktestEngine(
        initial_balance=INITIAL_BALANCE,
    )

    run_silently(
        engine,
        df,
    )

    assert list(
        df.columns
    ) == original_columns


# ============================================================
# Test 13
# ============================================================


def test_timestamp_order_is_deterministic():
    """
    Trade timestamps must remain identical between runs.
    """

    df = make_deterministic_dataframe()

    first_engine = BacktestEngine(
        initial_balance=INITIAL_BALANCE,
    )

    second_engine = BacktestEngine(
        initial_balance=INITIAL_BALANCE,
    )

    first_result = run_silently(
        first_engine,
        df,
    )

    second_result = run_silently(
        second_engine,
        df,
    )

    first_trades = normalize_trades(
        first_result.trades
    )

    second_trades = normalize_trades(
        second_result.trades
    )

    for first_trade, second_trade in zip(
        first_trades,
        second_trades,
    ):

        for field_name in (
            "entry_time",
            "exit_time",
        ):

            if (
                field_name in first_trade
                and field_name in second_trade
            ):

                assert (
                    first_trade[field_name]
                    == second_trade[field_name]
                )


# ============================================================
# Test 14
# ============================================================


def test_backtest_result_is_stored_on_engine():
    """
    After execution, the engine must retain the same
    BacktestResult object exposed by run().
    """

    df = make_deterministic_dataframe()

    engine = BacktestEngine(
        initial_balance=INITIAL_BALANCE,
    )

    result = run_silently(
        engine,
        df,
    )

    assert engine.result is result


# ============================================================
# Test 15
# ============================================================


def test_initial_balance_is_deterministic():
    """
    Every fresh engine must start from the configured
    INITIAL_BALANCE.
    """

    first_engine = BacktestEngine(
        initial_balance=INITIAL_BALANCE,
    )

    second_engine = BacktestEngine(
        initial_balance=INITIAL_BALANCE,
    )

    assert (
        first_engine.initial_balance
        == second_engine.initial_balance
    )

    assert (
        first_engine.initial_balance
        == INITIAL_BALANCE
    )


# ============================================================
# End
# ============================================================
