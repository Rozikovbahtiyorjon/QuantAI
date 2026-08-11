"""
QuantAI - Walk Forward Engine Validation Tests
================================================

Validation tests for:

    src/walk_forward_engine.py

Covered areas
-------------

- constructor validation
- configuration defaults
- data validation
- window generation
- train/test boundaries
- preserved DataFrame indexes
- returned DataFrame copies
- incomplete final windows
- step_size behavior
- run_window validation
- BacktestEngine orchestration
- balance rollover between windows
- aggregate trade statistics
- win-rate calculation
- result aliases
- result property lifecycle
- input DataFrame immutability
- convenience wrapper
- report validation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import pytest

import src.walk_forward_engine as walk_forward_module
from src.walk_forward_engine import (
    DEFAULT_INITIAL_BALANCE,
    DEFAULT_TEST_SIZE,
    DEFAULT_TRAIN_SIZE,
    MINIMUM_WINDOW_SIZE,
    WalkForwardEngine,
    WalkForwardResult,
    WalkForwardWindowResult,
    run_walk_forward,
)


# =========================================================
# TEST DATA
# =========================================================


def make_dataframe(rows: int = 30) -> pd.DataFrame:
    """
    Create deterministic OHLCV-like data.

    Index is deliberately non-default so that tests can verify
    that generate_windows() preserves original DataFrame indexes.
    """

    index = [
        1000 + i
        for i in range(rows)
    ]

    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01",
                periods=rows,
                freq="15min",
            ),
            "open": [
                100.0 + i
                for i in range(rows)
            ],
            "high": [
                101.0 + i
                for i in range(rows)
            ],
            "low": [
                99.0 + i
                for i in range(rows)
            ],
            "close": [
                100.5 + i
                for i in range(rows)
            ],
            "volume": [
                1000.0 + i
                for i in range(rows)
            ],
        },
        index=index,
    )


# =========================================================
# FAKE BACKTEST RESULT
# =========================================================


@dataclass
class FakeBacktestResult:
    """
    Minimal object compatible with the fields used by
    WalkForwardEngine.
    """

    initial_balance: float
    final_balance: float

    net_profit: float

    total_trades: int
    winning_trades: int
    losing_trades: int

    win_rate: float


# =========================================================
# FAKE BACKTEST ENGINE
# =========================================================


class FakeBacktestEngine:
    """
    Deterministic fake BacktestEngine used to validate
    WalkForwardEngine orchestration.

    Every test window increases balance by a fixed amount
    and returns deterministic trade statistics.
    """

    calls: list[dict[str, Any]] = []

    def __init__(
        self,
        initial_balance: float,
    ) -> None:

        self.initial_balance = float(
            initial_balance
        )

    def run(
        self,
        df: pd.DataFrame,
    ) -> FakeBacktestResult:

        FakeBacktestEngine.calls.append(
            {
                "initial_balance": self.initial_balance,
                "rows": len(df),
                "index": list(df.index),
            }
        )

        profit = 10.0

        final_balance = (
            self.initial_balance
            + profit
        )

        total_trades = 4
        winning_trades = 3
        losing_trades = 1

        win_rate = 75.0

        return FakeBacktestResult(
            initial_balance=self.initial_balance,
            final_balance=final_balance,
            net_profit=profit,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
        )


# =========================================================
# HELPERS
# =========================================================


def reset_fake_backtest_calls() -> None:
    FakeBacktestEngine.calls = []


# =========================================================
# CONSTRUCTOR VALIDATION
# =========================================================


def test_default_configuration() -> None:
    engine = WalkForwardEngine()

    assert engine.train_size == DEFAULT_TRAIN_SIZE
    assert engine.test_size == DEFAULT_TEST_SIZE
    assert engine.step_size == DEFAULT_TEST_SIZE
    assert engine.initial_balance == pytest.approx(
        DEFAULT_INITIAL_BALANCE
    )
    assert engine.result is None


def test_custom_configuration() -> None:
    engine = WalkForwardEngine(
        train_size=20,
        test_size=5,
        step_size=3,
        initial_balance=2500.0,
    )

    assert engine.train_size == 20
    assert engine.test_size == 5
    assert engine.step_size == 3
    assert engine.initial_balance == pytest.approx(
        2500.0
    )


def test_train_size_must_be_integer() -> None:
    with pytest.raises(
        TypeError,
        match="train_size must be an integer",
    ):
        WalkForwardEngine(
            train_size=10.5,
        )


@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
        -100,
    ],
)
def test_train_size_must_be_positive(
    value: int,
) -> None:

    with pytest.raises(
        ValueError,
        match="train_size must be greater than zero",
    ):
        WalkForwardEngine(
            train_size=value,
        )


def test_test_size_must_be_integer() -> None:
    with pytest.raises(
        TypeError,
        match="test_size must be an integer",
    ):
        WalkForwardEngine(
            test_size=5.5,
        )


@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
        -100,
    ],
)
def test_test_size_must_be_positive(
    value: int,
) -> None:

    with pytest.raises(
        ValueError,
        match="test_size must be greater than zero",
    ):
        WalkForwardEngine(
            test_size=value,
        )


def test_step_size_must_be_integer() -> None:
    with pytest.raises(
        TypeError,
        match="step_size must be an integer",
    ):
        WalkForwardEngine(
            step_size=2.5,
        )


@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
        -100,
    ],
)
def test_step_size_must_be_positive(
    value: int,
) -> None:

    with pytest.raises(
        ValueError,
        match="step_size must be greater than zero",
    ):
        WalkForwardEngine(
            step_size=value,
        )


def test_step_size_defaults_to_test_size() -> None:
    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
    )

    assert engine.step_size == 5


def test_initial_balance_must_be_positive() -> None:
    with pytest.raises(
        ValueError,
        match="initial_balance must be greater than zero",
    ):
        WalkForwardEngine(
            initial_balance=0,
        )


@pytest.mark.parametrize(
    "value",
    [
        -1,
        -1000,
    ],
)
def test_negative_initial_balance_is_rejected(
    value: float,
) -> None:

    with pytest.raises(
        ValueError,
        match="initial_balance must be greater than zero",
    ):
        WalkForwardEngine(
            initial_balance=value,
        )


def test_minimum_window_constant() -> None:
    assert MINIMUM_WINDOW_SIZE == 1


# =========================================================
# DATA VALIDATION
# =========================================================


def test_validate_data_requires_dataframe() -> None:
    engine = WalkForwardEngine(
        train_size=5,
        test_size=3,
    )

    with pytest.raises(
        TypeError,
        match="requires a pandas DataFrame",
    ):
        engine.validate_data(
            [
                1,
                2,
                3,
            ]
        )


def test_validate_data_rejects_empty_dataframe() -> None:
    engine = WalkForwardEngine(
        train_size=5,
        test_size=3,
    )

    df = pd.DataFrame()

    with pytest.raises(
        ValueError,
        match="Walk-forward data is empty",
    ):
        engine.validate_data(df)


def test_validate_data_rejects_insufficient_rows() -> None:
    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
    )

    df = make_dataframe(14)

    with pytest.raises(
        ValueError,
        match="Not enough rows",
    ):
        engine.validate_data(df)


def test_validate_data_accepts_exact_minimum_rows() -> None:
    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
    )

    df = make_dataframe(15)

    engine.validate_data(df)


# =========================================================
# WINDOW GENERATION
# =========================================================


def test_generate_windows_returns_list() -> None:
    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
    )

    df = make_dataframe(30)

    windows = engine.generate_windows(df)

    assert isinstance(
        windows,
        list,
    )


def test_generate_windows_returns_exact_tuple_shape() -> None:
    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
    )

    df = make_dataframe(30)

    windows = engine.generate_windows(df)

    assert len(windows) > 0

    for item in windows:

        assert isinstance(
            item,
            tuple,
        )

        assert len(item) == 3

        window_number, train_df, test_df = item

        assert isinstance(
            window_number,
            int,
        )

        assert isinstance(
            train_df,
            pd.DataFrame,
        )

        assert isinstance(
            test_df,
            pd.DataFrame,
        )


def test_first_window_boundaries() -> None:
    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
        step_size=5,
    )

    df = make_dataframe(30)

    windows = engine.generate_windows(df)

    window_number, train_df, test_df = windows[0]

    assert window_number == 1

    assert list(train_df.index) == list(
        range(
            1000,
            1010,
        )
    )

    assert list(test_df.index) == list(
        range(
            1010,
            1015,
        )
    )


def test_second_window_boundaries() -> None:
    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
        step_size=5,
    )

    df = make_dataframe(30)

    windows = engine.generate_windows(df)

    window_number, train_df, test_df = windows[1]

    assert window_number == 2

    assert list(train_df.index) == list(
        range(
            1005,
            1015,
        )
    )

    assert list(test_df.index) == list(
        range(
            1015,
            1020,
        )
    )


def test_window_numbers_are_sequential() -> None:
    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
        step_size=5,
    )

    df = make_dataframe(40)

    windows = engine.generate_windows(df)

    numbers = [
        item[0]
        for item in windows
    ]

    assert numbers == list(
        range(
            1,
            len(numbers) + 1,
        )
    )


def test_train_size_is_exact() -> None:
    engine = WalkForwardEngine(
        train_size=7,
        test_size=4,
        step_size=4,
    )

    df = make_dataframe(30)

    windows = engine.generate_windows(df)

    for _, train_df, _ in windows:

        assert len(train_df) == 7


def test_test_size_is_exact() -> None:
    engine = WalkForwardEngine(
        train_size=7,
        test_size=4,
        step_size=4,
    )

    df = make_dataframe(30)

    windows = engine.generate_windows(df)

    for _, _, test_df in windows:

        assert len(test_df) == 4


def test_incomplete_final_test_window_is_excluded() -> None:
    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
        step_size=5,
    )

    df = make_dataframe(23)

    windows = engine.generate_windows(df)

    # Valid:
    #
    # 0:10 + 10:15
    # 5:15 + 15:20
    #
    # 10:20 + 20:25 is incomplete.
    assert len(windows) == 2


def test_exact_final_test_window_is_included() -> None:
    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
        step_size=5,
    )

    df = make_dataframe(25)

    windows = engine.generate_windows(df)

    assert len(windows) == 3


def test_original_dataframe_index_is_preserved() -> None:
    engine = WalkForwardEngine(
        train_size=5,
        test_size=3,
        step_size=3,
    )

    df = make_dataframe(20)

    windows = engine.generate_windows(df)

    _, train_df, test_df = windows[0]

    assert list(train_df.index) == [
        1000,
        1001,
        1002,
        1003,
        1004,
    ]

    assert list(test_df.index) == [
        1005,
        1006,
        1007,
    ]


def test_generated_windows_are_copies() -> None:
    engine = WalkForwardEngine(
        train_size=5,
        test_size=3,
    )

    df = make_dataframe(20)

    windows = engine.generate_windows(df)

    _, train_df, test_df = windows[0]

    original_value = df.loc[
        1000,
        "close",
    ]

    train_df.loc[
        1000,
        "close",
    ] = 999999.0

    test_df.loc[
        1005,
        "close",
    ] = 888888.0

    assert df.loc[
        1000,
        "close",
    ] == original_value

    assert df.loc[
        1005,
        "close",
    ] != 888888.0


def test_generate_windows_does_not_modify_input() -> None:
    engine = WalkForwardEngine(
        train_size=5,
        test_size=3,
    )

    df = make_dataframe(20)

    original = df.copy(
        deep=True
    )

    engine.generate_windows(df)

    pd.testing.assert_frame_equal(
        df,
        original,
    )


def test_step_size_controls_train_window_shift() -> None:
    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
        step_size=2,
    )

    df = make_dataframe(25)

    windows = engine.generate_windows(df)

    first_train = windows[0][1]
    second_train = windows[1][1]

    assert list(first_train.index) == list(
        range(
            1000,
            1010,
        )
    )

    assert list(second_train.index) == list(
        range(
            1002,
            1012,
        )
    )


def test_step_size_can_be_larger_than_test_size() -> None:
    engine = WalkForwardEngine(
        train_size=5,
        test_size=3,
        step_size=7,
    )

    df = make_dataframe(30)

    windows = engine.generate_windows(df)

    first_train = windows[0][1]
    second_train = windows[1][1]

    assert list(first_train.index) == list(
        range(
            1000,
            1005,
        )
    )

    assert list(second_train.index) == list(
        range(
            1007,
            1012,
        )
    )


def test_step_size_one_produces_maximum_overlap() -> None:
    engine = WalkForwardEngine(
        train_size=5,
        test_size=2,
        step_size=1,
    )

    df = make_dataframe(10)

    windows = engine.generate_windows(df)

    assert len(windows) == 4

    assert list(
        windows[0][1].index
    ) == [
        1000,
        1001,
        1002,
        1003,
        1004,
    ]

    assert list(
        windows[1][1].index
    ) == [
        1001,
        1002,
        1003,
        1004,
        1005,
    ]


# =========================================================
# WINDOW RESULT DATACLASS
# =========================================================


def make_fake_window_result() -> WalkForwardWindowResult:
    backtest_result = FakeBacktestResult(
        initial_balance=1000.0,
        final_balance=1010.0,
        net_profit=10.0,
        total_trades=4,
        winning_trades=3,
        losing_trades=1,
        win_rate=75.0,
    )

    return WalkForwardWindowResult(
        window_id=7,
        train_start=10,
        train_end=20,
        test_start=20,
        test_end=25,
        train_size=10,
        test_size=5,
        backtest_result=backtest_result,
    )


def test_window_number_alias() -> None:
    result = make_fake_window_result()

    assert result.window_number == result.window_id
    assert result.window_number == 7


# =========================================================
# COMPLETE RESULT DATACLASS
# =========================================================


def test_walk_forward_result_aliases() -> None:
    window = make_fake_window_result()

    result = WalkForwardResult(
        initial_balance=1000.0,
        final_balance=1010.0,
        net_profit=10.0,
        total_trades=4,
        winning_trades=3,
        losing_trades=1,
        win_rate=75.0,
        windows=[window],
    )

    assert result.window_results is result.windows
    assert result.total_windows == 1


def test_empty_walk_forward_result_has_zero_windows() -> None:
    result = WalkForwardResult(
        initial_balance=1000.0,
        final_balance=1000.0,
        net_profit=0.0,
        total_trades=0,
        winning_trades=0,
        losing_trades=0,
        win_rate=0.0,
    )

    assert result.total_windows == 0
    assert result.window_results == []


# =========================================================
# RUN_WINDOW VALIDATION
# =========================================================


def test_run_window_rejects_negative_train_start() -> None:
    engine = WalkForwardEngine(
        train_size=5,
        test_size=3,
    )

    df = make_dataframe(20)

    with pytest.raises(
        ValueError,
        match="train_start cannot be negative",
    ):
        engine.run_window(
            df=df,
            window_id=1,
            train_start=-1,
            train_end=5,
            test_start=5,
            test_end=8,
            initial_balance=1000.0,
        )


def test_run_window_rejects_empty_train_window() -> None:
    engine = WalkForwardEngine(
        train_size=5,
        test_size=3,
    )

    df = make_dataframe(20)

    with pytest.raises(
        ValueError,
        match="Train window cannot be empty",
    ):
        engine.run_window(
            df=df,
            window_id=1,
            train_start=5,
            train_end=5,
            test_start=5,
            test_end=8,
            initial_balance=1000.0,
        )


def test_run_window_requires_test_after_train() -> None:
    engine = WalkForwardEngine(
        train_size=5,
        test_size=3,
    )

    df = make_dataframe(20)

    with pytest.raises(
        ValueError,
        match="Test window must start after",
    ):
        engine.run_window(
            df=df,
            window_id=1,
            train_start=0,
            train_end=5,
            test_start=4,
            test_end=7,
            initial_balance=1000.0,
        )


def test_run_window_rejects_empty_test_window() -> None:
    engine = WalkForwardEngine(
        train_size=5,
        test_size=3,
    )

    df = make_dataframe(20)

    with pytest.raises(
        ValueError,
        match="Test window cannot be empty",
    ):
        engine.run_window(
            df=df,
            window_id=1,
            train_start=0,
            train_end=5,
            test_start=5,
            test_end=5,
            initial_balance=1000.0,
        )


def test_run_window_rejects_test_beyond_dataframe() -> None:
    engine = WalkForwardEngine(
        train_size=5,
        test_size=3,
    )

    df = make_dataframe(10)

    with pytest.raises(
        ValueError,
        match="Test window exceeds available data",
    ):
        engine.run_window(
            df=df,
            window_id=1,
            train_start=0,
            train_end=5,
            test_start=5,
            test_end=11,
            initial_balance=1000.0,
        )


# =========================================================
# RUN_WINDOW ORCHESTRATION
# =========================================================


def test_run_window_calls_backtest_engine_with_test_data(
    monkeypatch,
) -> None:

    reset_fake_backtest_calls()

    monkeypatch.setattr(
        walk_forward_module,
        "BacktestEngine",
        FakeBacktestEngine,
    )

    engine = WalkForwardEngine(
        train_size=5,
        test_size=3,
    )

    df = make_dataframe(20)

    result = engine.run_window(
        df=df,
        window_id=1,
        train_start=0,
        train_end=5,
        test_start=5,
        test_end=8,
        initial_balance=1000.0,
    )

    assert isinstance(
        result,
        WalkForwardWindowResult,
    )

    assert result.window_id == 1
    assert result.train_start == 0
    assert result.train_end == 5
    assert result.test_start == 5
    assert result.test_end == 8

    assert result.train_size == 5
    assert result.test_size == 3

    assert len(
        FakeBacktestEngine.calls
    ) == 1

    call = FakeBacktestEngine.calls[0]

    assert call["initial_balance"] == pytest.approx(
        1000.0
    )

    assert call["rows"] == 3

    assert call["index"] == [
        1005,
        1006,
        1007,
    ]


def test_run_window_does_not_pass_train_data_to_backtest(
    monkeypatch,
) -> None:

    reset_fake_backtest_calls()

    monkeypatch.setattr(
        walk_forward_module,
        "BacktestEngine",
        FakeBacktestEngine,
    )

    engine = WalkForwardEngine(
        train_size=5,
        test_size=3,
    )

    df = make_dataframe(20)

    engine.run_window(
        df=df,
        window_id=1,
        train_start=0,
        train_end=5,
        test_start=5,
        test_end=8,
        initial_balance=1000.0,
    )

    call = FakeBacktestEngine.calls[0]

    assert call["rows"] == 3

    assert call["index"] == [
        1005,
        1006,
        1007,
    ]


def test_run_window_preserves_result_metrics(
    monkeypatch,
) -> None:

    reset_fake_backtest_calls()

    monkeypatch.setattr(
        walk_forward_module,
        "BacktestEngine",
        FakeBacktestEngine,
    )

    engine = WalkForwardEngine(
        train_size=5,
        test_size=3,
    )

    df = make_dataframe(20)

    result = engine.run_window(
        df=df,
        window_id=1,
        train_start=0,
        train_end=5,
        test_start=5,
        test_end=8,
        initial_balance=1000.0,
    )

    backtest = result.backtest_result

    assert backtest.initial_balance == pytest.approx(
        1000.0
    )

    assert backtest.final_balance == pytest.approx(
        1010.0
    )

    assert backtest.net_profit == pytest.approx(
        10.0
    )

    assert backtest.total_trades == 4
    assert backtest.winning_trades == 3
    assert backtest.losing_trades == 1
    assert backtest.win_rate == pytest.approx(
        75.0
    )


# =========================================================
# COMPLETE RUN
# =========================================================


def test_run_requires_valid_data() -> None:
    engine = WalkForwardEngine(
        train_size=5,
        test_size=3,
    )

    with pytest.raises(
        ValueError,
        match="Walk-forward data is empty",
    ):
        engine.run(
            pd.DataFrame()
        )


def test_run_rejects_data_without_complete_window() -> None:
    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
    )

    df = make_dataframe(14)

    with pytest.raises(
        ValueError,
        match="Not enough rows",
    ):
        engine.run(df)


def test_run_creates_result(
    monkeypatch,
) -> None:

    reset_fake_backtest_calls()

    monkeypatch.setattr(
        walk_forward_module,
        "BacktestEngine",
        FakeBacktestEngine,
    )

    engine = WalkForwardEngine(
        train_size=5,
        test_size=3,
        step_size=3,
        initial_balance=1000.0,
    )

    df = make_dataframe(20)

    result = engine.run(df)

    assert isinstance(
        result,
        WalkForwardResult,
    )

    assert engine.result is result


def test_run_creates_expected_number_of_windows(
    monkeypatch,
) -> None:

    reset_fake_backtest_calls()

    monkeypatch.setattr(
        walk_forward_module,
        "BacktestEngine",
        FakeBacktestEngine,
    )

    engine = WalkForwardEngine(
        train_size=5,
        test_size=3,
        step_size=3,
    )

    df = make_dataframe(20)

    result = engine.run(df)

    # Valid windows:
    #
    # 0:5 -> 5:8
    # 3:8 -> 8:11
    # 6:11 -> 11:14
    # 9:14 -> 14:17
    #
    # 12:17 -> 17:20 is also valid.
    assert result.total_windows == 5

    assert len(
        result.windows
    ) == 5


def test_run_rolls_balance_forward(
    monkeypatch,
) -> None:

    reset_fake_backtest_calls()

    monkeypatch.setattr(
        walk_forward_module,
        "BacktestEngine",
        FakeBacktestEngine,
    )

    engine = WalkForwardEngine(
        train_size=5,
        test_size=3,
        step_size=3,
        initial_balance=1000.0,
    )

    df = make_dataframe(20)

    result = engine.run(df)

    balances = [
        call["initial_balance"]
        for call in FakeBacktestEngine.calls
    ]

    assert balances == [
        1000.0,
        1010.0,
        1020.0,
        1030.0,
        1040.0,
    ]

    assert result.final_balance == pytest.approx(
        1050.0
    )


def test_run_aggregates_trade_counts(
    monkeypatch,
) -> None:

    reset_fake_backtest_calls()

    monkeypatch.setattr(
        walk_forward_module,
        "BacktestEngine",
        FakeBacktestEngine,
    )

    engine = WalkForwardEngine(
        train_size=5,
        test_size=3,
        step_size=3,
        initial_balance=1000.0,
    )

    df = make_dataframe(20)

    result = engine.run(df)

    windows = result.total_windows

    assert result.total_trades == (
        windows * 4
    )

    assert result.winning_trades == (
        windows * 3
    )

    assert result.losing_trades == (
        windows * 1
    )


def test_run_calculates_aggregate_win_rate(
    monkeypatch,
) -> None:

    reset_fake_backtest_calls()

    monkeypatch.setattr(
        walk_forward_module,
        "BacktestEngine",
        FakeBacktestEngine,
    )

    engine = WalkForwardEngine(
        train_size=5,
        test_size=3,
        step_size=3,
        initial_balance=1000.0,
    )

    df = make_dataframe(20)

    result = engine.run(df)

    assert result.win_rate == pytest.approx(
        75.0
    )


def test_run_calculates_net_profit(
    monkeypatch,
) -> None:

    reset_fake_backtest_calls()

    monkeypatch.setattr(
        walk_forward_module,
        "BacktestEngine",
        FakeBacktestEngine,
    )

    engine = WalkForwardEngine(
        train_size=5,
        test_size=3,
        step_size=3,
        initial_balance=1000.0,
    )

    df = make_dataframe(20)

    result = engine.run(df)

    assert result.net_profit == pytest.approx(
        result.final_balance
        - result.initial_balance
    )


def test_run_initial_balance_is_preserved(
    monkeypatch,
) -> None:

    reset_fake_backtest_calls()

    monkeypatch.setattr(
        walk_forward_module,
        "BacktestEngine",
        FakeBacktestEngine,
    )

    engine = WalkForwardEngine(
        train_size=5,
        test_size=3,
        step_size=3,
        initial_balance=2500.0,
    )

    df = make_dataframe(20)

    result = engine.run(df)

    assert result.initial_balance == pytest.approx(
        2500.0
    )


def test_run_does_not_modify_input_dataframe(
    monkeypatch,
) -> None:

    reset_fake_backtest_calls()

    monkeypatch.setattr(
        walk_forward_module,
        "BacktestEngine",
        FakeBacktestEngine,
    )

    engine = WalkForwardEngine(
        train_size=5,
        test_size=3,
    )

    df = make_dataframe(20)

    original = df.copy(
        deep=True
    )

    engine.run(df)

    pd.testing.assert_frame_equal(
        df,
        original,
    )


def test_run_stores_result_on_engine(
    monkeypatch,
) -> None:

    reset_fake_backtest_calls()

    monkeypatch.setattr(
        walk_forward_module,
        "BacktestEngine",
        FakeBacktestEngine,
    )

    engine = WalkForwardEngine(
        train_size=5,
        test_size=3,
    )

    df = make_dataframe(20)

    returned = engine.run(df)

    assert engine.result is returned
    assert engine.result is not None


def test_result_property_is_none_before_run() -> None:
    engine = WalkForwardEngine(
        train_size=5,
        test_size=3,
    )

    assert engine.result is None


def test_second_run_replaces_previous_result(
    monkeypatch,
) -> None:

    reset_fake_backtest_calls()

    monkeypatch.setattr(
        walk_forward_module,
        "BacktestEngine",
        FakeBacktestEngine,
    )

    engine = WalkForwardEngine(
        train_size=5,
        test_size=3,
    )

    df = make_dataframe(20)

    first = engine.run(df)
    second = engine.run(df)

    assert second is engine.result
    assert second is not first


# =========================================================
# WINDOW RESULTS INSIDE COMPLETE RESULT
# =========================================================


def test_window_results_contain_correct_boundaries(
    monkeypatch,
) -> None:

    reset_fake_backtest_calls()

    monkeypatch.setattr(
        walk_forward_module,
        "BacktestEngine",
        FakeBacktestEngine,
    )

    engine = WalkForwardEngine(
        train_size=5,
        test_size=3,
        step_size=3,
    )

    df = make_dataframe(20)

    result = engine.run(df)

    first = result.windows[0]

    assert first.window_id == 1
    assert first.train_start == 0
    assert first.train_end == 5
    assert first.test_start == 5
    assert first.test_end == 8

    assert first.train_size == 5
    assert first.test_size == 3


def test_window_number_alias_inside_result(
    monkeypatch,
) -> None:

    reset_fake_backtest_calls()

    monkeypatch.setattr(
        walk_forward_module,
        "BacktestEngine",
        FakeBacktestEngine,
    )

    engine = WalkForwardEngine(
        train_size=5,
        test_size=3,
    )

    df = make_dataframe(20)

    result = engine.run(df)

    for expected, window in enumerate(
        result.windows,
        start=1,
    ):
        assert window.window_number == expected
        assert window.window_id == expected


# =========================================================
# CONVENIENCE FUNCTION
# =========================================================


def test_run_walk_forward_returns_result(
    monkeypatch,
) -> None:

    reset_fake_backtest_calls()

    monkeypatch.setattr(
        walk_forward_module,
        "BacktestEngine",
        FakeBacktestEngine,
    )

    df = make_dataframe(20)

    result = run_walk_forward(
        df=df,
        train_size=5,
        test_size=3,
        step_size=3,
        initial_balance=1000.0,
    )

    assert isinstance(
        result,
        WalkForwardResult,
    )

    assert result.total_windows == 5


def test_run_walk_forward_prints_report(
    monkeypatch,
    capsys,
) -> None:

    reset_fake_backtest_calls()

    monkeypatch.setattr(
        walk_forward_module,
        "BacktestEngine",
        FakeBacktestEngine,
    )

    df = make_dataframe(20)

    run_walk_forward(
        df=df,
        train_size=5,
        test_size=3,
        step_size=3,
        initial_balance=1000.0,
    )

    captured = capsys.readouterr()

    assert (
        "QUANTAI WALK-FORWARD REPORT"
        in captured.out
    )

    assert (
        "Initial Balance"
        in captured.out
    )

    assert (
        "Final Balance"
        in captured.out
    )

    assert (
        "Total Trades"
        in captured.out
    )


# =========================================================
# REPORT VALIDATION
# =========================================================


def test_print_report_requires_correct_result_type(
    capsys,
) -> None:

    with pytest.raises(
        TypeError,
        match="result must be WalkForwardResult",
    ):
        WalkForwardEngine.print_report(
            object()
        )


def test_print_report_outputs_window_information(
    capsys,
) -> None:

    window = make_fake_window_result()

    result = WalkForwardResult(
        initial_balance=1000.0,
        final_balance=1010.0,
        net_profit=10.0,
        total_trades=4,
        winning_trades=3,
        losing_trades=1,
        win_rate=75.0,
        windows=[window],
    )

    WalkForwardEngine.print_report(
        result
    )

    captured = capsys.readouterr()

    assert "Window 7" in captured.out
    assert "TRAIN=10:20" in captured.out
    assert "TEST=20:25" in captured.out
    assert "trades=4" in captured.out


# =========================================================
# DETERMINISTIC WINDOW GENERATION
# =========================================================


def test_generate_windows_is_deterministic() -> None:
    engine = WalkForwardEngine(
        train_size=5,
        test_size=3,
        step_size=2,
    )

    df = make_dataframe(20)

    first = engine.generate_windows(df)
    second = engine.generate_windows(df)

    assert len(first) == len(second)

    for first_item, second_item in zip(
        first,
        second,
    ):

        first_number, first_train, first_test = first_item
        second_number, second_train, second_test = second_item

        assert first_number == second_number

        pd.testing.assert_frame_equal(
            first_train,
            second_train,
        )

        pd.testing.assert_frame_equal(
            first_test,
            second_test,
        )


# =========================================================
# EXPORT CONTRACT
# =========================================================


def test_public_exports_exist() -> None:
    expected_exports = {
        "DEFAULT_TRAIN_SIZE",
        "DEFAULT_TEST_SIZE",
        "DEFAULT_INITIAL_BALANCE",
        "MINIMUM_WINDOW_SIZE",
        "WalkForwardWindowResult",
        "WalkForwardResult",
        "WalkForwardEngine",
        "run_walk_forward",
    }

    for name in expected_exports:
        assert hasattr(
            walk_forward_module,
            name,
        )


# =========================================================
# FINAL SANITY TEST
# =========================================================


def test_complete_walk_forward_contract(
    monkeypatch,
) -> None:
    """
    High-level contract test combining:

        window generation
        backtest execution
        balance rollover
        aggregation
        result aliases
    """

    reset_fake_backtest_calls()

    monkeypatch.setattr(
        walk_forward_module,
        "BacktestEngine",
        FakeBacktestEngine,
    )

    engine = WalkForwardEngine(
        train_size=5,
        test_size=3,
        step_size=3,
        initial_balance=1000.0,
    )

    df = make_dataframe(20)

    result = engine.run(df)

    assert isinstance(
        result,
        WalkForwardResult,
    )

    assert result.initial_balance == pytest.approx(
        1000.0
    )

    assert result.final_balance == pytest.approx(
        1050.0
    )

    assert result.net_profit == pytest.approx(
        50.0
    )

    assert result.total_windows == 5

    assert result.total_trades == 20

    assert result.winning_trades == 15

    assert result.losing_trades == 5

    assert result.win_rate == pytest.approx(
        75.0
    )

    assert result.window_results is result.windows

    assert len(
        FakeBacktestEngine.calls
    ) == 5

    expected_balances = [
        1000.0,
        1010.0,
        1020.0,
        1030.0,
        1040.0,
    ]

    actual_balances = [
        call["initial_balance"]
        for call in FakeBacktestEngine.calls
    ]

    assert actual_balances == expected_balances
