"""
QuantAI - Walk Forward Engine Integration Tests
================================================

Integration tests for:

    WalkForwardEngine
        |
        +---- generate_windows()
        |
        +---- run_window()
        |
        +---- BacktestEngine
        |
        +---- balance roll-forward
        |
        +---- aggregate results

The real BacktestEngine is replaced with a deterministic
fake implementation where necessary so these tests verify
the integration contract of WalkForwardEngine itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd
import pytest

import src.walk_forward_engine as walk_forward_module
from src.walk_forward_engine import (
    WalkForwardEngine,
    WalkForwardResult,
    WalkForwardWindowResult,
)


# =========================================================
# CONSTANTS
# =========================================================

INITIAL_BALANCE = 1000.0

TRAIN_SIZE = 10
TEST_SIZE = 5
STEP_SIZE = 5


# =========================================================
# TEST DATA
# =========================================================

def make_dataframe(rows: int = 30) -> pd.DataFrame:
    """
    Create deterministic OHLCV-like DataFrame.

    The integration tests do not require real indicators because
    BacktestEngine is replaced with a deterministic fake.
    """

    timestamps = pd.date_range(
        start="2026-01-01",
        periods=rows,
        freq="15min",
    )

    return pd.DataFrame(
        {
            "timestamp": timestamps,
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
        }
    )


# =========================================================
# FAKE BACKTEST RESULT
# =========================================================

@dataclass
class FakeBacktestResult:
    """
    Minimal BacktestResult-compatible object.
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
    Deterministic replacement for BacktestEngine.

    Every test window produces:

        final_balance = initial_balance + 10 * test_size

    Therefore:

        5-row test window
        -> +50 profit

    This makes balance roll-forward easy to verify.
    """

    calls: List[dict] = []

    def __init__(
        self,
        initial_balance: float = 1000.0,
    ) -> None:

        self.initial_balance = float(
            initial_balance
        )

        FakeBacktestEngine.calls.append(
            {
                "initial_balance": self.initial_balance,
            }
        )

    def run(
        self,
        df: pd.DataFrame,
    ) -> FakeBacktestResult:

        test_size = len(df)

        net_profit = (
            float(test_size)
            * 10.0
        )

        final_balance = (
            self.initial_balance
            + net_profit
        )

        total_trades = test_size

        winning_trades = (
            test_size // 2
        )

        losing_trades = (
            test_size
            - winning_trades
        )

        win_rate = (
            (
                winning_trades
                / total_trades
            )
            * 100.0
            if total_trades > 0
            else 0.0
        )

        return FakeBacktestResult(
            initial_balance=self.initial_balance,

            final_balance=final_balance,

            net_profit=net_profit,

            total_trades=total_trades,

            winning_trades=winning_trades,

            losing_trades=losing_trades,

            win_rate=round(
                win_rate,
                2,
            ),
        )


# =========================================================
# FIXTURE
# =========================================================

@pytest.fixture
def fake_backtest_engine(monkeypatch):
    """
    Replace WalkForwardEngine's BacktestEngine dependency.
    """

    FakeBacktestEngine.calls = []

    monkeypatch.setattr(
        walk_forward_module,
        "BacktestEngine",
        FakeBacktestEngine,
    )

    return FakeBacktestEngine


# =========================================================
# INITIALIZATION
# =========================================================

def test_walk_forward_engine_creates_valid_configuration():
    """
    WalkForwardEngine must preserve supplied configuration.
    """

    engine = WalkForwardEngine(
        train_size=TRAIN_SIZE,
        test_size=TEST_SIZE,
        step_size=STEP_SIZE,
        initial_balance=INITIAL_BALANCE,
    )

    assert engine.train_size == TRAIN_SIZE

    assert engine.test_size == TEST_SIZE

    assert engine.step_size == STEP_SIZE

    assert (
        engine.initial_balance
        == pytest.approx(
            INITIAL_BALANCE
        )
    )


def test_default_step_size_equals_test_size():
    """
    If step_size is omitted, it must equal test_size.
    """

    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
    )

    assert engine.step_size == 5


# =========================================================
# WINDOW GENERATION
# =========================================================

def test_generate_windows_returns_expected_number():
    """
    30 rows with:

        train = 10
        test  = 5
        step  = 5

    produces:

        1: train 0:10, test 10:15
        2: train 5:15, test 15:20
        3: train 10:20, test 20:25
        4: train 15:25, test 25:30
    """

    df = make_dataframe(30)

    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
        step_size=5,
    )

    windows = engine.generate_windows(df)

    assert len(windows) == 4


def test_generate_windows_preserves_window_numbering():
    """
    Window numbers must start at 1 and increase sequentially.
    """

    df = make_dataframe(30)

    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
        step_size=5,
    )

    windows = engine.generate_windows(df)

    numbers = [
        item[0]
        for item in windows
    ]

    assert numbers == [
        1,
        2,
        3,
        4,
    ]


def test_generate_windows_preserves_original_dataframe_indexes():
    """
    iloc slicing must preserve original DataFrame indexes.
    """

    df = make_dataframe(30)

    df.index = range(
        100,
        130,
    )

    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
        step_size=5,
    )

    windows = engine.generate_windows(df)

    _, train_df, test_df = windows[0]

    assert train_df.index.tolist() == list(
        range(100, 110)
    )

    assert test_df.index.tolist() == list(
        range(110, 115)
    )


def test_generate_windows_returns_copies():
    """
    Generated windows must not share mutable DataFrame state
    with the caller.
    """

    df = make_dataframe(30)

    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
        step_size=5,
    )

    windows = engine.generate_windows(df)

    _, train_df, test_df = windows[0]

    original_value = df.iloc[0]["close"]

    train_df.loc[
        train_df.index[0],
        "close",
    ] = -999999.0

    test_df.loc[
        test_df.index[0],
        "close",
    ] = -888888.0

    assert (
        df.iloc[0]["close"]
        == original_value
    )


# =========================================================
# RUN WINDOW
# =========================================================

def test_run_window_uses_only_test_dataframe(
    fake_backtest_engine,
):
    """
    BacktestEngine must receive exactly the test window.
    """

    df = make_dataframe(30)

    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
        step_size=5,
        initial_balance=INITIAL_BALANCE,
    )

    result = engine.run_window(
        df=df,
        window_id=1,
        train_start=0,
        train_end=10,
        test_start=10,
        test_end=15,
        initial_balance=INITIAL_BALANCE,
    )

    assert result.train_size == 10

    assert result.test_size == 5

    assert result.window_id == 1

    assert len(
        fake_backtest_engine.calls
    ) == 1


def test_run_window_passes_initial_balance_to_backtest(
    fake_backtest_engine,
):
    """
    Initial balance must be transferred to BacktestEngine.
    """

    df = make_dataframe(30)

    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
        step_size=5,
    )

    engine.run_window(
        df=df,
        window_id=1,
        train_start=0,
        train_end=10,
        test_start=10,
        test_end=15,
        initial_balance=1234.56,
    )

    assert (
        fake_backtest_engine.calls[0][
            "initial_balance"
        ]
        == pytest.approx(
            1234.56
        )
    )


def test_run_window_returns_correct_result_type(
    fake_backtest_engine,
):
    """
    run_window() must return WalkForwardWindowResult.
    """

    df = make_dataframe(30)

    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
    )

    result = engine.run_window(
        df=df,
        window_id=1,
        train_start=0,
        train_end=10,
        test_start=10,
        test_end=15,
        initial_balance=1000.0,
    )

    assert isinstance(
        result,
        WalkForwardWindowResult,
    )


def test_run_window_contains_correct_boundaries(
    fake_backtest_engine,
):
    """
    Window metadata must describe the actual train/test ranges.
    """

    df = make_dataframe(30)

    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
    )

    result = engine.run_window(
        df=df,
        window_id=3,
        train_start=10,
        train_end=20,
        test_start=20,
        test_end=25,
        initial_balance=1000.0,
    )

    assert result.window_id == 3

    assert result.train_start == 10

    assert result.train_end == 20

    assert result.test_start == 20

    assert result.test_end == 25

    assert result.train_size == 10

    assert result.test_size == 5


# =========================================================
# FULL RUN
# =========================================================

def test_run_returns_walk_forward_result(
    fake_backtest_engine,
):
    """
    run() must return WalkForwardResult.
    """

    df = make_dataframe(30)

    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
        step_size=5,
        initial_balance=1000.0,
    )

    result = engine.run(df)

    assert isinstance(
        result,
        WalkForwardResult,
    )


def test_run_creates_expected_number_of_windows(
    fake_backtest_engine,
):
    """
    Full run must execute every valid window.
    """

    df = make_dataframe(30)

    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
        step_size=5,
        initial_balance=1000.0,
    )

    result = engine.run(df)

    assert result.total_windows == 4

    assert len(result.windows) == 4


def test_run_executes_backtest_for_each_window(
    fake_backtest_engine,
):
    """
    BacktestEngine must be instantiated once per window.
    """

    df = make_dataframe(30)

    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
        step_size=5,
    )

    engine.run(df)

    assert len(
        fake_backtest_engine.calls
    ) == 4


# =========================================================
# BALANCE ROLL-FORWARD
# =========================================================

def test_balance_rolls_forward_between_windows(
    fake_backtest_engine,
):
    """
    The final balance of one window must become
    the initial balance of the next window.

    Expected:

        Window 1:
            1000 -> 1050

        Window 2:
            1050 -> 1100

        Window 3:
            1100 -> 1150

        Window 4:
            1150 -> 1200
    """

    df = make_dataframe(30)

    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
        step_size=5,
        initial_balance=1000.0,
    )

    result = engine.run(df)

    balances = [
        call["initial_balance"]
        for call
        in fake_backtest_engine.calls
    ]

    assert balances == pytest.approx(
        [
            1000.0,
            1050.0,
            1100.0,
            1150.0,
        ]
    )

    assert (
        result.final_balance
        == pytest.approx(
            1200.0
        )
    )


def test_net_profit_is_based_on_original_initial_balance(
    fake_backtest_engine,
):
    """
    Walk-forward net profit must compare final balance
    against the original initial balance.
    """

    df = make_dataframe(30)

    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
        step_size=5,
        initial_balance=1000.0,
    )

    result = engine.run(df)

    assert (
        result.net_profit
        == pytest.approx(
            200.0
        )
    )


# =========================================================
# AGGREGATION
# =========================================================

def test_total_trades_are_aggregated(
    fake_backtest_engine,
):
    """
    Every fake window has 5 trades.

    Four windows -> 20 trades.
    """

    df = make_dataframe(30)

    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
        step_size=5,
    )

    result = engine.run(df)

    assert result.total_trades == 20


def test_winning_trades_are_aggregated(
    fake_backtest_engine,
):
    """
    Every five-trade window has two winners.

    Four windows -> 8 winners.
    """

    df = make_dataframe(30)

    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
        step_size=5,
    )

    result = engine.run(df)

    assert result.winning_trades == 8


def test_losing_trades_are_aggregated(
    fake_backtest_engine,
):
    """
    Every five-trade window has three losers.

    Four windows -> 12 losers.
    """

    df = make_dataframe(30)

    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
        step_size=5,
    )

    result = engine.run(df)

    assert result.losing_trades == 12


def test_aggregate_trade_counts_are_consistent(
    fake_backtest_engine,
):
    """
    Winners + losers must equal total trades.
    """

    df = make_dataframe(30)

    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
        step_size=5,
    )

    result = engine.run(df)

    assert (
        result.winning_trades
        + result.losing_trades
        == result.total_trades
    )


def test_aggregate_win_rate_is_correct(
    fake_backtest_engine,
):
    """
    8 winners / 20 trades = 40%.
    """

    df = make_dataframe(30)

    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
        step_size=5,
    )

    result = engine.run(df)

    assert result.win_rate == pytest.approx(
        40.0
    )


# =========================================================
# WINDOW RESULTS
# =========================================================

def test_each_window_contains_backtest_result(
    fake_backtest_engine,
):
    """
    Every WalkForwardWindowResult must contain
    the result returned by BacktestEngine.
    """

    df = make_dataframe(30)

    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
        step_size=5,
    )

    result = engine.run(df)

    for window in result.windows:

        assert (
            window.backtest_result
            is not None
        )

        assert (
            window.backtest_result.final_balance
            >= window.backtest_result.initial_balance
        )


def test_window_number_alias_matches_window_id(
    fake_backtest_engine,
):
    """
    Compatibility alias window_number must match window_id.
    """

    df = make_dataframe(30)

    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
    )

    result = engine.run(df)

    for window in result.windows:

        assert (
            window.window_number
            == window.window_id
        )


def test_window_results_alias_matches_windows(
    fake_backtest_engine,
):
    """
    Compatibility alias window_results must point to
    the same logical collection as windows.
    """

    df = make_dataframe(30)

    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
    )

    result = engine.run(df)

    assert (
        result.window_results
        == result.windows
    )


def test_total_windows_alias_matches_length(
    fake_backtest_engine,
):
    """
    total_windows must equal len(windows).
    """

    df = make_dataframe(30)

    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
    )

    result = engine.run(df)

    assert (
        result.total_windows
        == len(result.windows)
    )


# =========================================================
# RESULT PROPERTY
# =========================================================

def test_result_property_is_none_before_run():
    """
    result must be None before execution.
    """

    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
    )

    assert engine.result is None


def test_result_property_returns_latest_result(
    fake_backtest_engine,
):
    """
    result must return the latest completed result.
    """

    df = make_dataframe(30)

    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
    )

    result = engine.run(df)

    assert (
        engine.result
        is result
    )


# =========================================================
# REPEATED RUN
# =========================================================

def test_repeated_run_replaces_previous_result(
    fake_backtest_engine,
):
    """
    A second run must replace _result with the new result.
    """

    df = make_dataframe(30)

    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
    )

    first_result = engine.run(df)

    second_result = engine.run(df)

    assert (
        engine.result
        is second_result
    )

    assert (
        second_result
        is not first_result
    )


def test_repeated_run_produces_same_final_balance(
    fake_backtest_engine,
):
    """
    Deterministic fake BacktestEngine must produce
    the same final balance on repeated runs.
    """

    df = make_dataframe(30)

    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
        step_size=5,
        initial_balance=1000.0,
    )

    first_result = engine.run(df)

    second_result = engine.run(df)

    assert (
        first_result.final_balance
        == pytest.approx(
            second_result.final_balance
        )
    )

    assert (
        first_result.net_profit
        == pytest.approx(
            second_result.net_profit
        )
    )


# =========================================================
# OVERLAPPING TRAIN WINDOWS
# =========================================================

def test_train_windows_move_by_step_size(
    fake_backtest_engine,
):
    """
    With step_size=5:

        Window 1 train: 0:10
        Window 2 train: 5:15
        Window 3 train: 10:20
        Window 4 train: 15:25
    """

    df = make_dataframe(30)

    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
        step_size=5,
    )

    result = engine.run(df)

    boundaries = [
        (
            window.train_start,
            window.train_end,
        )
        for window in result.windows
    ]

    assert boundaries == [
        (0, 10),
        (5, 15),
        (10, 20),
        (15, 25),
    ]


def test_test_windows_are_sequential(
    fake_backtest_engine,
):
    """
    Test windows must move sequentially:

        10:15
        15:20
        20:25
        25:30
    """

    df = make_dataframe(30)

    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
        step_size=5,
    )

    result = engine.run(df)

    boundaries = [
        (
            window.test_start,
            window.test_end,
        )
        for window in result.windows
    ]

    assert boundaries == [
        (10, 15),
        (15, 20),
        (20, 25),
        (25, 30),
    ]


# =========================================================
# DATA INTEGRITY
# =========================================================

def test_run_does_not_modify_input_dataframe(
    fake_backtest_engine,
):
    """
    WalkForwardEngine must not mutate caller data.
    """

    df = make_dataframe(30)

    original = df.copy(
        deep=True
    )

    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
    )

    engine.run(df)

    pd.testing.assert_frame_equal(
        df,
        original,
    )


def test_run_window_does_not_modify_input_dataframe(
    fake_backtest_engine,
):
    """
    run_window() must not mutate caller data.
    """

    df = make_dataframe(30)

    original = df.copy(
        deep=True
    )

    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
    )

    engine.run_window(
        df=df,
        window_id=1,
        train_start=0,
        train_end=10,
        test_start=10,
        test_end=15,
        initial_balance=1000.0,
    )

    pd.testing.assert_frame_equal(
        df,
        original,
    )


# =========================================================
# DIFFERENT STEP SIZES
# =========================================================

def test_integration_with_step_size_equal_to_test_size(
    fake_backtest_engine,
):
    """
    Standard rolling walk-forward configuration.
    """

    df = make_dataframe(40)

    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
        step_size=5,
    )

    result = engine.run(df)

    assert result.total_windows == 6


def test_integration_with_larger_step_size(
    fake_backtest_engine,
):
    """
    step_size=10 creates non-overlapping train starts.
    """

    df = make_dataframe(50)

    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
        step_size=10,
    )

    result = engine.run(df)

    assert result.total_windows == 4

    boundaries = [
        (
            window.train_start,
            window.train_end,
            window.test_start,
            window.test_end,
        )
        for window in result.windows
    ]

    assert boundaries == [
        (0, 10, 10, 15),
        (10, 20, 20, 25),
        (20, 30, 30, 35),
        (30, 40, 40, 45),
    ]


def test_integration_with_overlapping_test_windows(
    fake_backtest_engine,
):
    """
    step_size smaller than test_size is supported.

    Example:

        train=10
        test=5
        step=2
    """

    df = make_dataframe(30)

    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
        step_size=2,
    )

    result = engine.run(df)

    assert result.total_windows > 1

    for previous, current in zip(
        result.windows,
        result.windows[1:],
    ):

        assert (
            current.train_start
            - previous.train_start
            == 2
        )


# =========================================================
# RUN_WINDOW BALANCE CONTRACT
# =========================================================

def test_run_window_preserves_supplied_initial_balance(
    fake_backtest_engine,
):
    """
    run_window() must preserve the balance supplied by run().
    """

    df = make_dataframe(30)

    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
    )

    result = engine.run_window(
        df=df,
        window_id=2,
        train_start=5,
        train_end=15,
        test_start=15,
        test_end=20,
        initial_balance=1777.25,
    )

    assert (
        result.backtest_result.initial_balance
        == pytest.approx(
            1777.25
        )
    )

    assert (
        result.backtest_result.final_balance
        == pytest.approx(
            1827.25
        )
    )


# =========================================================
# CONVENIENCE FUNCTION
# =========================================================

def test_run_walk_forward_convenience_function(
    monkeypatch,
):
    """
    run_walk_forward() must create the engine, execute it,
    print the report, and return WalkForwardResult.
    """

    FakeBacktestEngine.calls = []

    monkeypatch.setattr(
        walk_forward_module,
        "BacktestEngine",
        FakeBacktestEngine,
    )

    df = make_dataframe(30)

    result = walk_forward_module.run_walk_forward(
        df=df,
        train_size=10,
        test_size=5,
        step_size=5,
        initial_balance=1000.0,
    )

    assert isinstance(
        result,
        WalkForwardResult,
    )

    assert (
        result.final_balance
        == pytest.approx(
            1200.0
        )
    )


# =========================================================
# PRINT REPORT
# =========================================================

def test_print_report_accepts_valid_result(
    fake_backtest_engine,
    capsys,
):
    """
    print_report() must accept WalkForwardResult
    without raising an exception.
    """

    df = make_dataframe(30)

    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
    )

    result = engine.run(df)

    WalkForwardEngine.print_report(
        result
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


# =========================================================
# INVALID RESULT FOR REPORT
# =========================================================

def test_print_report_rejects_invalid_result():
    """
    print_report() must reject arbitrary objects.
    """

    with pytest.raises(
        TypeError
    ):

        WalkForwardEngine.print_report(
            object()
        )
