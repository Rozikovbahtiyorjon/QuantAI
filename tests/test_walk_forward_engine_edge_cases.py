from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

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


def make_dataframe(rows: int = 100) -> pd.DataFrame:
    timestamps = pd.date_range(
        start="2026-01-01 00:00:00",
        periods=rows,
        freq="15min",
    )

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100.0 + i for i in range(rows)],
            "high": [101.0 + i for i in range(rows)],
            "low": [99.0 + i for i in range(rows)],
            "close": [100.5 + i for i in range(rows)],
            "volume": [1000.0 + i for i in range(rows)],
        }
    )


def make_engine(
    train_size: int = 5,
    test_size: int = 5,
    step_size: int | None = None,
    initial_balance: float = 1000.0,
) -> WalkForwardEngine:
    return WalkForwardEngine(
        train_size=train_size,
        test_size=test_size,
        step_size=step_size,
        initial_balance=initial_balance,
    )


def make_mock_backtest_result(
    initial_balance: float = 1000.0,
    final_balance: float = 1000.0,
    total_trades: int = 0,
    winning_trades: int = 0,
    losing_trades: int = 0,
    win_rate: float = 0.0,
):
    from src.backtest_engine import BacktestResult

    net_profit = final_balance - initial_balance

    return BacktestResult(
        initial_balance=initial_balance,
        final_balance=final_balance,
        net_profit=net_profit,
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate=win_rate,
    )


def test_default_constants_are_valid():
    assert DEFAULT_TRAIN_SIZE > 0
    assert DEFAULT_TEST_SIZE > 0
    assert DEFAULT_INITIAL_BALANCE > 0
    assert MINIMUM_WINDOW_SIZE == 1


def test_default_configuration():
    engine = WalkForwardEngine()

    assert engine.train_size == DEFAULT_TRAIN_SIZE
    assert engine.test_size == DEFAULT_TEST_SIZE
    assert engine.step_size == DEFAULT_TEST_SIZE
    assert engine.initial_balance == DEFAULT_INITIAL_BALANCE
    assert engine.result is None


def test_custom_configuration():
    engine = WalkForwardEngine(
        train_size=20,
        test_size=10,
        step_size=7,
        initial_balance=2500.0,
    )

    assert engine.train_size == 20
    assert engine.test_size == 10
    assert engine.step_size == 7
    assert engine.initial_balance == 2500.0


def test_step_size_defaults_to_test_size():
    engine = WalkForwardEngine(
        train_size=10,
        test_size=7,
    )

    assert engine.step_size == 7


def test_initial_balance_is_converted_to_float():
    engine = WalkForwardEngine(
        train_size=5,
        test_size=5,
        initial_balance=1000,
    )

    assert engine.initial_balance == 1000.0
    assert isinstance(engine.initial_balance, float)


@pytest.mark.parametrize(
    "train_size",
    [
        0,
        -1,
        -10,
    ],
)
def test_invalid_train_size_value(train_size):
    with pytest.raises(ValueError, match="train_size"):
        WalkForwardEngine(
            train_size=train_size,
            test_size=5,
        )


@pytest.mark.parametrize(
    "test_size",
    [
        0,
        -1,
        -10,
    ],
)
def test_invalid_test_size_value(test_size):
    with pytest.raises(ValueError, match="test_size"):
        WalkForwardEngine(
            train_size=5,
            test_size=test_size,
        )


@pytest.mark.parametrize(
    "train_size",
    [
        1.0,
        5.0,
        "5",
        None,
        True,
        False,
    ],
)
def test_invalid_train_size_type(train_size):
    with pytest.raises(TypeError, match="train_size"):
        WalkForwardEngine(
            train_size=train_size,
            test_size=5,
        )


@pytest.mark.parametrize(
    "test_size",
    [
        1.0,
        5.0,
        "5",
        None,
        True,
        False,
    ],
)
def test_invalid_test_size_type(test_size):
    with pytest.raises(TypeError, match="test_size"):
        WalkForwardEngine(
            train_size=5,
            test_size=test_size,
        )


@pytest.mark.parametrize(
    "step_size",
    [
        0,
        -1,
        -10,
    ],
)
def test_invalid_step_size_value(step_size):
    with pytest.raises(ValueError, match="step_size"):
        WalkForwardEngine(
            train_size=5,
            test_size=5,
            step_size=step_size,
        )


@pytest.mark.parametrize(
    "step_size",
    [
        1.0,
        5.0,
        "5",
        None,
        True,
        False,
    ],
)
def test_invalid_step_size_type(step_size):
    if step_size is None:
        engine = WalkForwardEngine(
            train_size=5,
            test_size=5,
            step_size=None,
        )

        assert engine.step_size == 5
    else:
        with pytest.raises(TypeError, match="step_size"):
            WalkForwardEngine(
                train_size=5,
                test_size=5,
                step_size=step_size,
            )


@pytest.mark.parametrize(
    "initial_balance",
    [
        0,
        -1,
        -100,
        0.0,
    ],
)
def test_invalid_initial_balance(initial_balance):
    with pytest.raises(ValueError, match="initial_balance"):
        WalkForwardEngine(
            train_size=5,
            test_size=5,
            initial_balance=initial_balance,
        )


def test_validate_data_accepts_valid_dataframe():
    engine = make_engine(
        train_size=5,
        test_size=5,
    )

    df = make_dataframe(10)

    engine.validate_data(df)


@pytest.mark.parametrize(
    "invalid_data",
    [
        None,
        [],
        [1, 2, 3],
        {},
        "data",
        123,
    ],
)
def test_validate_data_rejects_non_dataframe(invalid_data):
    engine = make_engine(
        train_size=5,
        test_size=5,
    )

    with pytest.raises(TypeError, match="pandas DataFrame"):
        engine.validate_data(invalid_data)


def test_validate_data_rejects_empty_dataframe():
    engine = make_engine(
        train_size=5,
        test_size=5,
    )

    df = pd.DataFrame()

    with pytest.raises(ValueError, match="empty"):
        engine.validate_data(df)


def test_validate_data_rejects_insufficient_rows():
    engine = make_engine(
        train_size=10,
        test_size=5,
    )

    df = make_dataframe(14)

    with pytest.raises(ValueError, match="Not enough rows"):
        engine.validate_data(df)


def test_validate_data_accepts_exact_minimum_rows():
    engine = make_engine(
        train_size=5,
        test_size=5,
    )

    df = make_dataframe(10)

    engine.validate_data(df)


def test_generate_windows_returns_list():
    engine = make_engine(
        train_size=5,
        test_size=5,
    )

    df = make_dataframe(20)

    windows = engine.generate_windows(df)

    assert isinstance(windows, list)


def test_generate_windows_returns_expected_number_for_exact_fit():
    engine = make_engine(
        train_size=5,
        test_size=5,
        step_size=5,
    )

    df = make_dataframe(25)

    windows = engine.generate_windows(df)

    assert len(windows) == 4


def test_generate_windows_returns_expected_number_with_large_step():
    engine = make_engine(
        train_size=5,
        test_size=5,
        step_size=10,
    )

    df = make_dataframe(30)

    windows = engine.generate_windows(df)

    assert len(windows) == 3


def test_generate_windows_with_large_step_leaves_unused_tail():
    engine = make_engine(
        train_size=3,
        test_size=3,
        step_size=20,
    )

    df = make_dataframe(30)

    windows = engine.generate_windows(df)

    assert len(windows) == 2


def test_generate_windows_single_window():
    engine = make_engine(
        train_size=5,
        test_size=5,
    )

    df = make_dataframe(10)

    windows = engine.generate_windows(df)

    assert len(windows) == 1


def test_generate_windows_excludes_incomplete_final_window():
    engine = make_engine(
        train_size=5,
        test_size=5,
        step_size=5,
    )

    df = make_dataframe(23)

    windows = engine.generate_windows(df)

    assert len(windows) == 3


def test_generate_windows_window_numbers_are_sequential():
    engine = make_engine(
        train_size=5,
        test_size=5,
        step_size=5,
    )

    df = make_dataframe(30)

    windows = engine.generate_windows(df)

    numbers = [window[0] for window in windows]

    assert numbers == [1, 2, 3, 4, 5]


def test_generate_windows_each_item_has_three_elements():
    engine = make_engine(
        train_size=5,
        test_size=5,
    )

    df = make_dataframe(20)

    windows = engine.generate_windows(df)

    for window in windows:
        assert len(window) == 3


def test_generate_windows_train_size_is_constant():
    engine = make_engine(
        train_size=5,
        test_size=5,
        step_size=3,
    )

    df = make_dataframe(30)

    windows = engine.generate_windows(df)

    for _, train_df, _ in windows:
        assert len(train_df) == 5


def test_generate_windows_test_size_is_constant():
    engine = make_engine(
        train_size=5,
        test_size=5,
        step_size=3,
    )

    df = make_dataframe(30)

    windows = engine.generate_windows(df)

    for _, _, test_df in windows:
        assert len(test_df) == 5


def test_generate_windows_preserves_original_indexes():
    engine = make_engine(
        train_size=5,
        test_size=5,
        step_size=5,
    )

    df = make_dataframe(25)

    windows = engine.generate_windows(df)

    _, train_df, test_df = windows[0]

    assert list(train_df.index) == [0, 1, 2, 3, 4]
    assert list(test_df.index) == [5, 6, 7, 8, 9]


def test_generate_windows_preserves_non_default_index():
    engine = make_engine(
        train_size=5,
        test_size=5,
        step_size=5,
    )

    df = make_dataframe(20)
    df.index = range(100, 120)

    windows = engine.generate_windows(df)

    _, train_df, test_df = windows[0]

    assert list(train_df.index) == [100, 101, 102, 103, 104]
    assert list(test_df.index) == [105, 106, 107, 108, 109]


def test_generate_windows_returns_copies():
    engine = make_engine(
        train_size=5,
        test_size=5,
    )

    df = make_dataframe(20)

    windows = engine.generate_windows(df)

    _, train_df, test_df = windows[0]

    assert train_df is not df
    assert test_df is not df


def test_generate_windows_train_and_test_are_independent_copies():
    engine = make_engine(
        train_size=5,
        test_size=5,
    )

    df = make_dataframe(20)

    windows = engine.generate_windows(df)

    _, train_df, test_df = windows[0]

    original_train_value = df.loc[0, "close"]
    original_test_value = df.loc[5, "close"]

    train_df.loc[0, "close"] = -999999.0
    test_df.loc[5, "close"] = -888888.0

    assert df.loc[0, "close"] == original_train_value
    assert df.loc[5, "close"] == original_test_value


def test_generate_windows_does_not_modify_original_dataframe():
    engine = make_engine(
        train_size=5,
        test_size=5,
    )

    df = make_dataframe(20)
    original = df.copy(deep=True)

    engine.generate_windows(df)

    pd.testing.assert_frame_equal(df, original)


def test_generate_windows_uses_expanding_start_by_step_size():
    engine = make_engine(
        train_size=5,
        test_size=5,
        step_size=2,
    )

    df = make_dataframe(20)

    windows = engine.generate_windows(df)

    first_window = windows[0]
    second_window = windows[1]
    third_window = windows[2]

    assert list(first_window[1].index) == [0, 1, 2, 3, 4]
    assert list(first_window[2].index) == [5, 6, 7, 8, 9]

    assert list(second_window[1].index) == [2, 3, 4, 5, 6]
    assert list(second_window[2].index) == [7, 8, 9, 10, 11]

    assert list(third_window[1].index) == [4, 5, 6, 7, 8]
    assert list(third_window[2].index) == [9, 10, 11, 12, 13]


def test_generate_windows_supports_step_equal_to_test_size():
    engine = make_engine(
        train_size=5,
        test_size=5,
        step_size=5,
    )

    df = make_dataframe(25)

    windows = engine.generate_windows(df)

    assert list(windows[0][1].index) == [0, 1, 2, 3, 4]
    assert list(windows[0][2].index) == [5, 6, 7, 8, 9]

    assert list(windows[1][1].index) == [5, 6, 7, 8, 9]
    assert list(windows[1][2].index) == [10, 11, 12, 13, 14]


def test_generate_windows_supports_step_larger_than_test_size():
    engine = make_engine(
        train_size=5,
        test_size=5,
        step_size=10,
    )

    df = make_dataframe(30)

    windows = engine.generate_windows(df)

    assert list(windows[0][1].index) == [0, 1, 2, 3, 4]
    assert list(windows[0][2].index) == [5, 6, 7, 8, 9]

    assert list(windows[1][1].index) == [10, 11, 12, 13, 14]
    assert list(windows[1][2].index) == [15, 16, 17, 18, 19]

    assert list(windows[2][1].index) == [20, 21, 22, 23, 24]
    assert list(windows[2][2].index) == [25, 26, 27, 28, 29]


def test_generate_windows_all_test_windows_are_after_train_windows():
    engine = make_engine(
        train_size=5,
        test_size=5,
        step_size=3,
    )

    df = make_dataframe(30)

    windows = engine.generate_windows(df)

    for _, train_df, test_df in windows:
        assert train_df.index[-1] < test_df.index[0]


def test_generate_windows_train_and_test_do_not_overlap_within_window():
    engine = make_engine(
        train_size=5,
        test_size=5,
    )

    df = make_dataframe(20)

    windows = engine.generate_windows(df)

    for _, train_df, test_df in windows:
        assert set(train_df.index).isdisjoint(
            set(test_df.index)
        )


def test_generate_windows_first_window_starts_at_zero():
    engine = make_engine(
        train_size=7,
        test_size=3,
    )

    df = make_dataframe(20)

    windows = engine.generate_windows(df)

    _, train_df, test_df = windows[0]

    assert train_df.index[0] == 0
    assert test_df.index[0] == 7


def test_generate_windows_window_number_starts_at_one():
    engine = make_engine(
        train_size=5,
        test_size=5,
    )

    df = make_dataframe(20)

    windows = engine.generate_windows(df)

    assert windows[0][0] == 1


def test_generate_windows_empty_when_data_below_minimum():
    engine = make_engine(
        train_size=10,
        test_size=10,
    )

    df = make_dataframe(19)

    with pytest.raises(ValueError):
        engine.generate_windows(df)


def test_run_window_rejects_negative_train_start():
    engine = make_engine(
        train_size=5,
        test_size=5,
    )

    df = make_dataframe(20)

    with pytest.raises(ValueError, match="train_start"):
        engine.run_window(
            df=df,
            window_id=1,
            train_start=-1,
            train_end=4,
            test_start=4,
            test_end=9,
            initial_balance=1000.0,
        )


def test_run_window_rejects_empty_train_window():
    engine = make_engine(
        train_size=5,
        test_size=5,
    )

    df = make_dataframe(20)

    with pytest.raises(ValueError, match="Train window"):
        engine.run_window(
            df=df,
            window_id=1,
            train_start=5,
            train_end=5,
            test_start=5,
            test_end=10,
            initial_balance=1000.0,
        )


def test_run_window_rejects_test_before_train_end():
    engine = make_engine(
        train_size=5,
        test_size=5,
    )

    df = make_dataframe(20)

    with pytest.raises(ValueError, match="Test window"):
        engine.run_window(
            df=df,
            window_id=1,
            train_start=0,
            train_end=5,
            test_start=4,
            test_end=9,
            initial_balance=1000.0,
        )


def test_run_window_rejects_empty_test_window():
    engine = make_engine(
        train_size=5,
        test_size=5,
    )

    df = make_dataframe(20)

    with pytest.raises(ValueError, match="Test window"):
        engine.run_window(
            df=df,
            window_id=1,
            train_start=0,
            train_end=5,
            test_start=5,
            test_end=5,
            initial_balance=1000.0,
        )


def test_run_window_rejects_test_end_beyond_data():
    engine = make_engine(
        train_size=5,
        test_size=5,
    )

    df = make_dataframe(20)

    with pytest.raises(ValueError, match="exceeds"):
        engine.run_window(
            df=df,
            window_id=1,
            train_start=0,
            train_end=5,
            test_start=15,
            test_end=25,
            initial_balance=1000.0,
        )


def test_run_window_uses_only_test_dataframe():
    engine = make_engine(
        train_size=5,
        test_size=5,
    )

    df = make_dataframe(20)

    mock_result = make_mock_backtest_result(
        initial_balance=1000.0,
        final_balance=1100.0,
        total_trades=4,
        winning_trades=3,
        losing_trades=1,
        win_rate=75.0,
    )

    with patch(
        "src.walk_forward_engine.BacktestEngine"
    ) as mock_backtest_class:

        mock_backtest = mock_backtest_class.return_value
        mock_backtest.run.return_value = mock_result

        result = engine.run_window(
            df=df,
            window_id=1,
            train_start=0,
            train_end=5,
            test_start=5,
            test_end=10,
            initial_balance=1000.0,
        )

        assert result.window_id == 1
        assert result.train_start == 0
        assert result.train_end == 5
        assert result.test_start == 5
        assert result.test_end == 10
        assert result.train_size == 5
        assert result.test_size == 5

        mock_backtest.run.assert_called_once()

        passed_df = mock_backtest.run.call_args.args[0]

        pd.testing.assert_frame_equal(
            passed_df,
            df.iloc[5:10],
        )


def test_run_window_returns_correct_window_result():
    engine = make_engine(
        train_size=5,
        test_size=5,
    )

    df = make_dataframe(20)

    mock_result = make_mock_backtest_result(
        initial_balance=1000.0,
        final_balance=1050.0,
        total_trades=2,
        winning_trades=1,
        losing_trades=1,
        win_rate=50.0,
    )

    with patch(
        "src.walk_forward_engine.BacktestEngine"
    ) as mock_backtest_class:

        mock_backtest_class.return_value.run.return_value = mock_result

        result = engine.run_window(
            df=df,
            window_id=7,
            train_start=2,
            train_end=7,
            test_start=7,
            test_end=12,
            initial_balance=1000.0,
        )

    assert isinstance(
        result,
        WalkForwardWindowResult,
    )

    assert result.window_id == 7
    assert result.train_start == 2
    assert result.train_end == 7
    assert result.test_start == 7
    assert result.test_end == 12
    assert result.train_size == 5
    assert result.test_size == 5
    assert result.backtest_result is mock_result


def test_window_number_alias():
    result = WalkForwardWindowResult(
        window_id=3,
        train_start=0,
        train_end=5,
        test_start=5,
        test_end=10,
        train_size=5,
        test_size=5,
        backtest_result=make_mock_backtest_result(),
    )

    assert result.window_number == 3


def test_window_number_alias_matches_window_id():
    result = WalkForwardWindowResult(
        window_id=99,
        train_start=10,
        train_end=15,
        test_start=15,
        test_end=20,
        train_size=5,
        test_size=5,
        backtest_result=make_mock_backtest_result(),
    )

    assert result.window_number == result.window_id


def test_walk_forward_result_window_results_alias():
    result = WalkForwardResult(
        initial_balance=1000.0,
        final_balance=1100.0,
        net_profit=100.0,
        total_trades=2,
        winning_trades=1,
        losing_trades=1,
        win_rate=50.0,
        windows=[],
    )

    assert result.window_results is result.windows


def test_walk_forward_result_total_windows_empty():
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


def test_walk_forward_result_total_windows_matches_windows():
    window = WalkForwardWindowResult(
        window_id=1,
        train_start=0,
        train_end=5,
        test_start=5,
        test_end=10,
        train_size=5,
        test_size=5,
        backtest_result=make_mock_backtest_result(),
    )

    result = WalkForwardResult(
        initial_balance=1000.0,
        final_balance=1000.0,
        net_profit=0.0,
        total_trades=0,
        winning_trades=0,
        losing_trades=0,
        win_rate=0.0,
        windows=[window],
    )

    assert result.total_windows == 1
    assert result.window_results == result.windows


def test_run_returns_walk_forward_result():
    engine = make_engine(
        train_size=5,
        test_size=5,
        step_size=5,
    )

    df = make_dataframe(10)

    mock_result = make_mock_backtest_result(
        initial_balance=1000.0,
        final_balance=1100.0,
        total_trades=2,
        winning_trades=1,
        losing_trades=1,
        win_rate=50.0,
    )

    with patch(
        "src.walk_forward_engine.BacktestEngine"
    ) as mock_backtest_class:

        mock_backtest_class.return_value.run.return_value = mock_result

        result = engine.run(df)

    assert isinstance(result, WalkForwardResult)


def test_run_stores_latest_result():
    engine = make_engine(
        train_size=5,
        test_size=5,
    )

    df = make_dataframe(10)

    mock_result = make_mock_backtest_result(
        initial_balance=1000.0,
        final_balance=1025.0,
        total_trades=1,
        winning_trades=1,
        losing_trades=0,
        win_rate=100.0,
    )

    with patch(
        "src.walk_forward_engine.BacktestEngine"
    ) as mock_backtest_class:

        mock_backtest_class.return_value.run.return_value = mock_result

        result = engine.run(df)

    assert engine.result is result


def test_run_initial_balance_is_preserved():
    engine = make_engine(
        train_size=5,
        test_size=5,
        initial_balance=2500.0,
    )

    df = make_dataframe(10)

    mock_result = make_mock_backtest_result(
        initial_balance=2500.0,
        final_balance=2600.0,
        total_trades=1,
        winning_trades=1,
        losing_trades=0,
        win_rate=100.0,
    )

    with patch(
        "src.walk_forward_engine.BacktestEngine"
    ) as mock_backtest_class:

        mock_backtest_class.return_value.run.return_value = mock_result

        result = engine.run(df)

    assert result.initial_balance == 2500.0


def test_run_final_balance_matches_last_window():
    engine = make_engine(
        train_size=5,
        test_size=5,
        step_size=5,
        initial_balance=1000.0,
    )

    df = make_dataframe(15)

    mock_results = [
        make_mock_backtest_result(
            initial_balance=1000.0,
            final_balance=1100.0,
            total_trades=1,
            winning_trades=1,
            losing_trades=0,
            win_rate=100.0,
        ),
        make_mock_backtest_result(
            initial_balance=1100.0,
            final_balance=1200.0,
            total_trades=2,
            winning_trades=1,
            losing_trades=1,
            win_rate=50.0,
        ),
    ]

    with patch(
        "src.walk_forward_engine.BacktestEngine"
    ) as mock_backtest_class:

        mock_backtest_class.return_value.run.side_effect = mock_results

        result = engine.run(df)

    assert result.final_balance == 1200.0


def test_run_rolls_balance_forward_between_windows():
    engine = make_engine(
        train_size=5,
        test_size=5,
        step_size=5,
        initial_balance=1000.0,
    )

    df = make_dataframe(15)

    mock_results = [
        make_mock_backtest_result(
            initial_balance=1000.0,
            final_balance=1100.0,
            total_trades=1,
            winning_trades=1,
            losing_trades=0,
            win_rate=100.0,
        ),
        make_mock_backtest_result(
            initial_balance=1100.0,
            final_balance=1210.0,
            total_trades=1,
            winning_trades=1,
            losing_trades=0,
            win_rate=100.0,
        ),
    ]

    with patch(
        "src.walk_forward_engine.BacktestEngine"
    ) as mock_backtest_class:

        mock_backtest_class.return_value.run.side_effect = mock_results

        result = engine.run(df)

    assert result.windows[0].backtest_result.initial_balance == 1000.0
    assert result.windows[1].backtest_result.initial_balance == 1100.0


def test_run_calculates_net_profit_from_initial_balance():
    engine = make_engine(
        train_size=5,
        test_size=5,
        initial_balance=1000.0,
    )

    df = make_dataframe(10)

    mock_result = make_mock_backtest_result(
        initial_balance=1000.0,
        final_balance=1150.0,
        total_trades=1,
        winning_trades=1,
        losing_trades=0,
        win_rate=100.0,
    )

    with patch(
        "src.walk_forward_engine.BacktestEngine"
    ) as mock_backtest_class:

        mock_backtest_class.return_value.run.return_value = mock_result

        result = engine.run(df)

    assert result.net_profit == 150.0


def test_run_aggregates_total_trades():
    engine = make_engine(
        train_size=5,
        test_size=5,
        step_size=5,
    )

    df = make_dataframe(20)

    mock_results = [
        make_mock_backtest_result(
            initial_balance=1000.0,
            final_balance=1010.0,
            total_trades=2,
            winning_trades=1,
            losing_trades=1,
            win_rate=50.0,
        ),
        make_mock_backtest_result(
            initial_balance=1010.0,
            final_balance=1030.0,
            total_trades=3,
            winning_trades=2,
            losing_trades=1,
            win_rate=66.67,
        ),
        make_mock_backtest_result(
            initial_balance=1030.0,
            final_balance=1040.0,
            total_trades=4,
            winning_trades=3,
            losing_trades=1,
            win_rate=75.0,
        ),
    ]

    with patch(
        "src.walk_forward_engine.BacktestEngine"
    ) as mock_backtest_class:

        mock_backtest_class.return_value.run.side_effect = mock_results

        result = engine.run(df)

    assert result.total_trades == 9


def test_run_aggregates_winning_trades():
    engine = make_engine(
        train_size=5,
        test_size=5,
        step_size=5,
    )

    df = make_dataframe(20)

    mock_results = [
        make_mock_backtest_result(
            initial_balance=1000.0,
            final_balance=1010.0,
            total_trades=2,
            winning_trades=1,
            losing_trades=1,
            win_rate=50.0,
        ),
        make_mock_backtest_result(
            initial_balance=1010.0,
            final_balance=1030.0,
            total_trades=3,
            winning_trades=2,
            losing_trades=1,
            win_rate=66.67,
        ),
        make_mock_backtest_result(
            initial_balance=1030.0,
            final_balance=1040.0,
            total_trades=4,
            winning_trades=3,
            losing_trades=1,
            win_rate=75.0,
        ),
    ]

    with patch(
        "src.walk_forward_engine.BacktestEngine"
    ) as mock_backtest_class:

        mock_backtest_class.return_value.run.side_effect = mock_results

        result = engine.run(df)

    assert result.winning_trades == 6


def test_run_aggregates_losing_trades():
    engine = make_engine(
        train_size=5,
        test_size=5,
        step_size=5,
    )

    df = make_dataframe(20)

    mock_results = [
        make_mock_backtest_result(
            initial_balance=1000.0,
            final_balance=1010.0,
            total_trades=2,
            winning_trades=1,
            losing_trades=1,
            win_rate=50.0,
        ),
        make_mock_backtest_result(
            initial_balance=1010.0,
            final_balance=1030.0,
            total_trades=3,
            winning_trades=2,
            losing_trades=1,
            win_rate=66.67,
        ),
        make_mock_backtest_result(
            initial_balance=1030.0,
            final_balance=1040.0,
            total_trades=4,
            winning_trades=3,
            losing_trades=1,
            win_rate=75.0,
        ),
    ]

    with patch(
        "src.walk_forward_engine.BacktestEngine"
    ) as mock_backtest_class:

        mock_backtest_class.return_value.run.side_effect = mock_results

        result = engine.run(df)

    assert result.losing_trades == 3


def test_run_calculates_aggregate_win_rate():
    engine = make_engine(
        train_size=5,
        test_size=5,
        step_size=5,
    )

    df = make_dataframe(20)

    mock_results = [
        make_mock_backtest_result(
            initial_balance=1000.0,
            final_balance=1010.0,
            total_trades=2,
            winning_trades=1,
            losing_trades=1,
            win_rate=50.0,
        ),
        make_mock_backtest_result(
            initial_balance=1010.0,
            final_balance=1030.0,
            total_trades=3,
            winning_trades=2,
            losing_trades=1,
            win_rate=66.67,
        ),
        make_mock_backtest_result(
            initial_balance=1030.0,
            final_balance=1040.0,
            total_trades=4,
            winning_trades=3,
            losing_trades=1,
            win_rate=75.0,
        ),
    ]

    with patch(
        "src.walk_forward_engine.BacktestEngine"
    ) as mock_backtest_class:

        mock_backtest_class.return_value.run.side_effect = mock_results

        result = engine.run(df)

    assert result.total_trades == 9
    assert result.winning_trades == 6
    assert result.losing_trades == 3
    assert result.win_rate == 66.67


def test_run_handles_zero_trades():
    engine = make_engine(
        train_size=5,
        test_size=5,
    )

    df = make_dataframe(10)

    mock_result = make_mock_backtest_result(
        initial_balance=1000.0,
        final_balance=1000.0,
        total_trades=0,
        winning_trades=0,
        losing_trades=0,
        win_rate=0.0,
    )

    with patch(
        "src.walk_forward_engine.BacktestEngine"
    ) as mock_backtest_class:

        mock_backtest_class.return_value.run.return_value = mock_result

        result = engine.run(df)

    assert result.total_trades == 0
    assert result.winning_trades == 0
    assert result.losing_trades == 0
    assert result.win_rate == 0.0


def test_run_creates_window_results():
    engine = make_engine(
        train_size=5,
        test_size=5,
        step_size=5,
    )

    df = make_dataframe(20)

    mock_result = make_mock_backtest_result(
        initial_balance=1000.0,
        final_balance=1000.0,
        total_trades=0,
        winning_trades=0,
        losing_trades=0,
        win_rate=0.0,
    )

    with patch(
        "src.walk_forward_engine.BacktestEngine"
    ) as mock_backtest_class:

        mock_backtest_class.return_value.run.return_value = mock_result

        result = engine.run(df)

    assert len(result.windows) == 3

    for window in result.windows:
        assert isinstance(
            window,
            WalkForwardWindowResult,
        )


def test_run_window_ids_are_sequential():
    engine = make_engine(
        train_size=5,
        test_size=5,
        step_size=5,
    )

    df = make_dataframe(20)

    mock_result = make_mock_backtest_result()

    with patch(
        "src.walk_forward_engine.BacktestEngine"
    ) as mock_backtest_class:

        mock_backtest_class.return_value.run.return_value = mock_result

        result = engine.run(df)

    assert [
        window.window_id
        for window in result.windows
    ] == [1, 2, 3]


def test_run_window_positions_are_correct():
    engine = make_engine(
        train_size=5,
        test_size=5,
        step_size=5,
    )

    df = make_dataframe(20)

    mock_result = make_mock_backtest_result()

    with patch(
        "src.walk_forward_engine.BacktestEngine"
    ) as mock_backtest_class:

        mock_backtest_class.return_value.run.return_value = mock_result

        result = engine.run(df)

    first = result.windows[0]
    second = result.windows[1]
    third = result.windows[2]

    assert (
        first.train_start,
        first.train_end,
        first.test_start,
        first.test_end,
    ) == (0, 5, 5, 10)

    assert (
        second.train_start,
        second.train_end,
        second.test_start,
        second.test_end,
    ) == (5, 10, 10, 15)

    assert (
        third.train_start,
        third.train_end,
        third.test_start,
        third.test_end,
    ) == (10, 15, 15, 20)


def test_run_raises_when_no_windows_can_be_generated():
    engine = make_engine(
        train_size=10,
        test_size=10,
    )

    df = make_dataframe(19)

    with pytest.raises(ValueError):
        engine.run(df)


def test_result_property_is_none_before_run():
    engine = make_engine()

    assert engine.result is None


def test_print_report_accepts_valid_result(capsys):
    result = WalkForwardResult(
        initial_balance=1000.0,
        final_balance=1100.0,
        net_profit=100.0,
        total_trades=2,
        winning_trades=1,
        losing_trades=1,
        win_rate=50.0,
        windows=[],
    )

    WalkForwardEngine.print_report(result)

    output = capsys.readouterr().out

    assert "QUANTAI WALK-FORWARD REPORT" in output
    assert "Initial Balance" in output
    assert "Final Balance" in output
    assert "Net Profit" in output
    assert "Windows" in output
    assert "Total Trades" in output
    assert "Winning Trades" in output
    assert "Losing Trades" in output
    assert "Win Rate" in output


def test_print_report_rejects_invalid_result():
    with pytest.raises(TypeError, match="WalkForwardResult"):
        WalkForwardEngine.print_report(None)


def test_print_report_contains_window_information(capsys):
    window = WalkForwardWindowResult(
        window_id=1,
        train_start=0,
        train_end=5,
        test_start=5,
        test_end=10,
        train_size=5,
        test_size=5,
        backtest_result=make_mock_backtest_result(
            initial_balance=1000.0,
            final_balance=1050.0,
            total_trades=2,
            winning_trades=1,
            losing_trades=1,
            win_rate=50.0,
        ),
    )

    result = WalkForwardResult(
        initial_balance=1000.0,
        final_balance=1050.0,
        net_profit=50.0,
        total_trades=2,
        winning_trades=1,
        losing_trades=1,
        win_rate=50.0,
        windows=[window],
    )

    WalkForwardEngine.print_report(result)

    output = capsys.readouterr().out

    assert "Window 1" in output
    assert "TRAIN=0:5" in output
    assert "TEST=5:10" in output
    assert "trades=2" in output


def test_run_walk_forward_returns_result():
    df = make_dataframe(10)

    mock_result = make_mock_backtest_result(
        initial_balance=1000.0,
        final_balance=1050.0,
        total_trades=1,
        winning_trades=1,
        losing_trades=0,
        win_rate=100.0,
    )

    with patch(
        "src.walk_forward_engine.BacktestEngine"
    ) as mock_backtest_class:

        mock_backtest_class.return_value.run.return_value = mock_result

        with patch(
            "src.walk_forward_engine.WalkForwardEngine.print_report"
        ) as mock_print_report:

            result = run_walk_forward(
                df=df,
                train_size=5,
                test_size=5,
                step_size=5,
                initial_balance=1000.0,
            )

            assert isinstance(
                result,
                WalkForwardResult,
            )

            mock_print_report.assert_called_once_with(
                result
            )


def test_run_walk_forward_uses_configuration():
    df = make_dataframe(20)

    mock_result = make_mock_backtest_result()

    with patch(
        "src.walk_forward_engine.BacktestEngine"
    ) as mock_backtest_class:

        mock_backtest_class.return_value.run.return_value = mock_result

        with patch(
            "src.walk_forward_engine.WalkForwardEngine.print_report"
        ):
            result = run_walk_forward(
                df=df,
                train_size=4,
                test_size=3,
                step_size=2,
                initial_balance=1500.0,
            )

    assert result.initial_balance == 1500.0


def test_run_does_not_modify_original_dataframe():
    engine = make_engine(
        train_size=5,
        test_size=5,
    )

    df = make_dataframe(10)
    original = df.copy(deep=True)

    mock_result = make_mock_backtest_result()

    with patch(
        "src.walk_forward_engine.BacktestEngine"
    ) as mock_backtest_class:

        mock_backtest_class.return_value.run.return_value = mock_result

        engine.run(df)

    pd.testing.assert_frame_equal(
        df,
        original,
    )


def test_run_window_train_size_is_correct():
    engine = make_engine(
        train_size=7,
        test_size=3,
    )

    df = make_dataframe(20)

    mock_result = make_mock_backtest_result()

    with patch(
        "src.walk_forward_engine.BacktestEngine"
    ) as mock_backtest_class:

        mock_backtest_class.return_value.run.return_value = mock_result

        result = engine.run_window(
            df=df,
            window_id=1,
            train_start=0,
            train_end=7,
            test_start=7,
            test_end=10,
            initial_balance=1000.0,
        )

    assert result.train_size == 7


def test_run_window_test_size_is_correct():
    engine = make_engine(
        train_size=7,
        test_size=3,
    )

    df = make_dataframe(20)

    mock_result = make_mock_backtest_result()

    with patch(
        "src.walk_forward_engine.BacktestEngine"
    ) as mock_backtest_class:

        mock_backtest_class.return_value.run.return_value = mock_result

        result = engine.run_window(
            df=df,
            window_id=1,
            train_start=0,
            train_end=7,
            test_start=7,
            test_end=10,
            initial_balance=1000.0,
        )

    assert result.test_size == 3


def test_run_window_preserves_original_dataframe_index():
    engine = make_engine(
        train_size=5,
        test_size=5,
    )

    df = make_dataframe(20)
    df.index = range(100, 120)

    mock_result = make_mock_backtest_result()

    with patch(
        "src.walk_forward_engine.BacktestEngine"
    ) as mock_backtest_class:

        mock_backtest_class.return_value.run.return_value = mock_result

        result = engine.run_window(
            df=df,
            window_id=1,
            train_start=0,
            train_end=5,
            test_start=5,
            test_end=10,
            initial_balance=1000.0,
        )

        passed_df = (
            mock_backtest_class
            .return_value
            .run
            .call_args
            .args[0]
        )

    assert list(passed_df.index) == [
        105,
        106,
        107,
        108,
        109,
    ]

    assert result.test_start == 5
    assert result.test_end == 10


def test_generate_windows_supports_minimum_window_sizes():
    engine = make_engine(
        train_size=1,
        test_size=1,
        step_size=1,
    )

    df = make_dataframe(5)

    windows = engine.generate_windows(df)

    assert len(windows) == 4

    for _, train_df, test_df in windows:
        assert len(train_df) == 1
        assert len(test_df) == 1


def test_generate_windows_with_step_one():
    engine = make_engine(
        train_size=3,
        test_size=2,
        step_size=1,
    )

    df = make_dataframe(10)

    windows = engine.generate_windows(df)

    assert len(windows) == 6

    for i, (
        window_number,
        train_df,
        test_df,
    ) in enumerate(windows):
        assert window_number == i + 1
        assert len(train_df) == 3
        assert len(test_df) == 2


def test_generate_windows_with_non_overlapping_test_periods():
    engine = make_engine(
        train_size=5,
        test_size=5,
        step_size=5,
    )

    df = make_dataframe(25)

    windows = engine.generate_windows(df)

    previous_test_end = -1

    for _, _, test_df in windows:
        assert test_df.index[0] > previous_test_end
        previous_test_end = test_df.index[-1]


def test_generate_windows_with_step_larger_than_test_has_gaps():
    engine = make_engine(
        train_size=5,
        test_size=5,
        step_size=10,
    )

    df = make_dataframe(30)

    windows = engine.generate_windows(df)

    assert len(windows) == 3

    first_test = windows[0][2]
    second_test = windows[1][2]

    assert first_test.index[-1] + 1 < second_test.index[0]


def test_generate_windows_exact_boundary_is_valid():
    engine = make_engine(
        train_size=5,
        test_size=5,
        step_size=5,
    )

    df = make_dataframe(15)

    windows = engine.generate_windows(df)

    assert len(windows) == 2
    assert windows[-1][2].index[-1] == 14


def test_generate_windows_one_row_short_is_invalid():
    engine = make_engine(
        train_size=5,
        test_size=5,
        step_size=5,
    )

    df = make_dataframe(14)

    windows = engine.generate_windows(df)

    assert len(windows) == 1


def test_generate_windows_does_not_create_partial_test_window():
    engine = make_engine(
        train_size=5,
        test_size=5,
        step_size=5,
    )

    df = make_dataframe(16)

    windows = engine.generate_windows(df)

    for _, train_df, test_df in windows:
        assert len(train_df) == 5
        assert len(test_df) == 5


def test_run_uses_generated_windows_count():
    engine = make_engine(
        train_size=5,
        test_size=5,
        step_size=5,
    )

    df = make_dataframe(20)

    mock_result = make_mock_backtest_result()

    with patch(
        "src.walk_forward_engine.BacktestEngine"
    ) as mock_backtest_class:

        mock_backtest_class.return_value.run.return_value = mock_result

        result = engine.run(df)

    assert result.total_windows == 3


def test_run_calls_backtest_once_per_window():
    engine = make_engine(
        train_size=5,
        test_size=5,
        step_size=5,
    )

    df = make_dataframe(20)

    mock_result = make_mock_backtest_result()

    with patch(
        "src.walk_forward_engine.BacktestEngine"
    ) as mock_backtest_class:

        mock_backtest_class.return_value.run.return_value = mock_result

        engine.run(df)

        assert (
            mock_backtest_class.return_value.run.call_count
            == 3
        )


def test_run_creates_new_backtest_engine_for_each_window():
    engine = make_engine(
        train_size=5,
        test_size=5,
        step_size=5,
    )

    df = make_dataframe(20)

    mock_result = make_mock_backtest_result()

    with patch(
        "src.walk_forward_engine.BacktestEngine"
    ) as mock_backtest_class:

        mock_backtest_class.return_value.run.return_value = mock_result

        engine.run(df)

        assert mock_backtest_class.call_count == 3


def test_result_net_profit_is_rounded():
    engine = make_engine(
        train_size=5,
        test_size=5,
    )

    df = make_dataframe(10)

    mock_result = make_mock_backtest_result(
        initial_balance=1000.0,
        final_balance=1000.123456789123,
        total_trades=1,
        winning_trades=1,
        losing_trades=0,
        win_rate=100.0,
    )

    with patch(
        "src.walk_forward_engine.BacktestEngine"
    ) as mock_backtest_class:

        mock_backtest_class.return_value.run.return_value = mock_result

        result = engine.run(df)

    assert result.net_profit == round(
        result.final_balance - result.initial_balance,
        8,
    )


def test_result_window_results_alias_is_live():
    engine = make_engine(
        train_size=5,
        test_size=5,
    )

    df = make_dataframe(10)

    mock_result = make_mock_backtest_result()

    with patch(
        "src.walk_forward_engine.BacktestEngine"
    ) as mock_backtest_class:

        mock_backtest_class.return_value.run.return_value = mock_result

        result = engine.run(df)

    assert result.window_results is result.windows


def test_result_total_windows_matches_actual_results():
    engine = make_engine(
        train_size=5,
        test_size=5,
        step_size=5,
    )

    df = make_dataframe(25)

    mock_result = make_mock_backtest_result()

    with patch(
        "src.walk_forward_engine.BacktestEngine"
    ) as mock_backtest_class:

        mock_backtest_class.return_value.run.return_value = mock_result

        result = engine.run(df)

    assert result.total_windows == len(result.windows)
    assert result.total_windows == 4
