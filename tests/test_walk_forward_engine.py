from __future__ import annotations

import pandas as pd
import pytest

from src.walk_forward_engine import WalkForwardEngine


# =========================================================
# HELPERS
# =========================================================

def make_dataframe(rows: int = 30) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [
                f"2026-01-{i + 1:02d}"
                for i in range(rows)
            ],
            "open": [100.0 + i for i in range(rows)],
            "high": [101.0 + i for i in range(rows)],
            "low": [99.0 + i for i in range(rows)],
            "close": [100.0 + i for i in range(rows)],
            "volume": [1000.0] * rows,
            "atr": [1.0] * rows,
        }
    )


# =========================================================
# 1. CONSTRUCTOR VALIDATION
# =========================================================

def test_invalid_train_size():
    with pytest.raises(ValueError):
        WalkForwardEngine(
            train_size=0,
            test_size=5,
        )


def test_invalid_test_size():
    with pytest.raises(ValueError):
        WalkForwardEngine(
            train_size=10,
            test_size=0,
        )


def test_invalid_step_size():
    with pytest.raises(ValueError):
        WalkForwardEngine(
            train_size=10,
            test_size=5,
            step_size=0,
        )


def test_invalid_train_size_type():
    with pytest.raises(TypeError):
        WalkForwardEngine(
            train_size=10.5,
            test_size=5,
        )


def test_invalid_test_size_type():
    with pytest.raises(TypeError):
        WalkForwardEngine(
            train_size=10,
            test_size=5.5,
        )


def test_invalid_step_size_type():
    with pytest.raises(TypeError):
        WalkForwardEngine(
            train_size=10,
            test_size=5,
            step_size=2.5,
        )


def test_invalid_initial_balance():
    with pytest.raises(ValueError):
        WalkForwardEngine(
            train_size=10,
            test_size=5,
            initial_balance=0,
        )


# =========================================================
# 2. DATA VALIDATION
# =========================================================

def test_validate_data_requires_dataframe():
    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
    )

    with pytest.raises(TypeError):
        engine.validate_data([1, 2, 3])


def test_validate_data_rejects_empty_dataframe():
    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
    )

    with pytest.raises(ValueError):
        engine.validate_data(pd.DataFrame())


def test_validate_data_rejects_insufficient_rows():
    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
    )

    df = make_dataframe(14)

    with pytest.raises(ValueError):
        engine.validate_data(df)


def test_validate_data_accepts_minimum_required_rows():
    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
    )

    df = make_dataframe(15)

    engine.validate_data(df)


# =========================================================
# 3. CONFIGURATION
# =========================================================

def test_default_step_size_equals_test_size():
    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
    )

    assert engine.step_size == 5


def test_custom_step_size_is_preserved():
    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
        step_size=3,
    )

    assert engine.step_size == 3


def test_initial_balance_is_float():
    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
        initial_balance=1000,
    )

    assert isinstance(
        engine.initial_balance,
        float,
    )

    assert engine.initial_balance == 1000.0


# =========================================================
# 4. BASIC WINDOW GENERATION
# =========================================================

def test_generate_windows_basic():
    df = make_dataframe(20)

    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
    )

    windows = engine.generate_windows(df)

    assert len(windows) == 2


def test_generate_windows_returns_expected_tuple_shape():
    df = make_dataframe(20)

    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
    )

    windows = engine.generate_windows(df)

    for window in windows:
        assert isinstance(window, tuple)
        assert len(window) == 3

        window_number, train_df, test_df = window

        assert isinstance(window_number, int)
        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(test_df, pd.DataFrame)


# =========================================================
# 5. WINDOW SIZES
# =========================================================

def test_window_sizes_are_correct():
    df = make_dataframe(30)

    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
    )

    windows = engine.generate_windows(df)

    assert len(windows) > 0

    for _, train_df, test_df in windows:
        assert len(train_df) == 10
        assert len(test_df) == 5


# =========================================================
# 6. FIRST WINDOW ORDER
# =========================================================

def test_first_window_order():
    df = make_dataframe(20)

    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
    )

    windows = engine.generate_windows(df)

    (
        window_number,
        train_df,
        test_df,
    ) = windows[0]

    assert window_number == 1

    assert train_df.iloc[0]["close"] == 100.0
    assert train_df.iloc[-1]["close"] == 109.0

    assert test_df.iloc[0]["close"] == 110.0
    assert test_df.iloc[-1]["close"] == 114.0


# =========================================================
# 7. SECOND WINDOW ORDER
# =========================================================

def test_second_window_order():
    df = make_dataframe(20)

    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
    )

    windows = engine.generate_windows(df)

    (
        window_number,
        train_df,
        test_df,
    ) = windows[1]

    assert window_number == 2

    assert train_df.iloc[0]["close"] == 105.0
    assert train_df.iloc[-1]["close"] == 114.0

    assert test_df.iloc[0]["close"] == 115.0
    assert test_df.iloc[-1]["close"] == 119.0


# =========================================================
# 8. INDEX PRESERVATION
# =========================================================

def test_original_dataframe_indexes_are_preserved():
    df = make_dataframe(30)

    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
    )

    windows = engine.generate_windows(df)

    _, train_df, test_df = windows[0]

    assert train_df.index[0] == 0
    assert train_df.index[-1] == 9

    assert test_df.index[0] == 10
    assert test_df.index[-1] == 14


# =========================================================
# 9. GENERATED WINDOWS ARE COPIES
# =========================================================

def test_generated_windows_are_copies():
    df = make_dataframe(20)

    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
    )

    windows = engine.generate_windows(df)

    train_df = windows[0][1]

    original_value = df.iloc[0]["close"]

    train_df.iloc[
        0,
        train_df.columns.get_loc("close"),
    ] = 999999.0

    assert df.iloc[0]["close"] == original_value


# =========================================================
# 10. CUSTOM STEP SIZE
# =========================================================

def test_custom_step_size():
    df = make_dataframe(30)

    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
        step_size=3,
    )

    windows = engine.generate_windows(df)

    assert len(windows) > 2

    first_train = windows[0][1]
    second_train = windows[1][1]

    assert (
        second_train.iloc[0]["close"]
        - first_train.iloc[0]["close"]
        == 3
    )


# =========================================================
# 11. NO INCOMPLETE FINAL WINDOW
# =========================================================

def test_incomplete_final_window_is_not_generated():
    df = make_dataframe(22)

    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
    )

    windows = engine.generate_windows(df)

    for _, train_df, test_df in windows:
        assert len(train_df) == 10
        assert len(test_df) == 5


# =========================================================
# 12. TRAIN AND TEST DO NOT OVERLAP
# =========================================================

def test_train_and_test_do_not_overlap():
    df = make_dataframe(30)

    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
    )

    windows = engine.generate_windows(df)

    for _, train_df, test_df in windows:
        train_indexes = set(train_df.index)
        test_indexes = set(test_df.index)

        assert train_indexes.isdisjoint(
            test_indexes
        )


# =========================================================
# 13. WINDOWS ARE SEQUENTIAL
# =========================================================

def test_windows_are_sequential():
    df = make_dataframe(30)

    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
        step_size=5,
    )

    windows = engine.generate_windows(df)

    for i in range(len(windows) - 1):
        current_train = windows[i][1]
        next_train = windows[i + 1][1]

        expected_start = (
            current_train.index[0]
            + engine.step_size
        )

        actual_start = next_train.index[0]

        assert actual_start == expected_start


# =========================================================
# 14. WINDOW NUMBERS ARE SEQUENTIAL
# =========================================================

def test_window_numbers_are_sequential():
    df = make_dataframe(50)

    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
    )

    windows = engine.generate_windows(df)

    numbers = [
        window[0]
        for window in windows
    ]

    assert numbers == list(
        range(1, len(windows) + 1)
    )


# =========================================================
# 15. STEP SIZE GREATER THAN TEST SIZE
# =========================================================

def test_step_size_greater_than_test_size():
    df = make_dataframe(50)

    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
        step_size=10,
    )

    windows = engine.generate_windows(df)

    assert len(windows) > 0

    for i in range(len(windows) - 1):
        current_train = windows[i][1]
        next_train = windows[i + 1][1]

        expected_start = (
            current_train.index[0]
            + 10
        )

        assert next_train.index[0] == expected_start


# =========================================================
# 16. WINDOW DATA IS CHRONOLOGICAL
# =========================================================

def test_window_data_is_chronological():
    df = make_dataframe(50)

    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
    )

    windows = engine.generate_windows(df)

    for _, train_df, test_df in windows:
        assert train_df.index.is_monotonic_increasing
        assert test_df.index.is_monotonic_increasing

        assert (
            train_df.index[-1]
            < test_df.index[0]
        )


# =========================================================
# 17. RESULT STATE BEFORE RUN
# =========================================================

def test_result_is_none_before_run():
    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
    )

    assert engine.result is None