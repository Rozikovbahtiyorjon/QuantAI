"""
=========================================================
QuantAI WalkForwardEngine Tests
=========================================================

Tests for:

1. Constructor validation
2. Data validation
3. Sequential train/test windows
4. Window sizes
5. Window ordering
6. Step size
7. No incomplete final window
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.walk_forward_engine import (
    WalkForwardEngine,
)


# =========================================================
# HELPERS
# =========================================================

def make_dataframe(rows: int = 20) -> pd.DataFrame:

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
# 1. INVALID TRAIN SIZE
# =========================================================

def test_invalid_train_size():

    with pytest.raises(ValueError):

        WalkForwardEngine(
            train_size=0,
            test_size=5,
        )


# =========================================================
# 2. INVALID TEST SIZE
# =========================================================

def test_invalid_test_size():

    with pytest.raises(ValueError):

        WalkForwardEngine(
            train_size=10,
            test_size=0,
        )


# =========================================================
# 3. INVALID STEP SIZE
# =========================================================

def test_invalid_step_size():

    with pytest.raises(ValueError):

        WalkForwardEngine(
            train_size=10,
            test_size=5,
            step_size=0,
        )


# =========================================================
# 4. INVALID DATA TYPE
# =========================================================

def test_validate_data_requires_dataframe():

    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
    )

    with pytest.raises(TypeError):

        engine.validate_data(
            [1, 2, 3]
        )


# =========================================================
# 5. EMPTY DATA
# =========================================================

def test_validate_data_rejects_empty_dataframe():

    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
    )

    with pytest.raises(ValueError):

        engine.validate_data(
            pd.DataFrame()
        )


# =========================================================
# 6. DEFAULT STEP SIZE
# =========================================================

def test_default_step_size_equals_test_size():

    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
    )

    assert engine.step_size == 5


# =========================================================
# 7. BASIC WINDOW GENERATION
# =========================================================

def test_generate_windows_basic():

    df = make_dataframe(20)

    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
    )

    windows = list(
        engine.generate_windows(df)
    )

    assert len(windows) == 2


# =========================================================
# 8. WINDOW SIZES
# =========================================================

def test_window_sizes_are_correct():

    df = make_dataframe(20)

    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
    )

    windows = list(
        engine.generate_windows(df)
    )

    for (
        window_number,
        train_df,
        test_df,
    ) in windows:

        assert len(train_df) == 10
        assert len(test_df) == 5


# =========================================================
# 9. FIRST WINDOW ORDER
# =========================================================

def test_first_window_order():

    df = make_dataframe(20)

    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
    )

    windows = list(
        engine.generate_windows(df)
    )

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
# 10. SECOND WINDOW ORDER
# =========================================================

def test_second_window_order():

    df = make_dataframe(20)

    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
    )

    windows = list(
        engine.generate_windows(df)
    )

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
# 11. CUSTOM STEP SIZE
# =========================================================

def test_custom_step_size():

    df = make_dataframe(30)

    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
        step_size=3,
    )

    windows = list(
        engine.generate_windows(df)
    )

    assert len(windows) > 2

    first_train = windows[0][1]
    second_train = windows[1][1]

    assert (
        second_train.iloc[0]["close"]
        - first_train.iloc[0]["close"]
        == 3
    )


# =========================================================
# 12. NO INCOMPLETE FINAL WINDOW
# =========================================================

def test_incomplete_final_window_is_not_generated():

    df = make_dataframe(22)

    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
    )

    windows = list(
        engine.generate_windows(df)
    )

    for (
        _,
        train_df,
        test_df,
    ) in windows:

        assert len(train_df) == 10
        assert len(test_df) == 5


# =========================================================
# 13. TRAIN AND TEST DO NOT OVERLAP
# =========================================================

def test_train_and_test_do_not_overlap():

    df = make_dataframe(30)

    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
    )

    windows = list(
        engine.generate_windows(df)
    )

    for (
        _,
        train_df,
        test_df,
    ) in windows:

        train_indexes = set(
            train_df.index
        )

        test_indexes = set(
            test_df.index
        )

        assert train_indexes.isdisjoint(
            test_indexes
        )


# =========================================================
# 14. WINDOWS ARE SEQUENTIAL
# =========================================================

def test_windows_are_sequential():
    """
    Verify that walk-forward windows advance
    according to step_size.

    With:
        train_size = 10
        test_size  = 5
        step_size  = 5

    expected:

        Window 1:
            TRAIN 0-9
            TEST  10-14

        Window 2:
            TRAIN 5-14
            TEST  15-19

    Therefore the next training window starts
    step_size rows after the previous training start.
    """

    df = make_dataframe(30)

    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
    )

    windows = list(
        engine.generate_windows(df)
    )

    for i in range(
        len(windows) - 1
    ):

        current_train = windows[i][1]
        next_train = windows[i + 1][1]

        expected_start = (
            current_train.index[0]
            + engine.step_size
        )

        actual_start = next_train.index[0]

        assert actual_start == expected_start

    df = make_dataframe(30)

    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
    )

    windows = list(
        engine.generate_windows(df)
    )

    for i in range(
        len(windows) - 1
    ):
    
        current_train = windows[i][1]
        next_train = windows[i + 1][1]
    
        expected_start = (
            current_train.index[0]
            + engine.step_size
        )
    
        actual_start = next_train.index[0]
    
        assert actual_start == expected_start