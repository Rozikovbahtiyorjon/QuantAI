from __future__ import annotations

import pandas as pd
import pytest

from src.regime_walk_forward import (
    RegimeWalkForward,
    RegimeWalkForwardResult,
    RegimeWindow,
)


def create_market_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "close": [
                100.0,
                101.0,
                102.0,
                103.0,
                104.0,
                103.0,
                102.0,
                101.0,
                100.0,
                99.0,
                98.0,
                99.0,
            ],
            "regime": [
                "BULL",
                "BULL",
                "BULL",
                "BEAR",
                "BEAR",
                "BEAR",
                "RANGE",
                "RANGE",
                "RANGE",
                "BULL",
                "BULL",
                "BEAR",
            ],
        }
    )


def test_generate_windows() -> None:
    engine = RegimeWalkForward()

    windows = engine.generate_windows(
        create_market_data(),
        train_size=4,
        test_size=2,
    )

    assert windows
    assert all(
        isinstance(
            window,
            RegimeWindow,
        )
        for window in windows
    )

    assert all(
        window.regime
        for window in windows
    )


def test_generate_windows_preserves_regime_values() -> None:
    engine = RegimeWalkForward()

    windows = engine.generate_windows(
        create_market_data(),
        train_size=4,
        test_size=2,
    )

    regimes = {
        window.regime
        for window in windows
    }

    assert regimes == {
        "BULL",
        "BEAR",
        "RANGE",
    }


def test_regime_data_is_filtered_correctly() -> None:
    engine = RegimeWalkForward()

    windows = engine.generate_windows(
        create_market_data(),
        train_size=4,
        test_size=2,
    )

    for window in windows:
        if not window.train_df.empty:
            assert (
                window.train_df["regime"]
                == window.regime
            ).all()

        if not window.test_df.empty:
            assert (
                window.test_df["regime"]
                == window.regime
            ).all()


def test_run_returns_result() -> None:
    engine = RegimeWalkForward()

    result = engine.run(
        create_market_data(),
        train_size=4,
        test_size=2,
    )

    assert isinstance(
        result,
        RegimeWalkForwardResult,
    )

    assert result.total_windows > 0
    assert result.regimes == (
        "BEAR",
        "BULL",
        "RANGE",
    )


def test_result_property() -> None:
    engine = RegimeWalkForward()

    assert engine.result is None

    result = engine.run(
        create_market_data(),
        train_size=4,
        test_size=2,
    )

    assert engine.result == result


def test_reset() -> None:
    engine = RegimeWalkForward()

    engine.run(
        create_market_data(),
        train_size=4,
        test_size=2,
    )

    assert engine.result is not None

    engine.reset()

    assert engine.result is None


def test_custom_regime_column() -> None:
    df = pd.DataFrame(
        {
            "close": [
                100.0,
                101.0,
                102.0,
                103.0,
            ],
            "market_regime": [
                "bull",
                "bull",
                "bear",
                "bear",
            ],
        }
    )

    engine = RegimeWalkForward(
        regime_column="market_regime",
    )

    windows = engine.generate_windows(
        df,
        train_size=2,
        test_size=2,
    )

    regimes = {
        window.regime
        for window in windows
    }

    assert regimes == {
        "BULL",
        "BEAR",
    }


def test_regime_values_are_normalized() -> None:
    df = pd.DataFrame(
        {
            "close": [
                100.0,
                101.0,
                102.0,
                103.0,
            ],
            "regime": [
                " bull ",
                "BULL",
                " bear ",
                "BEAR",
            ],
        }
    )

    engine = RegimeWalkForward()

    windows = engine.generate_windows(
        df,
        train_size=2,
        test_size=2,
    )

    assert {
        window.regime
        for window in windows
    } == {
        "BULL",
        "BEAR",
    }


@pytest.mark.parametrize(
    "train_size",
    [0, -1],
)
def test_invalid_train_size(
    train_size: int,
) -> None:
    engine = RegimeWalkForward()

    with pytest.raises(ValueError):
        engine.generate_windows(
            create_market_data(),
            train_size=train_size,
            test_size=2,
        )


@pytest.mark.parametrize(
    "test_size",
    [0, -1],
)
def test_invalid_test_size(
    test_size: int,
) -> None:
    engine = RegimeWalkForward()

    with pytest.raises(ValueError):
        engine.generate_windows(
            create_market_data(),
            train_size=2,
            test_size=test_size,
        )


def test_window_sizes_cannot_exceed_dataframe() -> None:
    engine = RegimeWalkForward()

    with pytest.raises(ValueError):
        engine.generate_windows(
            create_market_data(),
            train_size=10,
            test_size=10,
        )


def test_empty_dataframe_rejected() -> None:
    engine = RegimeWalkForward()

    with pytest.raises(ValueError):
        engine.generate_windows(
            pd.DataFrame(
                {
                    "close": [],
                    "regime": [],
                }
            ),
            train_size=1,
            test_size=1,
        )


def test_missing_regime_column_rejected() -> None:
    engine = RegimeWalkForward()

    df = pd.DataFrame(
        {
            "close": [100.0, 101.0],
        }
    )

    with pytest.raises(ValueError):
        engine.generate_windows(
            df,
            train_size=1,
            test_size=1,
        )


def test_invalid_dataframe_type_rejected() -> None:
    engine = RegimeWalkForward()

    with pytest.raises(TypeError):
        engine.generate_windows(
            [1, 2, 3],  # type: ignore[arg-type]
            train_size=1,
            test_size=1,
        )


def test_empty_regime_rejected() -> None:
    engine = RegimeWalkForward()

    df = pd.DataFrame(
        {
            "close": [
                100.0,
                101.0,
                102.0,
                103.0,
            ],
            "regime": [
                "BULL",
                "",
                "BEAR",
                "BEAR",
            ],
        }
    )

    with pytest.raises(ValueError):
        engine.generate_windows(
            df,
            train_size=2,
            test_size=2,
        )


def test_non_string_regime_rejected() -> None:
    engine = RegimeWalkForward()

    df = pd.DataFrame(
        {
            "close": [
                100.0,
                101.0,
                102.0,
                103.0,
            ],
            "regime": [
                "BULL",
                1,
                "BEAR",
                "BEAR",
            ],
        }
    )

    with pytest.raises(TypeError):
        engine.generate_windows(
            df,
            train_size=2,
            test_size=2,
        )


def test_result_counts_rows_by_regime() -> None:
    engine = RegimeWalkForward()

    result = engine.run(
        create_market_data(),
        train_size=4,
        test_size=2,
    )

    assert set(
        result.windows_by_regime
    ) == {
        "BULL",
        "BEAR",
        "RANGE",
    }

    assert all(
        count > 0
        for count in result.windows_by_regime.values()
    )

    assert all(
        count > 0
        for count in result.rows_by_regime.values()
    )


def test_deterministic_results() -> None:
    df = create_market_data()

    first_engine = RegimeWalkForward()
    second_engine = RegimeWalkForward()

    first = first_engine.run(
        df,
        train_size=4,
        test_size=2,
    )

    second = second_engine.run(
        df,
        train_size=4,
        test_size=2,
    )

    assert first == second


def test_default_regime_column() -> None:
    engine = RegimeWalkForward()

    assert engine.regime_column == "regime"


def test_empty_regime_column_name_rejected() -> None:
    with pytest.raises(ValueError):
        RegimeWalkForward(
            regime_column="   "
        )


def test_invalid_regime_column_type_rejected() -> None:
    with pytest.raises(TypeError):
        RegimeWalkForward(
            regime_column=None,  # type: ignore[arg-type]
        )