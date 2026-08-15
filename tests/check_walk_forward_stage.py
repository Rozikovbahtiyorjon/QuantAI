from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def make_dataframe(rows: int = 40) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                start="2026-01-01",
                periods=rows,
                freq="15min",
            ),
            "open": [100.0 + i for i in range(rows)],
            "high": [101.0 + i for i in range(rows)],
            "low": [99.0 + i for i in range(rows)],
            "close": [100.5 + i for i in range(rows)],
            "volume": [1000.0 + i for i in range(rows)],
            "atr": [1.0] * rows,
        }
    )


def check_imports() -> None:
    from src.walk_forward_engine import WalkForwardEngine
    from src.walk_forward_validator import WalkForwardValidator

    assert WalkForwardEngine is not None
    assert WalkForwardValidator is not None

    print("[PASS] Imports")


def check_engine_configuration() -> None:
    from src.walk_forward_engine import WalkForwardEngine

    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
    )

    assert engine.train_size == 10
    assert engine.test_size == 5
    assert engine.step_size == 5
    assert engine.initial_balance == 1000.0

    print("[PASS] Engine configuration")


def check_window_generation() -> None:
    from src.walk_forward_engine import WalkForwardEngine

    df = make_dataframe(40)

    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
    )

    windows = engine.generate_windows(df)

    assert windows
    assert len(windows) == 6

    for window_number, train_df, test_df in windows:
        assert isinstance(window_number, int)
        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(test_df, pd.DataFrame)

        assert len(train_df) == 10
        assert len(test_df) == 5

        assert train_df.index.is_monotonic_increasing
        assert test_df.index.is_monotonic_increasing

        assert train_df.index[-1] < test_df.index[0]

        assert set(train_df.index).isdisjoint(
            set(test_df.index)
        )

    print(
        "[PASS] Window generation "
        f"({len(windows)} complete windows)"
    )


def check_window_sequence() -> None:
    from src.walk_forward_engine import WalkForwardEngine

    df = make_dataframe(40)

    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
        step_size=5,
    )

    windows = engine.generate_windows(df)

    numbers = [
        window[0]
        for window in windows
    ]

    assert numbers == list(
        range(1, len(windows) + 1)
    )

    for index in range(len(windows) - 1):
        current_train = windows[index][1]
        next_train = windows[index + 1][1]

        assert (
            next_train.index[0]
            == current_train.index[0]
            + engine.step_size
        )

    print("[PASS] Window sequence")


def check_window_copy_isolation() -> None:
    from src.walk_forward_engine import WalkForwardEngine

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

    print("[PASS] Window copy isolation")


def check_engine_validation() -> None:
    from src.walk_forward_engine import WalkForwardEngine

    engine = WalkForwardEngine(
        train_size=10,
        test_size=5,
    )

    try:
        engine.validate_data([])
    except TypeError:
        pass
    else:
        raise AssertionError(
            "WalkForwardEngine accepted non-DataFrame input."
        )

    try:
        engine.validate_data(pd.DataFrame())
    except ValueError:
        pass
    else:
        raise AssertionError(
            "WalkForwardEngine accepted empty DataFrame."
        )

    try:
        engine.validate_data(
            make_dataframe(14)
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "WalkForwardEngine accepted insufficient data."
        )

    assert engine.validate_data(
        make_dataframe(15)
    ) is None

    print("[PASS] Engine data validation")


def check_validator_configuration() -> None:
    from src.walk_forward_validator import WalkForwardValidator

    validator = WalkForwardValidator(
        train_size=10,
        test_size=5,
        step_size=5,
        initial_balance=1000.0,
        minimum_windows=1,
        require_positive_return=False,
        require_positive_window_rate=0.50,
    )

    assert validator.train_size == 10
    assert validator.test_size == 5
    assert validator.step_size == 5
    assert validator.initial_balance == 1000.0
    assert validator.minimum_windows == 1
    assert validator.require_positive_return is False
    assert validator.require_positive_window_rate == 0.50

    print("[PASS] Validator configuration")


def check_validator_data_validation() -> None:
    from src.walk_forward_validator import WalkForwardValidator

    validator = WalkForwardValidator(
        train_size=10,
        test_size=5,
    )

    try:
        validator.validate_data([])
    except TypeError:
        pass
    else:
        raise AssertionError(
            "Validator accepted non-DataFrame input."
        )

    try:
        validator.validate_data(pd.DataFrame())
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Validator accepted empty DataFrame."
        )

    try:
        validator.validate_data(
            make_dataframe(14)
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Validator accepted insufficient data."
        )

    assert validator.validate_data(
        make_dataframe(15)
    ) is True

    print("[PASS] Validator data validation")


def make_fake_backtest_result(
    initial_balance: float,
    net_profit: float,
    total_trades: int,
    winning_trades: int,
    losing_trades: int,
    win_rate: float,
):
    return type(
        "FakeBacktestResult",
        (),
        {
            "initial_balance": initial_balance,
            "final_balance": (
                initial_balance + net_profit
            ),
            "net_profit": net_profit,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
        },
    )()


def make_fake_window(
    window_id: int,
    profit: float,
    win_rate: float,
):
    from src.walk_forward_engine import (
        WalkForwardWindowResult,
    )

    total_trades = 10
    winning_trades = int(
        total_trades * win_rate / 100.0
    )
    losing_trades = (
        total_trades - winning_trades
    )

    return WalkForwardWindowResult(
        window_id=window_id,
        train_start=(window_id - 1) * 15,
        train_end=((window_id - 1) * 15) + 10,
        test_start=((window_id - 1) * 15) + 10,
        test_end=((window_id - 1) * 15) + 15,
        train_size=10,
        test_size=5,
        backtest_result=make_fake_backtest_result(
            initial_balance=1000.0,
            net_profit=profit,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
        ),
    )


def check_validator_result_analysis() -> None:
    from src.walk_forward_engine import WalkForwardResult
    from src.walk_forward_validator import (
        WalkForwardValidationResult,
        WalkForwardValidator,
    )

    windows = [
        make_fake_window(
            window_id=1,
            profit=100.0,
            win_rate=60.0,
        ),
        make_fake_window(
            window_id=2,
            profit=-50.0,
            win_rate=40.0,
        ),
        make_fake_window(
            window_id=3,
            profit=200.0,
            win_rate=80.0,
        ),
    ]

    total_trades = sum(
        window.backtest_result.total_trades
        for window in windows
    )

    winning_trades = sum(
        window.backtest_result.winning_trades
        for window in windows
    )

    losing_trades = sum(
        window.backtest_result.losing_trades
        for window in windows
    )

    initial_balance = 1000.0

    net_profit = sum(
        window.backtest_result.net_profit
        for window in windows
    )

    final_balance = (
        initial_balance + net_profit
    )

    win_rate = (
        winning_trades
        / total_trades
        * 100.0
    )

    result = WalkForwardResult(
        initial_balance=initial_balance,
        final_balance=final_balance,
        net_profit=net_profit,
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate=win_rate,
        windows=windows,
    )

    validator = WalkForwardValidator(
        train_size=10,
        test_size=5,
        step_size=5,
    )

    validation_result = (
        validator._build_validation_result(
            result
        )
    )

    assert isinstance(
        validation_result,
        WalkForwardValidationResult,
    )

    assert validation_result.total_windows == 3
    assert validation_result.profitable_windows == 2
    assert validation_result.losing_windows == 1
    assert validation_result.flat_windows == 0

    assert (
        validation_result.profitable_window_rate
        == 66.67
    )

    assert (
        validation_result.losing_window_rate
        == 33.33
    )

    assert validation_result.total_trades == 30
    assert validation_result.winning_trades == 18
    assert validation_result.losing_trades == 12

    assert validation_result.win_rate == 60.0

    assert validation_result.initial_balance == 1000.0
    assert validation_result.final_balance == 1250.0
    assert validation_result.net_profit == 250.0
    assert validation_result.return_percent == 25.0

    assert validation_result.best_window_profit == 200.0
    assert validation_result.worst_window_profit == -50.0

    expected_average_profit = (
        100.0 - 50.0 + 200.0
    ) / 3.0

    assert abs(
        validation_result.average_window_profit
        - expected_average_profit
    ) < 1e-8

    assert validation_result.best_window_win_rate == 80.0
    assert validation_result.worst_window_win_rate == 40.0
    assert validation_result.average_window_win_rate == 60.0

    assert validation_result.validation_passed is True
    assert 0.0 <= validation_result.validation_score <= 100.0

    print("[PASS] Validator result analysis")


def check_validator_result_validation() -> None:
    from src.walk_forward_engine import WalkForwardResult
    from src.walk_forward_validator import WalkForwardValidator

    valid_result = WalkForwardResult(
        initial_balance=1000.0,
        final_balance=1100.0,
        net_profit=100.0,
        total_trades=10,
        winning_trades=6,
        losing_trades=4,
        win_rate=60.0,
        windows=[
            make_fake_window(
                window_id=1,
                profit=100.0,
                win_rate=60.0,
            )
        ],
    )

    assert (
        WalkForwardValidator.validate_result(
            valid_result
        )
        is None
    )

    try:
        WalkForwardValidator.validate_result(
            None
        )
    except TypeError:
        pass
    else:
        raise AssertionError(
            "Validator accepted None as a valid result."
        )

    print("[PASS] Validator result validation")


def check_existing_pytest_suite() -> None:
    commands = [
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_walk_forward_engine.py",
        ],
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_walk_forward_validator.py",
        ],
    ]

    for command in commands:
        print(
            "[RUN] "
            + " ".join(command)
        )

        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
        )

        if completed.returncode != 0:
            raise RuntimeError(
                "Existing pytest suite failed: "
                + " ".join(command)
            )

    print("[PASS] Existing Walk-Forward pytest suites")


def main() -> None:
    print("=" * 72)
    print("QuantAI Walk-Forward Stage Check")
    print("=" * 72)

    checks = [
        check_imports,
        check_engine_configuration,
        check_window_generation,
        check_window_sequence,
        check_window_copy_isolation,
        check_engine_validation,
        check_validator_configuration,
        check_validator_data_validation,
        check_validator_result_analysis,
        check_validator_result_validation,
    ]

    for check in checks:
        check()

    check_existing_pytest_suite()

    print("=" * 72)
    print("WALK-FORWARD STAGE CHECK: SUCCESS")
    print("=" * 72)


if __name__ == "__main__":
    main()