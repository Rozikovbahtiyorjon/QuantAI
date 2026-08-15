import pandas as pd
import pytest

from src.walk_forward_engine import (
    DEFAULT_INITIAL_BALANCE,
    DEFAULT_TRAIN_SIZE,
    DEFAULT_TEST_SIZE,
    MINIMUM_WINDOW_SIZE,
    WalkForwardResult,
    WalkForwardWindowResult,
)

from src.walk_forward_validator import (
    WalkForwardValidationResult,
    WalkForwardValidator,
    validate_walk_forward,
)


def make_dataframe(rows: int = 30) -> pd.DataFrame:
    timestamps = pd.date_range(
        start="2025-01-01",
        periods=rows,
        freq="15min",
    )

    data = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100.0 + i for i in range(rows)],
            "high": [101.0 + i for i in range(rows)],
            "low": [99.0 + i for i in range(rows)],
            "close": [100.5 + i for i in range(rows)],
            "volume": [1000.0 + i for i in range(rows)],
            "atr": [1.0 for _ in range(rows)],
        }
    )

    return data


def make_backtest_result(
    initial_balance: float = 1000.0,
    final_balance: float = 1100.0,
    net_profit: float = 100.0,
    total_trades: int = 10,
    winning_trades: int = 6,
    losing_trades: int = 4,
    win_rate: float = 60.0,
):
    return type(
        "FakeBacktestResult",
        (),
        {
            "initial_balance": initial_balance,
            "final_balance": final_balance,
            "net_profit": net_profit,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
        },
    )()


def make_window(
    window_id: int = 1,
    profit: float = 100.0,
    win_rate: float = 60.0,
):
    return WalkForwardWindowResult(
        window_id=window_id,
        train_start=(window_id - 1) * 15,
        train_end=((window_id - 1) * 15) + 10,
        test_start=((window_id - 1) * 15) + 10,
        test_end=((window_id - 1) * 15) + 15,
        train_size=10,
        test_size=5,
        backtest_result=make_backtest_result(
            initial_balance=1000.0,
            final_balance=1000.0 + profit,
            net_profit=profit,
            total_trades=10,
            winning_trades=int(10 * win_rate / 100.0),
            losing_trades=10 - int(10 * win_rate / 100.0),
            win_rate=win_rate,
        ),
    )


def make_valid_walk_forward_result(
    windows=None,
) -> WalkForwardResult:
    if windows is None:
        windows = [
            make_window(1, 100.0, 60.0),
            make_window(2, 50.0, 70.0),
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
    final_balance = initial_balance + sum(
        window.backtest_result.net_profit
        for window in windows
    )

    net_profit = final_balance - initial_balance

    win_rate = (
        winning_trades / total_trades * 100.0
        if total_trades > 0
        else 0.0
    )

    return WalkForwardResult(
        initial_balance=initial_balance,
        final_balance=final_balance,
        net_profit=net_profit,
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate=win_rate,
        windows=windows,
    )


def test_validator_can_be_created():
    validator = WalkForwardValidator()

    assert validator is not None


def test_validator_default_configuration():
    validator = WalkForwardValidator()

    assert validator.train_size == DEFAULT_TRAIN_SIZE
    assert validator.test_size == DEFAULT_TEST_SIZE
    assert validator.step_size == DEFAULT_TEST_SIZE
    assert validator.initial_balance == pytest.approx(
        DEFAULT_INITIAL_BALANCE
    )
    assert validator.minimum_windows == 1
    assert validator.require_positive_return is False
    assert validator.require_positive_window_rate == pytest.approx(
        0.50
    )


def test_validator_preserves_configuration():
    validator = WalkForwardValidator(
        train_size=10,
        test_size=5,
        step_size=3,
        initial_balance=1234.56,
        minimum_windows=2,
        require_positive_return=True,
        require_positive_window_rate=0.75,
    )

    assert validator.train_size == 10
    assert validator.test_size == 5
    assert validator.step_size == 3
    assert validator.initial_balance == pytest.approx(
        1234.56
    )
    assert validator.minimum_windows == 2
    assert validator.require_positive_return is True
    assert validator.require_positive_window_rate == pytest.approx(
        0.75
    )


def test_validator_default_step_size_equals_test_size():
    validator = WalkForwardValidator(
        train_size=10,
        test_size=7,
    )

    assert validator.step_size == 7


def test_validator_rejects_invalid_train_size():
    with pytest.raises((TypeError, ValueError)):
        WalkForwardValidator(
            train_size=0,
            test_size=5,
        )


def test_validator_rejects_invalid_test_size():
    with pytest.raises((TypeError, ValueError)):
        WalkForwardValidator(
            train_size=10,
            test_size=0,
        )


def test_validator_rejects_invalid_step_size():
    with pytest.raises((TypeError, ValueError)):
        WalkForwardValidator(
            train_size=10,
            test_size=5,
            step_size=0,
        )


def test_validator_rejects_invalid_initial_balance():
    with pytest.raises((TypeError, ValueError)):
        WalkForwardValidator(
            train_size=10,
            test_size=5,
            initial_balance=0,
        )


def test_validator_rejects_invalid_minimum_windows():
    with pytest.raises((TypeError, ValueError)):
        WalkForwardValidator(
            minimum_windows=0,
        )


def test_validator_rejects_invalid_positive_return_flag():
    with pytest.raises(TypeError):
        WalkForwardValidator(
            require_positive_return=1,
        )


def test_validator_rejects_invalid_positive_window_rate():
    with pytest.raises((TypeError, ValueError)):
        WalkForwardValidator(
            require_positive_window_rate=1.5,
        )


def test_validator_rejects_negative_positive_window_rate():
    with pytest.raises((TypeError, ValueError)):
        WalkForwardValidator(
            require_positive_window_rate=-0.1,
        )


def test_validate_data_rejects_non_dataframe():
    validator = WalkForwardValidator(
        train_size=10,
        test_size=5,
    )

    with pytest.raises(TypeError):
        validator.validate_data([1, 2, 3])


def test_validate_data_rejects_empty_dataframe():
    validator = WalkForwardValidator(
        train_size=10,
        test_size=5,
    )

    with pytest.raises(ValueError):
        validator.validate_data(pd.DataFrame())


def test_validate_data_rejects_insufficient_data():
    validator = WalkForwardValidator(
        train_size=10,
        test_size=5,
    )

    df = make_dataframe(14)

    with pytest.raises(ValueError):
        validator.validate_data(df)


def test_validate_data_accepts_minimum_required_data():
    validator = WalkForwardValidator(
        train_size=10,
        test_size=5,
    )

    df = make_dataframe(15)

    assert validator.validate_data(df) is True


def test_validate_data_accepts_larger_dataframe():
    validator = WalkForwardValidator(
        train_size=10,
        test_size=5,
    )

    df = make_dataframe(100)

    assert validator.validate_data(df) is True


def test_validate_dataframe_returns_true():
    validator = WalkForwardValidator(
        train_size=10,
        test_size=5,
    )

    df = make_dataframe(30)

    assert validator.validate(df) is True


def test_validate_rejects_invalid_input():
    validator = WalkForwardValidator(
        train_size=10,
        test_size=5,
    )

    with pytest.raises(TypeError):
        validator.validate(None)


def test_validate_result_accepts_real_walk_forward_result():
    result = make_valid_walk_forward_result()

    validated = WalkForwardValidator.validate_result(
        result
    )

    assert validated is None


def test_validate_result_rejects_fake_result():
    class FakeWalkForwardResult:
        initial_balance = 1000.0
        final_balance = 1100.0
        net_profit = 100.0
        total_trades = 20
        winning_trades = 8
        losing_trades = 12
        win_rate = 40.0
        windows = []

    fake_result = FakeWalkForwardResult()

    with pytest.raises(
        TypeError,
        match="result must be WalkForwardResult",
    ):
        WalkForwardValidator.validate_result(
            fake_result
        )


def test_validate_result_rejects_none():
    with pytest.raises(
        TypeError,
        match="result must be WalkForwardResult",
    ):
        WalkForwardValidator.validate_result(None)


def test_validate_result_rejects_result_without_windows():
    result = WalkForwardResult(
        initial_balance=1000.0,
        final_balance=1000.0,
        net_profit=0.0,
        total_trades=0,
        winning_trades=0,
        losing_trades=0,
        win_rate=0.0,
        windows=[],
    )

    with pytest.raises(
        ValueError,
        match="no windows",
    ):
        WalkForwardValidator.validate_result(
            result
        )


def test_validate_result_rejects_negative_total_trades():
    result = WalkForwardResult(
        initial_balance=1000.0,
        final_balance=1000.0,
        net_profit=0.0,
        total_trades=-1,
        winning_trades=0,
        losing_trades=0,
        win_rate=0.0,
        windows=[make_window()],
    )

    with pytest.raises(
        ValueError,
        match="total_trades cannot be negative",
    ):
        WalkForwardValidator.validate_result(
            result
        )


def test_validate_result_rejects_inconsistent_trade_statistics():
    result = WalkForwardResult(
        initial_balance=1000.0,
        final_balance=1000.0,
        net_profit=0.0,
        total_trades=10,
        winning_trades=8,
        losing_trades=1,
        win_rate=80.0,
        windows=[make_window()],
    )

    with pytest.raises(
        ValueError,
        match="Winning and losing trades",
    ):
        WalkForwardValidator.validate_result(
            result
        )


def test_validate_result_accepts_valid_trade_statistics():
    result = make_valid_walk_forward_result()

    WalkForwardValidator.validate_result(
        result
    )


def test_build_validation_result_returns_correct_type():
    validator = WalkForwardValidator()

    result = make_valid_walk_forward_result()

    validation_result = (
        validator._build_validation_result(result)
    )

    assert isinstance(
        validation_result,
        WalkForwardValidationResult,
    )


def test_build_validation_result_calculates_windows():
    validator = WalkForwardValidator()

    result = make_valid_walk_forward_result(
        [
            make_window(1, 100.0, 60.0),
            make_window(2, -50.0, 40.0),
            make_window(3, 0.0, 50.0),
        ]
    )

    validation_result = (
        validator._build_validation_result(result)
    )

    assert validation_result.total_windows == 3
    assert validation_result.profitable_windows == 1
    assert validation_result.losing_windows == 1
    assert validation_result.flat_windows == 1
    assert validation_result.profitable_window_rate == pytest.approx(
        33.33,
        abs=0.01,
    )
    assert validation_result.losing_window_rate == pytest.approx(
        33.33,
        abs=0.01,
    )


def test_build_validation_result_calculates_profit_statistics():
    validator = WalkForwardValidator()

    result = make_valid_walk_forward_result(
        [
            make_window(1, 100.0, 60.0),
            make_window(2, -50.0, 40.0),
            make_window(3, 200.0, 80.0),
        ]
    )

    validation_result = (
        validator._build_validation_result(result)
    )

    assert validation_result.best_window_profit == pytest.approx(
        200.0
    )
    assert validation_result.worst_window_profit == pytest.approx(
        -50.0
    )
    assert validation_result.average_window_profit == pytest.approx(
        83.33333333,
        abs=1e-6,
    )


def test_build_validation_result_calculates_win_rate_statistics():
    validator = WalkForwardValidator()

    result = make_valid_walk_forward_result(
        [
            make_window(1, 100.0, 60.0),
            make_window(2, 50.0, 40.0),
            make_window(3, 200.0, 80.0),
        ]
    )

    validation_result = (
        validator._build_validation_result(result)
    )

    assert validation_result.best_window_win_rate == pytest.approx(
        80.0
    )
    assert validation_result.worst_window_win_rate == pytest.approx(
        40.0
    )
    assert validation_result.average_window_win_rate == pytest.approx(
        60.0
    )


def test_build_validation_result_calculates_return():
    validator = WalkForwardValidator()

    result = make_valid_walk_forward_result(
        [
            make_window(1, 100.0, 60.0),
            make_window(2, 50.0, 70.0),
        ]
    )

    validation_result = (
        validator._build_validation_result(result)
    )

    assert validation_result.initial_balance == pytest.approx(
        1000.0
    )
    assert validation_result.final_balance == pytest.approx(
        1150.0
    )
    assert validation_result.net_profit == pytest.approx(
        150.0
    )
    assert validation_result.return_percent == pytest.approx(
        15.0
    )


def test_build_validation_result_calculates_trade_statistics():
    validator = WalkForwardValidator()

    result = make_valid_walk_forward_result(
        [
            make_window(1, 100.0, 60.0),
            make_window(2, 50.0, 70.0),
        ]
    )

    validation_result = (
        validator._build_validation_result(result)
    )

    assert validation_result.total_trades == 20
    assert validation_result.winning_trades == 13
    assert validation_result.losing_trades == 7
    assert validation_result.win_rate == pytest.approx(
        65.0
    )


def test_validation_passes_with_default_requirements():
    validator = WalkForwardValidator()

    result = make_valid_walk_forward_result(
        [
            make_window(1, 100.0, 60.0),
            make_window(2, 50.0, 70.0),
        ]
    )

    validation_result = (
        validator.validate(result)
    )

    assert isinstance(
        validation_result,
        WalkForwardValidationResult,
    )

    assert validation_result.validation_passed is True


def test_validation_fails_when_profitable_window_rate_is_too_low():
    validator = WalkForwardValidator(
        require_positive_window_rate=0.75,
    )

    result = make_valid_walk_forward_result(
        [
            make_window(1, 100.0, 60.0),
            make_window(2, -50.0, 40.0),
        ]
    )

    validation_result = (
        validator.validate(result)
    )

    assert validation_result.validation_passed is False


def test_validation_requires_positive_return_when_configured():
    validator = WalkForwardValidator(
        require_positive_return=True,
    )

    result = make_valid_walk_forward_result(
        [
            make_window(1, -100.0, 40.0),
            make_window(2, 50.0, 60.0),
        ]
    )

    validation_result = (
        validator.validate(result)
    )

    assert validation_result.validation_passed is False


def test_validation_score_is_between_zero_and_one_hundred():
    validator = WalkForwardValidator()

    result = make_valid_walk_forward_result()

    validation_result = (
        validator.validate(result)
    )

    assert 0.0 <= validation_result.validation_score <= 100.0


def test_validate_result_stores_latest_validation_result():
    validator = WalkForwardValidator()

    result = make_valid_walk_forward_result()

    validation_result = validator.validate(result)

    assert validator.result is validation_result
    assert isinstance(
        validator.result,
        WalkForwardValidationResult,
    )


def test_run_returns_validation_result():
    validator = WalkForwardValidator(
        train_size=10,
        test_size=5,
        step_size=5,
    )

    df = make_dataframe(30)

    result = validator.run(df)

    assert isinstance(
        result,
        WalkForwardValidationResult,
    )


def test_run_stores_latest_validation_result():
    validator = WalkForwardValidator(
        train_size=10,
        test_size=5,
    )

    df = make_dataframe(30)

    result = validator.run(df)

    assert validator.result is result


def test_run_does_not_modify_input_dataframe():
    df = make_dataframe(30)
    original = df.copy(deep=True)

    validator = WalkForwardValidator(
        train_size=10,
        test_size=5,
    )

    result = validator.run(df)

    assert isinstance(
        result,
        WalkForwardValidationResult,
    )

    pd.testing.assert_frame_equal(
        df,
        original,
    )


def test_run_produces_at_least_one_window():
    validator = WalkForwardValidator(
        train_size=10,
        test_size=5,
        step_size=5,
    )

    df = make_dataframe(30)

    result = validator.run(df)

    assert result.total_windows > 0
    assert len(result.windows) > 0


def test_run_result_contains_balance_information():
    validator = WalkForwardValidator(
        train_size=10,
        test_size=5,
        initial_balance=1234.56,
    )

    df = make_dataframe(30)

    result = validator.run(df)

    assert result.initial_balance == pytest.approx(
        1234.56
    )
    assert isinstance(
        result.final_balance,
        (int, float),
    )
    assert isinstance(
        result.net_profit,
        (int, float),
    )


def test_run_result_contains_trade_statistics():
    validator = WalkForwardValidator(
        train_size=10,
        test_size=5,
    )

    df = make_dataframe(30)

    result = validator.run(df)

    assert isinstance(
        result.total_trades,
        int,
    )
    assert isinstance(
        result.winning_trades,
        int,
    )
    assert isinstance(
        result.losing_trades,
        int,
    )
    assert isinstance(
        result.win_rate,
        (int, float),
    )

    assert result.total_trades >= 0
    assert result.winning_trades >= 0
    assert result.losing_trades >= 0
    assert 0.0 <= result.win_rate <= 100.0


def test_run_result_contains_validation_metrics():
    validator = WalkForwardValidator(
        train_size=10,
        test_size=5,
    )

    df = make_dataframe(30)

    result = validator.run(df)

    assert isinstance(
        result.validation_passed,
        bool,
    )

    assert isinstance(
        result.validation_score,
        (int, float),
    )

    assert 0.0 <= result.validation_score <= 100.0


def test_run_is_deterministic():
    df = make_dataframe(30)

    validator = WalkForwardValidator(
        train_size=10,
        test_size=5,
        initial_balance=1000.0,
    )

    first_result = validator.run(df)
    second_result = validator.run(df)

    assert isinstance(
        first_result,
        WalkForwardValidationResult,
    )

    assert isinstance(
        second_result,
        WalkForwardValidationResult,
    )

    assert first_result.initial_balance == second_result.initial_balance
    assert first_result.final_balance == pytest.approx(
        second_result.final_balance
    )
    assert first_result.net_profit == pytest.approx(
        second_result.net_profit
    )
    assert first_result.total_trades == second_result.total_trades
    assert first_result.winning_trades == second_result.winning_trades
    assert first_result.losing_trades == second_result.losing_trades
    assert first_result.win_rate == pytest.approx(
        second_result.win_rate
    )
    assert first_result.total_windows == second_result.total_windows
    assert len(first_result.windows) == len(
        second_result.windows
    )


def test_print_report_accepts_valid_result(capsys):
    validator = WalkForwardValidator()

    result = make_valid_walk_forward_result()

    validation_result = validator.validate(result)

    validator.print_report(validation_result)

    captured = capsys.readouterr()

    assert "QUANTAI WALK-FORWARD VALIDATION REPORT" in captured.out
    assert "Validation Status" in captured.out
    assert "Validation Score" in captured.out


def test_print_report_rejects_invalid_result():
    with pytest.raises(
        TypeError,
        match="result must be WalkForwardValidationResult",
    ):
        WalkForwardValidator.print_report(
            object()
        )


def test_convenience_function_returns_validation_result():
    result = make_valid_walk_forward_result()

    validation_result = validate_walk_forward(
        result
    )

    assert isinstance(
        validation_result,
        WalkForwardValidationResult,
    )


def test_convenience_function_can_require_positive_return():
    result = make_valid_walk_forward_result(
        [
            make_window(1, -100.0, 40.0),
            make_window(2, -50.0, 40.0),
        ]
    )

    validation_result = validate_walk_forward(
        result,
        require_positive_return=True,
    )

    assert validation_result.validation_passed is False