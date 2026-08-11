from dataclasses import dataclass

import pytest

from src.walk.walk_forward_validator import (
    WalkForwardValidator,
    validate_walk_forward,
)


@dataclass
class FakeWindow:
    train_start: int
    train_end: int
    test_start: int
    test_end: int


@dataclass
class FakeResult:
    total_windows: int
    completed_windows: int
    failed_windows: int
    window_results: list


def make_result() -> FakeResult:
    windows = [
        FakeWindow(
            train_start=0,
            train_end=10,
            test_start=10,
            test_end=15,
        ),
        FakeWindow(
            train_start=5,
            train_end=15,
            test_start=15,
            test_end=20,
        ),
        FakeWindow(
            train_start=10,
            train_end=20,
            test_start=20,
            test_end=25,
        ),
    ]

    return FakeResult(
        total_windows=3,
        completed_windows=3,
        failed_windows=0,
        window_results=windows,
    )


def test_validator_creation() -> None:
    validator = WalkForwardValidator()

    assert validator.min_windows == 1
    assert validator.min_validation_rate == 1.0
    assert validator.latest_report is None


def test_min_windows_must_be_integer() -> None:
    with pytest.raises(TypeError):
        WalkForwardValidator(
            min_windows=1.5
        )


def test_min_windows_must_be_positive() -> None:
    with pytest.raises(ValueError):
        WalkForwardValidator(
            min_windows=0
        )


def test_validation_rate_must_be_valid() -> None:
    with pytest.raises(ValueError):
        WalkForwardValidator(
            min_validation_rate=1.5
        )


def test_validate_success() -> None:
    validator = WalkForwardValidator()

    result = make_result()

    report = validator.validate(
        result
    )

    assert report.total_windows == 3
    assert report.completed_windows == 3
    assert report.failed_windows == 0

    assert report.total_metrics == 5
    assert report.failed_metrics == 0

    assert report.passed is True


def test_latest_report_is_stored() -> None:
    validator = WalkForwardValidator()

    result = make_result()

    report = validator.validate(
        result
    )

    assert validator.latest_report is report


def test_validate_fails_when_not_enough_windows() -> None:
    validator = WalkForwardValidator(
        min_windows=5
    )

    result = make_result()

    report = validator.validate(
        result
    )

    assert report.passed is False


def test_validate_fails_when_window_failed() -> None:
    validator = WalkForwardValidator()

    result = make_result()

    result.failed_windows = 1
    result.completed_windows = 2

    report = validator.validate(
        result
    )

    assert report.passed is False


def test_validate_detects_invalid_boundaries() -> None:
    validator = WalkForwardValidator()

    result = make_result()

    result.window_results[1] = FakeWindow(
        train_start=10,
        train_end=5,
        test_start=5,
        test_end=2,
    )

    report = validator.validate(
        result
    )

    assert report.passed is False

    metric = next(
        metric
        for metric in report.metrics
        if metric.name
        == "valid_window_boundaries"
    )

    assert metric.passed is False


def test_validate_walk_forward_function() -> None:
    result = make_result()

    report = validate_walk_forward(
        result
    )

    assert report.passed is True


def test_window_results_are_counted() -> None:
    validator = WalkForwardValidator()

    result = make_result()

    result.total_windows = 4

    report = validator.validate(
        result
    )

    metric = next(
        metric
        for metric in report.metrics
        if metric.name
        == "window_results_available"
    )

    assert metric.passed is False


def test_completed_windows_must_match_total() -> None:
    validator = WalkForwardValidator()

    result = make_result()

    result.completed_windows = 2

    report = validator.validate(
        result
    )

    metric = next(
        metric
        for metric in report.metrics
        if metric.name
        == "completed_windows"
    )

    assert metric.passed is False


def test_failed_windows_must_be_zero() -> None:
    validator = WalkForwardValidator()

    result = make_result()

    result.failed_windows = 1

    report = validator.validate(
        result
    )

    metric = next(
        metric
        for metric in report.metrics
        if metric.name
        == "failed_windows"
    )

    assert metric.passed is False