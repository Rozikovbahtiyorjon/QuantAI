from dataclasses import dataclass

import pytest

from experimental.src.quantai_walk_forward_validation import (
    QuantAIWalkForwardValidationEngine,
    WalkForwardValidationResult,
    WalkForwardWindow,
)


@dataclass
class Result:
    passed: bool
    message: str = ""


def engine():
    return QuantAIWalkForwardValidationEngine(
        5,
        2,
        2,
    )


def test_configuration_and_minimum_size():
    item = engine()

    assert item.minimum_data_size == 9
    assert item.step_size == 2


@pytest.mark.parametrize(
    "name",
    [
        "train_size",
        "validation_size",
        "test_size",
    ],
)
def test_invalid_sizes(name):
    values = {
        "train_size": 5,
        "validation_size": 2,
        "test_size": 2,
    }

    values[name] = 0

    with pytest.raises(ValueError):
        QuantAIWalkForwardValidationEngine(
            **values
        )


def test_invalid_types_and_step():
    with pytest.raises(TypeError):
        QuantAIWalkForwardValidationEngine(
            5.0,
            2,
            2,
        )

    with pytest.raises(TypeError):
        QuantAIWalkForwardValidationEngine(
            5,
            2,
            2,
            1.5,
        )

    with pytest.raises(ValueError):
        QuantAIWalkForwardValidationEngine(
            5,
            2,
            2,
            0,
        )


def test_window_generation():
    windows = engine().generate_windows(20)

    assert len(windows) == 6

    assert windows[0] == WalkForwardWindow(
        0,
        0,
        5,
        5,
        7,
        7,
        9,
    )

    assert windows[1].train_start == 2

    assert windows[-1].test_end == 19


def test_insufficient_window_data():
    item = engine()

    assert item.generate_windows(8) == []

    with pytest.raises(ValueError):
        item.generate_windows(0)

    with pytest.raises(TypeError):
        item.generate_windows(8.0)


def test_all_windows_pass():
    calls = []

    def train_fn(data):
        calls.append(("train", list(data)))
        return "model"

    def validation_fn(model, data):
        calls.append(("validation", list(data)))
        return Result(True)

    def test_fn(model, data):
        calls.append(("test", list(data)))
        return Result(True)

    result = engine().validate(
        list(range(20)),
        train_fn,
        validation_fn,
        test_fn,
    )

    assert isinstance(
        result,
        WalkForwardValidationResult,
    )

    assert result.passed is True
    assert result.total_windows == 6
    assert result.passed_windows == 6
    assert result.failed_windows == 0
    assert result.errors == []
    assert len(calls) == 18


def test_segments_are_correct():
    captured = {}

    def train_fn(data):
        captured["train"] = list(data)
        return "model"

    def validation_fn(model, data):
        captured["validation"] = list(data)
        return True

    def test_fn(model, data):
        captured["test"] = list(data)
        return True

    result = engine().validate(
        list(range(9)),
        train_fn,
        validation_fn,
        test_fn,
    )

    assert result.passed is True

    assert captured["train"] == [
        0,
        1,
        2,
        3,
        4,
    ]

    assert captured["validation"] == [
        5,
        6,
    ]

    assert captured["test"] == [
        7,
        8,
    ]


def test_validation_failure_blocks_window():
    result = engine().validate(
        list(range(9)),
        lambda data: "model",
        lambda model, data: Result(
            False,
            "validation failed",
        ),
        lambda model, data: Result(True),
    )

    assert result.passed is False
    assert result.failed_windows == 1
    assert (
        "validation failed"
        in result.windows[0].message
    )


def test_test_failure_blocks_window():
    result = engine().validate(
        list(range(9)),
        lambda data: "model",
        lambda model, data: True,
        lambda model, data: Result(
            False,
            "test failed",
        ),
    )

    assert result.passed is False
    assert result.failed_windows == 1
    assert (
        "test failed"
        in result.windows[0].message
    )


def test_supported_status_attributes():
    result = engine().validate(
        list(range(9)),
        lambda data: "model",
        lambda model, data: type(
            "R",
            (),
            {"success": True},
        )(),
        lambda model, data: type(
            "R",
            (),
            {"valid": True},
        )(),
    )

    assert result.passed is True


def test_unsupported_status_is_recorded():
    result = engine().validate(
        list(range(9)),
        lambda data: "model",
        lambda model, data: object(),
        lambda model, data: True,
    )

    assert result.passed is False
    assert "TypeError" in result.errors[0]


@pytest.mark.parametrize(
    "argument",
    [
        "train",
        "validation",
        "test",
    ],
)
def test_non_callable_function_is_rejected(
    argument,
):
    functions = {
        "train": lambda data: "model",
        "validation": (
            lambda model, data: True
        ),
        "test": (
            lambda model, data: True
        ),
    }

    functions[argument] = None

    with pytest.raises(TypeError):
        engine().validate(
            list(range(9)),
            functions["train"],
            functions["validation"],
            functions["test"],
        )


def test_non_indexable_data_is_rejected():
    with pytest.raises(TypeError):
        engine().validate(
            iter(range(9)),
            lambda data: "model",
            lambda model, data: True,
            lambda model, data: True,
        )


def test_insufficient_data_returns_failure():
    result = engine().validate(
        list(range(8)),
        lambda data: "model",
        lambda model, data: True,
        lambda model, data: True,
    )

    assert result.passed is False
    assert result.total_windows == 0
    assert len(result.errors) == 1


def test_window_exception_isolated():
    result = engine().validate(
        list(range(9)),
        lambda data: (
            (_ for _ in ()).throw(
                RuntimeError(
                    "train failure"
                )
            )
        ),
        lambda model, data: True,
        lambda model, data: True,
    )

    assert result.passed is False
    assert result.failed_windows == 1
    assert (
        "RuntimeError"
        in result.errors[0]
    )


def test_validation_exception_is_recorded():
    def validate(model, data):
        raise ValueError(
            "validation failure"
        )

    result = engine().validate(
        list(range(9)),
        lambda data: "model",
        validate,
        lambda model, data: True,
    )

    assert result.passed is False
    assert "ValueError" in result.errors[0]


def test_test_exception_is_recorded():
    def test(model, data):
        raise RuntimeError(
            "test failure"
        )

    result = engine().validate(
        list(range(9)),
        lambda data: "model",
        lambda model, data: True,
        test,
    )

    assert result.passed is False
    assert "RuntimeError" in result.errors[0]


def test_custom_step():
    item = QuantAIWalkForwardValidationEngine(
        4,
        2,
        2,
        step_size=3,
    )

    windows = item.generate_windows(15)

    assert len(windows) == 3

    assert [
        window.train_start
        for window in windows
    ] == [
        0,
        3,
        6,
    ]


def test_results_preserve_outputs():
    model = object()
    validation = Result(True)
    testing = Result(True)

    result = engine().validate(
        list(range(9)),
        lambda data: model,
        lambda current, data: validation,
        lambda current, data: testing,
    )

    item = result.windows[0]

    assert item.train_result is model
    assert (
        item.validation_result
        is validation
    )
    assert item.test_result is testing


def test_counters_are_consistent():
    result = engine().validate(
        list(range(9)),
        lambda data: "model",
        lambda model, data: False,
        lambda model, data: True,
    )

    assert result.total_windows == 1
    assert result.passed_windows == 0
    assert result.failed_windows == 1

    assert (
        result.total_windows
        == result.passed_windows
        + result.failed_windows
    )