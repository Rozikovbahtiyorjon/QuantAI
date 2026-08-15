from dataclasses import dataclass

import pytest

from src.quantai_ml_walk_forward_integration import (
    MLWalkForwardIntegrationResult,
    QuantAIMLWalkForwardIntegration,
    validate_ml_walk_forward,
)


@dataclass
class ScoreResult:
    passed: bool
    score: float


def engine() -> QuantAIMLWalkForwardIntegration:
    return QuantAIMLWalkForwardIntegration(
        train_size=5,
        validation_size=2,
        test_size=2,
        step_size=2,
    )


def test_all_windows_pass():
    result = engine().validate(
        list(range(13)),
        lambda data: list(data),
        lambda model, data: ScoreResult(True, 0.8),
        lambda model, data: ScoreResult(True, 0.7),
    )

    assert isinstance(
        result,
        MLWalkForwardIntegrationResult,
    )

    assert result.passed is True
    assert result.total_windows == 3
    assert result.passed_windows == 3
    assert result.failed_windows == 0

    assert result.average_validation_score == pytest.approx(
        0.8
    )

    assert result.average_test_score == pytest.approx(
        0.7
    )


def test_window_sizes_and_indices():
    captured = []

    def trainer(data):
        captured.append(list(data))
        return "model"

    result = engine().validate(
        list(range(13)),
        trainer,
        lambda model, data: True,
        lambda model, data: True,
    )

    assert result.passed is True

    assert [
        window.index
        for window in result.windows
    ] == [0, 2, 4]

    assert captured == [
        [0, 1, 2, 3, 4],
        [2, 3, 4, 5, 6],
        [4, 5, 6, 7, 8],
    ]


def test_boolean_scorers_are_supported():
    result = engine().validate(
        list(range(9)),
        lambda data: "model",
        lambda model, data: True,
        lambda model, data: True,
    )

    assert result.passed is True
    assert result.validation_scores == []
    assert result.test_scores == []


def test_numeric_scores_are_supported():
    result = engine().validate(
        list(range(9)),
        lambda data: "model",
        lambda model, data: 0.75,
        lambda model, data: 0.65,
    )

    assert result.passed is True

    assert result.average_validation_score == pytest.approx(
        0.75
    )

    assert result.average_test_score == pytest.approx(
        0.65
    )


def test_failed_validation_blocks_result():
    result = engine().validate(
        list(range(9)),
        lambda data: "model",
        lambda model, data: ScoreResult(False, 0.2),
        lambda model, data: ScoreResult(True, 0.8),
    )

    assert result.passed is False
    assert result.failed_windows == result.total_windows

    assert any(
        "performance gate failed" in error
        for error in result.errors
    )


def test_failed_test_blocks_result():
    result = engine().validate(
        list(range(9)),
        lambda data: "model",
        lambda model, data: ScoreResult(True, 0.8),
        lambda model, data: ScoreResult(False, 0.2),
    )

    assert result.passed is False

    assert any(
        "performance gate failed" in error
        for error in result.errors
    )


def test_trainer_failure_is_captured():
    def trainer(data):
        raise RuntimeError(
            "training failure"
        )

    result = engine().validate(
        list(range(9)),
        trainer,
        lambda model, data: True,
        lambda model, data: True,
    )

    assert result.passed is False
    assert result.windows[0].error is not None
    assert "training failure" in result.errors[0]


def test_invalid_scorer_result_is_captured():
    result = engine().validate(
        list(range(9)),
        lambda data: "model",
        lambda model, data: "invalid",
        lambda model, data: True,
    )

    assert result.passed is False

    assert (
        "supported boolean status"
        in result.errors[0]
    )


def test_no_complete_windows():
    result = engine().validate(
        list(range(8)),
        lambda data: "model",
        lambda model, data: True,
        lambda model, data: True,
    )

    assert result.passed is False
    assert result.total_windows == 0

    assert result.warnings == [
        "No complete walk-forward windows were available."
    ]


def test_invalid_configuration():
    with pytest.raises(ValueError):
        QuantAIMLWalkForwardIntegration(
            train_size=0,
            validation_size=2,
            test_size=2,
        )

    with pytest.raises(TypeError):
        QuantAIMLWalkForwardIntegration(
            train_size=True,
            validation_size=2,
            test_size=2,
        )


def test_invalid_inputs():
    integration = engine()

    with pytest.raises(TypeError):
        integration.validate(
            "not-a-sequence",
            lambda data: "model",
            lambda model, data: True,
            lambda model, data: True,
        )

    with pytest.raises(TypeError):
        integration.validate(
            list(range(9)),
            None,
            lambda model, data: True,
            lambda model, data: True,
        )


def test_convenience_function():
    result = validate_ml_walk_forward(
        list(range(9)),
        lambda data: "model",
        lambda model, data: ScoreResult(
            True,
            0.9,
        ),
        lambda model, data: ScoreResult(
            True,
            0.85,
        ),
        train_size=5,
        validation_size=2,
        test_size=2,
    )

    assert result.passed is True

    assert result.average_validation_score == pytest.approx(
        0.9
    )

    assert result.average_test_score == pytest.approx(
        0.85
    )


def test_status_attribute_is_supported():
    class Result:
        success = True
        score = 0.91

    result = engine().validate(
        list(range(9)),
        lambda data: "model",
        lambda model, data: Result(),
        lambda model, data: Result(),
    )

    assert result.passed is True
    assert result.average_validation_score == pytest.approx(
        0.91
    )