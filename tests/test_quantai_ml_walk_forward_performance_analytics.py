from dataclasses import dataclass

import pytest

from src.quantai_ml_walk_forward_performance_analytics import (
    MLWalkForwardPerformanceAnalyticsResult,
    QuantAIMLWalkForwardPerformanceAnalytics,
    analyze_ml_walk_forward_performance,
)


@dataclass
class MockWindow:
    index: int
    validation_score: float | None
    test_score: float | None


@dataclass
class MockResult:
    windows: list[MockWindow]


def make_result(*test_scores: float) -> MockResult:
    return MockResult(
        windows=[
            MockWindow(
                index=index,
                validation_score=score + 0.05,
                test_score=score,
            )
            for index, score in enumerate(test_scores)
        ]
    )


def test_stable_performance_passes():
    result = QuantAIMLWalkForwardPerformanceAnalytics(
        degradation_threshold=0.10,
        stability_stddev_threshold=0.10,
    ).analyze(
        make_result(0.80, 0.82, 0.81)
    )

    assert isinstance(
        result,
        MLWalkForwardPerformanceAnalyticsResult,
    )

    assert result.stable is True
    assert result.degraded is False
    assert result.total_windows == 3

    assert result.average_test_score == pytest.approx(
        0.81
    )

    assert result.minimum_test_score == pytest.approx(
        0.80
    )


def test_degradation_is_detected():
    result = QuantAIMLWalkForwardPerformanceAnalytics(
        degradation_threshold=0.10,
        stability_stddev_threshold=1.0,
    ).analyze(
        make_result(0.90, 0.75)
    )

    assert result.stable is False
    assert result.degraded is True
    assert result.windows[1].degraded is True

    assert (
        result.windows[1].degradation_from_previous
        == pytest.approx(0.15)
    )

    assert any(
        "degradation" in warning
        for warning in result.warnings
    )


def test_variability_is_detected():
    result = QuantAIMLWalkForwardPerformanceAnalytics(
        degradation_threshold=1.0,
        stability_stddev_threshold=0.05,
    ).analyze(
        make_result(0.90, 0.70, 0.90)
    )

    assert result.stable is False
    assert result.degraded is False

    assert result.test_score_stddev is not None

    assert (
        result.test_score_stddev
        > 0.05
    )

    assert any(
        "variability" in warning
        for warning in result.warnings
    )


def test_minimum_test_score_gate():
    result = QuantAIMLWalkForwardPerformanceAnalytics(
        minimum_test_score=0.80,
    ).analyze(
        make_result(0.85, 0.75)
    )

    assert result.stable is False
    assert result.degraded is True

    assert any(
        "below" in warning
        for warning in result.warnings
    )


def test_missing_scores_are_allowed():
    result = QuantAIMLWalkForwardPerformanceAnalytics().analyze(
        MockResult(
            windows=[
                MockWindow(
                    0,
                    None,
                    None,
                ),
                MockWindow(
                    1,
                    0.8,
                    0.7,
                ),
            ]
        )
    )

    assert result.total_windows == 2

    assert result.average_validation_score == pytest.approx(
        0.8
    )

    assert result.average_test_score == pytest.approx(
        0.7
    )

    assert result.stable is True


def test_empty_windows_fail():
    result = QuantAIMLWalkForwardPerformanceAnalytics().analyze(
        MockResult(
            windows=[]
        )
    )

    assert result.stable is False
    assert result.degraded is False
    assert result.total_windows == 0

    assert result.errors == [
        "No analyzable walk-forward windows were provided."
    ]


def test_invalid_result_is_rejected():
    analytics = (
        QuantAIMLWalkForwardPerformanceAnalytics()
    )

    with pytest.raises(TypeError):
        analytics.analyze(None)

    with pytest.raises(TypeError):
        analytics.analyze("invalid")


def test_invalid_window_score_is_reported():
    result = QuantAIMLWalkForwardPerformanceAnalytics().analyze(
        MockResult(
            windows=[
                MockWindow(
                    0,
                    0.8,
                    "invalid",
                ),
            ]
        )
    )

    assert result.stable is False
    assert len(result.errors) == 1

    assert "test_score" in result.errors[0]


def test_threshold_validation():
    with pytest.raises(ValueError):
        QuantAIMLWalkForwardPerformanceAnalytics(
            degradation_threshold=-0.1,
        )

    with pytest.raises(TypeError):
        QuantAIMLWalkForwardPerformanceAnalytics(
            stability_stddev_threshold=True,
        )

    with pytest.raises(TypeError):
        QuantAIMLWalkForwardPerformanceAnalytics(
            minimum_test_score="bad",
        )


def test_convenience_function():
    result = analyze_ml_walk_forward_performance(
        make_result(
            0.80,
            0.81,
        )
    )

    assert result.stable is True

    assert result.average_test_score == pytest.approx(
        0.805
    )


def test_first_window_has_no_degradation_value():
    result = QuantAIMLWalkForwardPerformanceAnalytics().analyze(
        make_result(0.80)
    )

    assert (
        result.windows[0].degradation_from_previous
        is None
    )

    assert result.windows[0].degraded is False


def test_degradation_uses_previous_available_test_score():
    result = QuantAIMLWalkForwardPerformanceAnalytics(
        degradation_threshold=0.10,
        stability_stddev_threshold=1.0,
    ).analyze(
        MockResult(
            windows=[
                MockWindow(
                    0,
                    0.9,
                    0.9,
                ),
                MockWindow(
                    1,
                    0.8,
                    None,
                ),
                MockWindow(
                    2,
                    0.7,
                    0.7,
                ),
            ]
        )
    )

    assert (
        result.windows[2].degradation_from_previous
        == pytest.approx(0.2)
    )

    assert result.windows[2].degraded is True