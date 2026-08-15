from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Any, List, Optional, Sequence


@dataclass(frozen=True)
class MLPerformanceWindowAnalysis:
    index: int
    validation_score: Optional[float]
    test_score: Optional[float]
    degradation_from_previous: Optional[float]
    degraded: bool


@dataclass
class MLWalkForwardPerformanceAnalyticsResult:
    stable: bool
    degraded: bool
    windows: List[MLPerformanceWindowAnalysis] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def total_windows(self) -> int:
        return len(self.windows)

    @property
    def validation_scores(self) -> List[float]:
        return [
            window.validation_score
            for window in self.windows
            if window.validation_score is not None
        ]

    @property
    def test_scores(self) -> List[float]:
        return [
            window.test_score
            for window in self.windows
            if window.test_score is not None
        ]

    @property
    def average_validation_score(self) -> Optional[float]:
        scores = self.validation_scores
        return mean(scores) if scores else None

    @property
    def average_test_score(self) -> Optional[float]:
        scores = self.test_scores
        return mean(scores) if scores else None

    @property
    def minimum_validation_score(self) -> Optional[float]:
        scores = self.validation_scores
        return min(scores) if scores else None

    @property
    def minimum_test_score(self) -> Optional[float]:
        scores = self.test_scores
        return min(scores) if scores else None

    @property
    def test_score_stddev(self) -> Optional[float]:
        scores = self.test_scores

        if not scores:
            return None

        return pstdev(scores) if len(scores) > 1 else 0.0


class QuantAIMLWalkForwardPerformanceAnalytics:
    """Deterministic analytics for ML walk-forward stability and degradation."""

    def __init__(
        self,
        degradation_threshold: float = 0.10,
        stability_stddev_threshold: float = 0.10,
        minimum_test_score: Optional[float] = None,
    ) -> None:
        self._validate_threshold(
            "degradation_threshold",
            degradation_threshold,
        )

        self._validate_threshold(
            "stability_stddev_threshold",
            stability_stddev_threshold,
        )

        if minimum_test_score is not None:
            if not isinstance(
                minimum_test_score,
                (int, float),
            ):
                raise TypeError(
                    "minimum_test_score must be numeric or None."
                )

        self.degradation_threshold = float(
            degradation_threshold
        )

        self.stability_stddev_threshold = float(
            stability_stddev_threshold
        )

        self.minimum_test_score = (
            float(minimum_test_score)
            if minimum_test_score is not None
            else None
        )

    @staticmethod
    def _validate_threshold(
        name: str,
        value: float,
    ) -> None:
        if isinstance(value, bool) or not isinstance(
            value,
            (int, float),
        ):
            raise TypeError(
                f"{name} must be numeric."
            )

        if value < 0:
            raise ValueError(
                f"{name} cannot be negative."
            )

    @staticmethod
    def _extract_score(
        window: Any,
        attribute: str,
    ) -> Optional[float]:
        value = getattr(
            window,
            attribute,
            None,
        )

        if value is None:
            return None

        if isinstance(value, bool) or not isinstance(
            value,
            (int, float),
        ):
            raise TypeError(
                f"{attribute} must be numeric or None."
            )

        return float(value)

    def analyze(
        self,
        result: Any,
    ) -> MLWalkForwardPerformanceAnalyticsResult:
        windows = getattr(
            result,
            "windows",
            None,
        )

        if windows is None:
            raise TypeError(
                "result must expose a windows collection."
            )

        if isinstance(
            windows,
            (str, bytes),
        ) or not isinstance(
            windows,
            Sequence,
        ):
            raise TypeError(
                "result.windows must be a sequence."
            )

        analyses: List[
            MLPerformanceWindowAnalysis
        ] = []

        errors: List[str] = []
        warnings: List[str] = []

        previous_test_score: Optional[float] = None

        for position, window in enumerate(windows):
            index = getattr(
                window,
                "index",
                position,
            )

            try:
                validation_score = self._extract_score(
                    window,
                    "validation_score",
                )

                test_score = self._extract_score(
                    window,
                    "test_score",
                )

                degradation = None
                degraded = False

                if (
                    previous_test_score is not None
                    and test_score is not None
                ):
                    degradation = (
                        previous_test_score
                        - test_score
                    )

                    degraded = (
                        degradation
                        >= self.degradation_threshold
                    )

                analyses.append(
                    MLPerformanceWindowAnalysis(
                        index=int(index),
                        validation_score=validation_score,
                        test_score=test_score,
                        degradation_from_previous=degradation,
                        degraded=degraded,
                    )
                )

                if degraded:
                    warnings.append(
                        f"window_{index}: "
                        "test performance degradation "
                        f"of {degradation:.6f} detected."
                    )

                if test_score is not None:
                    previous_test_score = test_score

            except (
                TypeError,
                ValueError,
            ) as exc:
                errors.append(
                    f"window_{index}: "
                    f"{type(exc).__name__}: {exc}"
                )

        if not analyses and not errors:
            errors.append(
                "No analyzable walk-forward windows were provided."
            )

        test_scores = [
            window.test_score
            for window in analyses
            if window.test_score is not None
        ]

        degraded = any(
            window.degraded
            for window in analyses
        )

        unstable = False

        if len(test_scores) > 1:
            unstable = (
                pstdev(test_scores)
                > self.stability_stddev_threshold
            )

        if unstable:
            warnings.append(
                "Test performance variability exceeds "
                "the stability threshold."
            )

        if (
            self.minimum_test_score is not None
            and test_scores
            and min(test_scores)
            < self.minimum_test_score
        ):
            warnings.append(
                "Minimum test performance is below "
                "the configured threshold."
            )

            degraded = True

        stable = (
            bool(analyses)
            and not errors
            and not degraded
            and not unstable
        )

        return MLWalkForwardPerformanceAnalyticsResult(
            stable=stable,
            degraded=degraded,
            windows=analyses,
            errors=errors,
            warnings=warnings,
        )


def analyze_ml_walk_forward_performance(
    result: Any,
    degradation_threshold: float = 0.10,
    stability_stddev_threshold: float = 0.10,
    minimum_test_score: Optional[float] = None,
) -> MLWalkForwardPerformanceAnalyticsResult:
    analytics = QuantAIMLWalkForwardPerformanceAnalytics(
        degradation_threshold=degradation_threshold,
        stability_stddev_threshold=stability_stddev_threshold,
        minimum_test_score=minimum_test_score,
    )

    return analytics.analyze(result)


__all__ = [
    "MLPerformanceWindowAnalysis",
    "MLWalkForwardPerformanceAnalyticsResult",
    "QuantAIMLWalkForwardPerformanceAnalytics",
    "analyze_ml_walk_forward_performance",
]