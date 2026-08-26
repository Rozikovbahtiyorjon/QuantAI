from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, List, Optional, Sequence


@dataclass(frozen=True)
class MLWalkForwardWindowResult:
    index: int
    train_size: int
    validation_size: int
    test_size: int
    validation_passed: bool
    test_passed: bool
    validation_score: Optional[float] = None
    test_score: Optional[float] = None
    error: Optional[str] = None


@dataclass
class MLWalkForwardIntegrationResult:
    passed: bool
    windows: List[MLWalkForwardWindowResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def total_windows(self) -> int:
        return len(self.windows)

    @property
    def passed_windows(self) -> int:
        return sum(
            window.validation_passed and window.test_passed
            for window in self.windows
        )

    @property
    def failed_windows(self) -> int:
        return self.total_windows - self.passed_windows

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
        return sum(scores) / len(scores) if scores else None

    @property
    def average_test_score(self) -> Optional[float]:
        scores = self.test_scores
        return sum(scores) / len(scores) if scores else None


class QuantAIMLWalkForwardIntegration:
    """Integrates an ML train/score pipeline with rolling walk-forward windows."""

    def __init__(
        self,
        train_size: int,
        validation_size: int,
        test_size: int,
        step_size: int = 1,
    ) -> None:
        for name, value in (
            ("train_size", train_size),
            ("validation_size", validation_size),
            ("test_size", test_size),
            ("step_size", step_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")

            if value <= 0:
                raise ValueError(
                    f"{name} must be greater than zero."
                )

        self.train_size = train_size
        self.validation_size = validation_size
        self.test_size = test_size
        self.step_size = step_size

    def _windows(
        self,
        data: Sequence[Any],
    ) -> Iterable[
        tuple[int, Sequence[Any], Sequence[Any], Sequence[Any]]
    ]:
        required = (
            self.train_size
            + self.validation_size
            + self.test_size
        )

        index = 0

        while index + required <= len(data):
            train_end = index + self.train_size
            validation_end = (
                train_end + self.validation_size
            )
            test_end = validation_end + self.test_size

            yield (
                index,
                data[index:train_end],
                data[train_end:validation_end],
                data[validation_end:test_end],
            )

            index += self.step_size

    @staticmethod
    def _score(
        scorer: Callable[[Any, Sequence[Any]], Any],
        model: Any,
        data: Sequence[Any],
    ) -> tuple[bool, Optional[float]]:
        value = scorer(model, data)

        if isinstance(value, bool):
            return value, None

        if isinstance(value, (int, float)):
            return bool(value >= 0.0), float(value)

        status = getattr(
            value,
            "passed",
            None,
        )

        if isinstance(status, bool):
            score = getattr(
                value,
                "score",
                None,
            )

            if isinstance(score, (int, float)):
                return status, float(score)

            return status, None

        status = getattr(
            value,
            "success",
            None,
        )

        if isinstance(status, bool):
            score = getattr(
                value,
                "score",
                None,
            )

            if isinstance(score, (int, float)):
                return status, float(score)

            return status, None

        raise TypeError(
            "Scorer result must be a bool, numeric score, "
            "or expose a supported boolean status."
        )

    def validate(
        self,
        data: Sequence[Any],
        trainer: Callable[[Sequence[Any]], Any],
        validation_scorer: Callable[
            [Any, Sequence[Any]],
            Any,
        ],
        test_scorer: Callable[
            [Any, Sequence[Any]],
            Any,
        ],
    ) -> MLWalkForwardIntegrationResult:
        if isinstance(data, (str, bytes)) or not isinstance(data, Sequence):
            raise TypeError(
                "data must be a non-string sequence."
            )

        if not callable(trainer):
            raise TypeError(
                "trainer must be callable."
            )

        if not callable(validation_scorer):
            raise TypeError(
                "validation_scorer must be callable."
            )

        if not callable(test_scorer):
            raise TypeError(
                "test_scorer must be callable."
            )

        windows: List[MLWalkForwardWindowResult] = []
        errors: List[str] = []
        warnings: List[str] = []

        for (
            index,
            train,
            validation,
            test,
        ) in self._windows(data):
            try:
                model = trainer(train)

                (
                    validation_passed,
                    validation_score,
                ) = self._score(
                    validation_scorer,
                    model,
                    validation,
                )

                (
                    test_passed,
                    test_score,
                ) = self._score(
                    test_scorer,
                    model,
                    test,
                )

                windows.append(
                    MLWalkForwardWindowResult(
                        index=index,
                        train_size=len(train),
                        validation_size=len(validation),
                        test_size=len(test),
                        validation_passed=validation_passed,
                        test_passed=test_passed,
                        validation_score=validation_score,
                        test_score=test_score,
                    )
                )

                if (
                    not validation_passed
                    or not test_passed
                ):
                    errors.append(
                        f"window_{index}: "
                        "ML walk-forward performance gate failed."
                    )

            except Exception as exc:
                message = (
                    f"window_{index}: "
                    f"{type(exc).__name__}: {exc}"
                )

                errors.append(message)

                windows.append(
                    MLWalkForwardWindowResult(
                        index=index,
                        train_size=len(train),
                        validation_size=len(validation),
                        test_size=len(test),
                        validation_passed=False,
                        test_passed=False,
                        error=message,
                    )
                )

        if not windows:
            warnings.append(
                "No complete walk-forward windows were available."
            )

        return MLWalkForwardIntegrationResult(
            passed=bool(windows) and not errors,
            windows=windows,
            errors=errors,
            warnings=warnings,
        )


def validate_ml_walk_forward(
    data: Sequence[Any],
    trainer: Callable[[Sequence[Any]], Any],
    validation_scorer: Callable[
        [Any, Sequence[Any]],
        Any,
    ],
    test_scorer: Callable[
        [Any, Sequence[Any]],
        Any,
    ],
    train_size: int,
    validation_size: int,
    test_size: int,
    step_size: int = 1,
) -> MLWalkForwardIntegrationResult:
    return QuantAIMLWalkForwardIntegration(
        train_size=train_size,
        validation_size=validation_size,
        test_size=test_size,
        step_size=step_size,
    ).validate(
        data,
        trainer,
        validation_scorer,
        test_scorer,
    )


__all__ = [
    "MLWalkForwardWindowResult",
    "MLWalkForwardIntegrationResult",
    "QuantAIMLWalkForwardIntegration",
    "validate_ml_walk_forward",
]