from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Sequence


@dataclass(frozen=True)
class WalkForwardWindow:
    index: int
    train_start: int
    train_end: int
    validation_start: int
    validation_end: int
    test_start: int
    test_end: int


@dataclass(frozen=True)
class WalkForwardWindowResult:
    window: WalkForwardWindow
    train_result: Any
    validation_result: Any
    test_result: Any
    passed: bool
    message: str = ""


@dataclass
class WalkForwardValidationResult:
    passed: bool
    windows: List[WalkForwardWindowResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def total_windows(self) -> int:
        return len(self.windows)

    @property
    def passed_windows(self) -> int:
        return sum(item.passed for item in self.windows)

    @property
    def failed_windows(self) -> int:
        return sum(not item.passed for item in self.windows)


class QuantAIWalkForwardValidationEngine:
    """Deterministic walk-forward validation coordinator."""

    def __init__(
        self,
        train_size: int,
        validation_size: int,
        test_size: int,
        step_size: int | None = None,
    ) -> None:
        for name, value in (
            ("train_size", train_size),
            ("validation_size", validation_size),
            ("test_size", test_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero.")

        if step_size is None:
            step_size = test_size

        if isinstance(step_size, bool) or not isinstance(step_size, int):
            raise TypeError("step_size must be an integer.")
        if step_size <= 0:
            raise ValueError("step_size must be greater than zero.")

        self.train_size = train_size
        self.validation_size = validation_size
        self.test_size = test_size
        self.step_size = step_size

    @property
    def minimum_data_size(self) -> int:
        return self.train_size + self.validation_size + self.test_size

    def generate_windows(self, data_size: int) -> List[WalkForwardWindow]:
        if isinstance(data_size, bool) or not isinstance(data_size, int):
            raise TypeError("data_size must be an integer.")
        if data_size <= 0:
            raise ValueError("data_size must be greater than zero.")

        windows: List[WalkForwardWindow] = []
        start = 0
        index = 0

        while start + self.minimum_data_size <= data_size:
            train_end = start + self.train_size
            validation_start = train_end
            validation_end = validation_start + self.validation_size
            test_start = validation_end
            test_end = test_start + self.test_size

            windows.append(
                WalkForwardWindow(
                    index=index,
                    train_start=start,
                    train_end=train_end,
                    validation_start=validation_start,
                    validation_end=validation_end,
                    test_start=test_start,
                    test_end=test_end,
                )
            )

            start += self.step_size
            index += 1

        return windows

    @staticmethod
    def _status(result: Any) -> bool:
        if isinstance(result, bool):
            return result

        for attribute in ("passed", "success", "valid", "ready"):
            value = getattr(result, attribute, None)

            if isinstance(value, bool):
                return value

        raise TypeError(
            "Validation result must be a bool or expose "
            "a supported boolean status."
        )

    @staticmethod
    def _message(result: Any) -> str:
        value = getattr(result, "message", None)

        return "" if value is None else str(value)

    def validate(
        self,
        data: Sequence[Any],
        train_fn: Callable[[Sequence[Any]], Any],
        validation_fn: Callable[[Any, Sequence[Any]], Any],
        test_fn: Callable[[Any, Sequence[Any]], Any],
    ) -> WalkForwardValidationResult:
        if not hasattr(data, "__len__") or not hasattr(
            data,
            "__getitem__",
        ):
            raise TypeError(
                "data must be a sized, indexable sequence."
            )

        if not callable(train_fn):
            raise TypeError(
                "train_fn must be callable."
            )

        if not callable(validation_fn):
            raise TypeError(
                "validation_fn must be callable."
            )

        if not callable(test_fn):
            raise TypeError(
                "test_fn must be callable."
            )

        windows = self.generate_windows(len(data))

        if not windows:
            return WalkForwardValidationResult(
                passed=False,
                errors=[
                    (
                        "Insufficient data for one complete "
                        "walk-forward window."
                    )
                ],
            )

        results: List[WalkForwardWindowResult] = []
        errors: List[str] = []

        for window in windows:
            try:
                train_result = train_fn(
                    data[
                        window.train_start:
                        window.train_end
                    ]
                )

                validation_result = validation_fn(
                    train_result,
                    data[
                        window.validation_start:
                        window.validation_end
                    ],
                )

                test_result = test_fn(
                    train_result,
                    data[
                        window.test_start:
                        window.test_end
                    ],
                )

                validation_passed = self._status(
                    validation_result
                )

                test_passed = self._status(
                    test_result
                )

                passed = (
                    validation_passed
                    and test_passed
                )

                messages = [
                    message
                    for message in (
                        self._message(
                            validation_result
                        ),
                        self._message(
                            test_result
                        ),
                    )
                    if message
                ]

                results.append(
                    WalkForwardWindowResult(
                        window=window,
                        train_result=train_result,
                        validation_result=validation_result,
                        test_result=test_result,
                        passed=passed,
                        message="; ".join(messages),
                    )
                )

                if not passed:
                    errors.append(
                        (
                            f"window_{window.index}: "
                            "validation or test failed."
                        )
                    )

            except Exception as exc:
                message = (
                    f"window_{window.index}: "
                    f"{type(exc).__name__}: {exc}"
                )

                errors.append(message)

                results.append(
                    WalkForwardWindowResult(
                        window=window,
                        train_result=None,
                        validation_result=None,
                        test_result=None,
                        passed=False,
                        message=message,
                    )
                )

        return WalkForwardValidationResult(
            passed=(
                bool(results)
                and all(
                    item.passed
                    for item in results
                )
            ),
            windows=results,
            errors=errors,
        )


__all__ = [
    "WalkForwardWindow",
    "WalkForwardWindowResult",
    "WalkForwardValidationResult",
    "QuantAIWalkForwardValidationEngine",
]