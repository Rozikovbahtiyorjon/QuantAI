from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ValidationMetric:
    name: str
    value: float
    passed: bool
    threshold: float | None = None
    description: str = ""


@dataclass
class WalkForwardValidationReport:
    total_windows: int
    completed_windows: int
    failed_windows: int

    metrics: list[ValidationMetric] = field(
        default_factory=list
    )

    errors: list[str] = field(
        default_factory=list
    )

    passed: bool = False

    @property
    def total_metrics(self) -> int:
        return len(self.metrics)

    @property
    def passed_metrics(self) -> int:
        return sum(
            1
            for metric in self.metrics
            if metric.passed
        )

    @property
    def failed_metrics(self) -> int:
        return (
            self.total_metrics
            - self.passed_metrics
        )

    @property
    def validation_rate(self) -> float:
        if self.total_metrics == 0:
            return 0.0

        return (
            self.passed_metrics
            / self.total_metrics
        )

    def add_metric(
        self,
        name: str,
        value: float,
        passed: bool,
        threshold: float | None = None,
        description: str = "",
    ) -> None:
        self.metrics.append(
            ValidationMetric(
                name=name,
                value=float(value),
                passed=bool(passed),
                threshold=threshold,
                description=description,
            )
        )

    def add_error(
        self,
        message: str,
    ) -> None:
        self.errors.append(
            str(message)
        )

    def finalize(self) -> WalkForwardValidationReport:
        self.passed = (
            self.failed_windows == 0
            and len(self.errors) == 0
            and self.total_metrics > 0
            and self.failed_metrics == 0
        )

        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_windows": self.total_windows,
            "completed_windows": self.completed_windows,
            "failed_windows": self.failed_windows,
            "total_metrics": self.total_metrics,
            "passed_metrics": self.passed_metrics,
            "failed_metrics": self.failed_metrics,
            "validation_rate": self.validation_rate,
            "passed": self.passed,
            "errors": list(self.errors),
            "metrics": [
                {
                    "name": metric.name,
                    "value": metric.value,
                    "passed": metric.passed,
                    "threshold": metric.threshold,
                    "description": metric.description,
                }
                for metric in self.metrics
            ],
        }