from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping


class ReliabilityStatus(str, Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    CRITICAL = "CRITICAL"


class ReliabilityAction(str, Enum):
    CONTINUE = "CONTINUE"
    DEGRADE = "DEGRADE"
    HALT = "HALT"
    RECOVER = "RECOVER"


@dataclass(frozen=True)
class ReliabilityCheck:
    name: str
    healthy: bool
    severity: str = "INFO"
    message: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")

        if not isinstance(self.healthy, bool):
            raise TypeError("healthy must be a bool")

        if self.severity not in {"INFO", "WARNING", "CRITICAL"}:
            raise ValueError(
                "severity must be INFO, WARNING, or CRITICAL"
            )

        if not isinstance(self.message, str):
            raise TypeError("message must be a string")


@dataclass(frozen=True)
class ReliabilityReport:
    status: ReliabilityStatus
    action: ReliabilityAction
    score: float
    checks: tuple[ReliabilityCheck, ...]
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be between 0 and 1")

        if not isinstance(self.checks, tuple):
            raise TypeError("checks must be a tuple")

    @property
    def healthy(self) -> bool:
        return self.status is ReliabilityStatus.HEALTHY

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "action": self.action.value,
            "score": self.score,
            "checks": [
                {
                    "name": check.name,
                    "healthy": check.healthy,
                    "severity": check.severity,
                    "message": check.message,
                }
                for check in self.checks
            ],
            "timestamp": self.timestamp.isoformat(),
        }


class QuantAIReliability:
    DEFAULT_THRESHOLDS = {
        "healthy": 0.90,
        "degraded": 0.70,
    }

    def __init__(
        self,
        thresholds: Mapping[str, float] | None = None,
    ) -> None:
        selected = (
            dict(self.DEFAULT_THRESHOLDS)
            if thresholds is None
            else dict(thresholds)
        )

        self._validate_thresholds(selected)
        self._thresholds = selected

    @property
    def thresholds(self) -> dict[str, float]:
        return dict(self._thresholds)

    def evaluate(
        self,
        checks: list[ReliabilityCheck]
        | tuple[ReliabilityCheck, ...],
    ) -> ReliabilityReport:
        if not isinstance(checks, (list, tuple)):
            raise TypeError(
                "checks must be a list or tuple of ReliabilityCheck"
            )

        if not checks:
            raise ValueError("checks must not be empty")

        if not all(
            isinstance(check, ReliabilityCheck)
            for check in checks
        ):
            raise TypeError(
                "all checks must be ReliabilityCheck instances"
            )

        checks_tuple = tuple(checks)

        score = self._calculate_score(checks_tuple)
        status = self._status_from_score(
            score,
            checks_tuple,
        )
        action = self._action_from_status(status)

        return ReliabilityReport(
            status=status,
            action=action,
            score=score,
            checks=checks_tuple,
        )

    def evaluate_mapping(
        self,
        checks: Mapping[str, bool],
    ) -> ReliabilityReport:
        if not isinstance(checks, Mapping):
            raise TypeError("checks must be a mapping")

        if not checks:
            raise ValueError("checks must not be empty")

        converted: list[ReliabilityCheck] = []

        for name, healthy in checks.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    "check names must be non-empty strings"
                )

            if not isinstance(healthy, bool):
                raise TypeError(
                    "mapping check values must be bool"
                )

            converted.append(
                ReliabilityCheck(
                    name=name,
                    healthy=healthy,
                    severity="INFO" if healthy else "CRITICAL",
                    message="OK" if healthy else "FAILED",
                )
            )

        return self.evaluate(converted)

    @staticmethod
    def _validate_thresholds(
        thresholds: dict[str, float],
    ) -> None:
        required = {"healthy", "degraded"}

        if set(thresholds) != required:
            raise ValueError(
                "thresholds must contain exactly "
                "healthy and degraded"
            )

        for name, value in thresholds.items():
            if not isinstance(value, (int, float)):
                raise TypeError(
                    f"threshold '{name}' must be numeric"
                )

            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(
                    f"threshold '{name}' must be between 0 and 1"
                )

        if (
            float(thresholds["degraded"])
            >= float(thresholds["healthy"])
        ):
            raise ValueError(
                "degraded threshold must be lower "
                "than healthy threshold"
            )

    @staticmethod
    def _calculate_score(
        checks: tuple[ReliabilityCheck, ...],
    ) -> float:
        total = len(checks)
        healthy = sum(
            1
            for check in checks
            if check.healthy
        )

        return healthy / total

    def _status_from_score(
        self,
        score: float,
        checks: tuple[ReliabilityCheck, ...],
    ) -> ReliabilityStatus:
        if any(
            check.severity == "CRITICAL"
            and not check.healthy
            for check in checks
        ):
            return ReliabilityStatus.CRITICAL

        if score >= self._thresholds["healthy"]:
            return ReliabilityStatus.HEALTHY

        if score >= self._thresholds["degraded"]:
            return ReliabilityStatus.DEGRADED

        return ReliabilityStatus.CRITICAL

    @staticmethod
    def _action_from_status(
        status: ReliabilityStatus,
    ) -> ReliabilityAction:
        if status is ReliabilityStatus.HEALTHY:
            return ReliabilityAction.CONTINUE

        if status is ReliabilityStatus.DEGRADED:
            return ReliabilityAction.DEGRADE

        return ReliabilityAction.HALT

    def can_continue(
        self,
        report: ReliabilityReport,
    ) -> bool:
        if not isinstance(report, ReliabilityReport):
            raise TypeError(
                "report must be a ReliabilityReport"
            )

        return (
            report.action
            is ReliabilityAction.CONTINUE
        )

    def recovery_report(self) -> ReliabilityReport:
        check = ReliabilityCheck(
            name="recovery_check",
            healthy=True,
            severity="INFO",
            message="Recovery conditions are satisfied.",
        )

        return ReliabilityReport(
            status=ReliabilityStatus.HEALTHY,
            action=ReliabilityAction.RECOVER,
            score=1.0,
            checks=(check,),
        )