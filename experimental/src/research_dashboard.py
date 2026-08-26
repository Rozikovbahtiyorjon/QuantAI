from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping


@dataclass(frozen=True)
class DashboardSection:
    name: str
    status: str
    score: float | None = None
    message: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")

        if self.status not in {
            "HEALTHY",
            "DEGRADED",
            "CRITICAL",
            "UNKNOWN",
        }:
            raise ValueError("invalid section status")

        if self.score is not None:
            if not isinstance(self.score, (int, float)):
                raise TypeError("score must be numeric or None")

            if not 0.0 <= float(self.score) <= 1.0:
                raise ValueError("score must be between 0 and 1")

        if not isinstance(self.message, str):
            raise TypeError("message must be a string")


@dataclass(frozen=True)
class ResearchDashboardSnapshot:
    overall_status: str
    sections: tuple[DashboardSection, ...]
    metrics: dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def __post_init__(self) -> None:
        if self.overall_status not in {
            "HEALTHY",
            "DEGRADED",
            "CRITICAL",
            "UNKNOWN",
        }:
            raise ValueError("invalid overall status")

        if not isinstance(self.sections, tuple):
            raise TypeError("sections must be a tuple")

        if not isinstance(self.metrics, dict):
            raise TypeError("metrics must be a dictionary")

        for name, value in self.metrics.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    "metric names must be non-empty strings"
                )

            if not isinstance(value, (int, float)):
                raise TypeError(
                    "metric values must be numeric"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_status": self.overall_status,
            "sections": [
                {
                    "name": section.name,
                    "status": section.status,
                    "score": section.score,
                    "message": section.message,
                }
                for section in self.sections
            ],
            "metrics": dict(self.metrics),
            "timestamp": self.timestamp.isoformat(),
        }


class ResearchDashboard:
    DEFAULT_SECTIONS = (
        "market_regime",
        "market_intelligence",
        "strategy",
        "confidence",
        "risk",
        "exposure",
        "paper_trading",
        "ml_quality",
        "strategy_health",
        "derivatives",
        "liquidity",
        "monitoring",
        "reliability",
    )

    def __init__(
        self,
        section_names: tuple[str, ...]
        | list[str]
        | None = None,
    ) -> None:
        names = (
            self.DEFAULT_SECTIONS
            if section_names is None
            else tuple(section_names)
        )

        self._validate_section_names(names)
        self._section_names = names

    @property
    def section_names(self) -> tuple[str, ...]:
        return self._section_names

    def build_snapshot(
        self,
        sections: Mapping[str, DashboardSection],
        metrics: Mapping[str, float] | None = None,
    ) -> ResearchDashboardSnapshot:
        if not isinstance(sections, Mapping):
            raise TypeError("sections must be a mapping")

        if not sections:
            raise ValueError(
                "sections must not be empty"
            )

        for name, section in sections.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    "section names must be non-empty strings"
                )

            if not isinstance(section, DashboardSection):
                raise TypeError(
                    "section values must be DashboardSection"
                )

        if metrics is None:
            metric_values: dict[str, float] = {}
        else:
            if not isinstance(metrics, Mapping):
                raise TypeError(
                    "metrics must be a mapping"
                )

            metric_values = dict(metrics)

        for name, value in metric_values.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    "metric names must be non-empty strings"
                )

            if not isinstance(value, (int, float)):
                raise TypeError(
                    "metric values must be numeric"
                )

        section_values = tuple(sections.values())

        overall_status = self._calculate_overall_status(
            section_values
        )

        return ResearchDashboardSnapshot(
            overall_status=overall_status,
            sections=section_values,
            metrics={
                name: float(value)
                for name, value in metric_values.items()
            },
        )

    def build_from_statuses(
        self,
        statuses: Mapping[str, str],
        metrics: Mapping[str, float] | None = None,
    ) -> ResearchDashboardSnapshot:
        if not isinstance(statuses, Mapping):
            raise TypeError(
                "statuses must be a mapping"
            )

        if not statuses:
            raise ValueError(
                "statuses must not be empty"
            )

        sections: dict[str, DashboardSection] = {}

        for name, status in statuses.items():
            if name not in self._section_names:
                raise ValueError(
                    f"unknown dashboard section: {name}"
                )

            sections[name] = DashboardSection(
                name=name,
                status=status,
            )

        return self.build_snapshot(
            sections,
            metrics,
        )

    @staticmethod
    def _calculate_overall_status(
        sections: tuple[DashboardSection, ...],
    ) -> str:
        statuses = {
            section.status
            for section in sections
        }

        if "CRITICAL" in statuses:
            return "CRITICAL"

        if "DEGRADED" in statuses:
            return "DEGRADED"

        if statuses == {"HEALTHY"}:
            return "HEALTHY"

        return "UNKNOWN"

    @staticmethod
    def _validate_section_names(
        names: tuple[str, ...],
    ) -> None:
        if not names:
            raise ValueError(
                "section_names must not be empty"
            )

        if len(set(names)) != len(names):
            raise ValueError(
                "section_names must be unique"
            )

        for name in names:
            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    "section names must be non-empty strings"
                )

    @staticmethod
    def summarize(
        snapshot: ResearchDashboardSnapshot,
    ) -> dict[str, Any]:
        if not isinstance(
            snapshot,
            ResearchDashboardSnapshot,
        ):
            raise TypeError(
                "snapshot must be a ResearchDashboardSnapshot"
            )

        healthy = sum(
            section.status == "HEALTHY"
            for section in snapshot.sections
        )

        degraded = sum(
            section.status == "DEGRADED"
            for section in snapshot.sections
        )

        critical = sum(
            section.status == "CRITICAL"
            for section in snapshot.sections
        )

        total = len(snapshot.sections)

        return {
            "overall_status": snapshot.overall_status,
            "total_sections": total,
            "healthy_sections": healthy,
            "degraded_sections": degraded,
            "critical_sections": critical,
            "health_ratio": (
                healthy / total
                if total
                else 0.0
            ),
        }

    @staticmethod
    def to_dict(
        snapshot: ResearchDashboardSnapshot,
    ) -> dict[str, Any]:
        if not isinstance(
            snapshot,
            ResearchDashboardSnapshot,
        ):
            raise TypeError(
                "snapshot must be a ResearchDashboardSnapshot"
            )

        return snapshot.to_dict()