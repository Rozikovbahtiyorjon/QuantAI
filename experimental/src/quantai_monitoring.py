from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping


class HealthStatus(str, Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class ComponentHealth:
    name: str
    status: HealthStatus
    message: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("name must be a string.")

        if not self.name.strip():
            raise ValueError("name cannot be empty.")

        if not isinstance(self.status, HealthStatus):
            raise TypeError("status must be HealthStatus.")

        if not isinstance(self.message, str):
            raise TypeError("message must be a string.")


@dataclass(frozen=True)
class SystemHealth:
    status: HealthStatus
    components: tuple[ComponentHealth, ...]
    healthy_count: int
    degraded_count: int
    critical_count: int


class QuantAIMonitoring:
    def __init__(
        self,
        stale_threshold_seconds: float = 30.0,
    ) -> None:
        if not isinstance(
            stale_threshold_seconds,
            (int, float),
        ):
            raise TypeError(
                "stale_threshold_seconds must be numeric."
            )

        if stale_threshold_seconds <= 0:
            raise ValueError(
                "stale_threshold_seconds must be greater than zero."
            )

        self.stale_threshold_seconds = float(
            stale_threshold_seconds
        )

    def evaluate_component(
        self,
        name: str,
        *,
        healthy: bool,
        message: str = "",
    ) -> ComponentHealth:
        if not isinstance(healthy, bool):
            raise TypeError(
                "healthy must be a boolean."
            )

        return ComponentHealth(
            name=name,
            status=(
                HealthStatus.HEALTHY
                if healthy
                else HealthStatus.CRITICAL
            ),
            message=message,
        )

    def evaluate_staleness(
        self,
        name: str,
        age_seconds: float,
    ) -> ComponentHealth:
        if not isinstance(age_seconds, (int, float)):
            raise TypeError(
                "age_seconds must be numeric."
            )

        if age_seconds < 0:
            raise ValueError(
                "age_seconds cannot be negative."
            )

        if age_seconds <= self.stale_threshold_seconds:
            status = HealthStatus.HEALTHY
            message = "Data is fresh."
        elif age_seconds <= (
            self.stale_threshold_seconds * 2
        ):
            status = HealthStatus.DEGRADED
            message = "Data is becoming stale."
        else:
            status = HealthStatus.CRITICAL
            message = "Data is stale."

        return ComponentHealth(
            name=name,
            status=status,
            message=message,
        )

    @staticmethod
    def _validate_components(
        components: Mapping[str, ComponentHealth],
    ) -> dict[str, ComponentHealth]:
        if not isinstance(components, Mapping):
            raise TypeError(
                "components must be a mapping."
            )

        result = dict(components)

        for name, component in result.items():
            if not isinstance(name, str):
                raise TypeError(
                    "component names must be strings."
                )

            if not isinstance(
                component,
                ComponentHealth,
            ):
                raise TypeError(
                    "component values must be "
                    "ComponentHealth."
                )

            if component.name != name:
                raise ValueError(
                    "component key must match component name."
                )

        return result

    def evaluate_system(
        self,
        components: Mapping[str, ComponentHealth],
    ) -> SystemHealth:
        validated = self._validate_components(
            components
        )

        values = tuple(validated.values())

        healthy_count = sum(
            component.status is HealthStatus.HEALTHY
            for component in values
        )

        degraded_count = sum(
            component.status is HealthStatus.DEGRADED
            for component in values
        )

        critical_count = sum(
            component.status is HealthStatus.CRITICAL
            for component in values
        )

        if critical_count:
            status = HealthStatus.CRITICAL
        elif degraded_count:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.HEALTHY

        return SystemHealth(
            status=status,
            components=values,
            healthy_count=healthy_count,
            degraded_count=degraded_count,
            critical_count=critical_count,
        )