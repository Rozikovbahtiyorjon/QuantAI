from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from src.quantai_production_observability import (
    AuditEvent,
    AuditEventLevel,
    QuantAIProductionObservability,
)
from src.quantai_production_observability_integration import (
    QuantAIProductionObservabilityIntegration,
)


@dataclass(frozen=True)
class OperationalMetrics:
    total_events: int
    info_events: int
    warning_events: int
    error_events: int
    critical_events: int
    incident_events: int
    sources: tuple[str, ...]


@dataclass(frozen=True)
class Incident:
    event: AuditEvent
    reason: str


@dataclass
class IncidentDetectionResult:
    success: bool
    incidents: List[Incident] = field(
        default_factory=list
    )
    errors: List[str] = field(
        default_factory=list
    )
    warnings: List[str] = field(
        default_factory=list
    )

    @property
    def incident_count(self) -> int:
        return len(self.incidents)

    @property
    def has_incidents(self) -> bool:
        return bool(self.incidents)


class QuantAIProductionObservabilityAnalytics:
    """Operational metrics and deterministic incident detection."""

    def __init__(
        self,
        observability: Optional[
            QuantAIProductionObservability
        ] = None,
        integration: Optional[
            QuantAIProductionObservabilityIntegration
        ] = None,
    ) -> None:
        if integration is not None and not isinstance(
            integration,
            QuantAIProductionObservabilityIntegration,
        ):
            raise TypeError(
                "integration must be a "
                "QuantAIProductionObservabilityIntegration "
                "instance or None."
            )

        if observability is not None and not isinstance(
            observability,
            QuantAIProductionObservability,
        ):
            raise TypeError(
                "observability must be a "
                "QuantAIProductionObservability instance "
                "or None."
            )

        if integration is not None and observability is not None:
            if integration.observability is not observability:
                raise ValueError(
                    "integration and observability must reference "
                    "the same observability instance."
                )

        if integration is not None:
            self._integration = integration
            self._observability = integration.observability
        else:
            self._observability = (
                observability
                if observability is not None
                else QuantAIProductionObservability()
            )
            self._integration = (
                QuantAIProductionObservabilityIntegration(
                    observability=self._observability
                )
            )

    @property
    def observability(self) -> QuantAIProductionObservability:
        return self._observability

    @property
    def integration(
        self,
    ) -> QuantAIProductionObservabilityIntegration:
        return self._integration

    def calculate_metrics(
        self,
        source: Optional[str] = None,
    ) -> OperationalMetrics:
        events = self._observability.query(
            source=source
        )

        counts = {
            level: 0
            for level in AuditEventLevel
        }

        sources = set()

        for event in events:
            counts[event.level] += 1
            sources.add(event.source)

        incident_events = sum(
            event.level
            in (
                AuditEventLevel.ERROR,
                AuditEventLevel.CRITICAL,
            )
            for event in events
        )

        return OperationalMetrics(
            total_events=len(events),
            info_events=counts[AuditEventLevel.INFO],
            warning_events=counts[AuditEventLevel.WARNING],
            error_events=counts[AuditEventLevel.ERROR],
            critical_events=counts[AuditEventLevel.CRITICAL],
            incident_events=incident_events,
            sources=tuple(sorted(sources)),
        )

    def detect_incidents(
        self,
        source: Optional[str] = None,
        minimum_level: AuditEventLevel = AuditEventLevel.ERROR,
    ) -> IncidentDetectionResult:
        if not isinstance(
            minimum_level,
            AuditEventLevel,
        ):
            try:
                minimum_level = AuditEventLevel(
                    minimum_level
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "minimum_level must be a valid "
                    "AuditEventLevel."
                ) from exc

        severity = {
            AuditEventLevel.INFO: 0,
            AuditEventLevel.WARNING: 1,
            AuditEventLevel.ERROR: 2,
            AuditEventLevel.CRITICAL: 3,
        }

        events = self._observability.query(
            source=source
        )

        incidents = []

        for event in events:
            if severity[event.level] < severity[minimum_level]:
                continue

            incidents.append(
                Incident(
                    event=event,
                    reason=(
                        f"Event level {event.level.value} "
                        f"meets incident threshold "
                        f"{minimum_level.value}."
                    ),
                )
            )

        return IncidentDetectionResult(
            success=True,
            incidents=incidents,
        )

    def detect_critical_incidents(
        self,
        source: Optional[str] = None,
    ) -> IncidentDetectionResult:
        return self.detect_incidents(
            source=source,
            minimum_level=AuditEventLevel.CRITICAL,
        )

    def detect_correlated_incidents(
        self,
        correlation_id: str,
        minimum_level: AuditEventLevel = AuditEventLevel.ERROR,
    ) -> IncidentDetectionResult:
        events = self._integration.query(
            correlation_id=correlation_id,
        )

        severity = {
            AuditEventLevel.INFO: 0,
            AuditEventLevel.WARNING: 1,
            AuditEventLevel.ERROR: 2,
            AuditEventLevel.CRITICAL: 3,
        }

        if not isinstance(
            minimum_level,
            AuditEventLevel,
        ):
            try:
                minimum_level = AuditEventLevel(
                    minimum_level
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "minimum_level must be a valid "
                    "AuditEventLevel."
                ) from exc

        incidents = []

        for correlated in events:
            if severity[correlated.event.level] < severity[
                minimum_level
            ]:
                continue

            incidents.append(
                Incident(
                    event=correlated.event,
                    reason=(
                        f"Correlated event level "
                        f"{correlated.event.level.value} "
                        f"meets incident threshold "
                        f"{minimum_level.value}."
                    ),
                )
            )

        return IncidentDetectionResult(
            success=True,
            incidents=incidents,
        )

    def has_active_incidents(
        self,
        source: Optional[str] = None,
    ) -> bool:
        return self.detect_incidents(
            source=source
        ).has_incidents

    def clear(self) -> None:
        self._observability.clear()


def create_production_observability_analytics(
    observability: Optional[
        QuantAIProductionObservability
    ] = None,
    integration: Optional[
        QuantAIProductionObservabilityIntegration
    ] = None,
) -> QuantAIProductionObservabilityAnalytics:
    return QuantAIProductionObservabilityAnalytics(
        observability=observability,
        integration=integration,
    )


__all__ = [
    "OperationalMetrics",
    "Incident",
    "IncidentDetectionResult",
    "QuantAIProductionObservabilityAnalytics",
    "create_production_observability_analytics",
]