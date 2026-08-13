from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

from src.quantai_production_observability import (
    AuditEvent,
    AuditEventLevel,
    QuantAIProductionObservability,
)


@dataclass(frozen=True)
class CorrelatedAuditEvent:
    correlation_id: str
    event: AuditEvent


@dataclass
class ObservabilityIntegrationResult:
    success: bool
    correlation_id: str
    events: List[CorrelatedAuditEvent] = field(
        default_factory=list
    )
    errors: List[str] = field(
        default_factory=list
    )
    warnings: List[str] = field(
        default_factory=list
    )

    @property
    def event_count(self) -> int:
        return len(self.events)


class QuantAIProductionObservabilityIntegration:
    """
    Integration layer between QuantAI operational components
    and the production observability system.

    Responsibilities:

        - audit event creation
        - correlation ID management
        - control-event tracking
        - audit trail retrieval
        - event filtering

    This module does not implement trading logic or runtime control.
    """

    def __init__(
        self,
        observability: Optional[
            QuantAIProductionObservability
        ] = None,
    ) -> None:
        if observability is not None and not isinstance(
            observability,
            QuantAIProductionObservability,
        ):
            raise TypeError(
                "observability must be a "
                "QuantAIProductionObservability instance "
                "or None."
            )

        self._observability = (
            observability
            if observability is not None
            else QuantAIProductionObservability()
        )

    @property
    def observability(
        self,
    ) -> QuantAIProductionObservability:
        return self._observability

    @staticmethod
    def _validate_correlation_id(
        correlation_id: str,
    ) -> str:
        if not isinstance(
            correlation_id,
            str,
        ):
            raise TypeError(
                "correlation_id must be a string."
            )

        correlation_id = correlation_id.strip()

        if not correlation_id:
            raise ValueError(
                "correlation_id must be a non-empty string."
            )

        return correlation_id

    @staticmethod
    def _validate_event_type(
        event_type: str,
    ) -> str:
        if not isinstance(
            event_type,
            str,
        ):
            raise TypeError(
                "event_type must be a string."
            )

        event_type = event_type.strip()

        if not event_type:
            raise ValueError(
                "event_type must be a non-empty string."
            )

        return event_type

    @staticmethod
    def _validate_source(
        source: str,
    ) -> str:
        if not isinstance(
            source,
            str,
        ):
            raise TypeError(
                "source must be a string."
            )

        source = source.strip()

        if not source:
            raise ValueError(
                "source must be a non-empty string."
            )

        return source

    def record_event(
        self,
        correlation_id: str,
        event_type: str,
        message: str,
        level: AuditEventLevel = AuditEventLevel.INFO,
        source: str = "quantai",
        metadata: Optional[dict[str, Any]] = None,
    ) -> CorrelatedAuditEvent:
        correlation_id = self._validate_correlation_id(
            correlation_id
        )

        event_type = self._validate_event_type(
            event_type
        )

        source = self._validate_source(
            source
        )

        event_metadata = dict(
            metadata or {}
        )

        event_metadata[
            "correlation_id"
        ] = correlation_id

        event = self._observability.record(
            event_type=event_type,
            message=message,
            level=level,
            source=source,
            metadata=event_metadata,
        )

        return CorrelatedAuditEvent(
            correlation_id=correlation_id,
            event=event,
        )

    def info(
        self,
        correlation_id: str,
        event_type: str,
        message: str,
        source: str = "quantai",
        **metadata: Any,
    ) -> CorrelatedAuditEvent:
        return self.record_event(
            correlation_id=correlation_id,
            event_type=event_type,
            message=message,
            level=AuditEventLevel.INFO,
            source=source,
            metadata=metadata,
        )

    def warning(
        self,
        correlation_id: str,
        event_type: str,
        message: str,
        source: str = "quantai",
        **metadata: Any,
    ) -> CorrelatedAuditEvent:
        return self.record_event(
            correlation_id=correlation_id,
            event_type=event_type,
            message=message,
            level=AuditEventLevel.WARNING,
            source=source,
            metadata=metadata,
        )

    def error(
        self,
        correlation_id: str,
        event_type: str,
        message: str,
        source: str = "quantai",
        **metadata: Any,
    ) -> CorrelatedAuditEvent:
        return self.record_event(
            correlation_id=correlation_id,
            event_type=event_type,
            message=message,
            level=AuditEventLevel.ERROR,
            source=source,
            metadata=metadata,
        )

    def critical(
        self,
        correlation_id: str,
        event_type: str,
        message: str,
        source: str = "quantai",
        **metadata: Any,
    ) -> CorrelatedAuditEvent:
        return self.record_event(
            correlation_id=correlation_id,
            event_type=event_type,
            message=message,
            level=AuditEventLevel.CRITICAL,
            source=source,
            metadata=metadata,
        )

    def query_by_correlation(
        self,
        correlation_id: str,
    ) -> List[CorrelatedAuditEvent]:
        correlation_id = self._validate_correlation_id(
            correlation_id
        )

        events = self._observability.events

        return [
            CorrelatedAuditEvent(
                correlation_id=correlation_id,
                event=event,
            )
            for event in events
            if event.metadata.get(
                "correlation_id"
            ) == correlation_id
        ]

    def query(
        self,
        correlation_id: Optional[str] = None,
        event_type: Optional[str] = None,
        level: Optional[AuditEventLevel] = None,
        source: Optional[str] = None,
    ) -> List[CorrelatedAuditEvent]:
        if correlation_id is not None:
            correlation_id = self._validate_correlation_id(
                correlation_id
            )

        if event_type is not None:
            event_type = self._validate_event_type(
                event_type
            )

        if source is not None:
            source = self._validate_source(
                source
            )

        events = self._observability.query(
            event_type=event_type,
            level=level,
            source=source,
        )

        result: List[CorrelatedAuditEvent] = []

        for event in events:
            event_correlation_id = event.metadata.get(
                "correlation_id"
            )

            if (
                correlation_id is not None
                and event_correlation_id
                != correlation_id
            ):
                continue

            result.append(
                CorrelatedAuditEvent(
                    correlation_id=(
                        str(event_correlation_id)
                        if event_correlation_id is not None
                        else ""
                    ),
                    event=event,
                )
            )

        return result

    def audit_trail(
        self,
        correlation_id: str,
    ) -> ObservabilityIntegrationResult:
        correlation_id = self._validate_correlation_id(
            correlation_id
        )

        events = self.query_by_correlation(
            correlation_id
        )

        return ObservabilityIntegrationResult(
            success=True,
            correlation_id=correlation_id,
            events=events,
        )

    def clear(self) -> None:
        self._observability.clear()


def create_observability_integration(
    observability: Optional[
        QuantAIProductionObservability
    ] = None,
) -> QuantAIProductionObservabilityIntegration:
    return QuantAIProductionObservabilityIntegration(
        observability=observability
    )


__all__ = [
    "CorrelatedAuditEvent",
    "ObservabilityIntegrationResult",
    "QuantAIProductionObservabilityIntegration",
    "create_observability_integration",
]