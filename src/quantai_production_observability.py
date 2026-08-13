from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from threading import Lock
from typing import Any, List, Optional


class AuditEventLevel(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class AuditEvent:
    timestamp: datetime
    event_type: str
    level: AuditEventLevel
    message: str
    source: str = "quantai"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ObservabilitySnapshot:
    total_events: int
    info_events: int
    warning_events: int
    error_events: int
    critical_events: int
    last_event_type: Optional[str]
    last_event_level: Optional[AuditEventLevel]


class QuantAIProductionObservability:
    """Thread-safe in-memory audit log and operational event tracker."""

    def __init__(self, max_events: int = 1000) -> None:
        if not isinstance(max_events, int):
            raise TypeError("max_events must be an integer.")

        if isinstance(max_events, bool) or max_events <= 0:
            raise ValueError(
                "max_events must be greater than zero."
            )

        self._max_events = max_events
        self._events: List[AuditEvent] = []
        self._lock = Lock()

    @property
    def max_events(self) -> int:
        return self._max_events

    @property
    def events(self) -> List[AuditEvent]:
        with self._lock:
            return list(self._events)

    def record(
        self,
        event_type: str,
        message: str,
        level: AuditEventLevel = AuditEventLevel.INFO,
        source: str = "quantai",
        metadata: Optional[dict[str, Any]] = None,
    ) -> AuditEvent:
        if not isinstance(event_type, str) or not event_type.strip():
            raise ValueError(
                "event_type must be a non-empty string."
            )

        if not isinstance(message, str) or not message.strip():
            raise ValueError(
                "message must be a non-empty string."
            )

        if not isinstance(level, AuditEventLevel):
            try:
                level = AuditEventLevel(level)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "level must be a valid AuditEventLevel."
                ) from exc

        if not isinstance(source, str) or not source.strip():
            raise ValueError(
                "source must be a non-empty string."
            )

        if metadata is not None and not isinstance(
            metadata,
            dict,
        ):
            raise TypeError(
                "metadata must be a dictionary or None."
            )

        event = AuditEvent(
            timestamp=datetime.now(timezone.utc),
            event_type=event_type.strip(),
            level=level,
            message=message.strip(),
            source=source.strip(),
            metadata=dict(metadata or {}),
        )

        with self._lock:
            self._events.append(event)

            if len(self._events) > self._max_events:
                del self._events[
                    : len(self._events) - self._max_events
                ]

        return event

    def info(
        self,
        event_type: str,
        message: str,
        source: str = "quantai",
        **metadata: Any,
    ) -> AuditEvent:
        return self.record(
            event_type=event_type,
            message=message,
            level=AuditEventLevel.INFO,
            source=source,
            metadata=metadata,
        )

    def warning(
        self,
        event_type: str,
        message: str,
        source: str = "quantai",
        **metadata: Any,
    ) -> AuditEvent:
        return self.record(
            event_type=event_type,
            message=message,
            level=AuditEventLevel.WARNING,
            source=source,
            metadata=metadata,
        )

    def error(
        self,
        event_type: str,
        message: str,
        source: str = "quantai",
        **metadata: Any,
    ) -> AuditEvent:
        return self.record(
            event_type=event_type,
            message=message,
            level=AuditEventLevel.ERROR,
            source=source,
            metadata=metadata,
        )

    def critical(
        self,
        event_type: str,
        message: str,
        source: str = "quantai",
        **metadata: Any,
    ) -> AuditEvent:
        return self.record(
            event_type=event_type,
            message=message,
            level=AuditEventLevel.CRITICAL,
            source=source,
            metadata=metadata,
        )

    def query(
        self,
        event_type: Optional[str] = None,
        level: Optional[AuditEventLevel] = None,
        source: Optional[str] = None,
    ) -> List[AuditEvent]:
        with self._lock:
            events = list(self._events)

        if event_type is not None:
            events = [
                event
                for event in events
                if event.event_type == event_type
            ]

        if level is not None:
            if not isinstance(level, AuditEventLevel):
                level = AuditEventLevel(level)

            events = [
                event
                for event in events
                if event.level == level
            ]

        if source is not None:
            events = [
                event
                for event in events
                if event.source == source
            ]

        return events

    def snapshot(self) -> ObservabilitySnapshot:
        with self._lock:
            events = list(self._events)

        counts = {
            level: 0
            for level in AuditEventLevel
        }

        for event in events:
            counts[event.level] += 1

        last_event = events[-1] if events else None

        return ObservabilitySnapshot(
            total_events=len(events),
            info_events=counts[AuditEventLevel.INFO],
            warning_events=counts[AuditEventLevel.WARNING],
            error_events=counts[AuditEventLevel.ERROR],
            critical_events=counts[AuditEventLevel.CRITICAL],
            last_event_type=(
                last_event.event_type
                if last_event
                else None
            ),
            last_event_level=(
                last_event.level
                if last_event
                else None
            ),
        )

    def clear(self) -> None:
        with self._lock:
            self._events.clear()


def create_production_observability(
    max_events: int = 1000,
) -> QuantAIProductionObservability:
    return QuantAIProductionObservability(
        max_events=max_events
    )


__all__ = [
    "AuditEventLevel",
    "AuditEvent",
    "ObservabilitySnapshot",
    "QuantAIProductionObservability",
    "create_production_observability",
]