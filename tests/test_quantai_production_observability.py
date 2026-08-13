from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from src.quantai_production_observability import (
    AuditEvent,
    AuditEventLevel,
    ObservabilitySnapshot,
    QuantAIProductionObservability,
    create_production_observability,
)


def test_initial_state_is_empty():
    observability = QuantAIProductionObservability()

    assert observability.events == []

    snapshot = observability.snapshot()

    assert isinstance(
        snapshot,
        ObservabilitySnapshot,
    )

    assert snapshot.total_events == 0
    assert snapshot.last_event_type is None
    assert snapshot.last_event_level is None


def test_record_creates_audit_event():
    observability = QuantAIProductionObservability()

    event = observability.record(
        "runtime.started",
        "Runtime started.",
        source="runtime",
        metadata={
            "mode": "PAPER",
        },
    )

    assert isinstance(
        event,
        AuditEvent,
    )

    assert event.level is AuditEventLevel.INFO
    assert event.event_type == "runtime.started"
    assert event.source == "runtime"
    assert event.metadata == {
        "mode": "PAPER",
    }

    assert len(observability.events) == 1


def test_convenience_methods_use_correct_levels():
    observability = QuantAIProductionObservability()

    observability.info(
        "a",
        "info",
    )

    observability.warning(
        "b",
        "warning",
    )

    observability.error(
        "c",
        "error",
    )

    observability.critical(
        "d",
        "critical",
    )

    levels = [
        event.level
        for event in observability.events
    ]

    assert levels == [
        AuditEventLevel.INFO,
        AuditEventLevel.WARNING,
        AuditEventLevel.ERROR,
        AuditEventLevel.CRITICAL,
    ]


def test_snapshot_counts_levels():
    observability = QuantAIProductionObservability()

    observability.info(
        "a",
        "x",
    )

    observability.info(
        "b",
        "x",
    )

    observability.warning(
        "c",
        "x",
    )

    observability.error(
        "d",
        "x",
    )

    observability.critical(
        "e",
        "x",
    )

    snapshot = observability.snapshot()

    assert snapshot.total_events == 5
    assert snapshot.info_events == 2
    assert snapshot.warning_events == 1
    assert snapshot.error_events == 1
    assert snapshot.critical_events == 1

    assert snapshot.last_event_type == "e"

    assert (
        snapshot.last_event_level
        is AuditEventLevel.CRITICAL
    )


def test_query_filters_by_event_type():
    observability = QuantAIProductionObservability()

    observability.info(
        "runtime.started",
        "x",
    )

    observability.info(
        "runtime.stopped",
        "x",
    )

    observability.info(
        "runtime.started",
        "y",
    )

    result = observability.query(
        event_type="runtime.started"
    )

    assert len(result) == 2

    assert all(
        event.event_type == "runtime.started"
        for event in result
    )


def test_query_filters_by_level_and_source():
    observability = QuantAIProductionObservability()

    observability.warning(
        "a",
        "x",
        source="runtime",
    )

    observability.warning(
        "b",
        "x",
        source="supervisor",
    )

    observability.error(
        "c",
        "x",
        source="runtime",
    )

    assert len(
        observability.query(
            level=AuditEventLevel.WARNING
        )
    ) == 2

    assert len(
        observability.query(
            source="runtime"
        )
    ) == 2


def test_string_level_is_accepted():
    observability = QuantAIProductionObservability()

    event = observability.record(
        "runtime.warning",
        "warning",
        level="WARNING",
    )

    assert (
        event.level
        is AuditEventLevel.WARNING
    )


def test_max_events_retains_latest_events():
    observability = QuantAIProductionObservability(
        max_events=2
    )

    observability.info(
        "one",
        "1",
    )

    observability.info(
        "two",
        "2",
    )

    observability.info(
        "three",
        "3",
    )

    assert [
        event.event_type
        for event in observability.events
    ] == [
        "two",
        "three",
    ]


def test_clear_removes_all_events():
    observability = QuantAIProductionObservability()

    observability.info(
        "a",
        "x",
    )

    observability.clear()

    assert observability.events == []

    assert (
        observability.snapshot().total_events
        == 0
    )


def test_invalid_max_events():
    with pytest.raises(TypeError):
        QuantAIProductionObservability("10")

    with pytest.raises(ValueError):
        QuantAIProductionObservability(0)

    with pytest.raises(ValueError):
        QuantAIProductionObservability(-1)


@pytest.mark.parametrize(
    "value",
    [
        "",
        "   ",
        None,
        123,
    ],
)
def test_invalid_event_type(value):
    observability = QuantAIProductionObservability()

    with pytest.raises(ValueError):
        observability.record(
            value,
            "message",
        )


@pytest.mark.parametrize(
    "value",
    [
        "",
        "   ",
        None,
        123,
    ],
)
def test_invalid_message(value):
    observability = QuantAIProductionObservability()

    with pytest.raises(ValueError):
        observability.record(
            "event",
            value,
        )


def test_invalid_source():
    observability = QuantAIProductionObservability()

    with pytest.raises(ValueError):
        observability.record(
            "event",
            "message",
            source="",
        )


def test_invalid_metadata():
    observability = QuantAIProductionObservability()

    with pytest.raises(TypeError):
        observability.record(
            "event",
            "message",
            metadata=[],
        )


def test_events_are_returned_as_a_copy():
    observability = QuantAIProductionObservability()

    observability.info(
        "event",
        "message",
    )

    events = observability.events

    events.clear()

    assert len(
        observability.events
    ) == 1


def test_audit_event_is_immutable():
    observability = QuantAIProductionObservability()

    event = observability.info(
        "event",
        "message",
    )

    with pytest.raises(
        FrozenInstanceError
    ):
        event.message = "changed"


def test_factory_returns_observability():
    observability = create_production_observability(
        max_events=10
    )

    assert isinstance(
        observability,
        QuantAIProductionObservability,
    )

    assert observability.max_events == 10


def test_query_returns_copy():
    observability = QuantAIProductionObservability()

    observability.info(
        "event",
        "message",
    )

    result = observability.query()

    result.clear()

    assert len(
        observability.events
    ) == 1


def test_timestamps_are_utc():
    observability = QuantAIProductionObservability()

    event = observability.info(
        "event",
        "message",
    )

    assert event.timestamp.tzinfo is not None

    assert (
        event.timestamp.utcoffset().total_seconds()
        == 0
    )