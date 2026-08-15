from __future__ import annotations

import pytest

from src.quantai_production_observability import (
    AuditEventLevel,
    QuantAIProductionObservability,
)

from src.quantai_production_observability_integration import (
    CorrelatedAuditEvent,
    ObservabilityIntegrationResult,
    QuantAIProductionObservabilityIntegration,
    create_observability_integration,
)


def test_default_observability_is_created():
    integration = (
        QuantAIProductionObservabilityIntegration()
    )

    assert isinstance(
        integration.observability,
        QuantAIProductionObservability,
    )


def test_existing_observability_can_be_injected():
    observability = (
        QuantAIProductionObservability()
    )

    integration = (
        QuantAIProductionObservabilityIntegration(
            observability=observability
        )
    )

    assert integration.observability is observability


def test_invalid_observability_is_rejected():
    with pytest.raises(TypeError):
        QuantAIProductionObservabilityIntegration(
            observability=object()
        )


def test_record_event_creates_correlated_event():
    integration = (
        QuantAIProductionObservabilityIntegration()
    )

    result = integration.record_event(
        correlation_id="corr-001",
        event_type="runtime_start",
        message="Runtime started.",
        source="runtime",
    )

    assert isinstance(
        result,
        CorrelatedAuditEvent,
    )

    assert result.correlation_id == "corr-001"

    assert result.event.event_type == (
        "runtime_start"
    )

    assert result.event.source == "runtime"

    assert result.event.metadata[
        "correlation_id"
    ] == "corr-001"


def test_correlation_query_returns_matching_events():
    integration = (
        QuantAIProductionObservabilityIntegration()
    )

    integration.info(
        "corr-001",
        "runtime_start",
        "Runtime started.",
        source="runtime",
    )

    integration.warning(
        "corr-001",
        "runtime_warning",
        "Runtime warning.",
        source="supervisor",
    )

    integration.error(
        "corr-002",
        "runtime_error",
        "Runtime error.",
        source="runtime",
    )

    events = integration.query_by_correlation(
        "corr-001"
    )

    assert len(events) == 2

    assert all(
        event.correlation_id == "corr-001"
        for event in events
    )


def test_query_filters_by_level():
    integration = (
        QuantAIProductionObservabilityIntegration()
    )

    integration.info(
        "corr-001",
        "info",
        "Info.",
    )

    integration.warning(
        "corr-001",
        "warning",
        "Warning.",
    )

    integration.error(
        "corr-001",
        "error",
        "Error.",
    )

    result = integration.query(
        level=AuditEventLevel.WARNING
    )

    assert len(result) == 1

    assert result[0].event.level == (
        AuditEventLevel.WARNING
    )


def test_query_filters_by_source():
    integration = (
        QuantAIProductionObservabilityIntegration()
    )

    integration.info(
        "corr-001",
        "runtime_event",
        "Runtime.",
        source="runtime",
    )

    integration.warning(
        "corr-001",
        "supervisor_event",
        "Supervisor.",
        source="supervisor",
    )

    integration.error(
        "corr-002",
        "runtime_error",
        "Runtime error.",
        source="runtime",
    )

    result = integration.query(
        source="runtime"
    )

    assert len(result) == 2

    assert all(
        item.event.source == "runtime"
        for item in result
    )


def test_query_filters_by_event_type():
    integration = (
        QuantAIProductionObservabilityIntegration()
    )

    integration.info(
        "corr-001",
        "startup",
        "Started.",
    )

    integration.info(
        "corr-002",
        "shutdown",
        "Stopped.",
    )

    result = integration.query(
        event_type="startup"
    )

    assert len(result) == 1

    assert result[0].event.event_type == (
        "startup"
    )


def test_query_combines_filters():
    integration = (
        QuantAIProductionObservabilityIntegration()
    )

    integration.info(
        "corr-001",
        "runtime",
        "Runtime.",
        source="runtime",
    )

    integration.warning(
        "corr-001",
        "runtime",
        "Warning.",
        source="runtime",
    )

    integration.warning(
        "corr-002",
        "runtime",
        "Other warning.",
        source="runtime",
    )

    result = integration.query(
        correlation_id="corr-001",
        level=AuditEventLevel.WARNING,
        source="runtime",
        event_type="runtime",
    )

    assert len(result) == 1

    assert result[0].correlation_id == (
        "corr-001"
    )


def test_audit_trail_returns_result():
    integration = (
        QuantAIProductionObservabilityIntegration()
    )

    integration.info(
        "corr-001",
        "start",
        "Started.",
    )

    integration.info(
        "corr-001",
        "health",
        "Healthy.",
    )

    result = integration.audit_trail(
        "corr-001"
    )

    assert isinstance(
        result,
        ObservabilityIntegrationResult,
    )

    assert result.success is True

    assert result.correlation_id == (
        "corr-001"
    )

    assert result.event_count == 2


def test_audit_trail_for_unknown_correlation_is_empty():
    integration = (
        QuantAIProductionObservabilityIntegration()
    )

    result = integration.audit_trail(
        "unknown"
    )

    assert result.success is True

    assert result.event_count == 0

    assert result.events == []


def test_info_records_correct_level():
    integration = (
        QuantAIProductionObservabilityIntegration()
    )

    event = integration.info(
        "corr-001",
        "info",
        "Information.",
    )

    assert event.event.level == (
        AuditEventLevel.INFO
    )


def test_warning_records_correct_level():
    integration = (
        QuantAIProductionObservabilityIntegration()
    )

    event = integration.warning(
        "corr-001",
        "warning",
        "Warning.",
    )

    assert event.event.level == (
        AuditEventLevel.WARNING
    )


def test_error_records_correct_level():
    integration = (
        QuantAIProductionObservabilityIntegration()
    )

    event = integration.error(
        "corr-001",
        "error",
        "Error.",
    )

    assert event.event.level == (
        AuditEventLevel.ERROR
    )


def test_critical_records_correct_level():
    integration = (
        QuantAIProductionObservabilityIntegration()
    )

    event = integration.critical(
        "corr-001",
        "critical",
        "Critical.",
    )

    assert event.event.level == (
        AuditEventLevel.CRITICAL
    )


def test_metadata_is_preserved():
    integration = (
        QuantAIProductionObservabilityIntegration()
    )

    event = integration.info(
        "corr-001",
        "order",
        "Order event.",
        source="trade_engine",
        symbol="BTCUSDT",
        order_id="123",
    )

    assert event.event.metadata[
        "correlation_id"
    ] == "corr-001"

    assert event.event.metadata[
        "symbol"
    ] == "BTCUSDT"

    assert event.event.metadata[
        "order_id"
    ] == "123"


def test_empty_correlation_id_is_rejected():
    integration = (
        QuantAIProductionObservabilityIntegration()
    )

    with pytest.raises(ValueError):
        integration.info(
            "",
            "event",
            "Message.",
        )


def test_non_string_correlation_id_is_rejected():
    integration = (
        QuantAIProductionObservabilityIntegration()
    )

    with pytest.raises(TypeError):
        integration.info(
            123,
            "event",
            "Message.",
        )


def test_empty_event_type_is_rejected():
    integration = (
        QuantAIProductionObservabilityIntegration()
    )

    with pytest.raises(ValueError):
        integration.info(
            "corr-001",
            "",
            "Message.",
        )


def test_empty_source_is_rejected():
    integration = (
        QuantAIProductionObservabilityIntegration()
    )

    with pytest.raises(ValueError):
        integration.info(
            "corr-001",
            "event",
            "Message.",
            source="",
        )


def test_clear_removes_integrated_events():
    integration = (
        QuantAIProductionObservabilityIntegration()
    )

    integration.info(
        "corr-001",
        "event",
        "Message.",
    )

    assert integration.audit_trail(
        "corr-001"
    ).event_count == 1

    integration.clear()

    assert integration.audit_trail(
        "corr-001"
    ).event_count == 0


def test_factory_creates_integration():
    integration = (
        create_observability_integration()
    )

    assert isinstance(
        integration,
        QuantAIProductionObservabilityIntegration,
    )


def test_query_returns_empty_for_no_matches():
    integration = (
        QuantAIProductionObservabilityIntegration()
    )

    integration.info(
        "corr-001",
        "event",
        "Message.",
    )

    result = integration.query(
        correlation_id="corr-999"
    )

    assert result == []