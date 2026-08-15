from __future__ import annotations

import pytest

from src.quantai_production_observability import (
    AuditEventLevel,
    QuantAIProductionObservability,
)

from src.quantai_production_observability_integration import (
    QuantAIProductionObservabilityIntegration,
)

from src.quantai_production_observability_analytics import (
    Incident,
    IncidentDetectionResult,
    OperationalMetrics,
    QuantAIProductionObservabilityAnalytics,
    create_production_observability_analytics,
)


def test_default_dependencies_are_created():
    analytics = (
        QuantAIProductionObservabilityAnalytics()
    )

    assert isinstance(
        analytics.observability,
        QuantAIProductionObservability,
    )

    assert isinstance(
        analytics.integration,
        QuantAIProductionObservabilityIntegration,
    )


def test_existing_observability_can_be_injected():
    observability = QuantAIProductionObservability()

    analytics = (
        QuantAIProductionObservabilityAnalytics(
            observability=observability
        )
    )

    assert analytics.observability is observability
    assert (
        analytics.integration.observability
        is observability
    )


def test_existing_integration_can_be_injected():
    integration = (
        QuantAIProductionObservabilityIntegration()
    )

    analytics = (
        QuantAIProductionObservabilityAnalytics(
            integration=integration
        )
    )

    assert analytics.integration is integration


def test_invalid_integration_is_rejected():
    with pytest.raises(TypeError):
        QuantAIProductionObservabilityAnalytics(
            integration=object()
        )


def test_invalid_observability_is_rejected():
    with pytest.raises(TypeError):
        QuantAIProductionObservabilityAnalytics(
            observability=object()
        )


def test_mismatched_dependencies_are_rejected():
    first = QuantAIProductionObservability()
    second = QuantAIProductionObservability()

    integration = (
        QuantAIProductionObservabilityIntegration(
            observability=first
        )
    )

    with pytest.raises(ValueError):
        QuantAIProductionObservabilityAnalytics(
            observability=second,
            integration=integration,
        )


def test_metrics_are_calculated():
    analytics = (
        QuantAIProductionObservabilityAnalytics()
    )

    analytics.integration.info(
        "corr-1",
        "startup",
        "Runtime started.",
        source="runtime",
    )

    analytics.integration.warning(
        "corr-1",
        "warning",
        "Runtime warning.",
        source="supervisor",
    )

    analytics.integration.error(
        "corr-2",
        "error",
        "Runtime error.",
        source="runtime",
    )

    analytics.integration.critical(
        "corr-3",
        "critical",
        "Critical failure.",
        source="control",
    )

    metrics = analytics.calculate_metrics()

    assert isinstance(
        metrics,
        OperationalMetrics,
    )

    assert metrics.total_events == 4
    assert metrics.info_events == 1
    assert metrics.warning_events == 1
    assert metrics.error_events == 1
    assert metrics.critical_events == 1
    assert metrics.incident_events == 2
    assert metrics.sources == (
        "control",
        "runtime",
        "supervisor",
    )


def test_metrics_can_filter_by_source():
    analytics = (
        QuantAIProductionObservabilityAnalytics()
    )

    analytics.integration.info(
        "corr-1",
        "a",
        "A.",
        source="runtime",
    )

    analytics.integration.error(
        "corr-2",
        "b",
        "B.",
        source="runtime",
    )

    analytics.integration.warning(
        "corr-3",
        "c",
        "C.",
        source="supervisor",
    )

    metrics = analytics.calculate_metrics(
        source="runtime"
    )

    assert metrics.total_events == 2
    assert metrics.info_events == 1
    assert metrics.error_events == 1
    assert metrics.warning_events == 0
    assert metrics.incident_events == 1
    assert metrics.sources == ("runtime",)


def test_no_events_produces_empty_metrics():
    analytics = (
        QuantAIProductionObservabilityAnalytics()
    )

    metrics = analytics.calculate_metrics()

    assert metrics.total_events == 0
    assert metrics.info_events == 0
    assert metrics.warning_events == 0
    assert metrics.error_events == 0
    assert metrics.critical_events == 0
    assert metrics.incident_events == 0
    assert metrics.sources == ()


def test_error_events_are_detected_as_incidents():
    analytics = (
        QuantAIProductionObservabilityAnalytics()
    )

    analytics.integration.info(
        "corr-1",
        "info",
        "Info.",
    )

    analytics.integration.error(
        "corr-1",
        "error",
        "Error.",
    )

    result = analytics.detect_incidents()

    assert isinstance(
        result,
        IncidentDetectionResult,
    )

    assert result.success is True
    assert result.incident_count == 1
    assert result.has_incidents is True
    assert isinstance(
        result.incidents[0],
        Incident,
    )


def test_critical_incidents_are_detected():
    analytics = (
        QuantAIProductionObservabilityAnalytics()
    )

    analytics.integration.error(
        "corr-1",
        "error",
        "Error.",
    )

    analytics.integration.critical(
        "corr-2",
        "critical",
        "Critical.",
    )

    result = analytics.detect_critical_incidents()

    assert result.incident_count == 1
    assert (
        result.incidents[0].event.level
        == AuditEventLevel.CRITICAL
    )


def test_warning_threshold_can_detect_warnings():
    analytics = (
        QuantAIProductionObservabilityAnalytics()
    )

    analytics.integration.info(
        "corr-1",
        "info",
        "Info.",
    )

    analytics.integration.warning(
        "corr-2",
        "warning",
        "Warning.",
    )

    result = analytics.detect_incidents(
        minimum_level=AuditEventLevel.WARNING
    )

    assert result.incident_count == 1
    assert (
        result.incidents[0].event.level
        == AuditEventLevel.WARNING
    )


def test_info_threshold_detects_all_events():
    analytics = (
        QuantAIProductionObservabilityAnalytics()
    )

    analytics.integration.info(
        "corr-1",
        "info",
        "Info.",
    )

    analytics.integration.warning(
        "corr-2",
        "warning",
        "Warning.",
    )

    analytics.integration.error(
        "corr-3",
        "error",
        "Error.",
    )

    result = analytics.detect_incidents(
        minimum_level=AuditEventLevel.INFO
    )

    assert result.incident_count == 3


def test_incidents_can_filter_by_source():
    analytics = (
        QuantAIProductionObservabilityAnalytics()
    )

    analytics.integration.error(
        "corr-1",
        "runtime_error",
        "Runtime error.",
        source="runtime",
    )

    analytics.integration.error(
        "corr-2",
        "supervisor_error",
        "Supervisor error.",
        source="supervisor",
    )

    result = analytics.detect_incidents(
        source="runtime"
    )

    assert result.incident_count == 1
    assert (
        result.incidents[0].event.source
        == "runtime"
    )


def test_correlated_incidents_are_detected():
    analytics = (
        QuantAIProductionObservabilityAnalytics()
    )

    analytics.integration.info(
        "corr-1",
        "startup",
        "Started.",
    )

    analytics.integration.error(
        "corr-1",
        "runtime_error",
        "Failure.",
    )

    analytics.integration.error(
        "corr-2",
        "other_error",
        "Other failure.",
    )

    result = analytics.detect_correlated_incidents(
        "corr-1"
    )

    assert result.incident_count == 1

    assert (
        result.incidents[0].event.metadata[
            "correlation_id"
        ]
        == "corr-1"
    )


def test_unknown_correlation_has_no_incidents():
    analytics = (
        QuantAIProductionObservabilityAnalytics()
    )

    analytics.integration.error(
        "corr-1",
        "error",
        "Failure.",
    )

    result = analytics.detect_correlated_incidents(
        "unknown"
    )

    assert result.success is True
    assert result.incident_count == 0


def test_has_active_incidents():
    analytics = (
        QuantAIProductionObservabilityAnalytics()
    )

    assert (
        analytics.has_active_incidents()
        is False
    )

    analytics.integration.error(
        "corr-1",
        "error",
        "Failure.",
    )

    assert (
        analytics.has_active_incidents()
        is True
    )


def test_incident_reason_is_populated():
    analytics = (
        QuantAIProductionObservabilityAnalytics()
    )

    analytics.integration.critical(
        "corr-1",
        "critical",
        "Critical failure.",
    )

    result = analytics.detect_incidents()

    assert result.incident_count == 1
    assert result.incidents[0].reason
    assert "CRITICAL" in result.incidents[0].reason


def test_invalid_minimum_level_is_rejected():
    analytics = (
        QuantAIProductionObservabilityAnalytics()
    )

    with pytest.raises(ValueError):
        analytics.detect_incidents(
            minimum_level="INVALID"
        )


def test_string_level_can_be_normalized():
    analytics = (
        QuantAIProductionObservabilityAnalytics()
    )

    analytics.integration.error(
        "corr-1",
        "error",
        "Failure.",
    )

    result = analytics.detect_incidents(
        minimum_level="ERROR"
    )

    assert result.incident_count == 1


def test_factory_creates_analytics():
    analytics = (
        create_production_observability_analytics()
    )

    assert isinstance(
        analytics,
        QuantAIProductionObservabilityAnalytics,
    )


def test_clear_removes_events_and_incidents():
    analytics = (
        QuantAIProductionObservabilityAnalytics()
    )

    analytics.integration.error(
        "corr-1",
        "error",
        "Failure.",
    )

    assert analytics.has_active_incidents() is True

    analytics.clear()

    assert analytics.has_active_incidents() is False
    assert (
        analytics.calculate_metrics().total_events
        == 0
    )


def test_operational_metrics_are_immutable():
    analytics = (
        QuantAIProductionObservabilityAnalytics()
    )

    metrics = analytics.calculate_metrics()

    with pytest.raises(Exception):
        metrics.total_events = 10