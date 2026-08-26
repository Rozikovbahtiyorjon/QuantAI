import pytest

from experimental.src.quantai_monitoring import (
    ComponentHealth,
    HealthStatus,
    QuantAIMonitoring,
)


def test_healthy_component() -> None:
    monitor = QuantAIMonitoring()

    result = monitor.evaluate_component(
        "market_data",
        healthy=True,
    )

    assert result.status is HealthStatus.HEALTHY


def test_unhealthy_component_is_critical() -> None:
    monitor = QuantAIMonitoring()

    result = monitor.evaluate_component(
        "risk",
        healthy=False,
    )

    assert result.status is HealthStatus.CRITICAL


def test_fresh_data_is_healthy() -> None:
    monitor = QuantAIMonitoring(
        stale_threshold_seconds=30,
    )

    result = monitor.evaluate_staleness(
        "market_data",
        20,
    )

    assert result.status is HealthStatus.HEALTHY


def test_warning_staleness_is_degraded() -> None:
    monitor = QuantAIMonitoring(
        stale_threshold_seconds=30,
    )

    result = monitor.evaluate_staleness(
        "market_data",
        45,
    )

    assert result.status is HealthStatus.DEGRADED


def test_old_data_is_critical() -> None:
    monitor = QuantAIMonitoring(
        stale_threshold_seconds=30,
    )

    result = monitor.evaluate_staleness(
        "market_data",
        70,
    )

    assert result.status is HealthStatus.CRITICAL


def test_system_is_healthy_when_all_components_are_healthy() -> None:
    monitor = QuantAIMonitoring()

    result = monitor.evaluate_system(
        {
            "market_data": ComponentHealth(
                "market_data",
                HealthStatus.HEALTHY,
            ),
            "model": ComponentHealth(
                "model",
                HealthStatus.HEALTHY,
            ),
        }
    )

    assert result.status is HealthStatus.HEALTHY
    assert result.healthy_count == 2
    assert result.degraded_count == 0
    assert result.critical_count == 0


def test_system_is_degraded_when_component_is_degraded() -> None:
    monitor = QuantAIMonitoring()

    result = monitor.evaluate_system(
        {
            "market_data": ComponentHealth(
                "market_data",
                HealthStatus.HEALTHY,
            ),
            "model": ComponentHealth(
                "model",
                HealthStatus.DEGRADED,
            ),
        }
    )

    assert result.status is HealthStatus.DEGRADED


def test_system_is_critical_when_component_is_critical() -> None:
    monitor = QuantAIMonitoring()

    result = monitor.evaluate_system(
        {
            "market_data": ComponentHealth(
                "market_data",
                HealthStatus.CRITICAL,
            ),
            "model": ComponentHealth(
                "model",
                HealthStatus.DEGRADED,
            ),
        }
    )

    assert result.status is HealthStatus.CRITICAL


def test_critical_has_priority_over_degraded() -> None:
    monitor = QuantAIMonitoring()

    result = monitor.evaluate_system(
        {
            "a": ComponentHealth(
                "a",
                HealthStatus.DEGRADED,
            ),
            "b": ComponentHealth(
                "b",
                HealthStatus.CRITICAL,
            ),
        }
    )

    assert result.status is HealthStatus.CRITICAL
    assert result.degraded_count == 1
    assert result.critical_count == 1


def test_negative_age_is_invalid() -> None:
    with pytest.raises(ValueError):
        QuantAIMonitoring().evaluate_staleness(
            "data",
            -1,
        )


def test_invalid_age_type_is_rejected() -> None:
    with pytest.raises(TypeError):
        QuantAIMonitoring().evaluate_staleness(
            "data",
            "10",
        )


def test_invalid_threshold_is_rejected() -> None:
    with pytest.raises(ValueError):
        QuantAIMonitoring(
            stale_threshold_seconds=0,
        )


def test_invalid_component_name_is_rejected() -> None:
    with pytest.raises(ValueError):
        ComponentHealth(
            "",
            HealthStatus.HEALTHY,
        )


def test_invalid_component_status_is_rejected() -> None:
    with pytest.raises(TypeError):
        ComponentHealth(
            "model",
            "HEALTHY",
        )


def test_invalid_component_mapping_is_rejected() -> None:
    with pytest.raises(TypeError):
        QuantAIMonitoring().evaluate_system(
            {
                "model": "invalid",
            }
        )


def test_mismatched_component_key_is_rejected() -> None:
    with pytest.raises(ValueError):
        QuantAIMonitoring().evaluate_system(
            {
                "model": ComponentHealth(
                    "other",
                    HealthStatus.HEALTHY,
                )
            }
        )


def test_non_mapping_is_rejected() -> None:
    with pytest.raises(TypeError):
        QuantAIMonitoring().evaluate_system([])


def test_non_boolean_health_is_rejected() -> None:
    with pytest.raises(TypeError):
        QuantAIMonitoring().evaluate_component(
            "model",
            healthy=1,
        )