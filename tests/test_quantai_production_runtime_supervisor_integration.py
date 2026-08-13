from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from src.quantai_production_runtime_lifecycle import (
    QuantAIProductionRuntimeLifecycle,
    RuntimeLifecycleState,
)

from src.quantai_production_runtime_supervisor import (
    QuantAIProductionRuntimeSupervisor,
)

from src.quantai_production_runtime_supervisor_integration import (
    ProductionRuntimeSupervisorIntegrationResult,
    QuantAIProductionRuntimeSupervisorIntegration,
    SupervisorIntegrationCheck,
    supervise_production_runtime,
)


@dataclass
class MockHealth:
    healthy: bool = True
    errors: list[str] = field(
        default_factory=list
    )
    warnings: list[str] = field(
        default_factory=list
    )


def make_running_integration() -> (
    QuantAIProductionRuntimeSupervisorIntegration
):
    lifecycle = (
        QuantAIProductionRuntimeLifecycle()
    )

    lifecycle.start(
        lambda: {
            "status": "running"
        }
    )

    supervisor = (
        QuantAIProductionRuntimeSupervisor(
            lifecycle=lifecycle,
            max_recovery_attempts=1,
        )
    )

    return (
        QuantAIProductionRuntimeSupervisorIntegration(
            supervisor=supervisor
        )
    )


def test_valid_running_preflight_passes():
    integration = make_running_integration()

    result = integration.preflight()

    assert isinstance(
        result,
        ProductionRuntimeSupervisorIntegrationResult,
    )

    assert result.ready is True

    assert result.action == "preflight"

    assert result.total_checks == 2

    assert result.checks_passed == 2

    assert result.checks_failed == 0

    assert result.errors == []


def test_preflight_detects_stopped_runtime():
    integration = (
        QuantAIProductionRuntimeSupervisorIntegration()
    )

    result = integration.preflight(
        expected_running=True
    )

    assert result.ready is False

    assert result.checks_failed == 1

    assert any(
        check.name == "runtime_state"
        and not check.passed
        for check in result.checks
    )


def test_preflight_accepts_expected_stopped_state():
    integration = (
        QuantAIProductionRuntimeSupervisorIntegration()
    )

    result = integration.preflight(
        expected_running=False
    )

    assert result.ready is True

    assert result.checks_passed == 2


def test_supervision_passes_for_healthy_runtime():
    integration = make_running_integration()

    result = integration.supervise(
        health_checker=lambda: MockHealth(
            healthy=True
        )
    )

    assert result.ready is True

    assert result.action == "supervise"

    assert (
        result.state
        == RuntimeLifecycleState.RUNNING
    )

    assert (
        result.supervisor_result
        is not None
    )

    assert result.errors == []


def test_supervision_propagates_health_failure():
    integration = make_running_integration()

    result = integration.supervise(
        health_checker=lambda: MockHealth(
            healthy=False
        )
    )

    assert result.ready is False

    assert result.checks_failed == 1

    assert any(
        "health" in error.lower()
        for error in result.errors
    )


def test_supervision_with_recovery_succeeds():
    integration = (
        QuantAIProductionRuntimeSupervisorIntegration()
    )

    result = integration.recover(
        lambda: {
            "status": "recovered"
        }
    )

    assert result.ready is True

    assert result.action == "recover"

    assert (
        result.state
        == RuntimeLifecycleState.RUNNING
    )

    assert integration.is_running is True


def test_recovery_failure_is_propagated():
    integration = (
        QuantAIProductionRuntimeSupervisorIntegration()
    )

    def failing_runner():
        raise RuntimeError(
            "recovery execution failure"
        )

    result = integration.recover(
        failing_runner
    )

    assert result.ready is False

    assert any(
        "recovery execution failure"
        in error
        for error in result.errors
    )


def test_supervise_does_not_run_when_preflight_fails():
    integration = (
        QuantAIProductionRuntimeSupervisorIntegration()
    )

    called = False

    def health_checker():
        nonlocal called
        called = True
        return MockHealth(
            healthy=True
        )

    result = integration.supervise(
        health_checker=health_checker
    )

    assert result.ready is False

    assert called is False

    assert result.action == "preflight"


def test_invalid_health_checker_is_rejected():
    integration = make_running_integration()

    with pytest.raises(TypeError):
        integration.supervise(
            health_checker="invalid"
        )


def test_invalid_recovery_runner_is_rejected():
    integration = make_running_integration()

    with pytest.raises(TypeError):
        integration.supervise(
            health_checker=lambda: MockHealth(
                healthy=True
            ),
            recovery_runner="invalid",
        )


def test_invalid_recover_runner_is_rejected():
    integration = make_running_integration()

    with pytest.raises(TypeError):
        integration.recover(
            "invalid"
        )


def test_invalid_supervisor_is_rejected():
    with pytest.raises(TypeError):
        QuantAIProductionRuntimeSupervisorIntegration(
            supervisor="invalid"
        )


def test_warnings_are_propagated():
    integration = make_running_integration()

    result = integration.supervise(
        health_checker=lambda: MockHealth(
            healthy=True,
            warnings=[
                "runtime warning"
            ],
        )
    )

    assert result.ready is True

    assert (
        "runtime warning"
        in result.warnings
    )


def test_source_errors_are_propagated():
    integration = make_running_integration()

    result = integration.supervise(
        health_checker=lambda: MockHealth(
            healthy=True,
            errors=[
                "runtime source error"
            ],
        )
    )

    assert result.ready is False

    assert (
        "runtime source error"
        in result.errors
    )


def test_execute_delegates_to_supervise():
    integration = make_running_integration()

    result = integration.execute(
        health_checker=lambda: MockHealth(
            healthy=True
        )
    )

    assert result.ready is True

    assert result.action == "supervise"


def test_convenience_function_returns_result():
    integration = make_running_integration()

    result = supervise_production_runtime(
        health_checker=lambda: MockHealth(
            healthy=True
        ),
        supervisor=integration.supervisor,
    )

    assert isinstance(
        result,
        ProductionRuntimeSupervisorIntegrationResult,
    )

    assert result.ready is True


def test_recovery_counter_can_be_reset():
    integration = (
        QuantAIProductionRuntimeSupervisorIntegration()
    )

    integration.recover(
        lambda: (
            (_ for _ in ()).throw(
                RuntimeError(
                    "failure"
                )
            )
        )
    )

    assert (
        integration.recovery_attempts
        == 1
    )

    integration.reset_recovery_counter()

    assert (
        integration.recovery_attempts
        == 0
    )


def test_result_counters_are_consistent():
    integration = make_running_integration()

    result = integration.preflight()

    assert (
        result.checks_passed
        + result.checks_failed
        == result.total_checks
    )


def test_supervisor_integration_check_is_frozen():
    check = SupervisorIntegrationCheck(
        name="test",
        passed=True,
        message="ok",
    )

    with pytest.raises(Exception):
        check.passed = False


def test_default_supervisor_is_created():
    integration = (
        QuantAIProductionRuntimeSupervisorIntegration()
    )

    assert isinstance(
        integration.supervisor,
        QuantAIProductionRuntimeSupervisor,
    )

    assert integration.is_running is False
