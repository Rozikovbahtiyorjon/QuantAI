from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.quantai_production_model_runtime_incident_management import (
    ModelRuntimeIncidentState,
    QuantAIProductionModelRuntimeIncidentManager,
)
from src.quantai_production_model_runtime_recovery import (
    RecoveryState,
    QuantAIProductionModelRuntimeRecoveryOrchestrator,
)
from src.quantai_production_model_runtime_recovery_integration import (
    ProductionModelRuntimeRecoveryIntegrationResult,
    QuantAIProductionModelRuntimeRecoveryIntegration,
    RecoveryIntegrationCheck,
    coordinate_model_runtime_recovery,
)
from src.quantai_production_runtime_lifecycle import (
    QuantAIProductionRuntimeLifecycle,
    RuntimeLifecycleState,
)
from src.quantai_production_runtime_supervisor import (
    QuantAIProductionRuntimeSupervisor,
)


@dataclass
class Result:
    healthy: bool = True
    errors: list[str] | None = None
    warnings: list[str] | None = None


def integration() -> QuantAIProductionModelRuntimeRecoveryIntegration:
    return QuantAIProductionModelRuntimeRecoveryIntegration()


def running_integration() -> QuantAIProductionModelRuntimeRecoveryIntegration:
    lifecycle = QuantAIProductionRuntimeLifecycle()
    lifecycle.start(lambda: "runtime")

    return QuantAIProductionModelRuntimeRecoveryIntegration(
        lifecycle=lifecycle
    )


def test_default_components_are_wired():
    value = integration()

    assert isinstance(
        value.incident_manager,
        QuantAIProductionModelRuntimeIncidentManager,
    )
    assert isinstance(
        value.recovery_orchestrator,
        QuantAIProductionModelRuntimeRecoveryOrchestrator,
    )
    assert isinstance(
        value.supervisor,
        QuantAIProductionRuntimeSupervisor,
    )
    assert value.supervisor.lifecycle is value.lifecycle
    assert value.lifecycle.state == RuntimeLifecycleState.STOPPED


def test_custom_components_are_preserved():
    lifecycle = QuantAIProductionRuntimeLifecycle()
    incident = QuantAIProductionModelRuntimeIncidentManager()
    recovery = QuantAIProductionModelRuntimeRecoveryOrchestrator()
    supervisor = QuantAIProductionRuntimeSupervisor(
        lifecycle=lifecycle
    )

    value = QuantAIProductionModelRuntimeRecoveryIntegration(
        incident_manager=incident,
        recovery_orchestrator=recovery,
        supervisor=supervisor,
        lifecycle=lifecycle,
    )

    assert value.incident_manager is incident
    assert value.recovery_orchestrator is recovery
    assert value.supervisor is supervisor
    assert value.lifecycle is lifecycle


def test_mismatched_supervisor_and_lifecycle_are_rejected():
    lifecycle_one = QuantAIProductionRuntimeLifecycle()
    lifecycle_two = QuantAIProductionRuntimeLifecycle()

    supervisor = QuantAIProductionRuntimeSupervisor(
        lifecycle=lifecycle_one
    )

    with pytest.raises(ValueError):
        QuantAIProductionModelRuntimeRecoveryIntegration(
            supervisor=supervisor,
            lifecycle=lifecycle_two,
        )


def test_invalid_incident_manager_is_rejected():
    with pytest.raises(TypeError):
        QuantAIProductionModelRuntimeRecoveryIntegration(
            incident_manager=object()
        )


def test_invalid_recovery_orchestrator_is_rejected():
    with pytest.raises(TypeError):
        QuantAIProductionModelRuntimeRecoveryIntegration(
            recovery_orchestrator=object()
        )


def test_invalid_supervisor_is_rejected():
    with pytest.raises(TypeError):
        QuantAIProductionModelRuntimeRecoveryIntegration(
            supervisor=object()
        )


def test_invalid_lifecycle_is_rejected():
    with pytest.raises(TypeError):
        QuantAIProductionModelRuntimeRecoveryIntegration(
            lifecycle=object()
        )


def test_healthy_incident_requires_no_recovery():
    value = integration()

    result = value.coordinate(
        monitoring_result=Result(healthy=True),
        inference_result=Result(healthy=True),
    )

    assert isinstance(
        result,
        ProductionModelRuntimeRecoveryIntegrationResult,
    )
    assert result.success is True
    assert result.action == "no_action"
    assert result.incident_state == ModelRuntimeIncidentState.NORMAL
    assert result.lifecycle_state == RuntimeLifecycleState.STOPPED
    assert result.recovery_result is None
    assert result.supervisor_result is None


def test_healthy_incident_can_run_supervisor_health_check():
    value = running_integration()

    result = value.coordinate(
        monitoring_result=Result(healthy=True),
        inference_result=Result(healthy=True),
        supervisor_health_checker=lambda: Result(healthy=True),
    )

    assert result.success is True
    assert result.action == "supervise"
    assert result.supervisor_result is not None
    assert result.supervisor_result.healthy is True
    assert result.lifecycle_state == RuntimeLifecycleState.RUNNING


def test_supervisor_failure_is_propagated():
    value = running_integration()

    result = value.coordinate(
        monitoring_result=Result(healthy=True),
        inference_result=Result(healthy=True),
        supervisor_health_checker=lambda: Result(
            healthy=False,
            errors=["runtime unhealthy"],
        ),
    )

    assert result.success is False
    assert result.action == "supervise"
    assert "runtime unhealthy" in result.errors


def test_halted_incident_blocks_recovery():
    value = integration()

    result = value.coordinate(
        monitoring_result=Result(
            healthy=False,
            errors=["critical runtime incident"],
        ),
        inference_result=Result(healthy=False),
        recovery_runner=lambda: "unsafe",
        failover_runner=lambda: "unsafe",
        target_model_version="v2",
    )

    assert result.success is False
    assert result.action == "halt"
    assert result.incident_state == ModelRuntimeIncidentState.HALTED
    assert result.recovery_result is None
    assert result.supervisor_result is None

    assert any(
        check.name == "incident_safety_gate"
        and not check.passed
        for check in result.checks
    )


def test_recovery_path_executes_recovery_runner_when_fallback_disabled():
    incident = QuantAIProductionModelRuntimeIncidentManager(
        allow_fallback=False
    )

    value = QuantAIProductionModelRuntimeRecoveryIntegration(
        incident_manager=incident
    )

    calls: list[str] = []

    result = value.coordinate(
        monitoring_result=Result(healthy=False),
        inference_result=Result(healthy=True),
        recovery_runner=lambda: calls.append("recover") or "runtime",
    )

    assert result.success is True
    assert result.action == "recovered"
    assert calls == ["recover"]

    assert result.recovery_result is not None
    assert result.recovery_result.success is True
    assert result.recovery_result.state == RecoveryState.RECOVERING


def test_recovery_failure_is_propagated():
    incident = QuantAIProductionModelRuntimeIncidentManager(
        allow_fallback=False
    )

    value = QuantAIProductionModelRuntimeRecoveryIntegration(
        incident_manager=incident
    )

    def fail():
        raise RuntimeError("recovery failed")

    result = value.coordinate(
        monitoring_result=Result(healthy=False),
        inference_result=Result(healthy=True),
        recovery_runner=fail,
    )

    assert result.success is False
    assert result.action == "recovery_failed"
    assert result.recovery_result is not None
    assert result.recovery_result.halted is True

    assert any(
        "recovery failed" in message
        for message in result.errors
    )


def test_failover_path_updates_active_model_version():
    value = integration()
    calls: list[str] = []

    result = value.coordinate(
        monitoring_result=Result(healthy=False),
        inference_result=Result(healthy=True),
        failover_runner=lambda: calls.append("failover") or "runtime-v2",
        target_model_version="v2",
    )

    assert result.success is True
    assert result.action == "recovered"
    assert calls == ["failover"]
    assert result.model_version == "v2"

    assert result.recovery_result is not None
    assert result.recovery_result.state == RecoveryState.FAILED_OVER

    assert (
        value.recovery_orchestrator.active_model_version
        == "v2"
    )


def test_failover_requires_target_model_version():
    value = integration()

    result = value.coordinate(
        monitoring_result=Result(healthy=False),
        inference_result=Result(healthy=True),
        failover_runner=lambda: "runtime",
    )

    assert result.success is False
    assert result.action == "recovery_failed"
    assert result.recovery_result is not None
    assert result.recovery_result.halted is True

    assert any(
        "target model version" in message
        for message in result.errors
    )


def test_failover_runner_failure_halts_runtime_recovery():
    value = integration()

    def failover():
        raise RuntimeError("failover failed")

    result = value.coordinate(
        monitoring_result=Result(healthy=False),
        inference_result=Result(healthy=True),
        failover_runner=failover,
        target_model_version="v2",
    )

    assert result.success is False
    assert result.recovery_result is not None
    assert result.recovery_result.halted is True

    assert any(
        "failover failed" in message
        for message in result.errors
    )


def test_recovered_runtime_can_be_checked_by_supervisor():
    value = running_integration()

    result = value.coordinate(
        monitoring_result=Result(healthy=False),
        inference_result=Result(healthy=True),
        failover_runner=lambda: "runtime-v2",
        target_model_version="v2",
        supervisor_health_checker=lambda: Result(healthy=True),
    )

    assert result.success is True
    assert result.action == "recovered_and_supervised"
    assert result.supervisor_result is not None
    assert result.supervisor_result.healthy is True
    assert result.lifecycle_state == RuntimeLifecycleState.RUNNING


def test_recovered_runtime_with_bad_supervisor_is_not_healthy():
    value = running_integration()

    result = value.coordinate(
        monitoring_result=Result(healthy=False),
        inference_result=Result(healthy=True),
        failover_runner=lambda: "runtime-v2",
        target_model_version="v2",
        supervisor_health_checker=lambda: Result(
            healthy=False,
            errors=["post-recovery health failure"],
        ),
    )

    assert result.success is False
    assert result.action == "recovery_supervisor_failed"
    assert "post-recovery health failure" in result.errors


def test_recovery_without_supervisor_check_is_allowed():
    value = integration()

    result = value.coordinate(
        monitoring_result=Result(healthy=False),
        inference_result=Result(healthy=True),
        failover_runner=lambda: "runtime-v2",
        target_model_version="v2",
    )

    assert result.success is True
    assert result.action == "recovered"
    assert result.supervisor_result is None

    assert any(
        check.name == "supervisor_health"
        and check.passed
        for check in result.checks
    )


def test_source_warnings_are_preserved():
    value = integration()

    result = value.coordinate(
        monitoring_result=Result(
            healthy=True,
            warnings=["monitoring warning"],
        ),
        inference_result=Result(
            healthy=True,
            warnings=["inference warning"],
        ),
    )

    assert result.success is True
    assert "monitoring warning" in result.warnings
    assert "inference warning" in result.warnings


def test_source_errors_are_preserved():
    value = integration()

    result = value.coordinate(
        monitoring_result=Result(
            healthy=False,
            errors=["critical source error"],
        ),
        inference_result=Result(
            healthy=True,
            errors=["inference source error"],
        ),
    )

    assert result.success is False
    assert result.action == "halt"
    assert "critical source error" in result.errors
    assert "inference source error" in result.errors


def test_reset_restores_all_coordinated_state():
    value = running_integration()

    result = value.coordinate(
        monitoring_result=Result(healthy=False),
        inference_result=Result(healthy=True),
        failover_runner=lambda: "runtime-v2",
        target_model_version="v2",
    )

    assert result.success is True
    assert (
        value.recovery_orchestrator.active_model_version
        == "v2"
    )
    assert value.lifecycle.state == RuntimeLifecycleState.RUNNING

    value.reset()

    assert (
        value.recovery_orchestrator.active_model_version
        is None
    )
    assert value.supervisor.recovery_attempts == 0
    assert value.lifecycle.state == RuntimeLifecycleState.STOPPED


def test_result_properties_and_check_counts():
    result = ProductionModelRuntimeRecoveryIntegrationResult(
        success=False,
        action="test",
        incident_state=None,
        lifecycle_state=RuntimeLifecycleState.STOPPED,
        checks=[
            RecoveryIntegrationCheck("a", True, "ok"),
            RecoveryIntegrationCheck("b", False, "bad"),
        ],
        errors=["error"],
    )

    assert result.healthy is False
    assert result.blocked is True
    assert result.total_checks == 2
    assert result.checks_passed == 1
    assert result.checks_failed == 1


def test_convenience_function_uses_same_policy():
    result = coordinate_model_runtime_recovery(
        monitoring_result=Result(healthy=True),
        inference_result=Result(healthy=True),
    )

    assert result.success is True
    assert result.action == "no_action"
    assert result.incident_state == ModelRuntimeIncidentState.NORMAL


def test_inference_failure_without_recovery_permission_is_blocked():
    incident = QuantAIProductionModelRuntimeIncidentManager(
        allow_degraded_inference=False,
        allow_fallback=False,
        require_recovery_on_health_failure=False,
    )

    value = QuantAIProductionModelRuntimeRecoveryIntegration(
        incident_manager=incident
    )

    result = value.coordinate(
        monitoring_result=Result(healthy=True),
        inference_result=Result(healthy=False),
    )

    assert result.success is False
    assert result.action in {
        "recovery_failed",
        "halt",
    }
    assert (
        result.recovery_result is not None
        or result.incident_state is not None
    )