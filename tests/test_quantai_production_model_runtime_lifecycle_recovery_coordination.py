from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.quantai_production_model_runtime_lifecycle_recovery_coordination import (
    LifecycleRecoveryCoordinationCheck,
    ProductionModelRuntimeLifecycleRecoveryCoordinationResult,
    QuantAIProductionModelRuntimeLifecycleRecoveryCoordination,
    coordinate_model_runtime_lifecycle_recovery,
)
from src.quantai_production_model_runtime_recovery_integration import (
    QuantAIProductionModelRuntimeRecoveryIntegration,
)
from src.quantai_production_runtime_lifecycle import (
    QuantAIProductionRuntimeLifecycle,
    RuntimeLifecycleState,
)


@dataclass
class Result:
    healthy: bool = True
    errors: list[str] | None = None
    warnings: list[str] | None = None


def coordinator() -> (
    QuantAIProductionModelRuntimeLifecycleRecoveryCoordination
):
    return (
        QuantAIProductionModelRuntimeLifecycleRecoveryCoordination()
    )


def running_coordinator() -> (
    QuantAIProductionModelRuntimeLifecycleRecoveryCoordination
):
    lifecycle = QuantAIProductionRuntimeLifecycle()
    lifecycle.start(lambda: "runtime")

    return (
        QuantAIProductionModelRuntimeLifecycleRecoveryCoordination(
            lifecycle=lifecycle
        )
    )


def test_default_components_are_wired():
    value = coordinator()

    assert isinstance(
        value.recovery_integration,
        QuantAIProductionModelRuntimeRecoveryIntegration,
    )
    assert value.recovery_integration.lifecycle is value.lifecycle
    assert value.lifecycle.state == RuntimeLifecycleState.STOPPED


def test_custom_lifecycle_is_preserved():
    lifecycle = QuantAIProductionRuntimeLifecycle()

    value = (
        QuantAIProductionModelRuntimeLifecycleRecoveryCoordination(
            lifecycle=lifecycle
        )
    )

    assert value.lifecycle is lifecycle
    assert value.recovery_integration.lifecycle is lifecycle


def test_custom_integration_is_preserved():
    lifecycle = QuantAIProductionRuntimeLifecycle()

    integration = (
        QuantAIProductionModelRuntimeRecoveryIntegration(
            lifecycle=lifecycle
        )
    )

    value = (
        QuantAIProductionModelRuntimeLifecycleRecoveryCoordination(
            recovery_integration=integration
        )
    )

    assert value.recovery_integration is integration
    assert value.lifecycle is lifecycle


def test_mismatched_integration_and_lifecycle_are_rejected():
    lifecycle_one = QuantAIProductionRuntimeLifecycle()
    lifecycle_two = QuantAIProductionRuntimeLifecycle()

    integration = (
        QuantAIProductionModelRuntimeRecoveryIntegration(
            lifecycle=lifecycle_one
        )
    )

    with pytest.raises(ValueError):
        QuantAIProductionModelRuntimeLifecycleRecoveryCoordination(
            recovery_integration=integration,
            lifecycle=lifecycle_two,
        )


def test_invalid_integration_is_rejected():
    with pytest.raises(TypeError):
        QuantAIProductionModelRuntimeLifecycleRecoveryCoordination(
            recovery_integration=object()
        )


def test_invalid_lifecycle_is_rejected():
    with pytest.raises(TypeError):
        QuantAIProductionModelRuntimeLifecycleRecoveryCoordination(
            lifecycle=object()
        )


def test_healthy_runtime_requires_no_lifecycle_transition():
    value = coordinator()

    result = value.coordinate(
        monitoring_result=Result(healthy=True),
        inference_result=Result(healthy=True),
    )

    assert isinstance(
        result,
        ProductionModelRuntimeLifecycleRecoveryCoordinationResult,
    )
    assert result.success is True
    assert result.action == "no_action"
    assert result.lifecycle_state == RuntimeLifecycleState.STOPPED
    assert result.integration_result is not None
    assert result.integration_result.success is True
    assert result.total_checks == 2
    assert result.checks_failed == 0


def test_recovery_starts_stopped_lifecycle():
    value = coordinator()
    calls: list[str] = []

    result = value.coordinate(
        monitoring_result=Result(healthy=False),
        inference_result=Result(healthy=True),
        recovery_runner=(
            lambda: calls.append("recover") or "runtime-v2"
        ),
    )

    assert result.success is True
    assert result.action == "recovered_and_running"
    assert calls == ["recover"]
    assert result.runtime_result == "runtime-v2"
    assert result.lifecycle_state == RuntimeLifecycleState.RUNNING
    assert value.lifecycle.state == RuntimeLifecycleState.RUNNING
    assert result.integration_result is not None
    assert result.integration_result.success is True


def test_failover_starts_stopped_lifecycle():
    value = coordinator()
    calls: list[str] = []

    result = value.coordinate(
        monitoring_result=Result(healthy=False),
        inference_result=Result(healthy=True),
        failover_runner=(
            lambda: calls.append("failover") or "runtime-v2"
        ),
        target_model_version="v2",
    )

    assert result.success is True
    assert result.action == "recovered_and_running"
    assert calls == ["failover"]
    assert result.runtime_result == "runtime-v2"
    assert result.lifecycle_state == RuntimeLifecycleState.RUNNING

    assert result.integration_result is not None
    assert result.integration_result.model_version == "v2"


def test_running_lifecycle_is_not_restarted_after_recovery():
    value = running_coordinator()

    result = value.coordinate(
        monitoring_result=Result(healthy=False),
        inference_result=Result(healthy=True),
        failover_runner=lambda: "runtime-v2",
        target_model_version="v2",
    )

    assert result.success is True
    assert result.action == "recovered_and_running"
    assert result.lifecycle_state == RuntimeLifecycleState.RUNNING
    assert value.lifecycle.state == RuntimeLifecycleState.RUNNING


def test_supervised_recovery_keeps_lifecycle_running():
    value = running_coordinator()

    result = value.coordinate(
        monitoring_result=Result(healthy=False),
        inference_result=Result(healthy=True),
        failover_runner=lambda: "runtime-v2",
        target_model_version="v2",
        supervisor_health_checker=lambda: Result(
            healthy=True
        ),
    )

    assert result.success is True
    assert result.action == "recovered_and_running"
    assert result.lifecycle_state == RuntimeLifecycleState.RUNNING
    assert result.integration_result is not None
    assert (
        result.integration_result.action
        == "recovered_and_supervised"
    )


def test_failed_recovery_does_not_start_lifecycle():
    value = coordinator()

    def fail():
        raise RuntimeError("recovery failed")

    result = value.coordinate(
        monitoring_result=Result(healthy=False),
        inference_result=Result(healthy=True),
        recovery_runner=fail,
    )

    assert result.success is False
    assert result.action == "blocked"
    assert result.lifecycle_state == RuntimeLifecycleState.STOPPED
    assert value.lifecycle.state == RuntimeLifecycleState.STOPPED
    assert any(
        "recovery failed" in message
        for message in result.errors
    )


def test_halted_incident_blocks_lifecycle_transition():
    value = coordinator()

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
    assert result.action == "blocked"
    assert result.lifecycle_state == RuntimeLifecycleState.STOPPED
    assert value.lifecycle.state == RuntimeLifecycleState.STOPPED
    assert result.integration_result is not None
    assert result.integration_result.action == "halt"


def test_recovery_without_runner_does_not_start_lifecycle():
    value = coordinator()

    result = value.coordinate(
        monitoring_result=Result(healthy=False),
        inference_result=Result(healthy=True),
    )

    assert result.success is False
    assert result.action == "blocked"
    assert result.lifecycle_state == RuntimeLifecycleState.STOPPED


def test_supervisor_failure_does_not_start_stopped_lifecycle():
    value = coordinator()

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
    assert result.action == "blocked"
    assert result.lifecycle_state == RuntimeLifecycleState.STOPPED
    assert value.lifecycle.state == RuntimeLifecycleState.STOPPED
    assert "post-recovery health failure" in result.errors


def test_source_warnings_are_preserved():
    value = coordinator()

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


def test_reset_restores_lifecycle_and_recovery_state():
    value = coordinator()

    result = value.coordinate(
        monitoring_result=Result(healthy=False),
        inference_result=Result(healthy=True),
        failover_runner=lambda: "runtime-v2",
        target_model_version="v2",
    )

    assert result.success is True
    assert value.lifecycle.state == RuntimeLifecycleState.RUNNING
    assert (
        value.recovery_integration.recovery_orchestrator.active_model_version
        == "v2"
    )

    value.reset()

    assert value.lifecycle.state == RuntimeLifecycleState.STOPPED
    assert (
        value.recovery_integration.recovery_orchestrator.active_model_version
        is None
    )
    assert (
        value.recovery_integration.supervisor.recovery_attempts
        == 0
    )


def test_result_properties_and_check_counts():
    result = (
        ProductionModelRuntimeLifecycleRecoveryCoordinationResult(
            success=False,
            action="test",
            lifecycle_state=RuntimeLifecycleState.STOPPED,
            checks=[
                LifecycleRecoveryCoordinationCheck(
                    "a",
                    True,
                    "ok",
                ),
                LifecycleRecoveryCoordinationCheck(
                    "b",
                    False,
                    "bad",
                ),
            ],
            errors=["error"],
        )
    )

    assert result.healthy is False
    assert result.blocked is True
    assert result.total_checks == 2
    assert result.checks_passed == 1
    assert result.checks_failed == 1


def test_convenience_function_uses_same_policy():
    result = coordinate_model_runtime_lifecycle_recovery(
        monitoring_result=Result(healthy=True),
        inference_result=Result(healthy=True),
    )

    assert result.success is True
    assert result.action == "no_action"
    assert result.lifecycle_state == RuntimeLifecycleState.STOPPED