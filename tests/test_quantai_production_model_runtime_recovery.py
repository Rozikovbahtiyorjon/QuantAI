from dataclasses import dataclass, field

import pytest

from src.quantai_production_model_runtime_recovery import (
    ModelRuntimeRecoveryResult,
    QuantAIProductionModelRuntimeRecoveryOrchestrator,
    RecoveryAction,
    RecoveryState,
)


@dataclass
class MockIncident:
    state: str = "NORMAL"
    allow_inference: bool = True
    fallback_allowed: bool = False
    recovery_required: bool = False
    errors: list[str] = field(
        default_factory=list
    )
    warnings: list[str] = field(
        default_factory=list
    )


def test_normal_incident_requires_no_action():
    orchestrator = (
        QuantAIProductionModelRuntimeRecoveryOrchestrator()
    )

    result = orchestrator.recover(
        MockIncident()
    )

    assert isinstance(
        result,
        ModelRuntimeRecoveryResult,
    )

    assert result.success is True
    assert result.action == RecoveryAction.NO_ACTION
    assert result.state == RecoveryState.IDLE
    assert result.recovered is False
    assert result.halted is False


def test_recovery_required_selects_recovery():
    orchestrator = (
        QuantAIProductionModelRuntimeRecoveryOrchestrator(
            allow_failover=False
        )
    )

    incident = MockIncident(
        state="RECOVERY_REQUIRED",
        allow_inference=False,
        recovery_required=True,
        fallback_allowed=False,
    )

    assert (
        orchestrator.evaluate_action(incident)
        == RecoveryAction.RECOVER
    )


def test_recovery_runner_success():
    orchestrator = (
        QuantAIProductionModelRuntimeRecoveryOrchestrator(
            allow_failover=False
        )
    )

    incident = MockIncident(
        state="RECOVERY_REQUIRED",
        allow_inference=False,
        recovery_required=True,
        fallback_allowed=False,
    )

    result = orchestrator.recover(
        incident,
        recovery_runner=lambda: {
            "status": "recovered"
        },
    )

    assert result.success is True
    assert result.action == RecoveryAction.RECOVER
    assert result.state == RecoveryState.RECOVERING
    assert result.runtime_result == {
        "status": "recovered"
    }
    assert result.checks_failed == 0


def test_recovery_runner_failure_halts_runtime():
    orchestrator = (
        QuantAIProductionModelRuntimeRecoveryOrchestrator(
            allow_failover=False
        )
    )

    incident = MockIncident(
        state="RECOVERY_REQUIRED",
        allow_inference=False,
        recovery_required=True,
        fallback_allowed=False,
    )

    def failing_runner():
        raise RuntimeError(
            "recovery failure"
        )

    result = orchestrator.recover(
        incident,
        recovery_runner=failing_runner,
    )

    assert result.success is False
    assert result.state == RecoveryState.HALTED
    assert result.halted is True
    assert any(
        "recovery failure" in error
        for error in result.errors
    )


def test_recovery_requires_callable_runner():
    orchestrator = (
        QuantAIProductionModelRuntimeRecoveryOrchestrator(
            allow_failover=False
        )
    )

    incident = MockIncident(
        state="RECOVERY_REQUIRED",
        allow_inference=False,
        recovery_required=True,
    )

    result = orchestrator.recover(
        incident,
        recovery_runner=None,
    )

    assert result.success is False
    assert result.state == RecoveryState.HALTED
    assert result.checks_failed == 1


def test_failover_is_selected_when_allowed():
    orchestrator = (
        QuantAIProductionModelRuntimeRecoveryOrchestrator()
    )

    incident = MockIncident(
        state="RECOVERY_REQUIRED",
        allow_inference=False,
        recovery_required=True,
        fallback_allowed=True,
    )

    assert (
        orchestrator.evaluate_action(incident)
        == RecoveryAction.FAILOVER
    )


def test_failover_success_updates_active_model():
    orchestrator = (
        QuantAIProductionModelRuntimeRecoveryOrchestrator()
    )

    incident = MockIncident(
        state="RECOVERY_REQUIRED",
        allow_inference=False,
        recovery_required=True,
        fallback_allowed=True,
    )

    result = orchestrator.recover(
        incident,
        failover_runner=lambda: "challenger-running",
        target_model_version="model-v2",
    )

    assert result.success is True
    assert result.action == RecoveryAction.FAILOVER
    assert result.state == RecoveryState.FAILED_OVER
    assert result.model_version == "model-v2"
    assert (
        orchestrator.active_model_version
        == "model-v2"
    )
    assert result.runtime_result == (
        "challenger-running"
    )


def test_failover_requires_target_model():
    orchestrator = (
        QuantAIProductionModelRuntimeRecoveryOrchestrator()
    )

    incident = MockIncident(
        state="RECOVERY_REQUIRED",
        allow_inference=False,
        recovery_required=True,
        fallback_allowed=True,
    )

    result = orchestrator.recover(
        incident,
        failover_runner=lambda: "running",
    )

    assert result.success is False
    assert result.state == RecoveryState.HALTED
    assert any(
        "target model version" in error
        for error in result.errors
    )


def test_failover_requires_callable_runner():
    orchestrator = (
        QuantAIProductionModelRuntimeRecoveryOrchestrator()
    )

    incident = MockIncident(
        state="RECOVERY_REQUIRED",
        allow_inference=False,
        recovery_required=True,
        fallback_allowed=True,
    )

    result = orchestrator.recover(
        incident,
        failover_runner=None,
        target_model_version="model-v2",
    )

    assert result.success is False
    assert result.state == RecoveryState.HALTED


def test_failover_failure_halts_runtime():
    orchestrator = (
        QuantAIProductionModelRuntimeRecoveryOrchestrator()
    )

    incident = MockIncident(
        state="RECOVERY_REQUIRED",
        allow_inference=False,
        recovery_required=True,
        fallback_allowed=True,
    )

    def failing_runner():
        raise RuntimeError(
            "failover failure"
        )

    result = orchestrator.recover(
        incident,
        failover_runner=failing_runner,
        target_model_version="model-v2",
    )

    assert result.success is False
    assert result.state == RecoveryState.HALTED
    assert orchestrator.active_model_version is None
    assert any(
        "failover failure" in error
        for error in result.errors
    )


def test_halted_incident_blocks_everything():
    orchestrator = (
        QuantAIProductionModelRuntimeRecoveryOrchestrator()
    )

    incident = MockIncident(
        state="HALTED",
        allow_inference=False,
        fallback_allowed=False,
        recovery_required=True,
        errors=[
            "critical runtime incident"
        ],
    )

    result = orchestrator.recover(
        incident,
        recovery_runner=lambda: "unsafe",
        failover_runner=lambda: "unsafe",
        target_model_version="model-v2",
    )

    assert result.success is False
    assert result.action == RecoveryAction.HALT
    assert result.state == RecoveryState.HALTED
    assert result.halted is True
    assert result.runtime_result is None
    assert any(
        "blocked" in error.lower()
        for error in result.errors
    )


def test_halted_incident_cannot_trigger_failover():
    orchestrator = (
        QuantAIProductionModelRuntimeRecoveryOrchestrator()
    )

    incident = MockIncident(
        state="HALTED",
        allow_inference=False,
        fallback_allowed=True,
        recovery_required=True,
    )

    assert (
        orchestrator.evaluate_action(incident)
        == RecoveryAction.HALT
    )


def test_failover_can_be_disabled():
    orchestrator = (
        QuantAIProductionModelRuntimeRecoveryOrchestrator(
            allow_failover=False
        )
    )

    incident = MockIncident(
        state="RECOVERY_REQUIRED",
        allow_inference=False,
        fallback_allowed=True,
        recovery_required=True,
    )

    assert (
        orchestrator.evaluate_action(incident)
        == RecoveryAction.RECOVER
    )


def test_inference_block_without_recovery_halts():
    orchestrator = (
        QuantAIProductionModelRuntimeRecoveryOrchestrator()
    )

    incident = MockIncident(
        state="DEGRADED",
        allow_inference=False,
        fallback_allowed=False,
        recovery_required=False,
    )

    assert (
        orchestrator.evaluate_action(incident)
        == RecoveryAction.HALT
    )


def test_incident_errors_are_propagated():
    orchestrator = (
        QuantAIProductionModelRuntimeRecoveryOrchestrator(
            allow_failover=False
        )
    )

    incident = MockIncident(
        state="RECOVERY_REQUIRED",
        allow_inference=False,
        recovery_required=True,
        errors=["source incident"],
    )

    result = orchestrator.recover(
        incident,
        recovery_runner=lambda: "ok",
    )

    assert "source incident" in result.errors


def test_incident_warnings_are_propagated():
    orchestrator = (
        QuantAIProductionModelRuntimeRecoveryOrchestrator(
            allow_failover=False
        )
    )

    incident = MockIncident(
        state="RECOVERY_REQUIRED",
        allow_inference=False,
        recovery_required=True,
        warnings=["source warning"],
    )

    result = orchestrator.recover(
        incident,
        recovery_runner=lambda: "ok",
    )

    assert "source warning" in result.warnings


def test_recovery_result_counters_are_consistent():
    orchestrator = (
        QuantAIProductionModelRuntimeRecoveryOrchestrator(
            allow_failover=False
        )
    )

    incident = MockIncident(
        state="RECOVERY_REQUIRED",
        allow_inference=False,
        recovery_required=True,
    )

    result = orchestrator.recover(
        incident,
        recovery_runner=lambda: "ok",
    )

    assert (
        result.checks_passed
        + result.checks_failed
        == result.total_checks
    )


def test_invalid_allow_failover_type_is_rejected():
    with pytest.raises(TypeError):
        QuantAIProductionModelRuntimeRecoveryOrchestrator(
            allow_failover="yes"
        )


def test_reset_clears_recovery_state():
    orchestrator = (
        QuantAIProductionModelRuntimeRecoveryOrchestrator()
    )

    incident = MockIncident(
        state="RECOVERY_REQUIRED",
        allow_inference=False,
        recovery_required=True,
        fallback_allowed=True,
    )

    orchestrator.recover(
        incident,
        failover_runner=lambda: "ok",
        target_model_version="model-v2",
    )

    assert (
        orchestrator.state
        == RecoveryState.FAILED_OVER
    )

    orchestrator.reset()

    assert orchestrator.state == RecoveryState.IDLE
    assert orchestrator.active_model_version is None