from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from src.quantai_production_model_runtime_incident_management import (
    ModelRuntimeIncidentResult,
    ModelRuntimeIncidentState,
    QuantAIProductionModelRuntimeIncidentManager,
    evaluate_model_runtime_incident,
)


@dataclass
class MockResult:
    healthy: bool = False
    passed: bool = False
    valid: bool = False
    ready: bool = False
    success: bool = False

    errors: list[str] = field(
        default_factory=list
    )

    warnings: list[str] = field(
        default_factory=list
    )


def make_all_passed() -> tuple[
    MockResult,
    MockResult,
]:
    return (
        MockResult(
            healthy=True,
            passed=True,
        ),
        MockResult(
            healthy=True,
            success=True,
        ),
    )


def test_all_healthy_returns_normal():
    manager = (
        QuantAIProductionModelRuntimeIncidentManager()
    )

    result = manager.evaluate(
        *make_all_passed()
    )

    assert isinstance(
        result,
        ModelRuntimeIncidentResult,
    )

    assert (
        result.state
        == ModelRuntimeIncidentState.NORMAL
    )

    assert result.allow_inference is True
    assert result.fallback_allowed is False
    assert result.recovery_required is False
    assert result.healthy is True
    assert result.blocked is False
    assert result.checks_passed == 2
    assert result.checks_failed == 0


def test_health_failure_requires_recovery():
    manager = (
        QuantAIProductionModelRuntimeIncidentManager()
    )

    result = manager.evaluate(
        MockResult(
            healthy=False
        ),
        MockResult(
            healthy=True
        ),
    )

    assert (
        result.state
        == ModelRuntimeIncidentState.RECOVERY_REQUIRED
    )

    assert result.allow_inference is False
    assert result.fallback_allowed is True
    assert result.recovery_required is True
    assert result.blocked is True


def test_inference_failure_creates_degraded_state():
    manager = (
        QuantAIProductionModelRuntimeIncidentManager()
    )

    result = manager.evaluate(
        MockResult(
            healthy=True
        ),
        MockResult(
            healthy=False
        ),
    )

    assert (
        result.state
        == ModelRuntimeIncidentState.DEGRADED
    )

    assert result.allow_inference is False
    assert result.fallback_allowed is True
    assert result.recovery_required is False


def test_degraded_inference_can_be_allowed_by_policy():
    manager = (
        QuantAIProductionModelRuntimeIncidentManager(
            allow_degraded_inference=True
        )
    )

    result = manager.evaluate(
        MockResult(
            healthy=True
        ),
        MockResult(
            healthy=False
        ),
    )

    assert (
        result.state
        == ModelRuntimeIncidentState.DEGRADED
    )

    assert result.allow_inference is True
    assert result.fallback_allowed is True


def test_fallback_can_be_disabled():
    manager = (
        QuantAIProductionModelRuntimeIncidentManager(
            allow_fallback=False
        )
    )

    result = manager.evaluate(
        MockResult(
            healthy=False
        ),
        MockResult(
            healthy=True
        ),
    )

    assert result.fallback_allowed is False
    assert result.recovery_required is True


def test_health_failure_recovery_policy_can_be_disabled():
    manager = (
        QuantAIProductionModelRuntimeIncidentManager(
            require_recovery_on_health_failure=False
        )
    )

    result = manager.evaluate(
        MockResult(
            healthy=False
        ),
        MockResult(
            healthy=True
        ),
    )

    assert (
        result.state
        == ModelRuntimeIncidentState.DEGRADED
    )

    assert result.recovery_required is False


def test_missing_monitoring_result_blocks_inference():
    manager = (
        QuantAIProductionModelRuntimeIncidentManager()
    )

    result = manager.evaluate(
        None,
        MockResult(
            healthy=True
        ),
    )

    assert result.allow_inference is False
    assert result.blocked is True
    assert result.checks_failed == 1
    assert result.errors


def test_missing_inference_result_blocks_inference():
    manager = (
        QuantAIProductionModelRuntimeIncidentManager()
    )

    result = manager.evaluate(
        MockResult(
            healthy=True
        ),
        None,
    )

    assert result.allow_inference is False
    assert result.blocked is True
    assert result.checks_failed == 1


def test_source_errors_are_propagated():
    manager = (
        QuantAIProductionModelRuntimeIncidentManager()
    )

    result = manager.evaluate(
        MockResult(
            healthy=True,
            errors=["model health error"],
        ),
        MockResult(
            healthy=True
        ),
    )

    assert result.allow_inference is False
    assert "model health error" in result.errors


def test_source_warnings_are_propagated():
    manager = (
        QuantAIProductionModelRuntimeIncidentManager()
    )

    result = manager.evaluate(
        MockResult(
            healthy=True,
            warnings=["health warning"],
        ),
        MockResult(
            healthy=True
        ),
    )

    assert result.allow_inference is True
    assert "health warning" in result.warnings


def test_is_safe_matches_evaluation():
    manager = (
        QuantAIProductionModelRuntimeIncidentManager()
    )

    args = make_all_passed()

    result = manager.evaluate(*args)

    assert manager.is_safe(*args) is result.allow_inference


def test_convenience_function_returns_result():
    result = evaluate_model_runtime_incident(
        *make_all_passed()
    )

    assert isinstance(
        result,
        ModelRuntimeIncidentResult,
    )

    assert result.state == ModelRuntimeIncidentState.NORMAL


def test_invalid_health_status_is_rejected():
    manager = (
        QuantAIProductionModelRuntimeIncidentManager()
    )

    class InvalidResult:
        healthy = "yes"

    result = manager.evaluate(
        InvalidResult(),
        MockResult(
            healthy=True
        ),
    )

    assert result.allow_inference is False
    assert result.checks_failed == 1


def test_invalid_inference_status_is_rejected():
    manager = (
        QuantAIProductionModelRuntimeIncidentManager()
    )

    class InvalidResult:
        success = "yes"

    result = manager.evaluate(
        MockResult(
            healthy=True
        ),
        InvalidResult(),
    )

    assert result.allow_inference is False
    assert result.checks_failed == 1


def test_halted_state_blocks_everything():
    manager = (
        QuantAIProductionModelRuntimeIncidentManager()
    )

    result = manager.evaluate(
        MockResult(
            healthy=False,
            errors=[
                "critical runtime incident"
            ],
        ),
        MockResult(
            healthy=False
        ),
    )

    assert result.allow_inference is False
    assert result.fallback_allowed is False
    assert result.recovery_required is True


def test_checks_counter_is_consistent():
    manager = (
        QuantAIProductionModelRuntimeIncidentManager()
    )

    result = manager.evaluate(
        MockResult(
            healthy=False
        ),
        MockResult(
            healthy=False
        ),
    )

    assert (
        result.checks_passed
        + result.checks_failed
        == result.total_checks
    )


def test_empty_inputs_are_not_safe():
    manager = (
        QuantAIProductionModelRuntimeIncidentManager()
    )

    result = manager.evaluate()

    assert result.allow_inference is False
    assert result.blocked is True
    assert result.checks_failed == 2


def test_result_failed_state_is_detectable():
    manager = (
        QuantAIProductionModelRuntimeIncidentManager()
    )

    result = manager.evaluate(
        MockResult(
            healthy=True
        ),
        MockResult(
            healthy=False
        ),
    )

    assert result.healthy is False
    assert result.blocked is True


def test_manager_accepts_boolean_status_objects():
    manager = (
        QuantAIProductionModelRuntimeIncidentManager()
    )

    class BooleanResult:
        healthy = True

    result = manager.evaluate(
        BooleanResult(),
        BooleanResult(),
    )

    assert result.state == ModelRuntimeIncidentState.NORMAL
    assert result.allow_inference is True


def test_degraded_policy_does_not_override_health_recovery():
    manager = (
        QuantAIProductionModelRuntimeIncidentManager(
            allow_degraded_inference=True
        )
    )

    result = manager.evaluate(
        MockResult(
            healthy=False
        ),
        MockResult(
            healthy=True
        ),
    )

    assert (
        result.state
        == ModelRuntimeIncidentState.RECOVERY_REQUIRED
    )

    assert result.allow_inference is True
    assert result.recovery_required is True


def test_non_callable_configuration_is_normalized():
    manager = (
        QuantAIProductionModelRuntimeIncidentManager(
            allow_degraded_inference=1,
            allow_fallback=0,
            require_recovery_on_health_failure=1,
        )
    )

    assert manager.allow_degraded_inference is True
    assert manager.allow_fallback is False
    assert manager.require_recovery_on_health_failure is True


def test_manager_does_not_modify_source_results():
    monitoring = MockResult(
        healthy=True,
        warnings=["warning"],
    )

    inference = MockResult(
        healthy=True,
    )

    manager = (
        QuantAIProductionModelRuntimeIncidentManager()
    )

    manager.evaluate(
        monitoring,
        inference,
    )

    assert monitoring.warnings == ["warning"]
    assert inference.errors == []


def test_warning_is_generated_for_recovery():
    manager = (
        QuantAIProductionModelRuntimeIncidentManager()
    )

    result = manager.evaluate(
        MockResult(
            healthy=False
        ),
        MockResult(
            healthy=True
        ),
    )

    assert any(
        "recovery" in warning.lower()
        for warning in result.warnings
    )


def test_warning_is_generated_for_degraded_state():
    manager = (
        QuantAIProductionModelRuntimeIncidentManager()
    )

    result = manager.evaluate(
        MockResult(
            healthy=True
        ),
        MockResult(
            healthy=False
        ),
    )

    assert any(
        "degraded" in warning.lower()
        for warning in result.warnings
    )


def test_inference_block_warning_is_present():
    manager = (
        QuantAIProductionModelRuntimeIncidentManager()
    )

    result = manager.evaluate(
        MockResult(
            healthy=True
        ),
        MockResult(
            healthy=False
        ),
    )

    assert any(
        "blocked" in warning.lower()
        for warning in result.warnings
    )


def test_all_checks_have_names_and_messages():
    manager = (
        QuantAIProductionModelRuntimeIncidentManager()
    )

    result = manager.evaluate(
        *make_all_passed()
    )

    assert all(
        check.name
        and check.message
        for check in result.checks
    )


def test_all_passed_result_has_no_errors():
    result = (
        QuantAIProductionModelRuntimeIncidentManager()
        .evaluate(
            *make_all_passed()
        )
    )

    assert result.errors == []


def test_all_passed_result_has_no_safety_block():
    result = (
        QuantAIProductionModelRuntimeIncidentManager()
        .evaluate(
            *make_all_passed()
        )
    )

    assert result.blocked is False


def test_health_failure_can_disable_recovery_requirement():
    manager = (
        QuantAIProductionModelRuntimeIncidentManager(
            require_recovery_on_health_failure=False,
            allow_degraded_inference=False,
        )
    )

    result = manager.evaluate(
        MockResult(
            healthy=False
        ),
        MockResult(
            healthy=True
        ),
    )

    assert result.state == ModelRuntimeIncidentState.DEGRADED
    assert result.recovery_required is False
    assert result.allow_inference is False