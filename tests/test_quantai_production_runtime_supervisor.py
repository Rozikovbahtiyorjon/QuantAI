from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from src.quantai_production_runtime_lifecycle import (
    QuantAIProductionRuntimeLifecycle,
    RuntimeLifecycleState,
)

from src.quantai_production_runtime_supervisor import (
    ProductionRuntimeSupervisorResult,
    QuantAIProductionRuntimeSupervisor,
    SupervisorCheck,
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


def running_supervisor(
    max_recovery_attempts: int = 1,
) -> QuantAIProductionRuntimeSupervisor:
    lifecycle = (
        QuantAIProductionRuntimeLifecycle()
    )

    lifecycle.start(
        lambda: {
            "status": "running"
        }
    )

    return QuantAIProductionRuntimeSupervisor(
        lifecycle=lifecycle,
        max_recovery_attempts=max_recovery_attempts,
    )


def test_all_health_checks_pass():
    supervisor = running_supervisor()

    result = supervisor.check_health(
        lambda: MockHealth(
            healthy=True
        )
    )

    assert isinstance(
        result,
        ProductionRuntimeSupervisorResult,
    )

    assert result.healthy is True

    assert result.action == "health_check"

    assert result.checks_passed == 2

    assert result.checks_failed == 0

    assert result.errors == []


def test_not_running_state_fails_health():
    supervisor = (
        QuantAIProductionRuntimeSupervisor()
    )

    result = supervisor.check_health(
        lambda: MockHealth(
            healthy=True
        )
    )

    assert result.healthy is False

    assert result.checks_failed == 1

    assert any(
        check.name == "runtime_state"
        and check.passed is False
        for check in result.checks
    )


def test_failed_health_blocks_supervision_without_recovery():
    supervisor = running_supervisor()

    result = supervisor.supervise(
        lambda: MockHealth(
            healthy=False
        )
    )

    assert result.healthy is False

    assert result.action == "health_check"

    assert supervisor.recovery_attempts == 0


def test_health_checker_exception_is_captured():
    supervisor = running_supervisor()

    def failing_checker():
        raise RuntimeError(
            "health failure"
        )

    result = supervisor.check_health(
        failing_checker
    )

    assert result.healthy is False

    assert any(
        "health failure" in error
        for error in result.errors
    )


def test_invalid_health_checker_is_rejected():
    supervisor = running_supervisor()

    with pytest.raises(TypeError):
        supervisor.check_health(
            "invalid"
        )


def test_invalid_runner_is_rejected():
    supervisor = (
        QuantAIProductionRuntimeSupervisor()
    )

    with pytest.raises(TypeError):
        supervisor.recover(
            "invalid"
        )


def test_recovery_from_stopped_state_succeeds():
    supervisor = (
        QuantAIProductionRuntimeSupervisor()
    )

    result = supervisor.recover(
        lambda: {
            "status": "recovered"
        }
    )

    assert result.healthy is True

    assert result.action == "recover"

    assert (
        result.state
        == RuntimeLifecycleState.RUNNING
    )

    assert supervisor.is_running is True

    assert supervisor.recovery_attempts == 1


def test_recovery_from_failed_state_succeeds():
    lifecycle = (
        QuantAIProductionRuntimeLifecycle()
    )

    lifecycle.start(
        lambda: (
            (_ for _ in ()).throw(
                RuntimeError(
                    "startup failure"
                )
            )
        )
    )

    supervisor = (
        QuantAIProductionRuntimeSupervisor(
            lifecycle=lifecycle,
            max_recovery_attempts=1,
        )
    )

    result = supervisor.recover(
        lambda: {
            "status": "recovered"
        }
    )

    assert result.healthy is True

    assert (
        result.state
        == RuntimeLifecycleState.RUNNING
    )

    assert supervisor.recovery_attempts == 1


def test_recovery_failure_is_propagated():
    supervisor = (
        QuantAIProductionRuntimeSupervisor(
            max_recovery_attempts=1
        )
    )

    def failing_runner():
        raise RuntimeError(
            "recovery failure"
        )

    result = supervisor.recover(
        failing_runner
    )

    assert result.healthy is False

    assert any(
        "recovery failure" in error
        for error in result.errors
    )

    assert supervisor.recovery_attempts == 1


def test_recovery_attempt_limit_is_enforced():
    supervisor = (
        QuantAIProductionRuntimeSupervisor(
            max_recovery_attempts=1
        )
    )

    supervisor.recover(
        lambda: (
            (_ for _ in ()).throw(
                RuntimeError(
                    "failure"
                )
            )
        )
    )

    result = supervisor.recover(
        lambda: {
            "status": "recovered"
        }
    )

    assert result.healthy is False

    assert any(
        "Maximum recovery attempts exceeded"
        in error
        for error in result.errors
    )


def test_supervise_recovers_failed_health():
    supervisor = (
        QuantAIProductionRuntimeSupervisor(
            max_recovery_attempts=1
        )
    )

    result = supervisor.supervise(
        lambda: MockHealth(
            healthy=False
        ),
        recovery_runner=lambda: {
            "status": "recovered"
        },
    )

    assert result.healthy is True

    assert result.action == "supervise"

    assert (
        result.state
        == RuntimeLifecycleState.RUNNING
    )

    assert supervisor.is_running is True


def test_supervise_does_not_recover_healthy_runtime():
    supervisor = running_supervisor()

    result = supervisor.supervise(
        lambda: MockHealth(
            healthy=True
        ),
        recovery_runner=lambda: {
            "status": "unexpected"
        },
    )

    assert result.healthy is True

    assert result.action == "health_check"

    assert supervisor.recovery_attempts == 0


def test_warnings_are_propagated():
    supervisor = running_supervisor()

    result = supervisor.check_health(
        lambda: MockHealth(
            healthy=True,
            warnings=[
                "health warning"
            ],
        )
    )

    assert result.healthy is True

    assert (
        "health warning"
        in result.warnings
    )


def test_source_errors_make_health_check_fail():
    supervisor = running_supervisor()

    result = supervisor.check_health(
        lambda: MockHealth(
            healthy=True,
            errors=[
                "source error"
            ],
        )
    )

    assert result.healthy is False

    assert (
        "source error"
        in result.errors
    )


def test_reset_recovery_counter():
    supervisor = (
        QuantAIProductionRuntimeSupervisor(
            max_recovery_attempts=2
        )
    )

    supervisor.recover(
        lambda: (
            (_ for _ in ()).throw(
                RuntimeError(
                    "failure"
                )
            )
        )
    )

    assert supervisor.recovery_attempts == 1

    supervisor.reset_recovery_counter()

    assert supervisor.recovery_attempts == 0


def test_invalid_constructor_values_are_rejected():
    with pytest.raises(TypeError):
        QuantAIProductionRuntimeSupervisor(
            lifecycle="invalid"
        )

    with pytest.raises(TypeError):
        QuantAIProductionRuntimeSupervisor(
            max_recovery_attempts="1"
        )

    with pytest.raises(ValueError):
        QuantAIProductionRuntimeSupervisor(
            max_recovery_attempts=-1
        )


def test_supervisor_check_is_frozen():
    check = SupervisorCheck(
        name="test",
        passed=True,
        message="ok",
    )

    with pytest.raises(Exception):
        check.passed = False
