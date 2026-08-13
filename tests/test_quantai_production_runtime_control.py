from __future__ import annotations

import pytest

from src.quantai_production_runtime_control import (
    ProductionRuntimeControlResult,
    QuantAIProductionRuntimeControl,
    RuntimeControlCheck,
    execute_runtime_command,
)
from src.quantai_production_runtime_lifecycle import (
    QuantAIProductionRuntimeLifecycle,
    RuntimeLifecycleState,
)


def test_initial_status_is_stopped():
    controller = QuantAIProductionRuntimeControl()

    result = controller.status()

    assert isinstance(
        result,
        ProductionRuntimeControlResult,
    )

    assert result.success is True

    assert result.command == "status"

    assert result.state == RuntimeLifecycleState.STOPPED

    assert result.runtime_result == {
        "state": "STOPPED",
        "is_running": False,
    }

    assert result.checks_passed == 1

    assert result.checks_failed == 0


def test_start_command_starts_runtime():
    controller = QuantAIProductionRuntimeControl()

    result = controller.start(
        lambda: {
            "status": "running"
        }
    )

    assert result.success is True

    assert result.command == "start"

    assert result.state == RuntimeLifecycleState.RUNNING

    assert controller.is_running is True

    assert result.runtime_result == {
        "status": "running"
    }


def test_stop_command_stops_runtime():
    controller = QuantAIProductionRuntimeControl()

    controller.start(
        lambda: "running"
    )

    result = controller.stop(
        lambda: "stopped"
    )

    assert result.success is True

    assert result.command == "stop"

    assert result.state == RuntimeLifecycleState.STOPPED

    assert controller.is_running is False

    assert result.runtime_result is None


def test_emergency_stop_command():
    controller = QuantAIProductionRuntimeControl()

    controller.start(
        lambda: "running"
    )

    result = controller.emergency_stop(
        lambda: "emergency stopped"
    )

    assert result.success is True

    assert result.command == "emergency_stop"

    assert result.state == RuntimeLifecycleState.STOPPED

    assert controller.is_running is False


def test_recover_command():
    controller = QuantAIProductionRuntimeControl()

    controller.start(
        lambda: "running"
    )

    controller.stop()

    result = controller.recover(
        lambda: "recovered"
    )

    assert result.success is True

    assert result.command == "recover"

    assert result.state == RuntimeLifecycleState.RUNNING

    assert controller.is_running is True

    assert result.runtime_result == "recovered"


def test_start_failure_is_propagated():
    controller = QuantAIProductionRuntimeControl()

    def failing_runner():
        raise RuntimeError(
            "startup failure"
        )

    result = controller.start(
        failing_runner
    )

    assert result.success is False

    assert result.state == RuntimeLifecycleState.FAILED

    assert result.errors

    assert any(
        "startup failure" in error
        for error in result.errors
    )


def test_stop_failure_is_propagated():
    controller = QuantAIProductionRuntimeControl()

    controller.start(
        lambda: "running"
    )

    def failing_stopper():
        raise RuntimeError(
            "shutdown failure"
        )

    result = controller.stop(
        failing_stopper
    )

    assert result.success is False

    assert result.state == RuntimeLifecycleState.FAILED

    assert any(
        "shutdown failure" in error
        for error in result.errors
    )


def test_execute_status_command():
    controller = QuantAIProductionRuntimeControl()

    result = controller.execute(
        "status"
    )

    assert result.success is True

    assert result.command == "status"


def test_execute_start_command():
    controller = QuantAIProductionRuntimeControl()

    result = controller.execute(
        "START",
        runner=lambda: "running",
    )

    assert result.success is True

    assert result.state == RuntimeLifecycleState.RUNNING


def test_execute_stop_command():
    controller = QuantAIProductionRuntimeControl()

    controller.start(
        lambda: "running"
    )

    result = controller.execute(
        " stop ",
    )

    assert result.success is True

    assert result.state == RuntimeLifecycleState.STOPPED


def test_execute_emergency_stop_command():
    controller = QuantAIProductionRuntimeControl()

    controller.start(
        lambda: "running"
    )

    result = controller.execute(
        "emergency_stop"
    )

    assert result.success is True

    assert result.state == RuntimeLifecycleState.STOPPED


def test_execute_recover_command():
    controller = QuantAIProductionRuntimeControl()

    controller.start(
        lambda: "running"
    )

    controller.stop()

    result = controller.execute(
        "recover",
        runner=lambda: "recovered",
    )

    assert result.success is True

    assert result.state == RuntimeLifecycleState.RUNNING


def test_execute_unknown_command_is_rejected():
    controller = QuantAIProductionRuntimeControl()

    with pytest.raises(ValueError):
        controller.execute(
            "unknown"
        )


def test_execute_empty_command_is_rejected():
    controller = QuantAIProductionRuntimeControl()

    with pytest.raises(ValueError):
        controller.execute(
            "   "
        )


def test_execute_requires_runner_for_start():
    controller = QuantAIProductionRuntimeControl()

    with pytest.raises(ValueError):
        controller.execute(
            "start"
        )


def test_execute_requires_runner_for_recover():
    controller = QuantAIProductionRuntimeControl()

    with pytest.raises(ValueError):
        controller.execute(
            "recover"
        )


def test_invalid_runner_is_rejected():
    controller = QuantAIProductionRuntimeControl()

    with pytest.raises(TypeError):
        controller.start(
            "not callable"
        )


def test_invalid_stopper_is_rejected():
    controller = QuantAIProductionRuntimeControl()

    with pytest.raises(TypeError):
        controller.stop(
            "not callable"
        )


def test_invalid_lifecycle_is_rejected():
    with pytest.raises(TypeError):
        QuantAIProductionRuntimeControl(
            lifecycle="invalid"
        )


def test_reset_returns_controller_to_stopped():
    controller = QuantAIProductionRuntimeControl()

    controller.start(
        lambda: "running"
    )

    assert controller.is_running is True

    controller.reset()

    assert controller.state == RuntimeLifecycleState.STOPPED

    assert controller.is_running is False


def test_convenience_function_status():
    result = execute_runtime_command(
        "status"
    )

    assert isinstance(
        result,
        ProductionRuntimeControlResult,
    )

    assert result.success is True

    assert result.command == "status"


def test_custom_lifecycle_is_reused():
    lifecycle = QuantAIProductionRuntimeLifecycle()

    controller = QuantAIProductionRuntimeControl(
        lifecycle=lifecycle
    )

    result = controller.start(
        lambda: "running"
    )

    assert result.success is True

    assert lifecycle.state == RuntimeLifecycleState.RUNNING

    assert controller.lifecycle is lifecycle


def test_control_check_is_frozen():
    check = RuntimeControlCheck(
        name="test",
        passed=True,
        message="ok",
    )

    with pytest.raises(Exception):
        check.passed = False


def test_all_control_counters_are_consistent():
    controller = QuantAIProductionRuntimeControl()

    result = controller.status()

    assert (
        result.checks_passed
        + result.checks_failed
        == result.total_checks
    )