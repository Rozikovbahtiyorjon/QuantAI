from __future__ import annotations

import pytest

from src.quantai_production_runtime_lifecycle import (
    LifecycleEvent,
    ProductionRuntimeLifecycleResult,
    QuantAIProductionRuntimeLifecycle,
    RuntimeLifecycleState,
)


def test_initial_state_is_stopped():
    lifecycle = QuantAIProductionRuntimeLifecycle()

    assert lifecycle.state == RuntimeLifecycleState.STOPPED
    assert lifecycle.is_running is False


def test_start_moves_runtime_to_running():
    lifecycle = QuantAIProductionRuntimeLifecycle()

    result = lifecycle.start(
        lambda: {
            "status": "running"
        }
    )

    assert isinstance(
        result,
        ProductionRuntimeLifecycleResult,
    )

    assert result.success is True
    assert result.state == RuntimeLifecycleState.RUNNING
    assert lifecycle.is_running is True

    assert result.runtime_result == {
        "status": "running"
    }


def test_start_is_blocked_when_already_running():
    lifecycle = QuantAIProductionRuntimeLifecycle()

    lifecycle.start(
        lambda: "ok"
    )

    result = lifecycle.start(
        lambda: "second"
    )

    assert result.success is False
    assert result.state == RuntimeLifecycleState.RUNNING

    assert (
        "already running"
        in result.errors[0]
    )


def test_start_failure_moves_to_failed():
    lifecycle = QuantAIProductionRuntimeLifecycle()

    def failing_runner():
        raise RuntimeError(
            "startup failure"
        )

    result = lifecycle.start(
        failing_runner
    )

    assert result.success is False
    assert result.state == RuntimeLifecycleState.FAILED
    assert lifecycle.is_running is False

    assert any(
        "startup failure" in error
        for error in result.errors
    )


def test_stop_moves_running_runtime_to_stopped():
    lifecycle = QuantAIProductionRuntimeLifecycle()

    lifecycle.start(
        lambda: "running"
    )

    stopped = []

    result = lifecycle.stop(
        lambda: stopped.append(True)
    )

    assert result.success is True
    assert result.state == RuntimeLifecycleState.STOPPED
    assert lifecycle.is_running is False
    assert stopped == [True]


def test_stop_when_already_stopped_is_safe():
    lifecycle = QuantAIProductionRuntimeLifecycle()

    result = lifecycle.stop()

    assert result.success is True
    assert result.state == RuntimeLifecycleState.STOPPED

    assert result.warnings == [
        "Runtime is already stopped."
    ]


def test_stop_failure_moves_to_failed():
    lifecycle = QuantAIProductionRuntimeLifecycle()

    lifecycle.start(
        lambda: "running"
    )

    def failing_stopper():
        raise RuntimeError(
            "shutdown failure"
        )

    result = lifecycle.stop(
        failing_stopper
    )

    assert result.success is False
    assert result.state == RuntimeLifecycleState.FAILED

    assert any(
        "shutdown failure" in error
        for error in result.errors
    )


def test_emergency_stop_stops_running_runtime():
    lifecycle = QuantAIProductionRuntimeLifecycle()

    lifecycle.start(
        lambda: "running"
    )

    result = lifecycle.emergency_stop(
        lambda: "emergency complete"
    )

    assert result.success is True
    assert result.state == RuntimeLifecycleState.STOPPED
    assert lifecycle.is_running is False


def test_emergency_stop_failure_moves_to_failed():
    lifecycle = QuantAIProductionRuntimeLifecycle()

    lifecycle.start(
        lambda: "running"
    )

    def failing_stopper():
        raise RuntimeError(
            "emergency failure"
        )

    result = lifecycle.emergency_stop(
        failing_stopper
    )

    assert result.success is False
    assert result.state == RuntimeLifecycleState.FAILED

    assert any(
        "emergency failure" in error
        for error in result.errors
    )


def test_recovery_from_failed_restarts_runtime():
    lifecycle = QuantAIProductionRuntimeLifecycle()

    def failing_runner():
        raise RuntimeError(
            "initial failure"
        )

    lifecycle.start(
        failing_runner
    )

    result = lifecycle.recover(
        lambda: {
            "recovered": True
        }
    )

    assert result.success is True
    assert result.state == RuntimeLifecycleState.RUNNING
    assert lifecycle.is_running is True

    assert result.runtime_result == {
        "recovered": True
    }


def test_recovery_from_stopped_starts_runtime():
    lifecycle = QuantAIProductionRuntimeLifecycle()

    result = lifecycle.recover(
        lambda: "recovered"
    )

    assert result.success is True
    assert result.state == RuntimeLifecycleState.RUNNING


def test_recovery_is_blocked_while_running():
    lifecycle = QuantAIProductionRuntimeLifecycle()

    lifecycle.start(
        lambda: "running"
    )

    result = lifecycle.recover(
        lambda: "second"
    )

    assert result.success is False
    assert result.state == RuntimeLifecycleState.RUNNING


def test_invalid_runner_is_rejected():
    lifecycle = QuantAIProductionRuntimeLifecycle()

    with pytest.raises(TypeError):
        lifecycle.start(
            "not callable"
        )


def test_invalid_stopper_is_rejected():
    lifecycle = QuantAIProductionRuntimeLifecycle()

    with pytest.raises(TypeError):
        lifecycle.stop(
            "not callable"
        )


def test_lifecycle_events_are_recorded():
    lifecycle = QuantAIProductionRuntimeLifecycle()

    lifecycle.start(
        lambda: "running"
    )

    lifecycle.stop()

    events = lifecycle.events

    assert events

    assert all(
        isinstance(
            event,
            LifecycleEvent,
        )
        for event in events
    )

    messages = [
        event.message
        for event in events
    ]

    assert (
        "Runtime startup initiated."
        in messages
    )

    assert (
        "Runtime started successfully."
        in messages
    )

    assert (
        "Runtime shutdown initiated."
        in messages
    )

    assert (
        "Runtime stopped successfully."
        in messages
    )


def test_reset_returns_controller_to_clean_stopped_state():
    lifecycle = QuantAIProductionRuntimeLifecycle()

    lifecycle.start(
        lambda: "running"
    )

    lifecycle.stop()

    lifecycle.reset()

    assert (
        lifecycle.state
        == RuntimeLifecycleState.STOPPED
    )

    assert lifecycle.is_running is False
    assert lifecycle.events == []
    assert lifecycle.errors == []
    assert lifecycle.warnings == []