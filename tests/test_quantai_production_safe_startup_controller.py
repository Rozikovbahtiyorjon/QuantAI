from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from src.quantai_production_runtime import (
    QuantAIProductionRuntime,
)

from src.quantai_production_runtime_integration import (
    QuantAIProductionRuntimeIntegration,
)

from src.quantai_production_safe_startup_controller import (
    QuantAIProductionSafeStartupController,
    SafeStartupResult,
    StartupStep,
)


@dataclass
class MockResult:
    ready: bool = False
    prepared: bool = False
    success: bool = False
    valid: bool = False
    passed: bool = False
    healthy: bool = False

    errors: list[str] = field(
        default_factory=list
    )

    warnings: list[str] = field(
        default_factory=list
    )


def make_all_passed():
    return (
        MockResult(
            prepared=True
        ),
        MockResult(
            ready=True
        ),
        MockResult(
            ready=True
        ),
    )


def make_controller():

    integration = (
        QuantAIProductionRuntimeIntegration(
            QuantAIProductionRuntime()
        )
    )

    return (
        QuantAIProductionSafeStartupController(
            integration
        )
    )


def test_constructor_requires_integration():

    with pytest.raises(
        TypeError
    ):

        QuantAIProductionSafeStartupController(
            "invalid"
        )


def test_all_preflight_steps_pass_and_runtime_starts():

    controller = make_controller()

    result = controller.start(
        *make_all_passed(),
        runner=lambda: {
            "status": "running"
        },
    )

    assert isinstance(
        result,
        SafeStartupResult,
    )

    assert result.started is True

    assert result.startup_aborted is False

    assert result.steps_passed == 5

    assert result.steps_failed == 0

    assert result.errors == []

    assert controller.is_running is True

    controller.stop()


def test_deployment_failure_aborts_before_end_to_end():

    calls = []

    controller = make_controller()

    result = controller.start(
        MockResult(
            prepared=False
        ),
        MockResult(
            ready=True
        ),
        MockResult(
            ready=True
        ),
        runner=lambda: calls.append(
            "started"
        ),
    )

    assert result.started is False

    assert result.startup_aborted is True

    assert result.steps_failed == 1

    assert result.total_steps == 1

    assert calls == []

    assert controller.is_running is False


def test_missing_deployment_result_aborts():

    controller = make_controller()

    result = controller.start(
        None,
        MockResult(
            ready=True
        ),
        MockResult(
            ready=True
        ),
    )

    assert result.started is False

    assert result.startup_aborted is True

    assert result.steps_failed == 1

    assert (
        "deployment_preparation"
        in result.errors[0]
    )


def test_end_to_end_failure_aborts_before_runtime():

    calls = []

    controller = make_controller()

    result = controller.start(
        MockResult(
            prepared=True
        ),
        MockResult(
            ready=False
        ),
        MockResult(
            ready=True
        ),
        runner=lambda: calls.append(
            "started"
        ),
    )

    assert result.started is False

    assert result.startup_aborted is True

    assert result.steps_failed == 1

    assert result.total_steps == 2

    assert calls == []


def test_readiness_failure_aborts_before_runtime():

    calls = []

    controller = make_controller()

    result = controller.start(
        MockResult(
            prepared=True
        ),
        MockResult(
            ready=True
        ),
        MockResult(
            ready=False
        ),
        runner=lambda: calls.append(
            "started"
        ),
    )

    assert result.started is False

    assert result.startup_aborted is True

    assert result.steps_failed == 1

    assert result.total_steps == 3

    assert calls == []


def test_runtime_runner_failure_is_propagated():

    controller = make_controller()

    def failing_runner():

        raise RuntimeError(
            "runtime failure"
        )

    result = controller.start(
        *make_all_passed(),
        runner=failing_runner,
    )

    assert result.started is False

    assert result.startup_aborted is True

    assert result.runtime_result is not None

    assert any(
        "runtime failure" in error
        for error in result.errors
    )

    assert controller.is_running is False


def test_upstream_errors_are_propagated():

    controller = make_controller()

    result = controller.start(
        MockResult(
            prepared=True,
            errors=[
                "deployment error"
            ],
        ),
        MockResult(
            ready=True
        ),
        MockResult(
            ready=True
        ),
    )

    assert result.started is False

    assert result.startup_aborted is True

    assert (
        "deployment error"
        in result.errors
    )


def test_warnings_are_propagated():

    controller = make_controller()

    result = controller.start(
        MockResult(
            prepared=True,
            warnings=[
                "deployment warning"
            ],
        ),
        MockResult(
            ready=True,
            warnings=[
                "validation warning"
            ],
        ),
        MockResult(
            ready=True,
            warnings=[
                "readiness warning"
            ],
        ),
        runner=lambda: "running",
    )

    assert result.started is True

    assert (
        "deployment warning"
        in result.warnings
    )

    assert (
        "validation warning"
        in result.warnings
    )

    assert (
        "readiness warning"
        in result.warnings
    )

    controller.stop()


def test_stop_delegates_to_runtime():

    controller = make_controller()

    result = controller.start(
        *make_all_passed(),
        runner=lambda: "ok",
    )

    assert result.started is True

    assert controller.is_running is True

    runtime_result = controller.stop()

    assert runtime_result.errors == []

    assert controller.is_running is False


def test_start_is_blocked_when_runtime_is_already_running():

    controller = make_controller()

    first = controller.start(
        *make_all_passed(),
        runner=lambda: "ok",
    )

    assert first.started is True

    assert controller.is_running is True

    second = controller.start(
        *make_all_passed(),
        runner=lambda: "second",
    )

    assert second.started is False

    assert second.startup_aborted is True

    assert any(
        "already running" in error.lower()
        for error in second.errors
    )

    controller.stop()


def test_invalid_status_type_is_rejected():

    controller = make_controller()

    class InvalidResult:
        prepared = "yes"

    result = controller.start(
        InvalidResult(),
        MockResult(
            ready=True
        ),
        MockResult(
            ready=True
        ),
    )

    assert result.started is False

    assert result.startup_aborted is True

    assert result.steps_failed == 1


def test_startup_order_is_deterministic():

    controller = make_controller()

    order = []

    deployment = MockResult(
        prepared=True
    )

    end_to_end = MockResult(
        ready=True
    )

    readiness = MockResult(
        ready=True
    )

    original_start = (
        controller.integration.start
    )

    def tracked_start(**kwargs):

        order.append(
            "runtime_integration"
        )

        return original_start(
            **kwargs
        )

    controller.integration.start = (
        tracked_start
    )

    order.extend(
        [
            "deployment_preparation",
            "end_to_end_validation",
            "production_readiness",
        ]
    )

    result = controller.start(
        deployment,
        end_to_end,
        readiness,
        runner=lambda: order.append(
            "runtime"
        ),
    )

    assert result.started is True

    assert order[:3] == [
        "deployment_preparation",
        "end_to_end_validation",
        "production_readiness",
    ]

    assert (
        "runtime_integration"
        in order
    )

    assert (
        "runtime"
        in order
    )

    controller.stop()


def test_startup_result_contains_runtime_result():

    controller = make_controller()

    result = controller.start(
        *make_all_passed(),
        runner=lambda: {
            "ok": True
        },
    )

    assert (
        result.runtime_result
        is not None
    )

    assert (
        result.runtime_result.output
        == {
            "ok": True
        }
    )

    controller.stop()


def test_startup_step_is_frozen():

    step = StartupStep(
        name="test",
        passed=True,
        message="ok",
    )

    with pytest.raises(Exception):

        step.passed = False


def test_empty_inputs_never_claim_start():

    controller = make_controller()

    result = controller.start()

    assert result.started is False

    assert result.startup_aborted is True

    assert result.steps_failed == 1

    assert controller.is_running is False