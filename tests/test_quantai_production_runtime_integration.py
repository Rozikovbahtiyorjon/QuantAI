from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from src.quantai_production_runtime import (
    QuantAIProductionRuntime,
    RuntimeMode,
)

from src.quantai_production_runtime_integration import (
    DeploymentSafetyCheck,
    ProductionRuntimeIntegrationResult,
    QuantAIProductionRuntimeIntegration,
)


@dataclass
class MockResult:
    ready: bool = False
    prepared: bool = False
    success: bool = False
    valid: bool = False
    passed: bool = False

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


def test_constructor_requires_runtime():

    with pytest.raises(
        TypeError
    ):

        QuantAIProductionRuntimeIntegration(
            runtime="invalid"
        )


def test_all_safety_checks_pass():

    integration = (
        QuantAIProductionRuntimeIntegration(
            QuantAIProductionRuntime(
                mode="PAPER"
            )
        )
    )

    result = integration.preflight(
        *make_all_passed()
    )

    assert isinstance(
        result,
        ProductionRuntimeIntegrationResult,
    )

    assert result.ready_for_runtime is True

    assert result.checks_passed == 3

    assert result.checks_failed == 0

    assert result.errors == []


def test_missing_deployment_preparation_blocks():

    integration = (
        QuantAIProductionRuntimeIntegration(
            QuantAIProductionRuntime()
        )
    )

    result = integration.preflight(
        None,
        MockResult(
            ready=True
        ),
        MockResult(
            ready=True
        ),
    )

    assert result.ready_for_runtime is False

    assert result.checks_failed == 1


def test_failed_end_to_end_validation_blocks():

    integration = (
        QuantAIProductionRuntimeIntegration(
            QuantAIProductionRuntime()
        )
    )

    result = integration.preflight(
        MockResult(
            prepared=True
        ),
        MockResult(
            ready=False
        ),
        MockResult(
            ready=True
        ),
    )

    assert result.ready_for_runtime is False

    assert result.checks_failed == 1


def test_failed_readiness_blocks_runtime():

    calls = []

    integration = (
        QuantAIProductionRuntimeIntegration(
            QuantAIProductionRuntime(
                mode="LIVE"
            )
        )
    )

    result = integration.start(
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

    assert result.runtime_started is False

    assert integration.is_running is False

    assert calls == []


def test_all_passed_starts_runtime():

    integration = (
        QuantAIProductionRuntimeIntegration(
            QuantAIProductionRuntime(
                mode=RuntimeMode.PAPER
            )
        )
    )

    result = integration.start(
        *make_all_passed(),
        runner=lambda: {
            "status": "running"
        },
    )

    assert result.ready_for_runtime is True

    assert result.runtime_started is True

    assert result.runtime_result is not None

    assert result.runtime_result.output == {
        "status": "running"
    }

    assert integration.is_running is True

    integration.stop()


def test_runner_failure_is_propagated():

    integration = (
        QuantAIProductionRuntimeIntegration(
            QuantAIProductionRuntime()
        )
    )

    def failing_runner():

        raise RuntimeError(
            "execution failure"
        )

    result = integration.start(
        *make_all_passed(),
        runner=failing_runner,
    )

    assert result.runtime_started is False

    assert integration.is_running is False

    assert any(
        "execution failure" in error
        for error in result.errors
    )


def test_upstream_errors_block_start():

    calls = []

    integration = (
        QuantAIProductionRuntimeIntegration(
            QuantAIProductionRuntime()
        )
    )

    result = integration.start(
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
        runner=lambda: calls.append(
            "started"
        ),
    )

    assert result.runtime_started is False

    assert calls == []

    assert (
        "deployment error"
        in result.errors
    )


def test_warnings_are_propagated():

    integration = (
        QuantAIProductionRuntimeIntegration(
            QuantAIProductionRuntime()
        )
    )

    result = integration.preflight(
        MockResult(
            prepared=True,
            warnings=[
                "deploy warning"
            ],
        ),
        MockResult(
            ready=True
        ),
        MockResult(
            ready=True,
            warnings=[
                "ready warning"
            ],
        ),
    )

    assert result.ready_for_runtime is True

    assert (
        "deploy warning"
        in result.warnings
    )

    assert (
        "ready warning"
        in result.warnings
    )


def test_optional_upstream_checks_can_be_disabled():

    integration = (
        QuantAIProductionRuntimeIntegration(
            QuantAIProductionRuntime(),
            require_deployment_preparation=False,
            require_end_to_end_validation=False,
        )
    )

    result = integration.preflight(
        None,
        None,
        MockResult(
            ready=True
        ),
    )

    assert result.ready_for_runtime is True

    assert result.total_checks == 1


def test_stop_delegates_to_runtime():

    integration = (
        QuantAIProductionRuntimeIntegration(
            QuantAIProductionRuntime()
        )
    )

    integration.start(
        *make_all_passed(),
        runner=lambda: "ok",
    )

    assert integration.is_running is True

    result = integration.stop()

    assert result.errors == []

    assert integration.is_running is False


def test_check_dataclass():

    check = DeploymentSafetyCheck(
        name="test",
        passed=True,
        message="ok",
    )

    assert check.passed is True


def test_live_mode_requires_all_safety_checks():

    integration = (
        QuantAIProductionRuntimeIntegration(
            QuantAIProductionRuntime(
                mode="LIVE"
            )
        )
    )

    result = integration.preflight(
        MockResult(
            prepared=True
        ),
        MockResult(
            ready=True
        ),
        None,
    )

    assert result.ready_for_runtime is False

    assert result.checks_failed == 1

    assert (
        integration.runtime.mode
        is RuntimeMode.LIVE
    )


def test_no_checks_cannot_claim_ready():

    integration = (
        QuantAIProductionRuntimeIntegration(
            QuantAIProductionRuntime(),
            require_deployment_preparation=False,
            require_end_to_end_validation=False,
        )
    )

    result = integration.preflight(
        None,
        None,
        None,
    )

    assert result.ready_for_runtime is False

    assert result.checks_failed == 1