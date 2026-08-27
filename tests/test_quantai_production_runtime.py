from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from src.quantai_production_runtime import (
    ProductionRuntimeResult,
    QuantAIProductionRuntime,
    RuntimeMode,
)


@dataclass
class MockReadiness:
    ready: bool = False

    errors: list[str] = field(
        default_factory=list
    )

    warnings: list[str] = field(
        default_factory=list
    )


def test_default_mode_is_dry_run():

    runtime = QuantAIProductionRuntime()

    assert runtime.mode is RuntimeMode.DRY_RUN

    assert runtime.is_running is False


def test_string_mode_is_normalized():

    runtime = QuantAIProductionRuntime(
        mode="paper"
    )

    assert runtime.mode is RuntimeMode.PAPER


def test_invalid_mode_is_rejected():

    with pytest.raises(ValueError):

        QuantAIProductionRuntime(
            mode="invalid"
        )


def test_non_string_non_enum_mode_is_rejected():

    with pytest.raises(TypeError):

        QuantAIProductionRuntime(
            mode=123
        )


def test_preflight_passes_with_ready_result():

    runtime = QuantAIProductionRuntime(
        mode=RuntimeMode.PAPER
    )

    result = runtime.preflight(
        MockReadiness(
            ready=True
        )
    )

    assert isinstance(
        result,
        ProductionRuntimeResult,
    )

    assert result.started is False

    assert result.mode is RuntimeMode.PAPER

    assert result.checks_passed == 4

    assert result.checks_failed == 0

    assert result.errors == []


def test_preflight_fails_without_readiness_result():

    runtime = QuantAIProductionRuntime()

    result = runtime.preflight(
        None
    )

    assert result.started is False

    assert result.checks_failed == 1

    assert any(
        "readiness_gate" in error
        for error in result.errors
    )


def test_failed_readiness_blocks_start():

    calls = []

    runtime = QuantAIProductionRuntime(
        mode="LIVE"
    )

    result = runtime.start(
        MockReadiness(
            ready=False
        ),
        runner=lambda: calls.append(
            "executed"
        ),
    )

    assert result.started is False

    assert runtime.is_running is False

    assert calls == []


def test_ready_runtime_starts_runner():

    runtime = QuantAIProductionRuntime(
        mode="PAPER"
    )

    result = runtime.start(
        MockReadiness(
            ready=True
        ),
        runner=lambda: {
            "status": "ok"
        },
    )

    assert result.started is True

    assert result.output == {
        "status": "ok"
    }

    assert runtime.is_running is True


def test_runner_exception_blocks_runtime_state():

    runtime = QuantAIProductionRuntime()

    def failing_runner():

        raise RuntimeError(
            "runner failure"
        )

    result = runtime.start(
        MockReadiness(
            ready=True
        ),
        runner=failing_runner,
    )

    assert result.started is False

    assert runtime.is_running is False

    assert any(
        "runner failure" in error
        for error in result.errors
    )


def test_double_start_is_rejected():

    runtime = QuantAIProductionRuntime()

    first = runtime.start(
        MockReadiness(
            ready=True
        ),
        runner=lambda: "ok",
    )

    second = runtime.start(
        MockReadiness(
            ready=True
        ),
        runner=lambda: "again",
    )

    assert first.started is True

    assert second.started is False

    assert (
        "already running"
        in second.errors[0]
    )


def test_stop_changes_running_state():

    runtime = QuantAIProductionRuntime()

    runtime.start(
        MockReadiness(
            ready=True
        ),
        runner=lambda: "ok",
    )

    result = runtime.stop()

    assert result.started is False

    assert result.checks_failed == 0

    assert runtime.is_running is False


def test_stop_when_already_stopped_is_safe():

    runtime = QuantAIProductionRuntime()

    result = runtime.stop()

    assert result.errors == []

    assert result.checks_passed == 1

    assert runtime.is_running is False


def test_warnings_are_propagated():

    runtime = QuantAIProductionRuntime()

    result = runtime.preflight(
        MockReadiness(
            ready=True,
            warnings=[
                "test warning"
            ],
        )
    )

    assert result.errors == []

    assert result.warnings == [
        "test warning"
    ]


def test_source_errors_block_start():

    runtime = QuantAIProductionRuntime()

    result = runtime.start(
        MockReadiness(
            ready=True,
            errors=[
                "upstream error"
            ],
        )
    )

    assert result.started is False

    assert (
        "upstream error"
        in result.errors
    )

    assert runtime.is_running is False


def test_readiness_requirement_can_be_disabled():

    runtime = QuantAIProductionRuntime(
        require_readiness=False
    )

    result = runtime.start(
        runner=lambda: "dry-run"
    )

    assert result.started is True

    assert result.output == "dry-run"

    runtime.stop()


def test_running_property_reflects_lifecycle():

    runtime = QuantAIProductionRuntime()

    assert runtime.is_running is False

    runtime.start(
        MockReadiness(
            ready=True
        )
    )

    assert runtime.is_running is True

    runtime.stop()

    assert runtime.is_running is False