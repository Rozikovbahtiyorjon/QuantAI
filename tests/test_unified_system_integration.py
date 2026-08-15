from __future__ import annotations

import pytest

from src.unified_system_integration import (
    IntegrationStageResult,
    QuantAIUnifiedSystem,
    UnifiedSystemResult,
    create_default_integration,
)


def test_empty_system_succeeds():

    system = QuantAIUnifiedSystem()

    result = system.run(
        "data"
    )

    assert isinstance(
        result,
        UnifiedSystemResult,
    )

    assert result.success is True

    assert result.completed_stages == 0

    assert result.failed_stages == 0

    assert result.outputs == {}


def test_stages_execute_in_order_and_pass_output_forward():

    system = QuantAIUnifiedSystem()

    system.register_stage(
        "data",
        lambda value: value + 1,
    )

    system.register_stage(
        "features",
        lambda value: value * 2,
    )

    system.register_stage(
        "signal",
        lambda value: {
            "signal": value
        },
    )

    result = system.run(
        10
    )

    assert result.success is True

    assert result.stage_names == [
        "data",
        "features",
        "signal",
    ]

    assert result.completed_stages == 3

    assert result.failed_stages == 0

    assert result.outputs["data"] == 11

    assert result.outputs["features"] == 22

    assert result.outputs["signal"] == {
        "signal": 22
    }


def test_stage_result_structure():

    system = QuantAIUnifiedSystem()

    system.register_stage(
        "stage",
        lambda value: value,
    )

    result = system.run(
        "ok"
    )

    assert isinstance(
        result.stages[0],
        IntegrationStageResult,
    )

    assert result.stages[0].name == "stage"

    assert result.stages[0].success is True

    assert result.stages[0].output == "ok"

    assert result.stages[0].error is None


def test_failure_stops_pipeline_and_records_error():

    system = QuantAIUnifiedSystem()

    system.register_stage(
        "first",
        lambda value: value + 1,
    )

    def failing_stage(
        value,
    ):
        raise RuntimeError(
            "integration failure"
        )

    system.register_stage(
        "second",
        failing_stage,
    )

    system.register_stage(
        "third",
        lambda value: value + 100,
    )

    result = system.run(
        1
    )

    assert result.success is False

    assert result.completed_stages == 1

    assert result.failed_stages == 1

    assert result.stage_names == [
        "first",
        "second",
    ]

    assert (
        "second: RuntimeError: integration failure"
        in result.errors
    )

    assert "third" not in result.outputs


def test_duplicate_stage_is_rejected():

    system = QuantAIUnifiedSystem()

    system.register_stage(
        "data",
        lambda value: value,
    )

    with pytest.raises(
        ValueError,
        match="Stage already registered",
    ):

        system.register_stage(
            "data",
            lambda value: value,
        )


def test_invalid_stage_name_is_rejected():

    system = QuantAIUnifiedSystem()

    with pytest.raises(
        ValueError,
    ):

        system.register_stage(
            "",
            lambda value: value,
        )

    with pytest.raises(
        ValueError,
    ):

        system.register_stage(
            "   ",
            lambda value: value,
        )


def test_non_callable_handler_is_rejected():

    system = QuantAIUnifiedSystem()

    with pytest.raises(
        TypeError,
        match="Stage handler must be callable",
    ):

        system.register_stage(
            "data",
            None,
        )


def test_create_default_integration():

    system = create_default_integration(
        {
            "data": (
                lambda value: value + 1
            ),
            "risk": (
                lambda value: value - 1
            ),
        }
    )

    result = system.run(
        100
    )

    assert result.success is True

    assert system.stage_names == [
        "data",
        "risk",
    ]

    assert result.outputs["risk"] == 100


def test_create_default_integration_requires_mapping():

    with pytest.raises(
        TypeError,
        match="stages must be a mapping",
    ):

        create_default_integration(
            []
        )


def test_clear_removes_registered_stages():

    system = QuantAIUnifiedSystem()

    system.register_stage(
        "data",
        lambda value: value,
    )

    assert system.stage_names == [
        "data"
    ]

    system.clear()

    assert system.stage_names == []

    assert (
        system.run(
            "value"
        ).success
        is True
    )