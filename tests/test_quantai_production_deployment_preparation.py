from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from src.quantai_production_deployment_preparation import (
    DeploymentCheck,
    ProductionDeploymentPreparationResult,
    QuantAIProductionDeploymentPreparation,
    prepare_production_deployment,
)


@dataclass
class MockResult:
    ready: bool = False
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


def make_all_passed() -> tuple[
    MockResult,
    MockResult,
    MockResult,
    MockResult,
    MockResult,
    MockResult,
]:
    return (
        MockResult(
            ready=True,
            passed=True,
        ),
        MockResult(
            ready=True,
            passed=True,
        ),
        MockResult(
            success=True,
            passed=True,
        ),
        MockResult(
            healthy=True,
            passed=True,
        ),
        MockResult(
            valid=True,
            passed=True,
        ),
        MockResult(
            passed=True,
        ),
    )


def test_all_required_components_pass():
    preparation = QuantAIProductionDeploymentPreparation()

    result = preparation.evaluate(
        *make_all_passed()
    )

    assert isinstance(
        result,
        ProductionDeploymentPreparationResult,
    )

    assert result.ready is True
    assert result.total_checks == 6
    assert result.checks_passed == 6
    assert result.checks_failed == 0
    assert result.errors == []


def test_missing_end_to_end_result_fails():
    preparation = QuantAIProductionDeploymentPreparation()

    result = preparation.evaluate(
        None,
        MockResult(ready=True),
        MockResult(success=True),
        MockResult(healthy=True),
        MockResult(valid=True),
        MockResult(passed=True),
    )

    assert result.ready is False
    assert result.checks_failed == 1

    assert any(
        check.name
        == "end_to_end_system_validation"
        for check in result.checks
    )


def test_missing_readiness_gate_fails():
    preparation = QuantAIProductionDeploymentPreparation()

    result = preparation.evaluate(
        MockResult(ready=True),
        None,
        MockResult(success=True),
        MockResult(healthy=True),
        MockResult(valid=True),
        MockResult(passed=True),
    )

    assert result.ready is False
    assert result.checks_failed == 1


def test_failed_integration_blocks_deployment():
    preparation = QuantAIProductionDeploymentPreparation()

    result = preparation.evaluate(
        MockResult(ready=True),
        MockResult(ready=True),
        MockResult(success=False),
        MockResult(healthy=True),
        MockResult(valid=True),
        MockResult(passed=True),
    )

    assert result.ready is False
    assert result.checks_failed == 1

    assert any(
        "Unified System Integration failed"
        in error
        for error in result.errors
    )


def test_failed_system_health_blocks_deployment():
    preparation = QuantAIProductionDeploymentPreparation()

    result = preparation.evaluate(
        MockResult(ready=True),
        MockResult(ready=True),
        MockResult(success=True),
        MockResult(healthy=False),
        MockResult(valid=True),
        MockResult(passed=True),
    )

    assert result.ready is False
    assert result.checks_failed == 1


def test_failed_paper_validation_blocks_deployment():
    preparation = QuantAIProductionDeploymentPreparation()

    result = preparation.evaluate(
        MockResult(ready=True),
        MockResult(ready=True),
        MockResult(success=True),
        MockResult(healthy=True),
        MockResult(valid=False),
        MockResult(passed=True),
    )

    assert result.ready is False
    assert result.checks_failed == 1


def test_failed_quality_gate_blocks_deployment():
    preparation = QuantAIProductionDeploymentPreparation()

    result = preparation.evaluate(
        MockResult(ready=True),
        MockResult(ready=True),
        MockResult(success=True),
        MockResult(healthy=True),
        MockResult(valid=True),
        MockResult(passed=False),
    )

    assert result.ready is False
    assert result.checks_failed == 1


def test_source_errors_are_propagated():
    preparation = QuantAIProductionDeploymentPreparation()

    result = preparation.evaluate(
        MockResult(
            ready=True,
            errors=["deployment source error"],
        ),
        MockResult(ready=True),
        MockResult(success=True),
        MockResult(healthy=True),
        MockResult(valid=True),
        MockResult(passed=True),
    )

    assert result.ready is False
    assert "deployment source error" in result.errors


def test_source_warnings_are_propagated():
    preparation = QuantAIProductionDeploymentPreparation()

    result = preparation.evaluate(
        MockResult(
            ready=True,
            warnings=["deployment warning"],
        ),
        MockResult(ready=True),
        MockResult(success=True),
        MockResult(healthy=True),
        MockResult(valid=True),
        MockResult(passed=True),
    )

    assert result.ready is True
    assert "deployment warning" in result.warnings


def test_optional_checks_can_be_disabled():
    preparation = QuantAIProductionDeploymentPreparation(
        require_end_to_end=False,
        require_system_health=False,
    )

    result = preparation.evaluate(
        None,
        MockResult(ready=True),
        MockResult(success=True),
        None,
        MockResult(valid=True),
        MockResult(passed=True),
    )

    assert result.ready is True
    assert result.total_checks == 4
    assert result.checks_passed == 4


def test_is_ready_matches_evaluate():
    preparation = QuantAIProductionDeploymentPreparation()

    args = make_all_passed()

    assert preparation.is_ready(
        *args
    ) is True

    assert preparation.evaluate(
        *args
    ).ready is True


def test_convenience_function_returns_result():
    result = prepare_production_deployment(
        *make_all_passed()
    )

    assert isinstance(
        result,
        ProductionDeploymentPreparationResult,
    )

    assert result.ready is True


def test_empty_configuration_does_not_claim_readiness():
    preparation = QuantAIProductionDeploymentPreparation(
        require_end_to_end=False,
        require_readiness_gate=False,
        require_integration=False,
        require_system_health=False,
        require_paper_validation=False,
        require_quality_gate=False,
    )

    result = preparation.evaluate()

    assert result.ready is False
    assert result.total_checks == 0


def test_check_counters_are_consistent():
    preparation = QuantAIProductionDeploymentPreparation()

    result = preparation.evaluate(
        MockResult(ready=True),
        MockResult(ready=True),
        MockResult(success=False),
        MockResult(healthy=False),
        MockResult(valid=True),
        MockResult(passed=True),
    )

    assert result.checks_passed == 4
    assert result.checks_failed == 2

    assert (
        result.checks_passed
        + result.checks_failed
        == result.total_checks
    )


def test_invalid_status_type_is_rejected():
    preparation = QuantAIProductionDeploymentPreparation()

    class InvalidResult:
        ready = "yes"

    result = preparation.evaluate(
        InvalidResult(),
        MockResult(ready=True),
        MockResult(success=True),
        MockResult(healthy=True),
        MockResult(valid=True),
        MockResult(passed=True),
    )

    assert result.ready is False
    assert result.checks_failed == 1


def test_deployment_check_is_frozen():
    check = DeploymentCheck(
        name="test",
        passed=True,
        message="ok",
    )

    with pytest.raises(Exception):
        check.passed = False