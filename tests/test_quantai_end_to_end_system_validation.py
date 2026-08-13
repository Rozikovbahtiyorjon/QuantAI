from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from src.quantai_end_to_end_system_validation import (
    EndToEndValidationResult,
    QuantAIEndToEndSystemValidator,
    ValidationCheck,
    validate_end_to_end_system,
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
]:
    return (
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
    validator = QuantAIEndToEndSystemValidator()

    result = validator.evaluate(
        *make_all_passed()
    )

    assert isinstance(
        result,
        EndToEndValidationResult,
    )

    assert result.ready is True

    assert result.total_checks == 5

    assert result.checks_passed == 5

    assert result.checks_failed == 0

    assert result.errors == []


def test_missing_required_result_fails():
    validator = QuantAIEndToEndSystemValidator()

    result = validator.evaluate(
        readiness_gate_result=None,
        integration_result=MockResult(
            success=True
        ),
        system_health_result=MockResult(
            healthy=True
        ),
        paper_validation_result=MockResult(
            valid=True
        ),
        quality_gate_result=MockResult(
            passed=True
        ),
    )

    assert result.ready is False

    assert result.checks_failed == 1

    assert any(
        check.name
        == "production_readiness_gate"
        for check in result.checks
    )


def test_failed_integration_blocks_readiness():
    validator = QuantAIEndToEndSystemValidator()

    result = validator.evaluate(
        MockResult(
            ready=True
        ),
        MockResult(
            success=False
        ),
        MockResult(
            healthy=True
        ),
        MockResult(
            valid=True
        ),
        MockResult(
            passed=True
        ),
    )

    assert result.ready is False

    assert result.checks_failed == 1

    assert any(
        "Unified System Integration failed"
        in error
        for error in result.errors
    )


def test_failed_system_health_blocks_readiness():
    validator = QuantAIEndToEndSystemValidator()

    result = validator.evaluate(
        MockResult(
            ready=True
        ),
        MockResult(
            success=True
        ),
        MockResult(
            healthy=False
        ),
        MockResult(
            valid=True
        ),
        MockResult(
            passed=True
        ),
    )

    assert result.ready is False

    assert result.checks_failed == 1


def test_failed_paper_validation_blocks_readiness():
    validator = QuantAIEndToEndSystemValidator()

    result = validator.evaluate(
        MockResult(
            ready=True
        ),
        MockResult(
            success=True
        ),
        MockResult(
            healthy=True
        ),
        MockResult(
            valid=False
        ),
        MockResult(
            passed=True
        ),
    )

    assert result.ready is False

    assert result.checks_failed == 1


def test_failed_quality_gate_blocks_readiness():
    validator = QuantAIEndToEndSystemValidator()

    result = validator.evaluate(
        MockResult(
            ready=True
        ),
        MockResult(
            success=True
        ),
        MockResult(
            healthy=True
        ),
        MockResult(
            valid=True
        ),
        MockResult(
            passed=False
        ),
    )

    assert result.ready is False

    assert result.checks_failed == 1


def test_source_errors_are_propagated():
    validator = QuantAIEndToEndSystemValidator()

    result = validator.evaluate(
        MockResult(
            ready=True,
            errors=["gate error"],
        ),
        MockResult(
            success=True
        ),
        MockResult(
            healthy=True
        ),
        MockResult(
            valid=True
        ),
        MockResult(
            passed=True
        ),
    )

    assert result.ready is False

    assert "gate error" in result.errors


def test_source_warnings_are_propagated():
    validator = QuantAIEndToEndSystemValidator()

    result = validator.evaluate(
        MockResult(
            ready=True,
            warnings=["test warning"],
        ),
        MockResult(
            success=True
        ),
        MockResult(
            healthy=True
        ),
        MockResult(
            valid=True
        ),
        MockResult(
            passed=True
        ),
    )

    assert result.ready is True

    assert "test warning" in result.warnings


def test_optional_checks_can_be_disabled():
    validator = QuantAIEndToEndSystemValidator(
        require_integration=False,
        require_system_health=False,
    )

    result = validator.evaluate(
        MockResult(
            ready=True
        ),
        None,
        None,
        MockResult(
            valid=True
        ),
        MockResult(
            passed=True
        ),
    )

    assert result.ready is True

    assert result.total_checks == 3

    assert result.checks_passed == 3


def test_is_ready_matches_evaluate():
    validator = QuantAIEndToEndSystemValidator()

    args = make_all_passed()

    assert validator.is_ready(
        *args
    ) is True

    assert validator.evaluate(
        *args
    ).ready is True


def test_convenience_function_returns_result():
    result = validate_end_to_end_system(
        *make_all_passed()
    )

    assert isinstance(
        result,
        EndToEndValidationResult,
    )

    assert result.ready is True


def test_empty_configuration_does_not_claim_readiness():
    validator = QuantAIEndToEndSystemValidator(
        require_readiness_gate=False,
        require_integration=False,
        require_system_health=False,
        require_paper_validation=False,
        require_quality_gate=False,
    )

    result = validator.evaluate()

    assert result.ready is False

    assert result.total_checks == 0


def test_check_counters_are_consistent():
    validator = QuantAIEndToEndSystemValidator()

    result = validator.evaluate(
        MockResult(
            ready=True
        ),
        MockResult(
            success=False
        ),
        MockResult(
            healthy=True
        ),
        MockResult(
            valid=False
        ),
        MockResult(
            passed=True
        ),
    )

    assert result.checks_passed == 3

    assert result.checks_failed == 2

    assert (
        result.checks_passed
        + result.checks_failed
        == result.total_checks
    )


def test_invalid_status_type_is_rejected_as_missing_status():
    validator = QuantAIEndToEndSystemValidator()

    class InvalidResult:
        ready = "yes"

    result = validator.evaluate(
        InvalidResult(),
        MockResult(
            success=True
        ),
        MockResult(
            healthy=True
        ),
        MockResult(
            valid=True
        ),
        MockResult(
            passed=True
        ),
    )

    assert result.ready is False

    assert result.checks_failed == 1


def test_validation_check_is_frozen():
    check = ValidationCheck(
        name="test",
        passed=True,
        message="ok",
    )

    with pytest.raises(Exception):
        check.passed = False