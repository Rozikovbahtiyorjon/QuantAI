from __future__ import annotations

import pytest

from experimental.src.quantai_system_health import (
    HealthCheckResult,
    QuantAISystemHealth,
    QuantAISystemHealthResult,
    run_system_health_check,
)


def test_health_check_result_structure():

    result = HealthCheckResult(
        name="test",
        passed=True,
        message="ok",
    )

    assert result.name == "test"
    assert result.passed is True
    assert result.message == "ok"


def test_health_result_structure():

    result = QuantAISystemHealthResult(
        healthy=True,
        ready=True,
    )

    assert result.healthy is True
    assert result.ready is True
    assert result.checks == []
    assert result.errors == []
    assert result.warnings == []


def test_health_result_counters():

    result = QuantAISystemHealthResult(
        healthy=False,
        ready=False,
        checks=[
            HealthCheckResult(
                name="one",
                passed=True,
                message="ok",
            ),
            HealthCheckResult(
                name="two",
                passed=False,
                message="failed",
            ),
        ],
    )

    assert result.checks_passed == 1
    assert result.checks_failed == 1
    assert result.total_checks == 2


def test_default_modules_are_available():

    checker = QuantAISystemHealth()

    assert len(
        checker.required_modules
    ) > 0


def test_custom_modules_are_supported():

    checker = QuantAISystemHealth(
        required_modules=[
            "src.unified_system_integration",
        ]
    )

    assert checker.required_modules == (
        "src.unified_system_integration",
    )


def test_invalid_module_name_is_rejected():

    with pytest.raises(ValueError):

        QuantAISystemHealth(
            required_modules=[
                "",
            ]
        )


def test_invalid_module_type_is_rejected():

    with pytest.raises(ValueError):

        QuantAISystemHealth(
            required_modules=[
                "   ",
            ]
        )


def test_required_module_check_passes():

    checker = QuantAISystemHealth(
        required_modules=[
            "src.unified_system_integration",
        ]
    )

    checks = (
        checker.check_required_modules()
    )

    assert len(checks) == 1
    assert checks[0].passed is True


def test_missing_module_fails():

    checker = QuantAISystemHealth(
        required_modules=[
            "src.module_that_does_not_exist",
        ]
    )

    checks = (
        checker.check_required_modules()
    )

    assert len(checks) == 1
    assert checks[0].passed is False
    assert "Import failed" in checks[0].message


def test_integration_layer_is_available():

    result = (
        QuantAISystemHealth
        .check_integration_layer()
    )

    assert isinstance(
        result,
        HealthCheckResult,
    )

    assert result.passed is True


def test_orchestrator_is_operational():

    result = (
        QuantAISystemHealth
        .check_orchestrator()
    )

    assert isinstance(
        result,
        HealthCheckResult,
    )

    assert result.passed is True


def test_full_health_check():

    checker = QuantAISystemHealth(
        required_modules=[
            "src.unified_system_integration",
        ]
    )

    result = checker.check()

    assert isinstance(
        result,
        QuantAISystemHealthResult,
    )

    assert result.healthy is True
    assert result.ready is True
    assert result.checks_failed == 0
    assert result.checks_passed == result.total_checks


def test_full_health_check_contains_integration():

    checker = QuantAISystemHealth(
        required_modules=[
            "src.unified_system_integration",
        ]
    )

    result = checker.check()

    names = [
        check.name
        for check in result.checks
    ]

    assert "integration_layer" in names
    assert "orchestrator" in names


def test_failed_health_check_is_not_ready():

    checker = QuantAISystemHealth(
        required_modules=[
            "src.module_that_does_not_exist",
        ]
    )

    result = checker.check()

    assert result.healthy is False
    assert result.ready is False
    assert result.checks_failed >= 1
    assert len(result.errors) >= 1


def test_is_ready_returns_true():

    checker = QuantAISystemHealth(
        required_modules=[
            "src.unified_system_integration",
        ]
    )

    assert checker.is_ready() is True


def test_is_ready_returns_false_for_invalid_system():

    checker = QuantAISystemHealth(
        required_modules=[
            "src.module_that_does_not_exist",
        ]
    )

    assert checker.is_ready() is False


def test_convenience_function():

    result = run_system_health_check(
        required_modules=[
            "src.unified_system_integration",
        ]
    )

    assert isinstance(
        result,
        QuantAISystemHealthResult,
    )

    assert result.ready is True