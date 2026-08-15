from __future__ import annotations

from types import SimpleNamespace

from src.quantai_production_readiness_gate import (
    ProductionReadinessCheck,
    QuantAIProductionReadinessGate,
    QuantAIProductionReadinessResult,
    evaluate_production_readiness,
)


def make_all_passed_results():
    return {
        "end_to_end_result": SimpleNamespace(
            valid=True,
            errors=[],
            warnings=[],
        ),
        "paper_validation_result": SimpleNamespace(
            valid=True,
            errors=[],
            warnings=[],
        ),
        "quality_gate_result": SimpleNamespace(
            passed=True,
            errors=[],
            warnings=[],
        ),
        "integration_result": SimpleNamespace(
            success=True,
            errors=[],
            warnings=[],
        ),
        "system_health_result": SimpleNamespace(
            ready=True,
            healthy=True,
            errors=[],
            warnings=[],
        ),
    }


def test_check_structure():

    check = ProductionReadinessCheck(
        name="test",
        passed=True,
        message="ok",
    )

    assert check.name == "test"
    assert check.passed is True
    assert check.message == "ok"


def test_result_structure():

    result = QuantAIProductionReadinessResult(
        ready=True,
    )

    assert result.ready is True
    assert result.checks == []
    assert result.errors == []
    assert result.warnings == []


def test_result_counters():

    result = QuantAIProductionReadinessResult(
        ready=False,
        checks=[
            ProductionReadinessCheck(
                name="pass",
                passed=True,
                message="ok",
            ),
            ProductionReadinessCheck(
                name="fail",
                passed=False,
                message="failed",
            ),
        ],
    )

    assert result.checks_passed == 1
    assert result.checks_failed == 1
    assert result.total_checks == 2


def test_all_required_checks_pass():

    gate = QuantAIProductionReadinessGate()

    result = gate.evaluate(
        **make_all_passed_results()
    )

    assert isinstance(
        result,
        QuantAIProductionReadinessResult,
    )

    assert result.ready is True
    assert result.checks_failed == 0
    assert result.checks_passed == 5


def test_all_check_names_are_present():

    gate = QuantAIProductionReadinessGate()

    result = gate.evaluate(
        **make_all_passed_results()
    )

    names = [
        check.name
        for check in result.checks
    ]

    assert names == [
        "end_to_end_validation",
        "paper_trading_validation",
        "paper_trading_quality_gate",
        "unified_system_integration",
        "system_health",
    ]


def test_missing_end_to_end_result_fails():

    data = make_all_passed_results()

    data["end_to_end_result"] = None

    result = (
        QuantAIProductionReadinessGate()
        .evaluate(**data)
    )

    assert result.ready is False
    assert result.checks_failed >= 1

    assert any(
        "End-to-End Validation result"
        in error
        for error in result.errors
    )


def test_failed_paper_validation_fails():

    data = make_all_passed_results()

    data["paper_validation_result"] = (
        SimpleNamespace(
            valid=False,
            errors=[
                "paper validation failure"
            ],
            warnings=[],
        )
    )

    result = (
        QuantAIProductionReadinessGate()
        .evaluate(**data)
    )

    assert result.ready is False

    assert any(
        check.name
        == "paper_trading_validation"
        and not check.passed
        for check in result.checks
    )


def test_failed_quality_gate_fails():

    data = make_all_passed_results()

    data["quality_gate_result"] = (
        SimpleNamespace(
            passed=False,
            errors=[
                "quality gate failure"
            ],
            warnings=[],
        )
    )

    result = (
        QuantAIProductionReadinessGate()
        .evaluate(**data)
    )

    assert result.ready is False

    assert any(
        check.name
        == "paper_trading_quality_gate"
        and not check.passed
        for check in result.checks
    )


def test_failed_integration_fails():

    data = make_all_passed_results()

    data["integration_result"] = (
        SimpleNamespace(
            success=False,
            errors=[
                "integration failure"
            ],
            warnings=[],
        )
    )

    result = (
        QuantAIProductionReadinessGate()
        .evaluate(**data)
    )

    assert result.ready is False

    assert any(
        check.name
        == "unified_system_integration"
        and not check.passed
        for check in result.checks
    )


def test_failed_system_health_fails():

    data = make_all_passed_results()

    data["system_health_result"] = (
        SimpleNamespace(
            ready=False,
            healthy=False,
            errors=[
                "health failure"
            ],
            warnings=[],
        )
    )

    result = (
        QuantAIProductionReadinessGate()
        .evaluate(**data)
    )

    assert result.ready is False

    assert any(
        check.name
        == "system_health"
        and not check.passed
        for check in result.checks
    )


def test_source_errors_are_preserved():

    data = make_all_passed_results()

    data["paper_validation_result"] = (
        SimpleNamespace(
            valid=False,
            errors=[
                "source validation error"
            ],
            warnings=[
                "source validation warning"
            ],
        )
    )

    result = (
        QuantAIProductionReadinessGate()
        .evaluate(**data)
    )

    assert (
        "source validation error"
        in result.errors
    )

    assert (
        "source validation warning"
        in result.warnings
    )


def test_source_warnings_do_not_fail_gate():

    data = make_all_passed_results()

    data["system_health_result"] = (
        SimpleNamespace(
            ready=True,
            healthy=True,
            errors=[],
            warnings=[
                "non-critical warning"
            ],
        )
    )

    result = (
        QuantAIProductionReadinessGate()
        .evaluate(**data)
    )

    assert result.ready is True
    assert (
        "non-critical warning"
        in result.warnings
    )


def test_optional_checks_can_be_disabled():

    gate = QuantAIProductionReadinessGate(
        require_end_to_end=False,
        require_paper_validation=False,
        require_quality_gate=False,
        require_integration=True,
        require_system_health=True,
    )

    result = gate.evaluate(
        integration_result=SimpleNamespace(
            success=True
        ),
        system_health_result=SimpleNamespace(
            ready=True
        ),
    )

    assert result.ready is True
    assert result.total_checks == 2


def test_only_disabled_checks_do_not_create_false_failure():

    gate = QuantAIProductionReadinessGate(
        require_end_to_end=False,
        require_paper_validation=False,
        require_quality_gate=False,
        require_integration=False,
        require_system_health=False,
    )

    result = gate.evaluate()

    assert result.ready is False
    assert result.total_checks == 0


def test_is_ready_returns_true():

    gate = QuantAIProductionReadinessGate()

    assert gate.is_ready(
        **make_all_passed_results()
    ) is True


def test_is_ready_returns_false():

    data = make_all_passed_results()

    data["quality_gate_result"] = (
        SimpleNamespace(
            passed=False
        )
    )

    gate = QuantAIProductionReadinessGate()

    assert gate.is_ready(
        **data
    ) is False


def test_convenience_function():

    result = evaluate_production_readiness(
        **make_all_passed_results()
    )

    assert isinstance(
        result,
        QuantAIProductionReadinessResult,
    )

    assert result.ready is True