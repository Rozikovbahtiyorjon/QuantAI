# =========================================================
# FILE 2
# tests/test_quantai_production_readiness_integration.py
# =========================================================

from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.quantai_production_readiness_gate import (
    QuantAIProductionReadinessGate,
)

from src.quantai_production_readiness_integration import (
    ProductionReadinessIntegrationResult,
    QuantAIProductionReadinessIntegration,
    run_production_readiness_integration,
)


@dataclass
class EndToEndResult:
    valid: bool
    errors: list[str]
    warnings: list[str]


@dataclass
class PaperValidationResult:
    valid: bool
    errors: list[str]
    warnings: list[str]


@dataclass
class QualityGateResult:
    passed: bool
    errors: list[str]
    warnings: list[str]


@dataclass
class IntegrationResult:
    success: bool
    errors: list[str]
    warnings: list[str]


@dataclass
class SystemHealthResult:
    ready: bool
    errors: list[str]
    warnings: list[str]


def make_results(
    *,
    end_to_end: bool = True,
    paper_validation: bool = True,
    quality_gate: bool = True,
    integration: bool = True,
    system_health: bool = True,
):
    return (
        EndToEndResult(
            valid=end_to_end,
            errors=[],
            warnings=[],
        ),
        PaperValidationResult(
            valid=paper_validation,
            errors=[],
            warnings=[],
        ),
        QualityGateResult(
            passed=quality_gate,
            errors=[],
            warnings=[],
        ),
        IntegrationResult(
            success=integration,
            errors=[],
            warnings=[],
        ),
        SystemHealthResult(
            ready=system_health,
            errors=[],
            warnings=[],
        ),
    )


def test_full_production_readiness_pipeline_passes():

    results = make_results()

    integration = (
        QuantAIProductionReadinessIntegration()
    )

    result = integration.evaluate(
        end_to_end_result=results[0],
        paper_validation_result=results[1],
        quality_gate_result=results[2],
        integration_result=results[3],
        system_health_result=results[4],
    )

    assert isinstance(
        result,
        ProductionReadinessIntegrationResult,
    )

    assert result.ready is True
    assert result.total_checks == 5
    assert result.checks_passed == 5
    assert result.checks_failed == 0
    assert result.errors == []


def test_single_failed_layer_blocks_readiness():

    results = make_results(
        quality_gate=False,
    )

    result = (
        run_production_readiness_integration(
            end_to_end_result=results[0],
            paper_validation_result=results[1],
            quality_gate_result=results[2],
            integration_result=results[3],
            system_health_result=results[4],
        )
    )

    assert result.ready is False

    assert result.checks_passed == 4

    assert result.checks_failed == 1

    assert any(
        "paper_trading_quality_gate"
        in error
        for error in result.errors
    )


def test_missing_required_results_block_readiness():

    integration = (
        QuantAIProductionReadinessIntegration()
    )

    result = integration.evaluate()

    assert result.ready is False

    assert result.total_checks == 5

    assert result.checks_failed == 5

    assert len(
        result.errors
    ) >= 5


def test_is_ready_returns_true_for_complete_success():

    results = make_results()

    integration = (
        QuantAIProductionReadinessIntegration()
    )

    assert integration.is_ready(
        end_to_end_result=results[0],
        paper_validation_result=results[1],
        quality_gate_result=results[2],
        integration_result=results[3],
        system_health_result=results[4],
    ) is True


def test_is_ready_returns_false_for_failed_system_health():

    results = make_results(
        system_health=False,
    )

    integration = (
        QuantAIProductionReadinessIntegration()
    )

    assert integration.is_ready(
        end_to_end_result=results[0],
        paper_validation_result=results[1],
        quality_gate_result=results[2],
        integration_result=results[3],
        system_health_result=results[4],
    ) is False


def test_provider_execution_order():

    calls = []

    results = make_results()

    def make_provider(
        name,
        result,
    ):

        def provider():

            calls.append(name)

            return result

        return provider

    integration = (
        QuantAIProductionReadinessIntegration()
    )

    result = integration.run(
        {
            "end_to_end": make_provider(
                "end_to_end",
                results[0],
            ),
            "paper_validation": make_provider(
                "paper_validation",
                results[1],
            ),
            "quality_gate": make_provider(
                "quality_gate",
                results[2],
            ),
            "integration": make_provider(
                "integration",
                results[3],
            ),
            "system_health": make_provider(
                "system_health",
                results[4],
            ),
        }
    )

    assert result.ready is True

    assert calls == [
        "end_to_end",
        "paper_validation",
        "quality_gate",
        "integration",
        "system_health",
    ]


def test_missing_provider_is_rejected():

    integration = (
        QuantAIProductionReadinessIntegration()
    )

    with pytest.raises(
        ValueError,
        match="Missing readiness providers",
    ):

        integration.run({})


def test_non_callable_provider_is_rejected():

    integration = (
        QuantAIProductionReadinessIntegration()
    )

    providers = {
        "end_to_end": lambda: None,
        "paper_validation": lambda: None,
        "quality_gate": lambda: None,
        "integration": lambda: None,
        "system_health": "INVALID",
    }

    with pytest.raises(
        TypeError,
        match="must be callable",
    ):

        integration.run(
            providers
        )


def test_invalid_gate_type_is_rejected():

    with pytest.raises(TypeError):

        QuantAIProductionReadinessIntegration(
            gate="INVALID",
        )


def test_custom_gate_configuration_is_respected():

    gate = QuantAIProductionReadinessGate(
        require_end_to_end=False,
        require_paper_validation=False,
        require_quality_gate=False,
        require_integration=False,
        require_system_health=True,
    )

    integration = (
        QuantAIProductionReadinessIntegration(
            gate=gate,
        )
    )

    result = integration.evaluate(
        system_health_result=SystemHealthResult(
            ready=True,
            errors=[],
            warnings=[],
        )
    )

    assert result.ready is True

    assert result.total_checks == 1

    assert result.checks_passed == 1

    assert result.checks_failed == 0


def test_source_warnings_are_preserved():

    results = make_results()

    results = (
        results[0],
        PaperValidationResult(
            valid=True,
            errors=[],
            warnings=[
                "paper warning"
            ],
        ),
        results[2],
        results[3],
        results[4],
    )

    integration = (
        QuantAIProductionReadinessIntegration()
    )

    result = integration.evaluate(
        end_to_end_result=results[0],
        paper_validation_result=results[1],
        quality_gate_result=results[2],
        integration_result=results[3],
        system_health_result=results[4],
    )

    assert result.ready is True

    assert (
        "paper warning"
        in result.warnings
    )


def test_source_errors_make_final_result_fail():

    results = (
        EndToEndResult(
            valid=True,
            errors=[
                "end-to-end internal error"
            ],
            warnings=[],
        ),
        PaperValidationResult(
            valid=True,
            errors=[],
            warnings=[],
        ),
        QualityGateResult(
            passed=True,
            errors=[],
            warnings=[],
        ),
        IntegrationResult(
            success=True,
            errors=[],
            warnings=[],
        ),
        SystemHealthResult(
            ready=True,
            errors=[],
            warnings=[],
        ),
    )

    integration = (
        QuantAIProductionReadinessIntegration()
    )

    result = integration.evaluate(
        end_to_end_result=results[0],
        paper_validation_result=results[1],
        quality_gate_result=results[2],
        integration_result=results[3],
        system_health_result=results[4],
    )

    assert result.ready is False

    assert (
        "end-to-end internal error"
        in result.errors
    )


def test_convenience_function_returns_result():

    results = make_results()

    result = (
        run_production_readiness_integration(
            end_to_end_result=results[0],
            paper_validation_result=results[1],
            quality_gate_result=results[2],
            integration_result=results[3],
            system_health_result=results[4],
        )
    )

    assert isinstance(
        result,
        ProductionReadinessIntegrationResult,
    )

    assert result.ready is True

    assert result.gate_result.ready is True