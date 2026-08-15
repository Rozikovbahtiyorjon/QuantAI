# =========================================================
# FILE 1
# src/quantai_production_readiness_integration.py
# =========================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

from src.quantai_production_readiness_gate import (
    QuantAIProductionReadinessGate,
    QuantAIProductionReadinessResult,
)


@dataclass(frozen=True)
class ProductionReadinessIntegrationResult:
    ready: bool
    checks_passed: int
    checks_failed: int
    total_checks: int
    errors: list[str]
    warnings: list[str]
    gate_result: QuantAIProductionReadinessResult


class QuantAIProductionReadinessIntegration:
    """
    End-to-end integration runner for the existing
    QuantAI Production Readiness Gate.

    This component does not implement trading logic.

    It accepts the already-produced results from the
    five readiness layers, feeds them into the existing
    Production Readiness Gate, and exposes one
    deterministic integration result.
    """

    def __init__(
        self,
        gate: Optional[
            QuantAIProductionReadinessGate
        ] = None,
    ) -> None:

        self.gate = (
            gate
            if gate is not None
            else QuantAIProductionReadinessGate()
        )

        if not isinstance(
            self.gate,
            QuantAIProductionReadinessGate,
        ):

            raise TypeError(
                "gate must be QuantAIProductionReadinessGate."
            )

    def evaluate(
        self,
        end_to_end_result: Any = None,
        paper_validation_result: Any = None,
        quality_gate_result: Any = None,
        integration_result: Any = None,
        system_health_result: Any = None,
    ) -> ProductionReadinessIntegrationResult:
        """
        Run the complete production-readiness aggregation.
        """

        gate_result = self.gate.evaluate(
            end_to_end_result=end_to_end_result,
            paper_validation_result=paper_validation_result,
            quality_gate_result=quality_gate_result,
            integration_result=integration_result,
            system_health_result=system_health_result,
        )

        return ProductionReadinessIntegrationResult(
            ready=gate_result.ready,
            checks_passed=gate_result.checks_passed,
            checks_failed=gate_result.checks_failed,
            total_checks=gate_result.total_checks,
            errors=list(
                gate_result.errors
            ),
            warnings=list(
                gate_result.warnings
            ),
            gate_result=gate_result,
        )

    def is_ready(
        self,
        end_to_end_result: Any = None,
        paper_validation_result: Any = None,
        quality_gate_result: Any = None,
        integration_result: Any = None,
        system_health_result: Any = None,
    ) -> bool:
        """
        Return only the final readiness decision.
        """

        result = self.evaluate(
            end_to_end_result=end_to_end_result,
            paper_validation_result=paper_validation_result,
            quality_gate_result=quality_gate_result,
            integration_result=integration_result,
            system_health_result=system_health_result,
        )

        return result.ready

    def run(
        self,
        providers: Mapping[
            str,
            Callable[[], Any],
        ],
    ) -> ProductionReadinessIntegrationResult:
        """
        Execute readiness result providers and evaluate
        their results.

        Required provider names:

            end_to_end
            paper_validation
            quality_gate
            integration
            system_health

        Providers are executed in deterministic order.
        """

        if not isinstance(
            providers,
            Mapping,
        ):

            raise TypeError(
                "providers must be a mapping."
            )

        required_names = (
            "end_to_end",
            "paper_validation",
            "quality_gate",
            "integration",
            "system_health",
        )

        missing = [
            name
            for name in required_names
            if name not in providers
        ]

        if missing:

            raise ValueError(
                "Missing readiness providers: "
                + ", ".join(missing)
            )

        results: dict[str, Any] = {}

        for name in required_names:

            provider = providers[name]

            if not callable(provider):

                raise TypeError(
                    f"Provider '{name}' must be callable."
                )

            results[name] = provider()

        return self.evaluate(
            end_to_end_result=results[
                "end_to_end"
            ],
            paper_validation_result=results[
                "paper_validation"
            ],
            quality_gate_result=results[
                "quality_gate"
            ],
            integration_result=results[
                "integration"
            ],
            system_health_result=results[
                "system_health"
            ],
        )


def run_production_readiness_integration(
    end_to_end_result: Any = None,
    paper_validation_result: Any = None,
    quality_gate_result: Any = None,
    integration_result: Any = None,
    system_health_result: Any = None,
) -> ProductionReadinessIntegrationResult:
    """
    Convenience function for one-shot integration evaluation.
    """

    integration = (
        QuantAIProductionReadinessIntegration()
    )

    return integration.evaluate(
        end_to_end_result=end_to_end_result,
        paper_validation_result=paper_validation_result,
        quality_gate_result=quality_gate_result,
        integration_result=integration_result,
        system_health_result=system_health_result,
    )


__all__ = [
    "ProductionReadinessIntegrationResult",
    "QuantAIProductionReadinessIntegration",
    "run_production_readiness_integration",
]