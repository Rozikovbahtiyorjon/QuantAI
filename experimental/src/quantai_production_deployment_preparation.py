from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass(frozen=True)
class DeploymentCheck:
    name: str
    passed: bool
    message: str


@dataclass
class ProductionDeploymentPreparationResult:
    ready: bool
    checks: List[DeploymentCheck] = field(
        default_factory=list
    )
    errors: List[str] = field(
        default_factory=list
    )
    warnings: List[str] = field(
        default_factory=list
    )

    @property
    def checks_passed(self) -> int:
        return sum(
            check.passed
            for check in self.checks
        )

    @property
    def checks_failed(self) -> int:
        return sum(
            not check.passed
            for check in self.checks
        )

    @property
    def total_checks(self) -> int:
        return len(self.checks)


class QuantAIProductionDeploymentPreparation:
    """
    Deterministic production-deployment preparation gate.

    This module does not execute trades and does not modify
    exchange configuration.

    It validates results already produced by the QuantAI
    validation, readiness, integration, health and paper-trading
    layers.
    """

    def __init__(
        self,
        require_end_to_end: bool = True,
        require_readiness_gate: bool = True,
        require_integration: bool = True,
        require_system_health: bool = True,
        require_paper_validation: bool = True,
        require_quality_gate: bool = True,
    ) -> None:
        self.require_end_to_end = bool(
            require_end_to_end
        )

        self.require_readiness_gate = bool(
            require_readiness_gate
        )

        self.require_integration = bool(
            require_integration
        )

        self.require_system_health = bool(
            require_system_health
        )

        self.require_paper_validation = bool(
            require_paper_validation
        )

        self.require_quality_gate = bool(
            require_quality_gate
        )

    @staticmethod
    def _extract_boolean(
        result: Any,
        attributes: tuple[str, ...],
    ) -> Optional[bool]:
        if result is None:
            return None

        found_boolean = False

        for attribute in attributes:
            if not hasattr(
                result,
                attribute,
            ):
                continue

            value = getattr(
                result,
                attribute,
            )

            if not isinstance(
                value,
                bool,
            ):
                continue

            found_boolean = True

            if value:
                return True

        if found_boolean:
            return False

        return None

    @staticmethod
    def _extract_messages(
        result: Any,
        attribute: str,
    ) -> List[str]:
        if result is None:
            return []

        value = getattr(
            result,
            attribute,
            None,
        )

        if value is None:
            return []

        if isinstance(
            value,
            str,
        ):
            return [value]

        try:
            return [
                str(item)
                for item in value
            ]

        except TypeError:
            return [str(value)]

    @classmethod
    def _make_check(
        cls,
        name: str,
        result: Any,
        attributes: tuple[str, ...],
        success_message: str,
        missing_message: str,
        failure_message: str,
    ) -> DeploymentCheck:
        passed = cls._extract_boolean(
            result,
            attributes,
        )

        if passed is None:
            return DeploymentCheck(
                name=name,
                passed=False,
                message=missing_message,
            )

        if passed:
            return DeploymentCheck(
                name=name,
                passed=True,
                message=success_message,
            )

        return DeploymentCheck(
            name=name,
            passed=False,
            message=failure_message,
        )

    def _check_end_to_end(
        self,
        result: Any,
    ) -> DeploymentCheck:
        return self._make_check(
            "end_to_end_system_validation",
            result,
            (
                "ready",
                "passed",
                "valid",
                "success",
            ),
            "End-to-End System Validation passed.",
            (
                "End-to-End System Validation result "
                "was not provided or has no supported "
                "status."
            ),
            "End-to-End System Validation failed.",
        )

    def _check_readiness_gate(
        self,
        result: Any,
    ) -> DeploymentCheck:
        return self._make_check(
            "production_readiness_gate",
            result,
            (
                "ready",
                "passed",
                "valid",
                "success",
            ),
            "Production Readiness Gate passed.",
            (
                "Production Readiness Gate result "
                "was not provided or has no supported "
                "status."
            ),
            "Production Readiness Gate failed.",
        )

    def _check_integration(
        self,
        result: Any,
    ) -> DeploymentCheck:
        return self._make_check(
            "unified_system_integration",
            result,
            (
                "success",
                "passed",
                "valid",
            ),
            "Unified System Integration passed.",
            (
                "Unified System Integration result "
                "was not provided or has no supported "
                "status."
            ),
            "Unified System Integration failed.",
        )

    def _check_system_health(
        self,
        result: Any,
    ) -> DeploymentCheck:
        return self._make_check(
            "system_health",
            result,
            (
                "ready",
                "healthy",
                "passed",
                "valid",
            ),
            "System Health passed.",
            (
                "System Health result "
                "was not provided or has no supported "
                "status."
            ),
            "System Health failed.",
        )

    def _check_paper_validation(
        self,
        result: Any,
    ) -> DeploymentCheck:
        return self._make_check(
            "paper_trading_validation",
            result,
            (
                "valid",
                "passed",
                "success",
            ),
            "Paper Trading Validation passed.",
            (
                "Paper Trading Validation result "
                "was not provided or has no supported "
                "status."
            ),
            "Paper Trading Validation failed.",
        )

    def _check_quality_gate(
        self,
        result: Any,
    ) -> DeploymentCheck:
        return self._make_check(
            "paper_trading_quality_gate",
            result,
            (
                "passed",
                "valid",
                "success",
            ),
            "Paper Trading Quality Gate passed.",
            (
                "Paper Trading Quality Gate result "
                "was not provided or has no supported "
                "status."
            ),
            "Paper Trading Quality Gate failed.",
        )

    def evaluate(
        self,
        end_to_end_result: Any = None,
        readiness_gate_result: Any = None,
        integration_result: Any = None,
        system_health_result: Any = None,
        paper_validation_result: Any = None,
        quality_gate_result: Any = None,
    ) -> ProductionDeploymentPreparationResult:
        checks: List[DeploymentCheck] = []
        errors: List[str] = []
        warnings: List[str] = []

        if self.require_end_to_end:
            checks.append(
                self._check_end_to_end(
                    end_to_end_result
                )
            )

        if self.require_readiness_gate:
            checks.append(
                self._check_readiness_gate(
                    readiness_gate_result
                )
            )

        if self.require_integration:
            checks.append(
                self._check_integration(
                    integration_result
                )
            )

        if self.require_system_health:
            checks.append(
                self._check_system_health(
                    system_health_result
                )
            )

        if self.require_paper_validation:
            checks.append(
                self._check_paper_validation(
                    paper_validation_result
                )
            )

        if self.require_quality_gate:
            checks.append(
                self._check_quality_gate(
                    quality_gate_result
                )
            )

        for check in checks:
            if not check.passed:
                errors.append(
                    f"{check.name}: "
                    f"{check.message}"
                )

        source_results = (
            end_to_end_result,
            readiness_gate_result,
            integration_result,
            system_health_result,
            paper_validation_result,
            quality_gate_result,
        )

        for source_result in source_results:
            errors.extend(
                self._extract_messages(
                    source_result,
                    "errors",
                )
            )

            warnings.extend(
                self._extract_messages(
                    source_result,
                    "warnings",
                )
            )

        ready = bool(
            checks
            and all(
                check.passed
                for check in checks
            )
            and not errors
        )

        return ProductionDeploymentPreparationResult(
            ready=ready,
            checks=checks,
            errors=errors,
            warnings=warnings,
        )

    def is_ready(
        self,
        end_to_end_result: Any = None,
        readiness_gate_result: Any = None,
        integration_result: Any = None,
        system_health_result: Any = None,
        paper_validation_result: Any = None,
        quality_gate_result: Any = None,
    ) -> bool:
        result = self.evaluate(
            end_to_end_result=end_to_end_result,
            readiness_gate_result=readiness_gate_result,
            integration_result=integration_result,
            system_health_result=system_health_result,
            paper_validation_result=paper_validation_result,
            quality_gate_result=quality_gate_result,
        )

        return result.ready


def prepare_production_deployment(
    end_to_end_result: Any = None,
    readiness_gate_result: Any = None,
    integration_result: Any = None,
    system_health_result: Any = None,
    paper_validation_result: Any = None,
    quality_gate_result: Any = None,
) -> ProductionDeploymentPreparationResult:
    preparation = QuantAIProductionDeploymentPreparation()

    return preparation.evaluate(
        end_to_end_result=end_to_end_result,
        readiness_gate_result=readiness_gate_result,
        integration_result=integration_result,
        system_health_result=system_health_result,
        paper_validation_result=paper_validation_result,
        quality_gate_result=quality_gate_result,
    )


__all__ = [
    "DeploymentCheck",
    "ProductionDeploymentPreparationResult",
    "QuantAIProductionDeploymentPreparation",
    "prepare_production_deployment",
]