from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass(frozen=True)
class ProductionReadinessCheck:
    name: str
    passed: bool
    message: str


@dataclass
class QuantAIProductionReadinessResult:
    ready: bool
    checks: List[ProductionReadinessCheck] = field(
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


class QuantAIProductionReadinessGate:
    """
    Final readiness gate for the QuantAI system.

    This module does not implement trading logic.

    It aggregates results produced by already existing
    validation and integration components:

        - End-to-End Validation
        - Paper Trading Validator
        - Paper Trading Quality Gate
        - Unified System Integration
        - System Health

    Existing modules are not modified.
    """

    def __init__(
        self,
        require_end_to_end: bool = True,
        require_paper_validation: bool = True,
        require_quality_gate: bool = True,
        require_integration: bool = True,
        require_system_health: bool = True,
    ) -> None:

        self.require_end_to_end = bool(
            require_end_to_end
        )

        self.require_paper_validation = bool(
            require_paper_validation
        )

        self.require_quality_gate = bool(
            require_quality_gate
        )

        self.require_integration = bool(
            require_integration
        )

        self.require_system_health = bool(
            require_system_health
        )

    # =====================================================
    # GENERIC RESULT EXTRACTION
    # =====================================================

    @staticmethod
    def _extract_boolean(
        result: Any,
        attributes: tuple[str, ...],
    ) -> Optional[bool]:

        if result is None:
            return None

        for attribute in attributes:

            if hasattr(
                result,
                attribute,
            ):

                value = getattr(
                    result,
                    attribute,
                )

                if isinstance(
                    value,
                    bool,
                ):

                    return value

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

    # =====================================================
    # INDIVIDUAL CHECKS
    # =====================================================

    def _check_end_to_end(
        self,
        result: Any,
    ) -> ProductionReadinessCheck:

        if result is None:

            return ProductionReadinessCheck(
                name="end_to_end_validation",
                passed=False,
                message=(
                    "End-to-End Validation result "
                    "was not provided."
                ),
            )

        passed = self._extract_boolean(
            result,
            (
                "valid",
                "passed",
                "success",
                "healthy",
            ),
        )

        if passed is None:

            return ProductionReadinessCheck(
                name="end_to_end_validation",
                passed=False,
                message=(
                    "End-to-End Validation result "
                    "does not expose a supported "
                    "boolean status."
                ),
            )

        if passed:

            return ProductionReadinessCheck(
                name="end_to_end_validation",
                passed=True,
                message=(
                    "End-to-End validation passed."
                ),
            )

        return ProductionReadinessCheck(
            name="end_to_end_validation",
            passed=False,
            message=(
                "End-to-End validation failed."
            ),
        )

    def _check_paper_validation(
        self,
        result: Any,
    ) -> ProductionReadinessCheck:

        if result is None:

            return ProductionReadinessCheck(
                name="paper_trading_validation",
                passed=False,
                message=(
                    "Paper Trading Validator result "
                    "was not provided."
                ),
            )

        passed = self._extract_boolean(
            result,
            (
                "valid",
                "passed",
                "success",
            ),
        )

        if passed is None:

            return ProductionReadinessCheck(
                name="paper_trading_validation",
                passed=False,
                message=(
                    "Paper Trading validation result "
                    "does not expose a supported "
                    "boolean status."
                ),
            )

        if passed:

            return ProductionReadinessCheck(
                name="paper_trading_validation",
                passed=True,
                message=(
                    "Paper Trading validation passed."
                ),
            )

        return ProductionReadinessCheck(
            name="paper_trading_validation",
            passed=False,
            message=(
                "Paper Trading validation failed."
            ),
        )

    def _check_quality_gate(
        self,
        result: Any,
    ) -> ProductionReadinessCheck:

        if result is None:

            return ProductionReadinessCheck(
                name="paper_trading_quality_gate",
                passed=False,
                message=(
                    "Paper Trading Quality Gate result "
                    "was not provided."
                ),
            )

        passed = self._extract_boolean(
            result,
            (
                "passed",
                "valid",
                "success",
            ),
        )

        if passed is None:

            return ProductionReadinessCheck(
                name="paper_trading_quality_gate",
                passed=False,
                message=(
                    "Paper Trading Quality Gate result "
                    "does not expose a supported "
                    "boolean status."
                ),
            )

        if passed:

            return ProductionReadinessCheck(
                name="paper_trading_quality_gate",
                passed=True,
                message=(
                    "Paper Trading Quality Gate passed."
                ),
            )

        return ProductionReadinessCheck(
            name="paper_trading_quality_gate",
            passed=False,
            message=(
                "Paper Trading Quality Gate failed."
            ),
        )

    def _check_integration(
        self,
        result: Any,
    ) -> ProductionReadinessCheck:

        if result is None:

            return ProductionReadinessCheck(
                name="unified_system_integration",
                passed=False,
                message=(
                    "Unified System Integration result "
                    "was not provided."
                ),
            )

        passed = self._extract_boolean(
            result,
            (
                "success",
                "passed",
                "valid",
            ),
        )

        if passed is None:

            return ProductionReadinessCheck(
                name="unified_system_integration",
                passed=False,
                message=(
                    "Unified System Integration result "
                    "does not expose a supported "
                    "boolean status."
                ),
            )

        if passed:

            return ProductionReadinessCheck(
                name="unified_system_integration",
                passed=True,
                message=(
                    "Unified System Integration passed."
                ),
            )

        return ProductionReadinessCheck(
            name="unified_system_integration",
            passed=False,
            message=(
                "Unified System Integration failed."
            ),
        )

    def _check_system_health(
        self,
        result: Any,
    ) -> ProductionReadinessCheck:

        if result is None:

            return ProductionReadinessCheck(
                name="system_health",
                passed=False,
                message=(
                    "System Health result "
                    "was not provided."
                ),
            )

        passed = self._extract_boolean(
            result,
            (
                "ready",
                "healthy",
                "passed",
                "valid",
            ),
        )

        if passed is None:

            return ProductionReadinessCheck(
                name="system_health",
                passed=False,
                message=(
                    "System Health result "
                    "does not expose a supported "
                    "boolean status."
                ),
            )

        if passed:

            return ProductionReadinessCheck(
                name="system_health",
                passed=True,
                message=(
                    "System Health and Readiness passed."
                ),
            )

        return ProductionReadinessCheck(
            name="system_health",
            passed=False,
            message=(
                "System Health and Readiness failed."
            ),
        )

    # =====================================================
    # EVALUATION
    # =====================================================

    def evaluate(
        self,
        end_to_end_result: Any = None,
        paper_validation_result: Any = None,
        quality_gate_result: Any = None,
        integration_result: Any = None,
        system_health_result: Any = None,
    ) -> QuantAIProductionReadinessResult:
        """
        Aggregate all production-readiness results.

        A required check that is missing or failed makes
        the final readiness result False.
        """

        checks: List[
            ProductionReadinessCheck
        ] = []

        errors: List[str] = []

        warnings: List[str] = []

        # -------------------------------------------------
        # END-TO-END
        # -------------------------------------------------

        if self.require_end_to_end:

            check = self._check_end_to_end(
                end_to_end_result
            )

            checks.append(check)

        # -------------------------------------------------
        # PAPER VALIDATION
        # -------------------------------------------------

        if self.require_paper_validation:

            check = self._check_paper_validation(
                paper_validation_result
            )

            checks.append(check)

        # -------------------------------------------------
        # QUALITY GATE
        # -------------------------------------------------

        if self.require_quality_gate:

            check = self._check_quality_gate(
                quality_gate_result
            )

            checks.append(check)

        # -------------------------------------------------
        # INTEGRATION
        # -------------------------------------------------

        if self.require_integration:

            check = self._check_integration(
                integration_result
            )

            checks.append(check)

        # -------------------------------------------------
        # SYSTEM HEALTH
        # -------------------------------------------------

        if self.require_system_health:

            check = self._check_system_health(
                system_health_result
            )

            checks.append(check)

        # -------------------------------------------------
        # COLLECT ERRORS
        # -------------------------------------------------

        for check in checks:

            if not check.passed:

                errors.append(
                    f"{check.name}: "
                    f"{check.message}"
                )

        # -------------------------------------------------
        # COLLECT SOURCE ERRORS
        # -------------------------------------------------

        source_results = (
            end_to_end_result,
            paper_validation_result,
            quality_gate_result,
            integration_result,
            system_health_result,
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

        # -------------------------------------------------
        # FINAL DECISION
        # -------------------------------------------------

        ready = (
            len(checks) > 0
            and not errors
            and all(
                check.passed
                for check in checks
            )
        )

        return QuantAIProductionReadinessResult(
            ready=ready,
            checks=checks,
            errors=errors,
            warnings=warnings,
        )

    # =====================================================
    # READINESS
    # =====================================================

    def is_ready(
        self,
        end_to_end_result: Any = None,
        paper_validation_result: Any = None,
        quality_gate_result: Any = None,
        integration_result: Any = None,
        system_health_result: Any = None,
    ) -> bool:

        result = self.evaluate(
            end_to_end_result=end_to_end_result,
            paper_validation_result=paper_validation_result,
            quality_gate_result=quality_gate_result,
            integration_result=integration_result,
            system_health_result=system_health_result,
        )

        return result.ready


def evaluate_production_readiness(
    end_to_end_result: Any = None,
    paper_validation_result: Any = None,
    quality_gate_result: Any = None,
    integration_result: Any = None,
    system_health_result: Any = None,
) -> QuantAIProductionReadinessResult:

    gate = QuantAIProductionReadinessGate()

    return gate.evaluate(
        end_to_end_result=end_to_end_result,
        paper_validation_result=paper_validation_result,
        quality_gate_result=quality_gate_result,
        integration_result=integration_result,
        system_health_result=system_health_result,
    )


__all__ = [
    "ProductionReadinessCheck",
    "QuantAIProductionReadinessResult",
    "QuantAIProductionReadinessGate",
    "evaluate_production_readiness",
]