from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Dict


@dataclass(frozen=True)
class ProductionReadinessCheck:
    name: str
    passed: bool
    message: str


@dataclass(frozen=True)
class ProductionEvidenceContract:
    """Strict typed contract — no generic _extract_boolean (point 13)."""
    oos_pass: bool
    paper_pass: bool
    risk_pass: bool
    execution_pass: bool
    monitoring_pass: bool
    statistical_pass: bool
    metadata: Dict[str, Any] = field(default_factory=dict, compare=False, hash=False)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProductionEvidenceContract":
        required = ["oos_pass","paper_pass","risk_pass","execution_pass","monitoring_pass","statistical_pass"]
        missing = [k for k in required if k not in d]
        if missing:
            raise ValueError(f"ProductionEvidenceContract missing keys: {missing}")
        return cls(
            oos_pass=bool(d["oos_pass"]),
            paper_pass=bool(d["paper_pass"]),
            risk_pass=bool(d["risk_pass"]),
            execution_pass=bool(d["execution_pass"]),
            monitoring_pass=bool(d["monitoring_pass"]),
            statistical_pass=bool(d["statistical_pass"]),
            metadata={k:v for k,v in d.items() if k not in required},
        )


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
    def _has_typed_failure(result: Any) -> Optional[str]:
        """If result carries typed contract fields, any False must fail — prevents success=True bypass (point 13)."""
        if isinstance(result, dict):
            for k in ["oos_pass","paper_pass","risk_pass","execution_pass","monitoring_pass","statistical_pass"]:
                if k in result and not bool(result[k]):
                    return k
        # also check ProductionEvidenceContract instance
        if isinstance(result, ProductionEvidenceContract):
            for k in ["oos_pass","paper_pass","risk_pass","execution_pass","monitoring_pass","statistical_pass"]:
                if not bool(getattr(result, k)):
                    return k
        return None

    @staticmethod
    def _extract_boolean(
        result: Any,
        attributes: tuple[str, ...],
    ) -> Optional[bool]:
        """Deprecated generic extractor — kept for legacy 5-arg path but guarded by _has_typed_failure."""
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

        # Also support dict access
        if isinstance(result, dict):
            for attr in attributes:
                if attr in result and isinstance(result[attr], bool):
                    return bool(result[attr])

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

        # Typed contract guard — success=True cannot hide oos_pass=False
        tf = self._has_typed_failure(result)
        if tf:
            return ProductionReadinessCheck(name="end_to_end_validation", passed=False, message=f"Typed contract {tf}=False blocks bypass (success=True ignored)")

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

        tf = self._has_typed_failure(result)
        if tf:
            return ProductionReadinessCheck(name="paper_trading_validation", passed=False, message=f"Typed contract {tf}=False blocks bypass")

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

        tf = self._has_typed_failure(result)
        if tf:
            return ProductionReadinessCheck(name="paper_trading_quality_gate", passed=False, message=f"Typed contract {tf}=False blocks bypass")

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

        tf = self._has_typed_failure(result)
        if tf:
            return ProductionReadinessCheck(name="unified_system_integration", passed=False, message=f"Typed contract {tf}=False blocks bypass")

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

        tf = self._has_typed_failure(result)
        if tf:
            return ProductionReadinessCheck(name="system_health", passed=False, message=f"Typed contract {tf}=False blocks bypass")

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
    # TYPED CONTRACT — strict (point 13)
    # =====================================================

    def _check_typed_contract(self, contract: Any) -> List[ProductionReadinessCheck]:
        """Strict: requires ProductionEvidenceContract with 6 explicit bools — no _extract_boolean fallback."""
        checks: List[ProductionReadinessCheck] = []
        # Accept dataclass or dict
        if isinstance(contract, ProductionEvidenceContract):
            d = {
                "oos_pass": contract.oos_pass,
                "paper_pass": contract.paper_pass,
                "risk_pass": contract.risk_pass,
                "execution_pass": contract.execution_pass,
                "monitoring_pass": contract.monitoring_pass,
                "statistical_pass": contract.statistical_pass,
            }
        elif isinstance(contract, dict) and all(k in contract for k in ["oos_pass","paper_pass","risk_pass","execution_pass","monitoring_pass","statistical_pass"]):
            d = contract
        else:
            raise TypeError("Typed contract requires ProductionEvidenceContract or dict with oos_pass/paper_pass/risk_pass/execution_pass/monitoring_pass/statistical_pass")
        for name in ["oos_pass","paper_pass","risk_pass","execution_pass","monitoring_pass","statistical_pass"]:
            passed = bool(d[name])
            checks.append(ProductionReadinessCheck(
                name=name,
                passed=passed,
                message=f"{name}: {'PASS' if passed else 'FAIL'} (typed contract)",
            ))
        return checks

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
        contract: Any = None,
    ) -> QuantAIProductionReadinessResult:
        """
        Aggregate all production-readiness results — TYPED ONLY (P0.6).

        Production readiness REQUIRES ProductionEvidenceContract with 6 explicit bools:
        oos_pass, paper_pass, risk_pass, execution_pass, monitoring_pass, statistical_pass.
        Generic result["success"]/valid/passed/healthy/ready is FORBIDDEN — not trusted.
        Missing typed contract → FAIL (no fallback to generic _extract_boolean for production).
        """

        checks: List[
            ProductionReadinessCheck
        ] = []

        errors: List[str] = []

        warnings: List[str] = []

        # ---- Strict typed contract REQUIRED (P0.6) — no generic fallback for production ----
        typed_src = contract
        # Also detect contract smuggled as first arg dict with 6 keys
        if typed_src is None:
            for cand in [end_to_end_result, paper_validation_result, quality_gate_result, integration_result, system_health_result]:
                if isinstance(cand, ProductionEvidenceContract) or (isinstance(cand, dict) and all(k in cand for k in ["oos_pass","paper_pass","risk_pass","execution_pass","monitoring_pass","statistical_pass"])):
                    typed_src = cand
                    break
        if typed_src is not None:
            try:
                checks = self._check_typed_contract(typed_src)
                # Collect typed contract errors
                for c in checks:
                    if not c.passed:
                        errors.append(f"{c.name}: {c.message}")
                ready = len(checks) == 6 and all(c.passed for c in checks)
                return QuantAIProductionReadinessResult(ready=ready, checks=checks, errors=errors, warnings=warnings)
            except Exception as e:
                return QuantAIProductionReadinessResult(
                    ready=False,
                    checks=[],
                    errors=[f"Typed contract error: {e}"],
                    warnings=[],
                )
        # NO typed contract → FAIL (generic success/valid/passed is FORBIDDEN for production)
        return QuantAIProductionReadinessResult(
            ready=False,
            checks=[],
            errors=["ProductionEvidenceContract required: 6 explicit bools (oos_pass, paper_pass, risk_pass, execution_pass, monitoring_pass, statistical_pass) — generic result[\"success\"]/valid/passed not trusted (P0.6)"],
            warnings=[],
        )

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
    "ProductionEvidenceContract",
    "QuantAIProductionReadinessResult",
    "QuantAIProductionReadinessGate",
    "evaluate_production_readiness",
]