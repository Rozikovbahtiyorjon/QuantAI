from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass(frozen=True)
class ValidationCheck:
    name: str
    passed: bool
    message: str


@dataclass
class EndToEndValidationResult:
    ready: bool
    checks: List[ValidationCheck] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

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


class QuantAIEndToEndSystemValidator:
    """
    Final deterministic validation layer for production preparation.

    It validates already-produced results without implementing trading logic.
    """

    def __init__(
        self,
        require_readiness_gate: bool = True,
        require_integration: bool = True,
        require_system_health: bool = True,
        require_paper_validation: bool = True,
        require_quality_gate: bool = True,
    ) -> None:
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
        """
        Extract a supported boolean status from a result.

        The result objects used throughout QuantAI may expose
        several status attributes simultaneously.

        Example:

            ready=False
            healthy=True
            passed=True

        The presence of an earlier False value must not hide
        a later True value.

        Resolution rules:

            - any supported True value -> True
            - supported boolean values exist but all are False -> False
            - no supported boolean value -> None
        """

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
    ) -> ValidationCheck:
        passed = cls._extract_boolean(
            result,
            attributes,
        )

        if passed is None:
            return ValidationCheck(
                name=name,
                passed=False,
                message=missing_message,
            )

        if passed:
            return ValidationCheck(
                name=name,
                passed=True,
                message=success_message,
            )

        return ValidationCheck(
            name=name,
            passed=False,
            message=failure_message,
        )

    def evaluate(
        self,
        readiness_gate_result: Any = None,
        integration_result: Any = None,
        system_health_result: Any = None,
        paper_validation_result: Any = None,
        quality_gate_result: Any = None,
    ) -> EndToEndValidationResult:
        checks: List[ValidationCheck] = []
        errors: List[str] = []
        warnings: List[str] = []

        if self.require_readiness_gate:
            checks.append(
                self._make_check(
                    "production_readiness_gate",
                    readiness_gate_result,
                    (
                        "ready",
                        "passed",
                        "valid",
                        "success",
                    ),
                    "Production Readiness Gate passed.",
                    (
                        "Production Readiness Gate result "
                        "was not provided or has no "
                        "supported status."
                    ),
                    "Production Readiness Gate failed.",
                )
            )

        if self.require_integration:
            checks.append(
                self._make_check(
                    "unified_system_integration",
                    integration_result,
                    (
                        "success",
                        "passed",
                        "valid",
                    ),
                    "Unified System Integration passed.",
                    (
                        "Unified System Integration result "
                        "was not provided or has no "
                        "supported status."
                    ),
                    "Unified System Integration failed.",
                )
            )

        if self.require_system_health:
            checks.append(
                self._make_check(
                    "system_health",
                    system_health_result,
                    (
                        "ready",
                        "healthy",
                        "passed",
                        "valid",
                    ),
                    "System Health passed.",
                    (
                        "System Health result "
                        "was not provided or has no "
                        "supported status."
                    ),
                    "System Health failed.",
                )
            )

        if self.require_paper_validation:
            checks.append(
                self._make_check(
                    "paper_trading_validation",
                    paper_validation_result,
                    (
                        "valid",
                        "passed",
                        "success",
                    ),
                    "Paper Trading Validation passed.",
                    (
                        "Paper Trading Validation result "
                        "was not provided or has no "
                        "supported status."
                    ),
                    "Paper Trading Validation failed.",
                )
            )

        if self.require_quality_gate:
            checks.append(
                self._make_check(
                    "paper_trading_quality_gate",
                    quality_gate_result,
                    (
                        "passed",
                        "valid",
                        "success",
                    ),
                    "Paper Trading Quality Gate passed.",
                    (
                        "Paper Trading Quality Gate result "
                        "was not provided or has no "
                        "supported status."
                    ),
                    "Paper Trading Quality Gate failed.",
                )
            )

        for check in checks:
            if not check.passed:
                errors.append(
                    f"{check.name}: "
                    f"{check.message}"
                )

        source_results = (
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

        return EndToEndValidationResult(
            ready=ready,
            checks=checks,
            errors=errors,
            warnings=warnings,
        )

    def is_ready(
        self,
        readiness_gate_result: Any = None,
        integration_result: Any = None,
        system_health_result: Any = None,
        paper_validation_result: Any = None,
        quality_gate_result: Any = None,
    ) -> bool:
        result = self.evaluate(
            readiness_gate_result=readiness_gate_result,
            integration_result=integration_result,
            system_health_result=system_health_result,
            paper_validation_result=paper_validation_result,
            quality_gate_result=quality_gate_result,
        )

        return result.ready


def validate_end_to_end_system(
    readiness_gate_result: Any = None,
    integration_result: Any = None,
    system_health_result: Any = None,
    paper_validation_result: Any = None,
    quality_gate_result: Any = None,
) -> EndToEndValidationResult:
    validator = QuantAIEndToEndSystemValidator()

    return validator.evaluate(
        readiness_gate_result=readiness_gate_result,
        integration_result=integration_result,
        system_health_result=system_health_result,
        paper_validation_result=paper_validation_result,
        quality_gate_result=quality_gate_result,
    )


__all__ = [
    "ValidationCheck",
    "EndToEndValidationResult",
    "QuantAIEndToEndSystemValidator",
    "validate_end_to_end_system",
]