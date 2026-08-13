from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

from src.quantai_production_runtime import (
    ProductionRuntimeResult,
    QuantAIProductionRuntime,
    RuntimeMode,
)


@dataclass(frozen=True)
class DeploymentSafetyCheck:
    name: str
    passed: bool
    message: str


@dataclass
class ProductionRuntimeIntegrationResult:
    ready_for_runtime: bool
    runtime_started: bool
    mode: RuntimeMode
    checks: List[DeploymentSafetyCheck] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    runtime_result: Optional[ProductionRuntimeResult] = None

    @property
    def checks_passed(self) -> int:
        return sum(check.passed for check in self.checks)

    @property
    def checks_failed(self) -> int:
        return sum(not check.passed for check in self.checks)

    @property
    def total_checks(self) -> int:
        return len(self.checks)


class QuantAIProductionRuntimeIntegration:
    """
    Safety integration boundary between deployment preparation,
    end-to-end validation, production readiness and runtime.

    This module does not implement trading logic.
    """

    def __init__(
        self,
        runtime: QuantAIProductionRuntime,
        require_deployment_preparation: bool = True,
        require_end_to_end_validation: bool = True,
    ) -> None:
        if not isinstance(runtime, QuantAIProductionRuntime):
            raise TypeError(
                "runtime must be a "
                "QuantAIProductionRuntime instance."
            )

        self.runtime = runtime
        self.require_deployment_preparation = bool(
            require_deployment_preparation
        )
        self.require_end_to_end_validation = bool(
            require_end_to_end_validation
        )

    @staticmethod
    def _extract_boolean(
        result: Any,
        attributes: tuple[str, ...],
    ) -> Optional[bool]:
        if result is None:
            return None

        for attribute in attributes:
            if hasattr(result, attribute):
                value = getattr(result, attribute)

                if isinstance(value, bool):
                    return value

        return None

    @staticmethod
    def _extract_messages(
        result: Any,
        attribute: str,
    ) -> List[str]:
        if result is None:
            return []

        value = getattr(result, attribute, None)

        if value is None:
            return []

        if isinstance(value, str):
            return [value]

        try:
            return [str(item) for item in value]
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
    ) -> DeploymentSafetyCheck:
        passed = cls._extract_boolean(
            result,
            attributes,
        )

        if passed is None:
            return DeploymentSafetyCheck(
                name=name,
                passed=False,
                message=missing_message,
            )

        if passed:
            return DeploymentSafetyCheck(
                name=name,
                passed=True,
                message=success_message,
            )

        return DeploymentSafetyCheck(
            name=name,
            passed=False,
            message=failure_message,
        )

    def preflight(
        self,
        deployment_preparation_result: Any = None,
        end_to_end_validation_result: Any = None,
        readiness_result: Any = None,
    ) -> ProductionRuntimeIntegrationResult:
        checks: List[DeploymentSafetyCheck] = []

        if self.require_deployment_preparation:
            checks.append(
                self._make_check(
                    "deployment_preparation",
                    deployment_preparation_result,
                    (
                        "prepared",
                        "ready",
                        "passed",
                        "valid",
                        "success",
                    ),
                    "Deployment preparation passed.",
                    (
                        "Deployment preparation result "
                        "is missing or unsupported."
                    ),
                    "Deployment preparation failed.",
                )
            )

        if self.require_end_to_end_validation:
            checks.append(
                self._make_check(
                    "end_to_end_validation",
                    end_to_end_validation_result,
                    (
                        "ready",
                        "passed",
                        "valid",
                        "success",
                    ),
                    "End-to-End validation passed.",
                    (
                        "End-to-End validation result "
                        "is missing or unsupported."
                    ),
                    "End-to-End validation failed.",
                )
            )

        checks.append(
            self._make_check(
                "production_readiness",
                readiness_result,
                (
                    "ready",
                    "passed",
                    "valid",
                    "success",
                    "healthy",
                ),
                "Production readiness passed.",
                (
                    "Production readiness result "
                    "is missing or unsupported."
                ),
                "Production readiness failed.",
            )
        )

        errors: List[str] = []
        warnings: List[str] = []

        for check in checks:
            if not check.passed:
                errors.append(
                    f"{check.name}: {check.message}"
                )

        source_results = (
            deployment_preparation_result,
            end_to_end_validation_result,
            readiness_result,
        )

        for result in source_results:
            errors.extend(
                self._extract_messages(
                    result,
                    "errors",
                )
            )
            warnings.extend(
                self._extract_messages(
                    result,
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

        return ProductionRuntimeIntegrationResult(
            ready_for_runtime=ready,
            runtime_started=False,
            mode=self.runtime.mode,
            checks=checks,
            errors=errors,
            warnings=warnings,
        )

    def start(
        self,
        deployment_preparation_result: Any = None,
        end_to_end_validation_result: Any = None,
        readiness_result: Any = None,
        runner: Any = None,
    ) -> ProductionRuntimeIntegrationResult:
        integration_result = self.preflight(
            deployment_preparation_result=(
                deployment_preparation_result
            ),
            end_to_end_validation_result=(
                end_to_end_validation_result
            ),
            readiness_result=readiness_result,
        )

        if not integration_result.ready_for_runtime:
            return integration_result

        runtime_result = self.runtime.start(
            readiness_result=readiness_result,
            runner=runner,
        )

        integration_result.runtime_result = runtime_result
        integration_result.runtime_started = (
            runtime_result.started
        )

        integration_result.errors.extend(
            runtime_result.errors
        )
        integration_result.warnings.extend(
            runtime_result.warnings
        )

        integration_result.ready_for_runtime = (
            runtime_result.started
        )

        return integration_result

    def stop(self) -> ProductionRuntimeResult:
        return self.runtime.stop()

    @property
    def is_running(self) -> bool:
        return self.runtime.is_running


__all__ = [
    "DeploymentSafetyCheck",
    "ProductionRuntimeIntegrationResult",
    "QuantAIProductionRuntimeIntegration",
]