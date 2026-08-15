from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

from src.quantai_production_runtime import ProductionRuntimeResult
from src.quantai_production_runtime_integration import (
    ProductionRuntimeIntegrationResult,
    QuantAIProductionRuntimeIntegration,
)


@dataclass(frozen=True)
class StartupStep:
    name: str
    passed: bool
    message: str


@dataclass
class SafeStartupResult:
    started: bool
    startup_aborted: bool
    steps: List[StartupStep] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    integration_result: Optional[
        ProductionRuntimeIntegrationResult
    ] = None
    runtime_result: Optional[ProductionRuntimeResult] = None

    @property
    def steps_passed(self) -> int:
        return sum(
            step.passed
            for step in self.steps
        )

    @property
    def steps_failed(self) -> int:
        return sum(
            not step.passed
            for step in self.steps
        )

    @property
    def total_steps(self) -> int:
        return len(self.steps)


class QuantAIProductionSafeStartupController:
    """
    Deterministic safe-start orchestration for the
    QuantAI production runtime.

    The controller does not implement trading logic.

    Startup order:

        1. Deployment Preparation
        2. End-to-End Validation
        3. Production Readiness
        4. Production Runtime Integration
        5. Runtime Start

    A failed prerequisite immediately aborts startup.
    """

    def __init__(
        self,
        integration: QuantAIProductionRuntimeIntegration,
    ) -> None:

        if not isinstance(
            integration,
            QuantAIProductionRuntimeIntegration,
        ):
            raise TypeError(
                "integration must be a "
                "QuantAIProductionRuntimeIntegration "
                "instance."
            )

        self.integration = integration

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

    @classmethod
    def _validation_step(
        cls,
        name: str,
        result: Any,
        attributes: tuple[str, ...],
        success_message: str,
        missing_message: str,
        failure_message: str,
    ) -> StartupStep:

        passed = cls._extract_boolean(
            result,
            attributes,
        )

        if passed is None:

            return StartupStep(
                name=name,
                passed=False,
                message=missing_message,
            )

        if passed:

            return StartupStep(
                name=name,
                passed=True,
                message=success_message,
            )

        return StartupStep(
            name=name,
            passed=False,
            message=failure_message,
        )

    def start(
        self,
        deployment_preparation_result: Any = None,
        end_to_end_validation_result: Any = None,
        readiness_result: Any = None,
        runner: Any = None,
    ) -> SafeStartupResult:

        steps: List[StartupStep] = []
        errors: List[str] = []
        warnings: List[str] = []

        deployment_step = self._validation_step(
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

        steps.append(
            deployment_step
        )

        if not deployment_step.passed:

            errors.append(
                f"{deployment_step.name}: "
                f"{deployment_step.message}"
            )

            errors.extend(
                self._extract_messages(
                    deployment_preparation_result,
                    "errors",
                )
            )

            warnings.extend(
                self._extract_messages(
                    deployment_preparation_result,
                    "warnings",
                )
            )

            return SafeStartupResult(
                started=False,
                startup_aborted=True,
                steps=steps,
                errors=errors,
                warnings=warnings,
            )

        end_to_end_step = self._validation_step(
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

        steps.append(
            end_to_end_step
        )

        if not end_to_end_step.passed:

            errors.append(
                f"{end_to_end_step.name}: "
                f"{end_to_end_step.message}"
            )

            errors.extend(
                self._extract_messages(
                    end_to_end_validation_result,
                    "errors",
                )
            )

            warnings.extend(
                self._extract_messages(
                    end_to_end_validation_result,
                    "warnings",
                )
            )

            return SafeStartupResult(
                started=False,
                startup_aborted=True,
                steps=steps,
                errors=errors,
                warnings=warnings,
            )

        readiness_step = self._validation_step(
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

        steps.append(
            readiness_step
        )

        if not readiness_step.passed:

            errors.append(
                f"{readiness_step.name}: "
                f"{readiness_step.message}"
            )

            errors.extend(
                self._extract_messages(
                    readiness_result,
                    "errors",
                )
            )

            warnings.extend(
                self._extract_messages(
                    readiness_result,
                    "warnings",
                )
            )

            return SafeStartupResult(
                started=False,
                startup_aborted=True,
                steps=steps,
                errors=errors,
                warnings=warnings,
            )

        integration_result = self.integration.start(
            deployment_preparation_result=(
                deployment_preparation_result
            ),
            end_to_end_validation_result=(
                end_to_end_validation_result
            ),
            readiness_result=readiness_result,
            runner=runner,
        )

        steps.append(
            StartupStep(
                name="runtime_integration",
                passed=(
                    integration_result.ready_for_runtime
                ),
                message=(
                    "Production Runtime Integration "
                    "passed."
                    if integration_result.ready_for_runtime
                    else
                    "Production Runtime Integration "
                    "failed."
                ),
            )
        )

        errors.extend(
            integration_result.errors
        )

        warnings.extend(
            integration_result.warnings
        )

        if not integration_result.runtime_started:

            return SafeStartupResult(
                started=False,
                startup_aborted=True,
                steps=steps,
                errors=errors,
                warnings=warnings,
                integration_result=integration_result,
                runtime_result=(
                    integration_result.runtime_result
                ),
            )

        steps.append(
            StartupStep(
                name="runtime_started",
                passed=True,
                message=(
                    "Production Runtime started safely."
                ),
            )
        )

        return SafeStartupResult(
            started=True,
            startup_aborted=False,
            steps=steps,
            errors=errors,
            warnings=warnings,
            integration_result=integration_result,
            runtime_result=(
                integration_result.runtime_result
            ),
        )

    def stop(self) -> ProductionRuntimeResult:
        return self.integration.stop()

    @property
    def is_running(self) -> bool:
        return self.integration.is_running


__all__ = [
    "StartupStep",
    "SafeStartupResult",
    "QuantAIProductionSafeStartupController",
]