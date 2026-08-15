from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional


class ModelRuntimeIncidentState(str, Enum):
    NORMAL = "NORMAL"
    DEGRADED = "DEGRADED"
    RECOVERY_REQUIRED = "RECOVERY_REQUIRED"
    HALTED = "HALTED"


@dataclass(frozen=True)
class ModelRuntimeIncidentCheck:
    name: str
    passed: bool
    message: str


@dataclass
class ModelRuntimeIncidentResult:
    state: ModelRuntimeIncidentState
    allow_inference: bool
    fallback_allowed: bool
    recovery_required: bool
    checks: List[ModelRuntimeIncidentCheck] = field(
        default_factory=list
    )
    errors: List[str] = field(
        default_factory=list
    )
    warnings: List[str] = field(
        default_factory=list
    )

    @property
    def healthy(self) -> bool:
        return self.state == ModelRuntimeIncidentState.NORMAL

    @property
    def blocked(self) -> bool:
        return not self.allow_inference

    @property
    def total_checks(self) -> int:
        return len(self.checks)

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


class QuantAIProductionModelRuntimeIncidentManager:
    """
    Deterministic incident and degradation policy layer
    for production model runtime.

    This module does not execute trading logic.

    Responsibilities:

        - evaluate model runtime health
        - classify runtime incidents
        - control inference permission
        - determine fallback eligibility
        - determine recovery requirement
        - preserve explicit incident reasons
    """

    def __init__(
        self,
        allow_degraded_inference: bool = False,
        allow_fallback: bool = True,
        require_recovery_on_health_failure: bool = True,
    ) -> None:
        self.allow_degraded_inference = bool(
            allow_degraded_inference
        )
        self.allow_fallback = bool(
            allow_fallback
        )
        self.require_recovery_on_health_failure = bool(
            require_recovery_on_health_failure
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
                value = getattr(
                    result,
                    attribute,
                )

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

        value = getattr(
            result,
            attribute,
            None,
        )

        if value is None:
            return []

        if isinstance(value, str):
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
    ) -> ModelRuntimeIncidentCheck:
        passed = cls._extract_boolean(
            result,
            attributes,
        )

        if passed is None:
            return ModelRuntimeIncidentCheck(
                name=name,
                passed=False,
                message=missing_message,
            )

        if passed:
            return ModelRuntimeIncidentCheck(
                name=name,
                passed=True,
                message=success_message,
            )

        return ModelRuntimeIncidentCheck(
            name=name,
            passed=False,
            message=failure_message,
        )

    def evaluate(
        self,
        monitoring_result: Any = None,
        inference_result: Any = None,
    ) -> ModelRuntimeIncidentResult:
        checks: List[ModelRuntimeIncidentCheck] = []
        errors: List[str] = []
        warnings: List[str] = []

        health_check = self._make_check(
            name="model_runtime_health",
            result=monitoring_result,
            attributes=(
                "healthy",
                "passed",
                "valid",
                "ready",
            ),
            success_message=(
                "Model runtime health check passed."
            ),
            missing_message=(
                "Model runtime health result "
                "was not provided or has no "
                "supported status."
            ),
            failure_message=(
                "Model runtime health check failed."
            ),
        )

        checks.append(health_check)

        inference_check = self._make_check(
            name="inference_health",
            result=inference_result,
            attributes=(
                "healthy",
                "passed",
                "valid",
                "success",
            ),
            success_message=(
                "Inference health check passed."
            ),
            missing_message=(
                "Inference health result "
                "was not provided or has no "
                "supported status."
            ),
            failure_message=(
                "Inference health check failed."
            ),
        )

        checks.append(inference_check)

        for check in checks:
            if not check.passed:
                errors.append(
                    f"{check.name}: {check.message}"
                )

        for source_result in (
            monitoring_result,
            inference_result,
        ):
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

        health_failed = not health_check.passed
        inference_failed = not inference_check.passed

        critical_source_error = bool(
            self._extract_messages(
                monitoring_result,
                "errors",
            )
            or self._extract_messages(
                inference_result,
                "errors",
            )
        )

        if critical_source_error:
            state = ModelRuntimeIncidentState.HALTED
            allow_inference = False
            fallback_allowed = False
            recovery_required = True

            warnings.append(
                "Critical runtime incident detected. "
                "Inference and fallback are blocked."
            )

        elif health_failed:
            state = (
                ModelRuntimeIncidentState.RECOVERY_REQUIRED
                if self.require_recovery_on_health_failure
                else ModelRuntimeIncidentState.DEGRADED
            )

            allow_inference = (
                self.allow_degraded_inference
                and not inference_failed
            )

            fallback_allowed = self.allow_fallback
            recovery_required = (
                self.require_recovery_on_health_failure
            )

        elif inference_failed:
            state = ModelRuntimeIncidentState.DEGRADED

            allow_inference = (
                self.allow_degraded_inference
            )

            fallback_allowed = self.allow_fallback
            recovery_required = False

        else:
            state = ModelRuntimeIncidentState.NORMAL
            allow_inference = True
            fallback_allowed = False
            recovery_required = False

        if state == ModelRuntimeIncidentState.RECOVERY_REQUIRED:
            warnings.append(
                "Model runtime recovery is required."
            )

        if state == ModelRuntimeIncidentState.DEGRADED:
            warnings.append(
                "Model runtime is operating in degraded state."
            )

        if (
            not allow_inference
            and state != ModelRuntimeIncidentState.HALTED
        ):
            warnings.append(
                "Model inference is blocked by runtime safety policy."
            )

        if state == ModelRuntimeIncidentState.HALTED:
            allow_inference = False
            fallback_allowed = False
            recovery_required = True

        return ModelRuntimeIncidentResult(
            state=state,
            allow_inference=allow_inference,
            fallback_allowed=fallback_allowed,
            recovery_required=recovery_required,
            checks=checks,
            errors=errors,
            warnings=warnings,
        )

    def is_safe(
        self,
        monitoring_result: Any = None,
        inference_result: Any = None,
    ) -> bool:
        result = self.evaluate(
            monitoring_result=monitoring_result,
            inference_result=inference_result,
        )

        return result.allow_inference


def evaluate_model_runtime_incident(
    monitoring_result: Any = None,
    inference_result: Any = None,
) -> ModelRuntimeIncidentResult:
    manager = (
        QuantAIProductionModelRuntimeIncidentManager()
    )

    return manager.evaluate(
        monitoring_result=monitoring_result,
        inference_result=inference_result,
    )


__all__ = [
    "ModelRuntimeIncidentState",
    "ModelRuntimeIncidentCheck",
    "ModelRuntimeIncidentResult",
    "QuantAIProductionModelRuntimeIncidentManager",
    "evaluate_model_runtime_incident",
]