from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from .quantai_production_runtime_lifecycle import (
    QuantAIProductionRuntimeLifecycle,
    RuntimeLifecycleState,
)


@dataclass(frozen=True)
class SupervisorCheck:
    name: str
    passed: bool
    message: str


@dataclass
class ProductionRuntimeSupervisorResult:
    healthy: bool
    action: str
    state: RuntimeLifecycleState
    checks: List[SupervisorCheck] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    runtime_result: Any = None

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


class QuantAIProductionRuntimeSupervisor:
    """
    Deterministic runtime supervisor.

    Responsibilities:

        - runtime health checks
        - lifecycle state verification
        - controlled recovery
        - recovery attempt limiting
        - safe runtime supervision

    This module does not implement trading logic.
    """

    def __init__(
        self,
        lifecycle: Optional[
            QuantAIProductionRuntimeLifecycle
        ] = None,
        max_recovery_attempts: int = 1,
    ) -> None:
        if lifecycle is not None and not isinstance(
            lifecycle,
            QuantAIProductionRuntimeLifecycle,
        ):
            raise TypeError(
                "lifecycle must be QuantAIProductionRuntimeLifecycle or None."
            )

        if not isinstance(
            max_recovery_attempts,
            int,
        ):
            raise TypeError(
                "max_recovery_attempts must be an integer."
            )

        if max_recovery_attempts < 0:
            raise ValueError(
                "max_recovery_attempts must be non-negative."
            )

        self.lifecycle = (
            lifecycle
            if lifecycle is not None
            else QuantAIProductionRuntimeLifecycle()
        )

        self.max_recovery_attempts = (
            max_recovery_attempts
        )

        self._recovery_attempts = 0

    @property
    def recovery_attempts(self) -> int:
        return self._recovery_attempts

    @property
    def is_running(self) -> bool:
        return self.lifecycle.is_running

    def _check_lifecycle_state(
        self,
    ) -> SupervisorCheck:
        passed = (
            self.lifecycle.state
            == RuntimeLifecycleState.RUNNING
        )

        return SupervisorCheck(
            name="runtime_state",
            passed=passed,
            message=(
                "Runtime is running."
                if passed
                else (
                    "Runtime is not running: "
                    f"{self.lifecycle.state.value}."
                )
            ),
        )

    @staticmethod
    def _check_health_result(
        health_result: Any,
    ) -> SupervisorCheck:
        if health_result is None:
            return SupervisorCheck(
                name="health_check",
                passed=False,
                message=(
                    "Health check result "
                    "was not provided."
                ),
            )

        for attribute in (
            "healthy",
            "ready",
            "passed",
            "valid",
            "success",
        ):
            if hasattr(
                health_result,
                attribute,
            ):
                value = getattr(
                    health_result,
                    attribute,
                )

                if isinstance(
                    value,
                    bool,
                ):
                    return SupervisorCheck(
                        name="health_check",
                        passed=value,
                        message=(
                            "Runtime health check passed."
                            if value
                            else (
                                "Runtime health check failed."
                            )
                        ),
                    )

        return SupervisorCheck(
            name="health_check",
            passed=False,
            message=(
                "Health check result does not expose "
                "a supported boolean status."
            ),
        )

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

    def check_health(
        self,
        health_checker: Callable[[], Any],
    ) -> ProductionRuntimeSupervisorResult:
        if not callable(
            health_checker
        ):
            raise TypeError(
                "health_checker must be callable."
            )

        checks: List[
            SupervisorCheck
        ] = []

        errors: List[str] = []

        warnings: List[str] = []

        state_check = (
            self._check_lifecycle_state()
        )

        checks.append(
            state_check
        )

        try:
            health_result = health_checker()

        except Exception as exc:
            message = (
                "Health check execution failed: "
                f"{type(exc).__name__}: {exc}"
            )

            errors.append(
                message
            )

            checks.append(
                SupervisorCheck(
                    name="health_check",
                    passed=False,
                    message=message,
                )
            )

            return ProductionRuntimeSupervisorResult(
                healthy=False,
                action="health_check",
                state=self.lifecycle.state,
                checks=checks,
                errors=errors,
                warnings=warnings,
            )

        health_check = (
            self._check_health_result(
                health_result
            )
        )

        checks.append(
            health_check
        )

        errors.extend(
            self._extract_messages(
                health_result,
                "errors",
            )
        )

        warnings.extend(
            self._extract_messages(
                health_result,
                "warnings",
            )
        )

        healthy = (
            all(
                check.passed
                for check in checks
            )
            and not errors
        )

        return ProductionRuntimeSupervisorResult(
            healthy=healthy,
            action="health_check",
            state=self.lifecycle.state,
            checks=checks,
            errors=errors,
            warnings=warnings,
        )

    def recover(
        self,
        runner: Callable[[], Any],
    ) -> ProductionRuntimeSupervisorResult:
        if not callable(
            runner
        ):
            raise TypeError(
                "runner must be callable."
            )

        if (
            self._recovery_attempts
            >= self.max_recovery_attempts
        ):
            return ProductionRuntimeSupervisorResult(
                healthy=False,
                action="recover",
                state=self.lifecycle.state,
                errors=[
                    "Maximum recovery attempts exceeded."
                ],
            )

        if self.lifecycle.state not in (
            RuntimeLifecycleState.FAILED,
            RuntimeLifecycleState.STOPPED,
        ):
            return ProductionRuntimeSupervisorResult(
                healthy=False,
                action="recover",
                state=self.lifecycle.state,
                errors=[
                    (
                        "Recovery is not allowed from "
                        f"state {self.lifecycle.state.value}."
                    )
                ],
            )

        self._recovery_attempts += 1

        result = self.lifecycle.recover(
            runner
        )

        return ProductionRuntimeSupervisorResult(
            healthy=result.success,
            action="recover",
            state=result.state,
            errors=list(result.errors),
            warnings=list(result.warnings),
            runtime_result=result.runtime_result,
        )

    def supervise(
        self,
        health_checker: Callable[[], Any],
        recovery_runner: Optional[
            Callable[[], Any]
        ] = None,
    ) -> ProductionRuntimeSupervisorResult:
        health_result = self.check_health(
            health_checker
        )

        if health_result.healthy:
            return health_result

        if recovery_runner is None:
            return health_result

        recovery_result = self.recover(
            recovery_runner
        )

        combined_errors = (
            list(health_result.errors)
            + list(recovery_result.errors)
        )

        combined_warnings = (
            list(health_result.warnings)
            + list(recovery_result.warnings)
        )

        return ProductionRuntimeSupervisorResult(
            healthy=recovery_result.healthy,
            action="supervise",
            state=recovery_result.state,
            checks=health_result.checks,
            errors=combined_errors,
            warnings=combined_warnings,
            runtime_result=recovery_result.runtime_result,
        )

    def reset_recovery_counter(self) -> None:
        self._recovery_attempts = 0


__all__ = [
    "SupervisorCheck",
    "ProductionRuntimeSupervisorResult",
    "QuantAIProductionRuntimeSupervisor",
]
