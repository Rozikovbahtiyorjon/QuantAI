from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from .quantai_production_runtime_lifecycle import (
    QuantAIProductionRuntimeLifecycle,
    RuntimeLifecycleState,
)
from .quantai_production_runtime_supervisor import (
    ProductionRuntimeSupervisorResult,
    QuantAIProductionRuntimeSupervisor,
)


@dataclass(frozen=True)
class SupervisorIntegrationCheck:
    name: str
    passed: bool
    message: str


@dataclass
class ProductionRuntimeSupervisorIntegrationResult:
    ready: bool
    action: str
    state: RuntimeLifecycleState
    checks: List[SupervisorIntegrationCheck] = field(
        default_factory=list
    )
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    supervisor_result: Optional[
        ProductionRuntimeSupervisorResult
    ] = None
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

    @property
    def healthy(self) -> bool:
        return self.ready


class QuantAIProductionRuntimeSupervisorIntegration:
    """
    Integration and orchestration layer for the
    Production Runtime Supervisor.

    Responsibilities:

        - validate supervisor configuration
        - execute health supervision
        - coordinate automatic recovery
        - propagate supervisor errors and warnings
        - expose deterministic integration results

    This module does not implement trading logic.
    """

    def __init__(
        self,
        supervisor: Optional[
            QuantAIProductionRuntimeSupervisor
        ] = None,
    ) -> None:
        if supervisor is not None and not isinstance(
            supervisor,
            QuantAIProductionRuntimeSupervisor,
        ):
            raise TypeError(
                "supervisor must be "
                "QuantAIProductionRuntimeSupervisor "
                "or None."
            )

        self.supervisor = (
            supervisor
            if supervisor is not None
            else QuantAIProductionRuntimeSupervisor()
        )

    @property
    def lifecycle(
        self,
    ) -> QuantAIProductionRuntimeLifecycle:
        return self.supervisor.lifecycle

    @property
    def is_running(self) -> bool:
        return self.supervisor.is_running

    @property
    def recovery_attempts(self) -> int:
        return self.supervisor.recovery_attempts

    def _configuration_check(
        self,
    ) -> SupervisorIntegrationCheck:
        lifecycle = self.supervisor.lifecycle

        if not isinstance(
            lifecycle,
            QuantAIProductionRuntimeLifecycle,
        ):
            return SupervisorIntegrationCheck(
                name="supervisor_configuration",
                passed=False,
                message=(
                    "Supervisor lifecycle configuration "
                    "is invalid."
                ),
            )

        return SupervisorIntegrationCheck(
            name="supervisor_configuration",
            passed=True,
            message=(
                "Supervisor configuration is valid."
            ),
        )

    def _state_check(
        self,
        expected_running: bool,
    ) -> SupervisorIntegrationCheck:
        actual_running = self.is_running

        if actual_running == expected_running:
            return SupervisorIntegrationCheck(
                name="runtime_state",
                passed=True,
                message=(
                    "Runtime state matches the "
                    "expected state."
                ),
            )

        return SupervisorIntegrationCheck(
            name="runtime_state",
            passed=False,
            message=(
                "Runtime state does not match "
                "the expected state."
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

    @staticmethod
    def _collect_check_errors(
        checks: List[SupervisorIntegrationCheck],
    ) -> List[str]:
        return [
            f"{check.name}: {check.message}"
            for check in checks
            if not check.passed
        ]

    def preflight(
        self,
        expected_running: bool = True,
    ) -> ProductionRuntimeSupervisorIntegrationResult:
        checks = [
            self._configuration_check(),
            self._state_check(
                expected_running
            ),
        ]

        errors = self._collect_check_errors(
            checks
        )

        return ProductionRuntimeSupervisorIntegrationResult(
            ready=(
                all(
                    check.passed
                    for check in checks
                )
                and not errors
            ),
            action="preflight",
            state=self.supervisor.lifecycle.state,
            checks=checks,
            errors=errors,
        )

    def supervise(
        self,
        health_checker: Callable[[], Any],
        recovery_runner: Optional[
            Callable[[], Any]
        ] = None,
    ) -> ProductionRuntimeSupervisorIntegrationResult:
        if not callable(
            health_checker
        ):
            raise TypeError(
                "health_checker must be callable."
            )

        if (
            recovery_runner is not None
            and not callable(recovery_runner)
        ):
            raise TypeError(
                "recovery_runner must be callable "
                "when provided."
            )

        preflight = self.preflight(
            expected_running=True
        )

        if not preflight.ready:
            return preflight

        supervisor_result = (
            self.supervisor.supervise(
                health_checker=health_checker,
                recovery_runner=recovery_runner,
            )
        )

        checks = [
            SupervisorIntegrationCheck(
                name="supervisor_health",
                passed=supervisor_result.healthy,
                message=(
                    "Supervisor health check passed."
                    if supervisor_result.healthy
                    else (
                        "Supervisor health check "
                        "failed."
                    )
                ),
            )
        ]

        errors = self._collect_check_errors(
            checks
        )

        errors.extend(
            self._extract_messages(
                supervisor_result,
                "errors",
            )
        )

        warnings = self._extract_messages(
            supervisor_result,
            "warnings",
        )

        return ProductionRuntimeSupervisorIntegrationResult(
            ready=(
                supervisor_result.healthy
                and not errors
            ),
            action="supervise",
            state=supervisor_result.state,
            checks=checks,
            errors=errors,
            warnings=warnings,
            supervisor_result=supervisor_result,
            runtime_result=(
                supervisor_result.runtime_result
            ),
        )

    def recover(
        self,
        runner: Callable[[], Any],
    ) -> ProductionRuntimeSupervisorIntegrationResult:
        if not callable(
            runner
        ):
            raise TypeError(
                "runner must be callable."
            )

        supervisor_result = (
            self.supervisor.recover(
                runner
            )
        )

        check = SupervisorIntegrationCheck(
            name="recovery",
            passed=supervisor_result.healthy,
            message=(
                "Runtime recovery succeeded."
                if supervisor_result.healthy
                else "Runtime recovery failed."
            ),
        )

        checks = [check]

        errors = self._collect_check_errors(
            checks
        )

        errors.extend(
            self._extract_messages(
                supervisor_result,
                "errors",
            )
        )

        warnings = self._extract_messages(
            supervisor_result,
            "warnings",
        )

        return ProductionRuntimeSupervisorIntegrationResult(
            ready=(
                supervisor_result.healthy
                and not errors
            ),
            action="recover",
            state=supervisor_result.state,
            checks=checks,
            errors=errors,
            warnings=warnings,
            supervisor_result=supervisor_result,
            runtime_result=(
                supervisor_result.runtime_result
            ),
        )

    def execute(
        self,
        health_checker: Callable[[], Any],
        recovery_runner: Optional[
            Callable[[], Any]
        ] = None,
    ) -> ProductionRuntimeSupervisorIntegrationResult:
        return self.supervise(
            health_checker=health_checker,
            recovery_runner=recovery_runner,
        )

    def reset_recovery_counter(self) -> None:
        self.supervisor.reset_recovery_counter()


def supervise_production_runtime(
    health_checker: Callable[[], Any],
    recovery_runner: Optional[
        Callable[[], Any]
    ] = None,
    supervisor: Optional[
        QuantAIProductionRuntimeSupervisor
    ] = None,
) -> ProductionRuntimeSupervisorIntegrationResult:
    integration = (
        QuantAIProductionRuntimeSupervisorIntegration(
            supervisor=supervisor
        )
    )

    return integration.supervise(
        health_checker=health_checker,
        recovery_runner=recovery_runner,
    )


__all__ = [
    "SupervisorIntegrationCheck",
    "ProductionRuntimeSupervisorIntegrationResult",
    "QuantAIProductionRuntimeSupervisorIntegration",
    "supervise_production_runtime",
]