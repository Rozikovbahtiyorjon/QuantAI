from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from .quantai_production_model_runtime_incident_management import (
    ModelRuntimeIncidentState,
    QuantAIProductionModelRuntimeIncidentManager,
)
from .quantai_production_model_runtime_recovery import (
    ModelRuntimeRecoveryResult,
    QuantAIProductionModelRuntimeRecoveryOrchestrator,
)
from .quantai_production_runtime_lifecycle import (
    QuantAIProductionRuntimeLifecycle,
    RuntimeLifecycleState,
)
from .quantai_production_runtime_supervisor import (
    ProductionRuntimeSupervisorResult,
    QuantAIProductionRuntimeSupervisor,
)


@dataclass(frozen=True)
class RecoveryIntegrationCheck:
    name: str
    passed: bool
    message: str


@dataclass
class ProductionModelRuntimeRecoveryIntegrationResult:
    success: bool
    action: str
    incident_state: Optional[ModelRuntimeIncidentState]
    lifecycle_state: RuntimeLifecycleState
    recovery_state: Any = None
    model_version: Optional[str] = None
    checks: List[RecoveryIntegrationCheck] = field(
        default_factory=list
    )
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    incident_result: Any = None
    recovery_result: Optional[ModelRuntimeRecoveryResult] = None
    supervisor_result: Optional[ProductionRuntimeSupervisorResult] = None

    @property
    def healthy(self) -> bool:
        return self.success and not self.errors

    @property
    def blocked(self) -> bool:
        return not self.success

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


class QuantAIProductionModelRuntimeRecoveryIntegration:
    """
    Coordinates:

        Incident Manager
            ->
        Recovery Orchestrator
            ->
        Runtime Supervisor
            ->
        Runtime Lifecycle

    This module does not implement trading logic.

    Incident errors remain active blocking errors until
    recovery succeeds. After successful recovery or failover,
    the original incident errors are considered resolved
    diagnostics and are not propagated as active fatal errors.

    Errors produced by recovery or post-recovery supervision
    remain active fatal errors.
    """

    def __init__(
        self,
        incident_manager: Optional[
            QuantAIProductionModelRuntimeIncidentManager
        ] = None,
        recovery_orchestrator: Optional[
            QuantAIProductionModelRuntimeRecoveryOrchestrator
        ] = None,
        supervisor: Optional[
            QuantAIProductionRuntimeSupervisor
        ] = None,
        lifecycle: Optional[
            QuantAIProductionRuntimeLifecycle
        ] = None,
    ) -> None:
        if incident_manager is not None and not isinstance(
            incident_manager,
            QuantAIProductionModelRuntimeIncidentManager,
        ):
            raise TypeError(
                "incident_manager must be "
                "QuantAIProductionModelRuntimeIncidentManager "
                "or None."
            )

        if recovery_orchestrator is not None and not isinstance(
            recovery_orchestrator,
            QuantAIProductionModelRuntimeRecoveryOrchestrator,
        ):
            raise TypeError(
                "recovery_orchestrator must be "
                "QuantAIProductionModelRuntimeRecoveryOrchestrator "
                "or None."
            )

        if lifecycle is not None and not isinstance(
            lifecycle,
            QuantAIProductionRuntimeLifecycle,
        ):
            raise TypeError(
                "lifecycle must be "
                "QuantAIProductionRuntimeLifecycle or None."
            )

        if supervisor is not None and not isinstance(
            supervisor,
            QuantAIProductionRuntimeSupervisor,
        ):
            raise TypeError(
                "supervisor must be "
                "QuantAIProductionRuntimeSupervisor or None."
            )

        if supervisor is not None and lifecycle is not None:
            if supervisor.lifecycle is not lifecycle:
                raise ValueError(
                    "supervisor and lifecycle must reference "
                    "the same lifecycle instance."
                )

        if supervisor is not None:
            resolved_lifecycle = supervisor.lifecycle
        elif lifecycle is not None:
            resolved_lifecycle = lifecycle
        else:
            resolved_lifecycle = QuantAIProductionRuntimeLifecycle()

        self.lifecycle = resolved_lifecycle

        self.incident_manager = (
            incident_manager
            if incident_manager is not None
            else QuantAIProductionModelRuntimeIncidentManager()
        )

        self.recovery_orchestrator = (
            recovery_orchestrator
            if recovery_orchestrator is not None
            else QuantAIProductionModelRuntimeRecoveryOrchestrator()
        )

        self.supervisor = (
            supervisor
            if supervisor is not None
            else QuantAIProductionRuntimeSupervisor(
                lifecycle=self.lifecycle
            )
        )

    @staticmethod
    def _messages(
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

    def _base_result(
        self,
        action: str,
        incident_result: Any,
        checks: List[RecoveryIntegrationCheck],
        errors: List[str],
        warnings: List[str],
        recovery_result: Optional[
            ModelRuntimeRecoveryResult
        ] = None,
        supervisor_result: Optional[
            ProductionRuntimeSupervisorResult
        ] = None,
    ) -> ProductionModelRuntimeRecoveryIntegrationResult:
        recovery_state = getattr(
            recovery_result,
            "state",
            None,
        )

        model_version = getattr(
            recovery_result,
            "model_version",
            None,
        )

        return ProductionModelRuntimeRecoveryIntegrationResult(
            success=not errors,
            action=action,
            incident_state=getattr(
                incident_result,
                "state",
                None,
            ),
            lifecycle_state=self.lifecycle.state,
            recovery_state=recovery_state,
            model_version=model_version,
            checks=checks,
            errors=list(errors),
            warnings=list(warnings),
            incident_result=incident_result,
            recovery_result=recovery_result,
            supervisor_result=supervisor_result,
        )

    def coordinate(
        self,
        monitoring_result: Any = None,
        inference_result: Any = None,
        recovery_runner: Optional[
            Callable[[], Any]
        ] = None,
        failover_runner: Optional[
            Callable[[], Any]
        ] = None,
        supervisor_health_checker: Optional[
            Callable[[], Any]
        ] = None,
        target_model_version: Optional[str] = None,
    ) -> ProductionModelRuntimeRecoveryIntegrationResult:
        incident_result = self.incident_manager.evaluate(
            monitoring_result=monitoring_result,
            inference_result=inference_result,
        )

        checks: List[RecoveryIntegrationCheck] = [
            RecoveryIntegrationCheck(
                name="incident_evaluation",
                passed=incident_result is not None,
                message=(
                    "Incident evaluation completed."
                    if incident_result is not None
                    else (
                        "Incident evaluation did not "
                        "produce a result."
                    )
                ),
            )
        ]

        incident_errors = self._messages(
            incident_result,
            "errors",
        )

        warnings = self._messages(
            incident_result,
            "warnings",
        )

        incident_state = getattr(
            incident_result,
            "state",
            None,
        )

        if incident_state == ModelRuntimeIncidentState.HALTED:
            checks.append(
                RecoveryIntegrationCheck(
                    name="incident_safety_gate",
                    passed=False,
                    message=(
                        "Runtime is halted by incident policy; "
                        "recovery and failover are blocked."
                    ),
                )
            )

            return self._base_result(
                action="halt",
                incident_result=incident_result,
                checks=checks,
                errors=incident_errors,
                warnings=warnings,
            )

        checks.append(
            RecoveryIntegrationCheck(
                name="incident_safety_gate",
                passed=True,
                message=(
                    "Incident policy permits coordination."
                ),
            )
        )

        recovery_action = (
            self.recovery_orchestrator.evaluate_action(
                incident_result
            )
        )

        if recovery_action.value == "NO_ACTION":
            checks.append(
                RecoveryIntegrationCheck(
                    name="recovery_policy",
                    passed=True,
                    message=(
                        "No recovery action is required."
                    ),
                )
            )

            if supervisor_health_checker is not None:
                supervisor_result = self.supervisor.check_health(
                    supervisor_health_checker
                )

                supervisor_errors = self._messages(
                    supervisor_result,
                    "errors",
                )

                supervisor_warnings = self._messages(
                    supervisor_result,
                    "warnings",
                )

                errors = (
                    list(incident_errors)
                    + supervisor_errors
                )

                warnings.extend(
                    supervisor_warnings
                )

                checks.append(
                    RecoveryIntegrationCheck(
                        name="supervisor_health",
                        passed=supervisor_result.healthy,
                        message=(
                            "Runtime supervisor health "
                            "check passed."
                            if supervisor_result.healthy
                            else (
                                "Runtime supervisor health "
                                "check failed."
                            )
                        ),
                    )
                )

                return self._base_result(
                    action="supervise",
                    incident_result=incident_result,
                    checks=checks,
                    errors=errors,
                    warnings=warnings,
                    supervisor_result=supervisor_result,
                )

            return self._base_result(
                action="no_action",
                incident_result=incident_result,
                checks=checks,
                errors=incident_errors,
                warnings=warnings,
            )

        checks.append(
            RecoveryIntegrationCheck(
                name="recovery_policy",
                passed=True,
                message=(
                    "Recovery policy selected "
                    f"{recovery_action.value}."
                ),
            )
        )

        recovery_result = (
            self.recovery_orchestrator.recover(
                incident_result=incident_result,
                recovery_runner=recovery_runner,
                failover_runner=failover_runner,
                target_model_version=target_model_version,
            )
        )

        recovery_errors = self._messages(
            recovery_result,
            "errors",
        )

        recovery_warnings = self._messages(
            recovery_result,
            "warnings",
        )

        warnings.extend(
            recovery_warnings
        )

        recovery_success = bool(
            getattr(
                recovery_result,
                "success",
                False,
            )
        )

        checks.append(
            RecoveryIntegrationCheck(
                name="recovery_execution",
                passed=recovery_success,
                message=(
                    "Recovery orchestration completed successfully."
                    if recovery_success
                    else "Recovery orchestration failed."
                ),
            )
        )

        if not recovery_success:
            return self._base_result(
                action="recovery_failed",
                incident_result=incident_result,
                checks=checks,
                errors=recovery_errors,
                warnings=warnings,
                recovery_result=recovery_result,
            )

        checks.append(
            RecoveryIntegrationCheck(
                name="incident_resolution",
                passed=True,
                message=(
                    "Original runtime incident was resolved "
                    "by successful recovery orchestration."
                ),
            )
        )

        if supervisor_health_checker is None:
            checks.append(
                RecoveryIntegrationCheck(
                    name="supervisor_health",
                    passed=True,
                    message=(
                        "Supervisor health check was not requested."
                    ),
                )
            )

            return self._base_result(
                action="recovered",
                incident_result=incident_result,
                checks=checks,
                errors=[],
                warnings=warnings,
                recovery_result=recovery_result,
            )

        supervisor_result = self.supervisor.check_health(
            supervisor_health_checker
        )

        supervisor_errors = self._messages(
            supervisor_result,
            "errors",
        )

        supervisor_warnings = self._messages(
            supervisor_result,
            "warnings",
        )

        warnings.extend(
            supervisor_warnings
        )

        supervisor_success = bool(
            supervisor_result.healthy
        )

        checks.append(
            RecoveryIntegrationCheck(
                name="supervisor_health",
                passed=supervisor_success,
                message=(
                    "Runtime supervisor health "
                    "check passed after recovery."
                    if supervisor_success
                    else (
                        "Runtime supervisor health "
                        "check failed after recovery."
                    )
                ),
            )
        )

        if supervisor_success:
            return self._base_result(
                action="recovered_and_supervised",
                incident_result=incident_result,
                checks=checks,
                errors=[],
                warnings=warnings,
                recovery_result=recovery_result,
                supervisor_result=supervisor_result,
            )

        return self._base_result(
            action="recovery_supervisor_failed",
            incident_result=incident_result,
            checks=checks,
            errors=supervisor_errors,
            warnings=warnings,
            recovery_result=recovery_result,
            supervisor_result=supervisor_result,
        )

    def reset(self) -> None:
        self.recovery_orchestrator.reset()
        self.supervisor.reset_recovery_counter()
        self.lifecycle.reset()


def coordinate_model_runtime_recovery(
    monitoring_result: Any = None,
    inference_result: Any = None,
    recovery_runner: Optional[
        Callable[[], Any]
    ] = None,
    failover_runner: Optional[
        Callable[[], Any]
    ] = None,
    supervisor_health_checker: Optional[
        Callable[[], Any]
    ] = None,
    target_model_version: Optional[str] = None,
) -> ProductionModelRuntimeRecoveryIntegrationResult:
    integration = (
        QuantAIProductionModelRuntimeRecoveryIntegration()
    )

    return integration.coordinate(
        monitoring_result=monitoring_result,
        inference_result=inference_result,
        recovery_runner=recovery_runner,
        failover_runner=failover_runner,
        supervisor_health_checker=supervisor_health_checker,
        target_model_version=target_model_version,
    )


__all__ = [
    "RecoveryIntegrationCheck",
    "ProductionModelRuntimeRecoveryIntegrationResult",
    "QuantAIProductionModelRuntimeRecoveryIntegration",
    "coordinate_model_runtime_recovery",
]