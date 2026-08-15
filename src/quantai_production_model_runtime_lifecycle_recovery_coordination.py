from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from .quantai_production_model_runtime_recovery_integration import (
    ProductionModelRuntimeRecoveryIntegrationResult,
    QuantAIProductionModelRuntimeRecoveryIntegration,
)
from .quantai_production_runtime_lifecycle import (
    QuantAIProductionRuntimeLifecycle,
    RuntimeLifecycleState,
)


@dataclass(frozen=True)
class LifecycleRecoveryCoordinationCheck:
    name: str
    passed: bool
    message: str


@dataclass
class ProductionModelRuntimeLifecycleRecoveryCoordinationResult:
    success: bool
    action: str
    lifecycle_state: RuntimeLifecycleState
    runtime_result: Any = None
    integration_result: Optional[
        ProductionModelRuntimeRecoveryIntegrationResult
    ] = None
    checks: List[LifecycleRecoveryCoordinationCheck] = field(
        default_factory=list
    )
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def healthy(self) -> bool:
        return self.success and not self.errors

    @property
    def blocked(self) -> bool:
        return not self.success

    @property
    def checks_passed(self) -> int:
        return sum(check.passed for check in self.checks)

    @property
    def checks_failed(self) -> int:
        return sum(not check.passed for check in self.checks)

    @property
    def total_checks(self) -> int:
        return len(self.checks)


class QuantAIProductionModelRuntimeLifecycleRecoveryCoordination:
    """
    Coordinates recovery integration with the runtime lifecycle.

    The coordination layer owns lifecycle transitions only.
    Recovery and failover execution remain delegated to
    QuantAIProductionModelRuntimeRecoveryIntegration.

    Recovery policy:

    - Healthy runtime -> NO_ACTION.
    - An explicitly supplied recovery runner is preferred for
      normal recovery when no failover runner is supplied.
    - An explicitly supplied failover runner is used for failover.
    - Recovery/failover runners are executed exactly once.
    - Successful recovery/failover may start a stopped lifecycle.
    - Failed recovery/failover never starts the lifecycle.
    - HALTED recovery never starts the lifecycle.
    - The exact runner return value is exposed through runtime_result.
    - Recovery integration errors are preserved.
    - reset() restores recovery integration and lifecycle state.
    """

    def __init__(
        self,
        recovery_integration: Optional[
            QuantAIProductionModelRuntimeRecoveryIntegration
        ] = None,
        lifecycle: Optional[
            QuantAIProductionRuntimeLifecycle
        ] = None,
        integration: Optional[
            QuantAIProductionModelRuntimeRecoveryIntegration
        ] = None,
    ) -> None:
        if (
            recovery_integration is not None
            and integration is not None
            and recovery_integration is not integration
        ):
            raise ValueError(
                "recovery_integration and integration must "
                "reference the same integration instance."
            )

        resolved_integration = (
            recovery_integration
            if recovery_integration is not None
            else integration
        )

        if resolved_integration is not None and not isinstance(
            resolved_integration,
            QuantAIProductionModelRuntimeRecoveryIntegration,
        ):
            raise TypeError(
                "recovery_integration must be "
                "QuantAIProductionModelRuntimeRecoveryIntegration "
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

        if (
            resolved_integration is not None
            and lifecycle is not None
            and resolved_integration.lifecycle is not lifecycle
        ):
            raise ValueError(
                "recovery_integration and lifecycle must "
                "reference the same lifecycle instance."
            )

        if resolved_integration is not None:
            resolved_lifecycle = resolved_integration.lifecycle
        elif lifecycle is not None:
            resolved_lifecycle = lifecycle
        else:
            resolved_lifecycle = QuantAIProductionRuntimeLifecycle()

        self.lifecycle = resolved_lifecycle

        self.recovery_integration = (
            resolved_integration
            if resolved_integration is not None
            else QuantAIProductionModelRuntimeRecoveryIntegration(
                lifecycle=self.lifecycle
            )
        )

        self.integration = self.recovery_integration

    @staticmethod
    def _messages(
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

    @staticmethod
    def _extract_runtime_result(
        integration_result: Any,
        captured_runtime_result: Any,
    ) -> Any:
        if captured_runtime_result is not None:
            return captured_runtime_result

        if integration_result is None:
            return None

        direct = getattr(
            integration_result,
            "runtime_result",
            None,
        )

        if direct is not None:
            return direct

        recovery_result = getattr(
            integration_result,
            "recovery_result",
            None,
        )

        if recovery_result is None:
            return None

        for attribute in (
            "runtime_result",
            "runtime",
            "result",
            "execution_result",
            "recovery_runtime",
            "failover_runtime",
        ):
            value = getattr(
                recovery_result,
                attribute,
                None,
            )

            if value is not None:
                return value

        return None

    @staticmethod
    def _extract_runner_error(
        runner_error: Optional[BaseException],
    ) -> List[str]:
        if runner_error is None:
            return []

        message = str(runner_error)

        if message:
            return [message]

        return [runner_error.__class__.__name__]

    def _make_result(
        self,
        success: bool,
        action: str,
        integration_result: Optional[
            ProductionModelRuntimeRecoveryIntegrationResult
        ],
        checks: List[LifecycleRecoveryCoordinationCheck],
        errors: List[str],
        warnings: List[str],
        runtime_result: Any = None,
    ) -> ProductionModelRuntimeLifecycleRecoveryCoordinationResult:
        return ProductionModelRuntimeLifecycleRecoveryCoordinationResult(
            success=success,
            action=action,
            lifecycle_state=self.lifecycle.state,
            runtime_result=runtime_result,
            integration_result=integration_result,
            checks=checks,
            errors=errors,
            warnings=warnings,
        )

    def _start_lifecycle(
        self,
        runtime_result: Any,
    ) -> tuple[bool, Optional[str]]:
        if self.lifecycle.state == RuntimeLifecycleState.RUNNING:
            return True, None

        if self.lifecycle.state != RuntimeLifecycleState.STOPPED:
            return (
                False,
                (
                    "Runtime lifecycle cannot be started "
                    f"from state {self.lifecycle.state.value}."
                ),
            )

        try:
            self.lifecycle.start(lambda: runtime_result)
        except Exception as exc:
            return False, f"lifecycle start failed: {exc}"

        if self.lifecycle.state != RuntimeLifecycleState.RUNNING:
            return (
                False,
                "Runtime lifecycle start did not reach RUNNING state.",
            )

        return True, None

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
    ) -> ProductionModelRuntimeLifecycleRecoveryCoordinationResult:
        captured_runtime_result: Any = None
        runner_error: Optional[BaseException] = None

        def wrapped_recovery_runner() -> Any:
            nonlocal captured_runtime_result
            nonlocal runner_error

            if recovery_runner is None:
                return None

            try:
                captured_runtime_result = recovery_runner()
                return captured_runtime_result
            except BaseException as exc:
                runner_error = exc
                raise

        def wrapped_failover_runner() -> Any:
            nonlocal captured_runtime_result
            nonlocal runner_error

            if failover_runner is None:
                return None

            try:
                captured_runtime_result = failover_runner()
                return captured_runtime_result
            except BaseException as exc:
                runner_error = exc
                raise

        orchestrator = getattr(
            self.recovery_integration,
            "recovery_orchestrator",
            None,
        )

        original_allow_failover: Optional[bool] = None

        prefer_recovery = (
            recovery_runner is not None
            and failover_runner is None
            and orchestrator is not None
            and hasattr(orchestrator, "allow_failover")
        )

        if prefer_recovery:
            original_allow_failover = bool(
                orchestrator.allow_failover
            )
            orchestrator.allow_failover = False

        try:
            integration_result = self.recovery_integration.coordinate(
                monitoring_result=monitoring_result,
                inference_result=inference_result,
                recovery_runner=(
                    wrapped_recovery_runner
                    if recovery_runner is not None
                    else None
                ),
                failover_runner=(
                    wrapped_failover_runner
                    if failover_runner is not None
                    else None
                ),
                supervisor_health_checker=supervisor_health_checker,
                target_model_version=target_model_version,
            )
        finally:
            if (
                prefer_recovery
                and original_allow_failover is not None
            ):
                orchestrator.allow_failover = (
                    original_allow_failover
                )

        checks: List[
            LifecycleRecoveryCoordinationCheck
        ] = [
            LifecycleRecoveryCoordinationCheck(
                name="recovery_integration",
                passed=integration_result is not None,
                message=(
                    "Recovery integration completed."
                    if integration_result is not None
                    else (
                        "Recovery integration did not "
                        "produce a result."
                    )
                ),
            )
        ]

        if integration_result is None:
            checks.append(
                LifecycleRecoveryCoordinationCheck(
                    name="recovery_gate",
                    passed=False,
                    message=(
                        "Lifecycle transition is blocked because "
                        "recovery integration returned no result."
                    ),
                )
            )

            errors = [
                "Recovery integration returned no result."
            ]
            errors.extend(
                self._extract_runner_error(runner_error)
            )

            return self._make_result(
                success=False,
                action="blocked",
                integration_result=None,
                checks=checks,
                errors=errors,
                warnings=[],
                runtime_result=captured_runtime_result,
            )

        integration_errors = self._messages(
            integration_result,
            "errors",
        )
        integration_warnings = self._messages(
            integration_result,
            "warnings",
        )
        runner_errors = self._extract_runner_error(
            runner_error
        )

        runtime_result = self._extract_runtime_result(
            integration_result,
            captured_runtime_result,
        )

        if (
            integration_result.success
            and integration_result.action == "no_action"
        ):
            checks.append(
                LifecycleRecoveryCoordinationCheck(
                    name="lifecycle_transition",
                    passed=True,
                    message=(
                        "Healthy runtime requires no "
                        "lifecycle transition."
                    ),
                )
            )

            return self._make_result(
                success=True,
                action="no_action",
                integration_result=integration_result,
                checks=checks,
                errors=[],
                warnings=integration_warnings,
                runtime_result=runtime_result,
            )

        if not integration_result.success:
            checks.append(
                LifecycleRecoveryCoordinationCheck(
                    name="recovery_gate",
                    passed=False,
                    message=(
                        "Recovery integration failed; "
                        "lifecycle transition is blocked."
                    ),
                )
            )

            errors = list(integration_errors)

            for message in runner_errors:
                if message not in errors:
                    errors.append(message)

            return self._make_result(
                success=False,
                action="blocked",
                integration_result=integration_result,
                checks=checks,
                errors=errors,
                warnings=integration_warnings,
                runtime_result=runtime_result,
            )

        checks.append(
            LifecycleRecoveryCoordinationCheck(
                name="recovery_gate",
                passed=True,
                message=(
                    "Recovery integration completed successfully."
                ),
            )
        )

        if self.lifecycle.state == RuntimeLifecycleState.RUNNING:
            checks.append(
                LifecycleRecoveryCoordinationCheck(
                    name="lifecycle_transition",
                    passed=True,
                    message="Runtime lifecycle is already running.",
                )
            )

            return self._make_result(
                success=True,
                action="recovered_and_running",
                integration_result=integration_result,
                checks=checks,
                errors=[],
                warnings=integration_warnings,
                runtime_result=runtime_result,
            )

        started, lifecycle_error = self._start_lifecycle(
            runtime_result
        )

        checks.append(
            LifecycleRecoveryCoordinationCheck(
                name="lifecycle_transition",
                passed=started,
                message=(
                    "Runtime lifecycle started successfully."
                    if started
                    else "Runtime lifecycle start failed."
                ),
            )
        )

        if not started:
            errors = list(integration_errors)

            for message in runner_errors:
                if message not in errors:
                    errors.append(message)

            if lifecycle_error:
                errors.append(lifecycle_error)

            return self._make_result(
                success=False,
                action="blocked",
                integration_result=integration_result,
                checks=checks,
                errors=errors,
                warnings=integration_warnings,
                runtime_result=runtime_result,
            )

        return self._make_result(
            success=True,
            action="recovered_and_running",
            integration_result=integration_result,
            checks=checks,
            errors=[],
            warnings=integration_warnings,
            runtime_result=runtime_result,
        )

    def reset(self) -> None:
        self.recovery_integration.reset()

        if self.lifecycle.state != RuntimeLifecycleState.STOPPED:
            try:
                self.lifecycle.stop()
            except Exception:
                pass

        self.lifecycle.reset()


def coordinate_model_runtime_lifecycle_recovery(
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
) -> ProductionModelRuntimeLifecycleRecoveryCoordinationResult:
    coordinator = (
        QuantAIProductionModelRuntimeLifecycleRecoveryCoordination()
    )

    return coordinator.coordinate(
        monitoring_result=monitoring_result,
        inference_result=inference_result,
        recovery_runner=recovery_runner,
        failover_runner=failover_runner,
        supervisor_health_checker=supervisor_health_checker,
        target_model_version=target_model_version,
    )


__all__ = [
    "LifecycleRecoveryCoordinationCheck",
    "ProductionModelRuntimeLifecycleRecoveryCoordinationResult",
    "QuantAIProductionModelRuntimeLifecycleRecoveryCoordination",
    "coordinate_model_runtime_lifecycle_recovery",
]