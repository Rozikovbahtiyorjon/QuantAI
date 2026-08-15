from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, List, Optional


class RecoveryAction(str, Enum):
    NO_ACTION = "NO_ACTION"
    RECOVER = "RECOVER"
    FAILOVER = "FAILOVER"
    HALT = "HALT"


class RecoveryState(str, Enum):
    IDLE = "IDLE"
    RECOVERING = "RECOVERING"
    FAILED_OVER = "FAILED_OVER"
    HALTED = "HALTED"


@dataclass(frozen=True)
class RecoveryCheck:
    name: str
    passed: bool
    message: str


@dataclass
class ModelRuntimeRecoveryResult:
    success: bool
    state: RecoveryState
    action: RecoveryAction
    model_version: Optional[str] = None
    runtime_result: Any = None
    checks: List[RecoveryCheck] = field(
        default_factory=list
    )
    errors: List[str] = field(
        default_factory=list
    )
    warnings: List[str] = field(
        default_factory=list
    )

    @property
    def recovered(self) -> bool:
        return (
            self.success
            and self.state
            in (
                RecoveryState.RECOVERING,
                RecoveryState.FAILED_OVER,
            )
        )

    @property
    def halted(self) -> bool:
        return self.state == RecoveryState.HALTED

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


class QuantAIProductionModelRuntimeRecoveryOrchestrator:
    """
    Deterministic recovery and safe model failover policy layer.

    Responsibilities:

        - evaluate incident state
        - determine recovery action
        - execute safe runtime recovery
        - execute controlled model failover
        - block unsafe recovery
        - transition to HALTED on recovery failure
        - preserve incident errors and warnings

    This module does not implement trading logic.
    """

    def __init__(
        self,
        allow_failover: bool = True,
    ) -> None:
        if not isinstance(
            allow_failover,
            bool,
        ):
            raise TypeError(
                "allow_failover must be a boolean."
            )

        self.allow_failover = allow_failover
        self._state = RecoveryState.IDLE
        self._active_model_version: Optional[str] = None

    @property
    def state(self) -> RecoveryState:
        return self._state

    @property
    def active_model_version(
        self,
    ) -> Optional[str]:
        return self._active_model_version

    @staticmethod
    def _status(
        result: Any,
        names: tuple[str, ...],
    ) -> Optional[bool]:
        if result is None:
            return None

        for name in names:
            value = getattr(
                result,
                name,
                None,
            )

            if isinstance(value, bool):
                return value

        return None

    @staticmethod
    def _messages(
        result: Any,
        name: str,
    ) -> List[str]:
        if result is None:
            return []

        value = getattr(
            result,
            name,
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

    def evaluate_action(
        self,
        incident_result: Any,
    ) -> RecoveryAction:
        state = getattr(
            incident_result,
            "state",
            None,
        )

        allow_inference = self._status(
            incident_result,
            ("allow_inference",),
        )

        fallback_allowed = self._status(
            incident_result,
            ("fallback_allowed",),
        )

        recovery_required = self._status(
            incident_result,
            ("recovery_required",),
        )

        normalized_state = getattr(
            state,
            "value",
            state,
        )

        if normalized_state == "HALTED":
            return RecoveryAction.HALT

        if (
            recovery_required is True
            and fallback_allowed is True
            and self.allow_failover
        ):
            return RecoveryAction.FAILOVER

        if recovery_required is True:
            return RecoveryAction.RECOVER

        if allow_inference is False:
            return RecoveryAction.HALT

        return RecoveryAction.NO_ACTION

    def recover(
        self,
        incident_result: Any,
        recovery_runner: Optional[
            Callable[[], Any]
        ] = None,
        failover_runner: Optional[
            Callable[[], Any]
        ] = None,
        target_model_version: Optional[
            str
        ] = None,
    ) -> ModelRuntimeRecoveryResult:
        action = self.evaluate_action(
            incident_result
        )

        checks: List[RecoveryCheck] = []

        errors = self._messages(
            incident_result,
            "errors",
        )

        warnings = self._messages(
            incident_result,
            "warnings",
        )

        if action == RecoveryAction.NO_ACTION:
            checks.append(
                RecoveryCheck(
                    name="incident_policy",
                    passed=True,
                    message=(
                        "No recovery action is required."
                    ),
                )
            )

            return ModelRuntimeRecoveryResult(
                success=True,
                state=RecoveryState.IDLE,
                action=action,
                model_version=(
                    self._active_model_version
                ),
                checks=checks,
                errors=errors,
                warnings=warnings,
            )

        if action == RecoveryAction.HALT:
            self._state = RecoveryState.HALTED

            checks.append(
                RecoveryCheck(
                    name="incident_policy",
                    passed=False,
                    message=(
                        "Runtime recovery is blocked "
                        "by safety policy."
                    ),
                )
            )

            errors.append(
                "Runtime recovery is blocked; "
                "runtime must remain halted."
            )

            return ModelRuntimeRecoveryResult(
                success=False,
                state=self._state,
                action=action,
                model_version=(
                    self._active_model_version
                ),
                checks=checks,
                errors=errors,
                warnings=warnings,
            )

        if action == RecoveryAction.FAILOVER:
            if (
                not target_model_version
                or not isinstance(
                    target_model_version,
                    str,
                )
            ):
                self._state = (
                    RecoveryState.HALTED
                )

                checks.append(
                    RecoveryCheck(
                        name="target_model",
                        passed=False,
                        message=(
                            "A valid failover model "
                            "version is required."
                        ),
                    )
                )

                errors.append(
                    "Safe failover requires "
                    "a target model version."
                )

                return ModelRuntimeRecoveryResult(
                    success=False,
                    state=self._state,
                    action=action,
                    model_version=(
                        self._active_model_version
                    ),
                    checks=checks,
                    errors=errors,
                    warnings=warnings,
                )

            if not callable(
                failover_runner
            ):
                self._state = (
                    RecoveryState.HALTED
                )

                checks.append(
                    RecoveryCheck(
                        name="failover_runner",
                        passed=False,
                        message=(
                            "A callable failover "
                            "runner is required."
                        ),
                    )
                )

                errors.append(
                    "Safe failover requires "
                    "a callable failover runner."
                )

                return ModelRuntimeRecoveryResult(
                    success=False,
                    state=self._state,
                    action=action,
                    model_version=(
                        self._active_model_version
                    ),
                    checks=checks,
                    errors=errors,
                    warnings=warnings,
                )

            self._state = (
                RecoveryState.RECOVERING
            )

            checks.append(
                RecoveryCheck(
                    name="failover_runner",
                    passed=True,
                    message=(
                        "Failover runner accepted."
                    ),
                )
            )

            try:
                runtime_result = (
                    failover_runner()
                )

            except Exception as exc:
                self._state = (
                    RecoveryState.HALTED
                )

                message = (
                    "Model failover failed: "
                    f"{type(exc).__name__}: {exc}"
                )

                errors.append(message)

                checks.append(
                    RecoveryCheck(
                        name="failover_execution",
                        passed=False,
                        message=message,
                    )
                )

                return ModelRuntimeRecoveryResult(
                    success=False,
                    state=self._state,
                    action=action,
                    model_version=(
                        self._active_model_version
                    ),
                    checks=checks,
                    errors=errors,
                    warnings=warnings,
                )

            self._active_model_version = (
                target_model_version
            )

            self._state = (
                RecoveryState.FAILED_OVER
            )

            checks.append(
                RecoveryCheck(
                    name="failover_execution",
                    passed=True,
                    message=(
                        "Model failover completed "
                        "successfully."
                    ),
                )
            )

            return ModelRuntimeRecoveryResult(
                success=True,
                state=self._state,
                action=action,
                model_version=(
                    self._active_model_version
                ),
                runtime_result=runtime_result,
                checks=checks,
                errors=errors,
                warnings=warnings,
            )

        if not callable(
            recovery_runner
        ):
            self._state = (
                RecoveryState.HALTED
            )

            checks.append(
                RecoveryCheck(
                    name="recovery_runner",
                    passed=False,
                    message=(
                        "A callable recovery runner "
                        "is required."
                    ),
                )
            )

            errors.append(
                "Runtime recovery requires "
                "a callable recovery runner."
            )

            return ModelRuntimeRecoveryResult(
                success=False,
                state=self._state,
                action=action,
                model_version=(
                    self._active_model_version
                ),
                checks=checks,
                errors=errors,
                warnings=warnings,
            )

        self._state = (
            RecoveryState.RECOVERING
        )

        checks.append(
            RecoveryCheck(
                name="recovery_runner",
                passed=True,
                message=(
                    "Recovery runner accepted."
                ),
            )
        )

        try:
            runtime_result = (
                recovery_runner()
            )

        except Exception as exc:
            self._state = (
                RecoveryState.HALTED
            )

            message = (
                "Runtime recovery failed: "
                f"{type(exc).__name__}: {exc}"
            )

            errors.append(message)

            checks.append(
                RecoveryCheck(
                    name="recovery_execution",
                    passed=False,
                    message=message,
                )
            )

            return ModelRuntimeRecoveryResult(
                success=False,
                state=self._state,
                action=action,
                model_version=(
                    self._active_model_version
                ),
                checks=checks,
                errors=errors,
                warnings=warnings,
            )

        checks.append(
            RecoveryCheck(
                name="recovery_execution",
                passed=True,
                message=(
                    "Runtime recovery completed "
                    "successfully."
                ),
            )
        )

        return ModelRuntimeRecoveryResult(
            success=True,
            state=self._state,
            action=action,
            model_version=(
                self._active_model_version
            ),
            runtime_result=runtime_result,
            checks=checks,
            errors=errors,
            warnings=warnings,
        )

    def reset(self) -> None:
        self._state = RecoveryState.IDLE
        self._active_model_version = None


__all__ = [
    "RecoveryAction",
    "RecoveryState",
    "RecoveryCheck",
    "ModelRuntimeRecoveryResult",
    "QuantAIProductionModelRuntimeRecoveryOrchestrator",
]