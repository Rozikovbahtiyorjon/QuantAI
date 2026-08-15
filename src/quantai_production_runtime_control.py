from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from .quantai_production_runtime_lifecycle import (
    ProductionRuntimeLifecycleResult,
    QuantAIProductionRuntimeLifecycle,
    RuntimeLifecycleState,
)


@dataclass(frozen=True)
class RuntimeControlCheck:
    name: str
    passed: bool
    message: str


@dataclass
class ProductionRuntimeControlResult:
    success: bool
    command: str
    state: RuntimeLifecycleState
    checks: List[RuntimeControlCheck] = field(
        default_factory=list
    )
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    lifecycle_result: Optional[
        ProductionRuntimeLifecycleResult
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


class QuantAIProductionRuntimeControl:
    """
    Central operational control interface for the
    QuantAI production runtime.

    Supported commands:

        - status
        - start
        - stop
        - emergency_stop
        - recover

    This layer coordinates the already validated
    production runtime lifecycle and does not
    implement trading logic.
    """

    def __init__(
        self,
        lifecycle: Optional[
            QuantAIProductionRuntimeLifecycle
        ] = None,
    ) -> None:
        if lifecycle is not None and not isinstance(
            lifecycle,
            QuantAIProductionRuntimeLifecycle,
        ):
            raise TypeError(
                "lifecycle must be "
                "QuantAIProductionRuntimeLifecycle "
                "or None."
            )

        self.lifecycle = (
            lifecycle
            if lifecycle is not None
            else QuantAIProductionRuntimeLifecycle()
        )

    @property
    def state(self) -> RuntimeLifecycleState:
        return self.lifecycle.state

    @property
    def is_running(self) -> bool:
        return self.lifecycle.is_running

    @staticmethod
    def _check(
        name: str,
        passed: bool,
        success_message: str,
        failure_message: str,
    ) -> RuntimeControlCheck:
        return RuntimeControlCheck(
            name=name,
            passed=passed,
            message=(
                success_message
                if passed
                else failure_message
            ),
        )

    @staticmethod
    def _collect_errors(
        checks: List[RuntimeControlCheck],
    ) -> List[str]:
        return [
            f"{check.name}: {check.message}"
            for check in checks
            if not check.passed
        ]

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

    def status(
        self,
    ) -> ProductionRuntimeControlResult:
        running = self.is_running

        check = self._check(
            name="runtime_state",
            passed=True,
            success_message=(
                "Runtime state is available."
            ),
            failure_message=(
                "Runtime state is unavailable."
            ),
        )

        return ProductionRuntimeControlResult(
            success=True,
            command="status",
            state=self.state,
            checks=[check],
            runtime_result={
                "state": self.state.value,
                "is_running": running,
            },
        )

    def start(
        self,
        runner: Callable[[], Any],
    ) -> ProductionRuntimeControlResult:
        if not callable(runner):
            raise TypeError(
                "runner must be callable."
            )

        lifecycle_result = self.lifecycle.start(
            runner
        )

        return self._from_lifecycle_result(
            command="start",
            lifecycle_result=lifecycle_result,
        )

    def stop(
        self,
        stopper: Optional[Callable[[], Any]] = None,
    ) -> ProductionRuntimeControlResult:
        if (
            stopper is not None
            and not callable(stopper)
        ):
            raise TypeError(
                "stopper must be callable "
                "when provided."
            )

        lifecycle_result = self.lifecycle.stop(
            stopper
        )

        return self._from_lifecycle_result(
            command="stop",
            lifecycle_result=lifecycle_result,
        )

    def emergency_stop(
        self,
        stopper: Optional[Callable[[], Any]] = None,
    ) -> ProductionRuntimeControlResult:
        if (
            stopper is not None
            and not callable(stopper)
        ):
            raise TypeError(
                "stopper must be callable "
                "when provided."
            )

        lifecycle_result = (
            self.lifecycle.emergency_stop(
                stopper
            )
        )

        return self._from_lifecycle_result(
            command="emergency_stop",
            lifecycle_result=lifecycle_result,
        )

    def recover(
        self,
        runner: Callable[[], Any],
    ) -> ProductionRuntimeControlResult:
        if not callable(runner):
            raise TypeError(
                "runner must be callable."
            )

        lifecycle_result = self.lifecycle.recover(
            runner
        )

        return self._from_lifecycle_result(
            command="recover",
            lifecycle_result=lifecycle_result,
        )

    def execute(
        self,
        command: str,
        runner: Optional[
            Callable[[], Any]
        ] = None,
        stopper: Optional[
            Callable[[], Any]
        ] = None,
    ) -> ProductionRuntimeControlResult:
        if not isinstance(command, str):
            raise TypeError(
                "command must be a string."
            )

        normalized = command.strip().lower()

        if not normalized:
            raise ValueError(
                "command cannot be empty."
            )

        if normalized == "status":
            return self.status()

        if normalized == "start":
            if runner is None:
                raise ValueError(
                    "runner is required for start."
                )

            return self.start(
                runner
            )

        if normalized == "stop":
            return self.stop(
                stopper
            )

        if normalized == "emergency_stop":
            return self.emergency_stop(
                stopper
            )

        if normalized == "recover":
            if runner is None:
                raise ValueError(
                    "runner is required for recover."
                )

            return self.recover(
                runner
            )

        raise ValueError(
            f"Unsupported runtime command: {command}"
        )

    def reset(self) -> None:
        self.lifecycle.reset()

    def _from_lifecycle_result(
        self,
        command: str,
        lifecycle_result: ProductionRuntimeLifecycleResult,
    ) -> ProductionRuntimeControlResult:
        checks = [
            self._check(
                name="lifecycle_command",
                passed=lifecycle_result.success,
                success_message=(
                    "Lifecycle command executed successfully."
                ),
                failure_message=(
                    lifecycle_result.errors[0]
                    if lifecycle_result.errors
                    else (
                        "Lifecycle command failed."
                    )
                ),
            )
        ]

        errors = self._collect_errors(
            checks
        )

        errors.extend(
            self._extract_messages(
                lifecycle_result,
                "errors",
            )
        )

        warnings = self._extract_messages(
            lifecycle_result,
            "warnings",
        )

        return ProductionRuntimeControlResult(
            success=(
                lifecycle_result.success
                and not errors
            ),
            command=command,
            state=lifecycle_result.state,
            checks=checks,
            errors=errors,
            warnings=warnings,
            lifecycle_result=lifecycle_result,
            runtime_result=(
                lifecycle_result.runtime_result
            ),
        )


def execute_runtime_command(
    command: str,
    runner: Optional[
        Callable[[], Any]
    ] = None,
    stopper: Optional[
        Callable[[], Any]
    ] = None,
    lifecycle: Optional[
        QuantAIProductionRuntimeLifecycle
    ] = None,
) -> ProductionRuntimeControlResult:
    controller = QuantAIProductionRuntimeControl(
        lifecycle=lifecycle
    )

    return controller.execute(
        command=command,
        runner=runner,
        stopper=stopper,
    )


__all__ = [
    "RuntimeControlCheck",
    "ProductionRuntimeControlResult",
    "QuantAIProductionRuntimeControl",
    "execute_runtime_command",
]