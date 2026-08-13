from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, List, Optional


class RuntimeLifecycleState(str, Enum):
    STOPPED = "STOPPED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    FAILED = "FAILED"


@dataclass(frozen=True)
class LifecycleEvent:
    state: RuntimeLifecycleState
    message: str


@dataclass
class ProductionRuntimeLifecycleResult:
    success: bool
    state: RuntimeLifecycleState
    action: str
    runtime_result: Any = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    events: List[LifecycleEvent] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return not self.success


class QuantAIProductionRuntimeLifecycle:
    """
    Deterministic lifecycle controller for production runtime.

    Responsibilities:

        - start
        - safe stop
        - emergency stop
        - recovery
        - lifecycle state management
        - lifecycle event recording

    This module does not implement trading logic.
    """

    def __init__(self) -> None:
        self._state = RuntimeLifecycleState.STOPPED
        self._runtime_result: Any = None
        self._errors: List[str] = []
        self._warnings: List[str] = []
        self._events: List[LifecycleEvent] = []

    @property
    def state(self) -> RuntimeLifecycleState:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._state == RuntimeLifecycleState.RUNNING

    @property
    def events(self) -> List[LifecycleEvent]:
        return list(self._events)

    @property
    def errors(self) -> List[str]:
        return list(self._errors)

    @property
    def warnings(self) -> List[str]:
        return list(self._warnings)

    def _record_event(
        self,
        message: str,
    ) -> None:
        self._events.append(
            LifecycleEvent(
                state=self._state,
                message=message,
            )
        )

    def _build_result(
        self,
        action: str,
        success: bool,
        runtime_result: Any = None,
        errors: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None,
    ) -> ProductionRuntimeLifecycleResult:
        return ProductionRuntimeLifecycleResult(
            success=success,
            state=self._state,
            action=action,
            runtime_result=runtime_result,
            errors=list(errors or []),
            warnings=list(warnings or []),
            events=self.events,
        )

    def start(
        self,
        runner: Callable[[], Any],
    ) -> ProductionRuntimeLifecycleResult:
        if not callable(runner):
            raise TypeError(
                "runner must be callable."
            )

        if self._state == RuntimeLifecycleState.RUNNING:
            return self._build_result(
                action="start",
                success=False,
                errors=[
                    "Runtime is already running."
                ],
            )

        if self._state in (
            RuntimeLifecycleState.STARTING,
            RuntimeLifecycleState.STOPPING,
        ):
            return self._build_result(
                action="start",
                success=False,
                errors=[
                    (
                        "Runtime is busy in state "
                        f"{self._state.value}."
                    )
                ],
            )

        self._state = RuntimeLifecycleState.STARTING

        self._record_event(
            "Runtime startup initiated."
        )

        try:
            self._runtime_result = runner()

        except Exception as exc:
            message = (
                "Runtime startup failed: "
                f"{type(exc).__name__}: {exc}"
            )

            self._state = RuntimeLifecycleState.FAILED

            self._errors.append(
                message
            )

            self._record_event(
                message
            )

            return self._build_result(
                action="start",
                success=False,
                errors=[message],
            )

        self._state = RuntimeLifecycleState.RUNNING

        self._record_event(
            "Runtime started successfully."
        )

        return self._build_result(
            action="start",
            success=True,
            runtime_result=self._runtime_result,
        )

    def stop(
        self,
        stopper: Optional[Callable[[], Any]] = None,
    ) -> ProductionRuntimeLifecycleResult:
        if stopper is not None and not callable(stopper):
            raise TypeError(
                "stopper must be callable or None."
            )

        if self._state == RuntimeLifecycleState.STOPPED:
            return self._build_result(
                action="stop",
                success=True,
                warnings=[
                    "Runtime is already stopped."
                ],
            )

        if self._state == RuntimeLifecycleState.STOPPING:
            return self._build_result(
                action="stop",
                success=False,
                errors=[
                    "Runtime is already stopping."
                ],
            )

        if self._state == RuntimeLifecycleState.STARTING:
            return self._build_result(
                action="stop",
                success=False,
                errors=[
                    (
                        "Cannot stop runtime while "
                        "it is starting."
                    )
                ],
            )

        self._state = RuntimeLifecycleState.STOPPING

        self._record_event(
            "Runtime shutdown initiated."
        )

        try:
            if stopper is not None:
                self._runtime_result = stopper()

        except Exception as exc:
            message = (
                "Runtime shutdown failed: "
                f"{type(exc).__name__}: {exc}"
            )

            self._state = RuntimeLifecycleState.FAILED

            self._errors.append(
                message
            )

            self._record_event(
                message
            )

            return self._build_result(
                action="stop",
                success=False,
                errors=[message],
            )

        self._state = RuntimeLifecycleState.STOPPED
        self._runtime_result = None

        self._record_event(
            "Runtime stopped successfully."
        )

        return self._build_result(
            action="stop",
            success=True,
        )

    def emergency_stop(
        self,
        stopper: Optional[Callable[[], Any]] = None,
    ) -> ProductionRuntimeLifecycleResult:
        if stopper is not None and not callable(stopper):
            raise TypeError(
                "stopper must be callable or None."
            )

        if self._state == RuntimeLifecycleState.STOPPED:
            return self._build_result(
                action="emergency_stop",
                success=True,
                warnings=[
                    "Runtime is already stopped."
                ],
            )

        self._state = RuntimeLifecycleState.STOPPING

        self._record_event(
            "Emergency shutdown initiated."
        )

        try:
            if stopper is not None:
                self._runtime_result = stopper()

        except Exception as exc:
            message = (
                "Emergency shutdown failed: "
                f"{type(exc).__name__}: {exc}"
            )

            self._state = RuntimeLifecycleState.FAILED

            self._errors.append(
                message
            )

            self._record_event(
                message
            )

            return self._build_result(
                action="emergency_stop",
                success=False,
                errors=[message],
            )

        self._state = RuntimeLifecycleState.STOPPED
        self._runtime_result = None

        self._record_event(
            "Emergency shutdown completed."
        )

        return self._build_result(
            action="emergency_stop",
            success=True,
        )

    def recover(
        self,
        runner: Callable[[], Any],
    ) -> ProductionRuntimeLifecycleResult:
        if not callable(runner):
            raise TypeError(
                "runner must be callable."
            )

        if self._state not in (
            RuntimeLifecycleState.FAILED,
            RuntimeLifecycleState.STOPPED,
        ):
            return self._build_result(
                action="recover",
                success=False,
                errors=[
                    (
                        "Recovery is not allowed from "
                        f"state {self._state.value}."
                    )
                ],
            )

        self._record_event(
            "Runtime recovery initiated."
        )

        return self.start(
            runner
        )

    def reset(self) -> None:
        self._state = RuntimeLifecycleState.STOPPED
        self._runtime_result = None
        self._errors.clear()
        self._warnings.clear()
        self._events.clear()


__all__ = [
    "RuntimeLifecycleState",
    "LifecycleEvent",
    "ProductionRuntimeLifecycleResult",
    "QuantAIProductionRuntimeLifecycle",
]