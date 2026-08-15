from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from src.quantai_production_model_runtime_monitoring import (
    PredictionHealthResult,
    PredictionHealthSnapshot,
    QuantAIProductionModelRuntimeMonitoring,
)


@dataclass
class ModelHealthSupervisorResult:
    ready: bool
    health_result: Optional[PredictionHealthResult] = None
    supervisor_result: Any = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recovery_attempted: bool = False
    recovery_succeeded: bool = False

    @property
    def healthy(self) -> bool:
        return self.ready


class QuantAIProductionModelRuntimeMonitoringIntegration:
    """
    Integration layer between model inference health monitoring
    and the production runtime supervisor.

    This module does not implement trading logic.
    """

    def __init__(
        self,
        monitoring: QuantAIProductionModelRuntimeMonitoring,
        supervisor: Any = None,
    ) -> None:
        if not isinstance(
            monitoring,
            QuantAIProductionModelRuntimeMonitoring,
        ):
            raise TypeError(
                "monitoring must be a "
                "QuantAIProductionModelRuntimeMonitoring."
            )

        if supervisor is not None:
            required_methods = (
                "supervise",
            )

            for method_name in required_methods:
                if not callable(
                    getattr(
                        supervisor,
                        method_name,
                        None,
                    )
                ):
                    raise TypeError(
                        "supervisor must provide a callable "
                        f"{method_name} method."
                    )

        self._monitoring = monitoring
        self._supervisor = supervisor

    @property
    def monitoring(
        self,
    ) -> QuantAIProductionModelRuntimeMonitoring:
        return self._monitoring

    @property
    def supervisor(self) -> Any:
        return self._supervisor

    @property
    def is_healthy(self) -> bool:
        return self._monitoring.is_healthy

    def _supervisor_health_check(
        self,
        health_result: PredictionHealthResult,
    ) -> Any:
        if self._supervisor is None:
            return None

        try:
            return self._supervisor.supervise(
                health_checker=lambda: health_result,
            )
        except TypeError:
            return self._supervisor.supervise(
                health_checker=lambda: health_result.healthy,
            )

    @staticmethod
    def _extract_supervisor_errors(
        supervisor_result: Any,
    ) -> List[str]:
        if supervisor_result is None:
            return []

        errors = getattr(
            supervisor_result,
            "errors",
            None,
        )

        if errors is None:
            return []

        if isinstance(errors, str):
            return [errors]

        try:
            return [
                str(error)
                for error in errors
            ]
        except TypeError:
            return [str(errors)]

    @staticmethod
    def _extract_supervisor_warnings(
        supervisor_result: Any,
    ) -> List[str]:
        if supervisor_result is None:
            return []

        warnings = getattr(
            supervisor_result,
            "warnings",
            None,
        )

        if warnings is None:
            return []

        if isinstance(warnings, str):
            return [warnings]

        try:
            return [
                str(warning)
                for warning in warnings
            ]
        except TypeError:
            return [str(warnings)]

    @staticmethod
    def _extract_supervisor_ready(
        supervisor_result: Any,
    ) -> Optional[bool]:
        if supervisor_result is None:
            return None

        for attribute in (
            "ready",
            "healthy",
            "success",
            "passed",
        ):
            value = getattr(
                supervisor_result,
                attribute,
                None,
            )

            if isinstance(value, bool):
                return value

        return None

    def monitor(
        self,
        data: Any,
    ) -> ModelHealthSupervisorResult:
        health_result = self._monitoring.monitor(
            data
        )

        errors = list(
            health_result.errors
        )

        warnings = list(
            health_result.warnings
        )

        supervisor_result = None

        if not health_result.healthy:
            errors.append(
                "Production model health check failed."
            )

        if self._supervisor is not None:
            try:
                supervisor_result = (
                    self._supervisor_health_check(
                        health_result
                    )
                )

            except Exception as exc:
                errors.append(
                    (
                        "Runtime supervisor health "
                        "check failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
                )

            else:
                errors.extend(
                    self._extract_supervisor_errors(
                        supervisor_result
                    )
                )

                warnings.extend(
                    self._extract_supervisor_warnings(
                        supervisor_result
                    )
                )

                supervisor_ready = (
                    self._extract_supervisor_ready(
                        supervisor_result
                    )
                )

                if supervisor_ready is False:
                    errors.append(
                        "Runtime supervisor reported "
                        "an unhealthy state."
                    )

        ready = (
            health_result.healthy
            and not errors
        )

        return ModelHealthSupervisorResult(
            ready=ready,
            health_result=health_result,
            supervisor_result=supervisor_result,
            errors=errors,
            warnings=warnings,
        )

    def monitor_and_recover(
        self,
        data: Any,
        recovery: Optional[
            Callable[[], Any]
        ] = None,
    ) -> ModelHealthSupervisorResult:
        result = self.monitor(data)

        if result.ready:
            return result

        if recovery is None:
            return result

        if not callable(recovery):
            raise TypeError(
                "recovery must be callable or None."
            )

        result.recovery_attempted = True

        try:
            recovery_result = recovery()

        except Exception as exc:
            result.recovery_succeeded = False
            result.errors.append(
                (
                    "Model recovery failed: "
                    f"{type(exc).__name__}: {exc}"
                )
            )
            return result

        if isinstance(
            recovery_result,
            bool,
        ):
            result.recovery_succeeded = (
                recovery_result
            )
        else:
            result.recovery_succeeded = bool(
                getattr(
                    recovery_result,
                    "success",
                    getattr(
                        recovery_result,
                        "ready",
                        False,
                    ),
                )
            )

        if not result.recovery_succeeded:
            result.errors.append(
                "Model recovery did not succeed."
            )

        return result

    def snapshot(
        self,
    ) -> PredictionHealthSnapshot:
        return self._monitoring.snapshot()

    def reset(self) -> None:
        self._monitoring.reset()


__all__ = [
    "ModelHealthSupervisorResult",
    "QuantAIProductionModelRuntimeMonitoringIntegration",
]