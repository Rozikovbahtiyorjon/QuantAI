from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, List, Optional

try:
    from src.feature_store.live_logger import LiveFeatureLogger, LiveLoggerConfig

    _LIVE_LOGGER_AVAILABLE = True
except ImportError:
    LiveFeatureLogger = None  # type: ignore
    LiveLoggerConfig = None  # type: ignore
    _LIVE_LOGGER_AVAILABLE = False

try:
    from src.automation.drift_trigger import DriftRetrainTrigger, DriftTriggerConfig

    _DRIFT_TRIGGER_AVAILABLE = True
except ImportError:
    DriftRetrainTrigger = None  # type: ignore
    DriftTriggerConfig = None  # type: ignore
    _DRIFT_TRIGGER_AVAILABLE = False


class RuntimeMode(str, Enum):
    DRY_RUN = "DRY_RUN"
    PAPER = "PAPER"
    LIVE = "LIVE"


@dataclass(frozen=True)
class RuntimeCheck:
    name: str
    passed: bool
    message: str


@dataclass
class ProductionRuntimeResult:
    started: bool
    mode: RuntimeMode
    checks: List[RuntimeCheck] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    output: Any = None

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


class QuantAIProductionRuntime:
    """
    Controlled runtime execution layer for QuantAI.

    This module does not implement trading logic.
    It provides a deterministic execution boundary
    around an already validated deployment target.
    """

    def __init__(
        self,
        mode: RuntimeMode | str = RuntimeMode.DRY_RUN,
        require_readiness: bool = True,
        enable_live_features: bool = False,
        live_logger: Any | None = None,
        live_logger_config: Any | None = None,
        enable_drift_trigger: bool = False,
        drift_trigger_config: Any | None = None,
        drift_retrain_fn: Any | None = None,
    ) -> None:
        self.mode = self._normalize_mode(
            mode
        )

        self.require_readiness = bool(
            require_readiness
        )

        self._running = False

        # Live Feature Store (auto-logging + drift)
        self.enable_live_features = bool(enable_live_features)
        self.live_logger: Any | None = live_logger
        if self.enable_live_features and self.live_logger is None:
            if not _LIVE_LOGGER_AVAILABLE:
                raise ImportError("LiveFeatureLogger not available")
            cfg = live_logger_config
            if cfg is None:
                cfg = LiveLoggerConfig()  # type: ignore
            self.live_logger = LiveFeatureLogger(config=cfg)  # type: ignore

        # Drift-triggered retraining
        self.enable_drift_trigger = bool(enable_drift_trigger)
        self.drift_trigger: Any | None = None
        if self.enable_drift_trigger:
            if not _DRIFT_TRIGGER_AVAILABLE:
                raise ImportError("DriftRetrainTrigger not available")
            self.drift_trigger = DriftRetrainTrigger(  # type: ignore
                config=drift_trigger_config,
                retrain_fn=drift_retrain_fn,
            )

    @staticmethod
    def _normalize_mode(
        mode: RuntimeMode | str,
    ) -> RuntimeMode:

        if isinstance(
            mode,
            RuntimeMode,
        ):
            return mode

        if isinstance(
            mode,
            str,
        ):

            try:
                return RuntimeMode(
                    mode.strip().upper()
                )

            except ValueError as exc:

                raise ValueError(
                    "mode must be DRY_RUN, PAPER, or LIVE."
                ) from exc

        raise TypeError(
            "mode must be a RuntimeMode or string."
        )

    @staticmethod
    def _extract_ready(
        result: Any,
    ) -> Optional[bool]:

        if result is None:
            return None

        for attribute in (
            "ready",
            "passed",
            "valid",
            "success",
            "healthy",
        ):

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

    def _readiness_check(
        self,
        readiness_result: Any,
    ) -> RuntimeCheck:

        if not self.require_readiness:

            return RuntimeCheck(
                name="readiness_gate",
                passed=True,
                message=(
                    "Readiness requirement disabled."
                ),
            )

        ready = self._extract_ready(
            readiness_result
        )

        if ready is None:

            return RuntimeCheck(
                name="readiness_gate",
                passed=False,
                message=(
                    "Readiness result was not "
                    "provided or has no supported "
                    "status."
                ),
            )

        if ready:

            return RuntimeCheck(
                name="readiness_gate",
                passed=True,
                message=(
                    "Production readiness check passed."
                ),
            )

        return RuntimeCheck(
            name="readiness_gate",
            passed=False,
            message=(
                "Production readiness check failed."
            ),
        )

    def _mode_check(self) -> RuntimeCheck:

        return RuntimeCheck(
            name="runtime_mode",
            passed=True,
            message=(
                f"Runtime mode accepted: "
                f"{self.mode.value}."
            ),
        )

    def _live_feature_check(self) -> RuntimeCheck:
        if not self.enable_live_features:
            return RuntimeCheck(
                name="live_feature_store",
                passed=True,
                message="Live feature logging disabled.",
            )
        if self.live_logger is None:
            return RuntimeCheck(
                name="live_feature_store",
                passed=False,
                message="Live feature logging enabled but logger not initialized.",
            )
        try:
            # Probe store writability
            _ = self.live_logger.store.root
            return RuntimeCheck(
                name="live_feature_store",
                passed=True,
                message=(
                    f"Live feature store ready: view="
                    f"{self.live_logger.config.view_name}, "
                    f"buffer={self.live_logger.buffered_count}."
                ),
            )
        except Exception as exc:
            return RuntimeCheck(
                name="live_feature_store",
                passed=False,
                message=f"Live feature store not ready: {exc}",
            )

    def log_live_features(self, features: dict) -> bool:
        """Auto-log live features (non-blocking). Returns True if flushed."""
        if not self.enable_live_features or self.live_logger is None:
            return False
        return bool(self.live_logger.log(features))

    def flush_live_features(self) -> Any:
        if self.live_logger is None:
            return None
        return self.live_logger.flush()

    def check_drift_and_maybe_retrain(self) -> Any:
        """Check drift and trigger retrain if needed (call periodically)."""
        if not self.enable_drift_trigger or self.drift_trigger is None:
            return None
        return self.drift_trigger.maybe_trigger_retrain()

    def _drift_trigger_check(self) -> RuntimeCheck:
        if not self.enable_drift_trigger:
            return RuntimeCheck(
                name="drift_trigger",
                passed=True,
                message="Drift trigger disabled.",
            )
        if self.drift_trigger is None:
            return RuntimeCheck(
                name="drift_trigger",
                passed=False,
                message="Drift trigger enabled but not initialized.",
            )
        return RuntimeCheck(
            name="drift_trigger",
            passed=True,
            message=(
                f"Drift trigger ready: view={self.drift_trigger.config.live_view} "
                f"PSI>{self.drift_trigger.config.psi_threshold}, "
                f"triggers={self.drift_trigger.trigger_count}."
            ),
        )

    def preflight(
        self,
        readiness_result: Any = None,
    ) -> ProductionRuntimeResult:

        checks = [
            self._mode_check(),
            self._readiness_check(
                readiness_result
            ),
            self._live_feature_check(),
            self._drift_trigger_check(),
        ]

        errors: List[str] = []

        warnings = self._extract_messages(
            readiness_result,
            "warnings",
        )

        errors.extend(
            self._extract_messages(
                readiness_result,
                "errors",
            )
        )

        for check in checks:

            if not check.passed:

                errors.append(
                    f"{check.name}: "
                    f"{check.message}"
                )

        return ProductionRuntimeResult(
            started=False,
            mode=self.mode,
            checks=checks,
            errors=errors,
            warnings=warnings,
        )

    def start(
        self,
        readiness_result: Any = None,
        runner: Optional[
            Callable[[], Any]
        ] = None,
    ) -> ProductionRuntimeResult:

        if self._running:

            return ProductionRuntimeResult(
                started=False,
                mode=self.mode,
                checks=[],
                errors=[
                    "Runtime is already running."
                ],
            )

        result = self.preflight(
            readiness_result
        )

        if (
            result.errors
            or result.checks_failed
        ):

            return result

        try:

            output = (
                runner()
                if runner is not None
                else None
            )

        except Exception as exc:

            message = (
                f"runner: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            result.errors.append(
                message
            )

            return result

        self._running = True

        result.started = True
        result.output = output

        return result

    def stop(self) -> ProductionRuntimeResult:

        if not self._running:

            return ProductionRuntimeResult(
                started=False,
                mode=self.mode,
                checks=[
                    RuntimeCheck(
                        name="runtime_state",
                        passed=True,
                        message=(
                            "Runtime is already stopped."
                        ),
                    )
                ],
            )

        # Flush live features before stopping
        if self.enable_live_features and self.live_logger is not None:
            try:
                self.live_logger.flush()
            except Exception:
                pass

        self._running = False

        return ProductionRuntimeResult(
            started=False,
            mode=self.mode,
            checks=[
                RuntimeCheck(
                    name="runtime_state",
                    passed=True,
                    message=(
                        "Runtime stopped successfully."
                    ),
                )
            ],
        )

    @property
    def is_running(self) -> bool:
        return self._running


__all__ = [
    "RuntimeMode",
    "RuntimeCheck",
    "ProductionRuntimeResult",
    "QuantAIProductionRuntime",
]