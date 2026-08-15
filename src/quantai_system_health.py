from __future__ import annotations

from dataclasses import dataclass, field
import importlib
from typing import List, Sequence


@dataclass(frozen=True)
class HealthCheckResult:
    name: str
    passed: bool
    message: str


@dataclass
class QuantAISystemHealthResult:
    healthy: bool
    ready: bool
    checks: List[HealthCheckResult] = field(
        default_factory=list
    )
    errors: List[str] = field(
        default_factory=list
    )
    warnings: List[str] = field(
        default_factory=list
    )

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


class QuantAISystemHealth:
    """
    Lightweight health and readiness checker for QuantAI.

    This module does not execute trading logic.
    It only verifies that required system components
    can be imported and that the integration layer
    is structurally available.
    """

    DEFAULT_MODULES: Sequence[str] = (
        "src.feature_engine",
        "src.dataset_builder",
        "src.ml_engine",
        "src.model_manager",
        "src.confidence_engine",
        "src.strategy",
        "src.trade_engine",
        "src.backtest_engine",
        "src.performance_analyzer",
        "src.portfolio_risk_engine",
        "src.drawdown_guard",
        "src.risk_aggregator",
        "src.portfolio_exposure_engine",
        "src.research_dashboard",
        "src.quantai_architecture_audit",
        "src.quantai_end_to_end_validation",
        "src.paper_trading_validator",
        "src.paper_trading_quality_gate",
        "src.unified_system_integration",
    )

    def __init__(
        self,
        required_modules: Sequence[str] | None = None,
    ) -> None:

        modules = (
            required_modules
            if required_modules is not None
            else self.DEFAULT_MODULES
        )

        if not isinstance(
            modules,
            Sequence,
        ):

            raise TypeError(
                "required_modules must be a sequence."
            )

        normalized = []

        for module_name in modules:

            if not isinstance(
                module_name,
                str,
            ) or not module_name.strip():

                raise ValueError(
                    "Module names must be non-empty strings."
                )

            normalized.append(
                module_name.strip()
            )

        self.required_modules = tuple(
            normalized
        )

    # =====================================================
    # IMPORT CHECK
    # =====================================================

    @staticmethod
    def _check_module(
        module_name: str,
    ) -> HealthCheckResult:

        try:

            importlib.import_module(
                module_name
            )

        except Exception as exc:

            return HealthCheckResult(
                name=f"module:{module_name}",
                passed=False,
                message=(
                    f"Import failed: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            )

        return HealthCheckResult(
            name=f"module:{module_name}",
            passed=True,
            message="Module imported successfully.",
        )

    # =====================================================
    # REQUIRED MODULES
    # =====================================================

    def check_required_modules(
        self,
    ) -> List[HealthCheckResult]:

        return [
            self._check_module(
                module_name
            )
            for module_name in self.required_modules
        ]

    # =====================================================
    # INTEGRATION CHECK
    # =====================================================

    @staticmethod
    def check_integration_layer() -> HealthCheckResult:

        try:

            module = importlib.import_module(
                "src.unified_system_integration"
            )

            required_names = (
                "IntegrationStageResult",
                "UnifiedSystemResult",
                "QuantAIUnifiedSystem",
                "create_default_integration",
            )

            missing = [
                name
                for name in required_names
                if not hasattr(
                    module,
                    name,
                )
            ]

            if missing:

                return HealthCheckResult(
                    name="integration_layer",
                    passed=False,
                    message=(
                        "Missing integration exports: "
                        + ", ".join(missing)
                    ),
                )

        except Exception as exc:

            return HealthCheckResult(
                name="integration_layer",
                passed=False,
                message=(
                    f"Integration import failed: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            )

        return HealthCheckResult(
            name="integration_layer",
            passed=True,
            message="Integration layer is available.",
        )

    # =====================================================
    # ORCHESTRATOR CHECK
    # =====================================================

    @staticmethod
    def check_orchestrator() -> HealthCheckResult:

        try:

            from src.unified_system_integration import (
                QuantAIUnifiedSystem,
            )

            system = QuantAIUnifiedSystem()

            system.register_stage(
                "health_check",
                lambda value: value + 1,
            )

            result = system.run(
                0
            )

            if not result.success:

                return HealthCheckResult(
                    name="orchestrator",
                    passed=False,
                    message=(
                        "Integration orchestrator "
                        "execution failed."
                    ),
                )

            if result.outputs.get(
                "health_check"
            ) != 1:

                return HealthCheckResult(
                    name="orchestrator",
                    passed=False,
                    message=(
                        "Integration orchestrator "
                        "returned an unexpected output."
                    ),
                )

            if result.stage_names != [
                "health_check"
            ]:

                return HealthCheckResult(
                    name="orchestrator",
                    passed=False,
                    message=(
                        "Integration orchestrator "
                        "stage tracking is invalid."
                    ),
                )

        except Exception as exc:

            return HealthCheckResult(
                name="orchestrator",
                passed=False,
                message=(
                    f"Orchestrator check failed: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            )

        return HealthCheckResult(
            name="orchestrator",
            passed=True,
            message="Integration orchestrator is operational.",
        )

    # =====================================================
    # FULL HEALTH CHECK
    # =====================================================

    def check(
        self,
    ) -> QuantAISystemHealthResult:

        checks: List[
            HealthCheckResult
        ] = []

        errors: List[str] = []

        warnings: List[str] = []

        module_checks = (
            self.check_required_modules()
        )

        checks.extend(
            module_checks
        )

        integration_check = (
            self.check_integration_layer()
        )

        checks.append(
            integration_check
        )

        orchestrator_check = (
            self.check_orchestrator()
        )

        checks.append(
            orchestrator_check
        )

        for check in checks:

            if not check.passed:

                errors.append(
                    f"{check.name}: "
                    f"{check.message}"
                )

        healthy = not errors

        ready = (
            healthy
            and len(checks) > 0
        )

        return QuantAISystemHealthResult(
            healthy=healthy,
            ready=ready,
            checks=checks,
            errors=errors,
            warnings=warnings,
        )

    # =====================================================
    # CONVENIENCE
    # =====================================================

    def is_ready(self) -> bool:

        return self.check().ready


def run_system_health_check(
    required_modules: Sequence[str] | None = None,
) -> QuantAISystemHealthResult:

    checker = QuantAISystemHealth(
        required_modules=required_modules,
    )

    return checker.check()


__all__ = [
    "HealthCheckResult",
    "QuantAISystemHealthResult",
    "QuantAISystemHealth",
    "run_system_health_check",
]