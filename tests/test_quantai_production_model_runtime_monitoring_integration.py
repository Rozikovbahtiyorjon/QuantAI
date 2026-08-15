from dataclasses import dataclass, field

import pytest

from src.quantai_production_model_registry_integration import (
    QuantAIProductionModelRegistryIntegration,
)
from src.quantai_production_model_runtime_binding import (
    QuantAIProductionModelRuntimeBinding,
)
from src.quantai_production_model_runtime_execution import (
    QuantAIProductionModelRuntimeExecution,
)
from src.quantai_production_model_runtime_monitoring import (
    QuantAIProductionModelRuntimeMonitoring,
)
from src.quantai_production_model_runtime_monitoring_integration import (
    ModelHealthSupervisorResult,
    QuantAIProductionModelRuntimeMonitoringIntegration,
)


@dataclass
class Model:
    name: str
    version: str
    artifact: object = field(
        default_factory=object
    )

    @property
    def identifier(self):
        return f"{self.name}:{self.version}"

    def predict(self, data):
        return {"size": len(data)}


class Registry:
    def __init__(self):
        self._champion = Model(
            "xgb",
            "1",
        )

    @property
    def champion(self):
        return self._champion

    def promote(self, identifier):
        return type(
            "Result",
            (),
            {
                "success": True,
                "errors": [],
                "warnings": [],
            },
        )()

    def rollback(self):
        return type(
            "Result",
            (),
            {
                "success": True,
                "errors": [],
                "warnings": [],
            },
        )()

    def select_best_challenger(self):
        return self._champion


class Supervisor:
    def __init__(
        self,
        ready=True,
    ):
        self.ready = ready
        self.calls = 0
        self.last_health_checker = None

    def supervise(
        self,
        health_checker,
    ):
        self.calls += 1
        self.last_health_checker = health_checker

        health = health_checker()

        if hasattr(
            health,
            "healthy",
        ):
            healthy = health.healthy
        else:
            healthy = bool(health)

        return type(
            "Result",
            (),
            {
                "ready": (
                    self.ready
                    and healthy
                ),
                "errors": (
                    []
                    if self.ready and healthy
                    else [
                        "supervisor error"
                    ]
                ),
                "warnings": [],
            },
        )()


def make_integration(
    supervisor=None,
    max_failure_rate=0.0,
):
    integration = (
        QuantAIProductionModelRegistryIntegration(
            Registry()
        )
    )

    binding = (
        QuantAIProductionModelRuntimeBinding(
            integration
        )
    )

    binding.bind()

    execution = (
        QuantAIProductionModelRuntimeExecution(
            binding
        )
    )

    monitoring = (
        QuantAIProductionModelRuntimeMonitoring(
            execution,
            max_failure_rate=max_failure_rate,
        )
    )

    return (
        QuantAIProductionModelRuntimeMonitoringIntegration(
            monitoring,
            supervisor,
        )
    )


def test_constructor_accepts_monitoring_without_supervisor():
    integration = make_integration()

    assert integration.supervisor is None
    assert integration.is_healthy is False


def test_constructor_rejects_invalid_monitoring():
    with pytest.raises(TypeError):
        QuantAIProductionModelRuntimeMonitoringIntegration(
            object()
        )


def test_constructor_rejects_invalid_supervisor():
    with pytest.raises(TypeError):
        QuantAIProductionModelRuntimeMonitoringIntegration(
            make_integration().monitoring,
            object(),
        )


def test_healthy_prediction_is_ready():
    supervisor = Supervisor()

    integration = make_integration(
        supervisor=supervisor
    )

    result = integration.monitor(
        [1, 2, 3]
    )

    assert isinstance(
        result,
        ModelHealthSupervisorResult,
    )

    assert result.ready is True
    assert result.errors == []
    assert result.health_result is not None
    assert result.health_result.healthy is True
    assert result.supervisor_result is not None
    assert supervisor.calls == 1


def test_supervisor_receives_health_checker():
    supervisor = Supervisor()

    integration = make_integration(
        supervisor=supervisor
    )

    integration.monitor(
        [1]
    )

    assert callable(
        supervisor.last_health_checker
    )

    health = (
        supervisor.last_health_checker()
    )

    assert health.healthy is True


def test_unhealthy_model_blocks_runtime_readiness():
    integration = make_integration()

    result = integration.monitor(
        None
    )

    assert result.ready is False

    assert any(
        "health" in error.lower()
        for error in result.errors
    )


def test_supervisor_health_failure_is_propagated():
    supervisor = Supervisor(
        ready=False
    )

    integration = make_integration(
        supervisor=supervisor
    )

    result = integration.monitor(
        [1]
    )

    assert result.ready is False

    assert any(
        "supervisor" in error.lower()
        for error in result.errors
    )


def test_supervisor_exception_is_propagated():
    class FailingSupervisor:
        def supervise(
            self,
            health_checker,
        ):
            raise RuntimeError(
                "supervisor failure"
            )

    integration = make_integration(
        supervisor=FailingSupervisor()
    )

    result = integration.monitor(
        [1]
    )

    assert result.ready is False

    assert any(
        "supervisor failure" in error
        for error in result.errors
    )


def test_warnings_are_propagated():
    class WarningSupervisor:
        def supervise(
            self,
            health_checker,
        ):
            return type(
                "Result",
                (),
                {
                    "ready": True,
                    "errors": [],
                    "warnings": [
                        "runtime warning"
                    ],
                },
            )()

    integration = make_integration(
        supervisor=WarningSupervisor()
    )

    result = integration.monitor(
        [1]
    )

    assert result.ready is True
    assert "runtime warning" in (
        result.warnings
    )


def test_monitoring_without_supervisor_still_works():
    integration = make_integration()

    result = integration.monitor(
        [1, 2]
    )

    assert result.ready is True
    assert result.supervisor_result is None


def test_snapshot_is_delegated():
    integration = make_integration()

    integration.monitor(
        [1, 2]
    )

    snapshot = integration.snapshot()

    assert snapshot.total_predictions == 1
    assert snapshot.successful_predictions == 1
    assert snapshot.failed_predictions == 0


def test_reset_is_delegated():
    integration = make_integration()

    integration.monitor(
        [1]
    )

    integration.reset()

    snapshot = integration.snapshot()

    assert snapshot.total_predictions == 0
    assert snapshot.successful_predictions == 0
    assert snapshot.failed_predictions == 0


def test_recovery_is_not_attempted_when_healthy():
    integration = make_integration()

    calls = []

    result = integration.monitor_and_recover(
        [1],
        recovery=lambda: calls.append(
            "recovered"
        ),
    )

    assert result.ready is True
    assert result.recovery_attempted is False
    assert result.recovery_succeeded is False
    assert calls == []


def test_recovery_is_attempted_after_health_failure():
    integration = make_integration()

    calls = []

    result = integration.monitor_and_recover(
        None,
        recovery=lambda: calls.append(
            "recovered"
        ) or True,
    )

    assert result.ready is False
    assert result.recovery_attempted is True
    assert result.recovery_succeeded is True
    assert calls == ["recovered"]


def test_failed_recovery_is_reported():
    integration = make_integration()

    result = integration.monitor_and_recover(
        None,
        recovery=lambda: False,
    )

    assert result.ready is False
    assert result.recovery_attempted is True
    assert result.recovery_succeeded is False

    assert any(
        "recovery" in error.lower()
        for error in result.errors
    )


def test_recovery_exception_is_reported():
    integration = make_integration()

    def recovery():
        raise RuntimeError(
            "recovery failure"
        )

    result = integration.monitor_and_recover(
        None,
        recovery=recovery,
    )

    assert result.ready is False
    assert result.recovery_attempted is True
    assert result.recovery_succeeded is False

    assert any(
        "recovery failure" in error
        for error in result.errors
    )


def test_invalid_recovery_is_rejected():
    integration = make_integration()

    with pytest.raises(TypeError):
        integration.monitor_and_recover(
            None,
            recovery=object(),
        )


def test_supervisor_boolean_health_result_is_supported():
    class BooleanSupervisor:
        def supervise(
            self,
            health_checker,
        ):
            return type(
                "Result",
                (),
                {
                    "ready": True,
                    "errors": [],
                    "warnings": [],
                },
            )()

    integration = make_integration(
        supervisor=BooleanSupervisor()
    )

    result = integration.monitor(
        [1]
    )

    assert result.ready is True


def test_model_failure_reaches_supervisor():
    supervisor = Supervisor()

    integration = make_integration(
        supervisor=supervisor
    )

    integration.monitor(
        None
    )

    assert supervisor.calls == 1


def test_result_contains_health_result():
    integration = make_integration()

    result = integration.monitor(
        [1, 2, 3]
    )

    assert isinstance(
        result.health_result,
        type(
            integration.monitoring.monitor(
                [4]
            )
        ),
    )


def test_result_healthy_property_matches_ready():
    integration = make_integration()

    healthy_result = integration.monitor(
        [1]
    )

    unhealthy_result = integration.monitor(
        None
    )

    assert (
        healthy_result.healthy
        == healthy_result.ready
    )

    assert (
        unhealthy_result.healthy
        == unhealthy_result.ready
    )


def test_supervisor_is_optional_for_model_health():
    integration = make_integration()

    result = integration.monitor(
        [1]
    )

    assert result.supervisor_result is None
    assert result.ready is True