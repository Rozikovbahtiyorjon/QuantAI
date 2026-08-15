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
    PredictionHealthResult,
    PredictionHealthSnapshot,
    QuantAIProductionModelRuntimeMonitoring,
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
        self._champion = Model("xgb", "1")

    @property
    def champion(self):
        return self._champion

    def promote(self, identifier):
        return type(
            "Result",
            (),
            {"success": True, "errors": [], "warnings": []},
        )()

    def rollback(self):
        return type(
            "Result",
            (),
            {"success": True, "errors": [], "warnings": []},
        )()

    def select_best_challenger(self):
        return self._champion


def make_monitor(max_failure_rate=0.0):
    integration = QuantAIProductionModelRegistryIntegration(
        Registry()
    )
    binding = QuantAIProductionModelRuntimeBinding(
        integration
    )
    binding.bind()
    execution = QuantAIProductionModelRuntimeExecution(
        binding
    )
    return QuantAIProductionModelRuntimeMonitoring(
        execution,
        max_failure_rate=max_failure_rate,
    )


def test_constructor_rejects_invalid_execution():
    with pytest.raises(TypeError):
        QuantAIProductionModelRuntimeMonitoring(
            object()
        )


def test_constructor_rejects_invalid_threshold_type():
    integration = QuantAIProductionModelRegistryIntegration(
        Registry()
    )
    binding = QuantAIProductionModelRuntimeBinding(
        integration
    )
    binding.bind()
    execution = QuantAIProductionModelRuntimeExecution(
        binding
    )

    with pytest.raises(TypeError):
        QuantAIProductionModelRuntimeMonitoring(
            execution,
            max_failure_rate="0.1",
        )


def test_constructor_rejects_invalid_threshold_value():
    integration = QuantAIProductionModelRegistryIntegration(
        Registry()
    )
    binding = QuantAIProductionModelRuntimeBinding(
        integration
    )
    binding.bind()
    execution = QuantAIProductionModelRuntimeExecution(
        binding
    )

    with pytest.raises(ValueError):
        QuantAIProductionModelRuntimeMonitoring(
            execution,
            max_failure_rate=1.1,
        )


def test_initial_monitor_is_not_healthy():
    monitor = make_monitor()

    assert monitor.total_predictions == 0
    assert monitor.successful_predictions == 0
    assert monitor.failed_predictions == 0
    assert monitor.failure_rate == 0.0
    assert monitor.success_rate == 0.0
    assert monitor.is_healthy is False


def test_successful_prediction_is_healthy():
    monitor = make_monitor()

    result = monitor.monitor([1, 2, 3])

    assert isinstance(
        result,
        PredictionHealthResult,
    )
    assert result.healthy is True
    assert result.prediction == {"size": 3}
    assert result.model_identifier == "xgb:1"
    assert result.errors == []
    assert monitor.total_predictions == 1
    assert monitor.successful_predictions == 1
    assert monitor.failed_predictions == 0


def test_success_rate_is_calculated():
    monitor = make_monitor()

    monitor.monitor([1])
    monitor.monitor([1, 2])

    assert monitor.success_rate == 1.0
    assert monitor.failure_rate == 0.0


def test_none_input_is_recorded_as_failure():
    monitor = make_monitor()

    result = monitor.monitor(None)

    assert result.healthy is False
    assert result.errors
    assert monitor.total_predictions == 1
    assert monitor.successful_predictions == 0
    assert monitor.failed_predictions == 1
    assert monitor.failure_rate == 1.0


def test_failure_rate_threshold_is_applied():
    monitor = make_monitor(
        max_failure_rate=0.5
    )

    monitor.monitor([1])
    result = monitor.monitor(None)

    assert result.healthy is False
    assert monitor.failure_rate == 0.5
    assert monitor.is_healthy is True


def test_failure_rate_above_threshold_marks_unhealthy():
    monitor = make_monitor(
        max_failure_rate=0.0
    )

    result = monitor.monitor(None)

    assert result.healthy is False
    assert monitor.is_healthy is False
    assert any(
        "failure rate" in warning.lower()
        for warning in result.warnings
    )


def test_batch_monitoring_returns_results():
    monitor = make_monitor()

    results = monitor.monitor_batch(
        [
            [1],
            [1, 2],
            [1, 2, 3],
        ]
    )

    assert len(results) == 3
    assert all(
        isinstance(
            result,
            PredictionHealthResult,
        )
        for result in results
    )
    assert all(
        result.healthy
        for result in results
    )
    assert monitor.total_predictions == 3


def test_batch_monitoring_counts_failures():
    monitor = make_monitor()

    results = monitor.monitor_batch(
        [
            [1],
            None,
            [1, 2],
            None,
        ]
    )

    assert len(results) == 4
    assert monitor.total_predictions == 4
    assert monitor.successful_predictions == 2
    assert monitor.failed_predictions == 2
    assert monitor.failure_rate == 0.5


def test_batch_monitoring_rejects_non_list():
    monitor = make_monitor()

    with pytest.raises(TypeError):
        monitor.monitor_batch(
            (1, 2)
        )


def test_snapshot_contains_current_metrics():
    monitor = make_monitor()

    monitor.monitor([1])

    snapshot = monitor.snapshot()

    assert isinstance(
        snapshot,
        PredictionHealthSnapshot,
    )
    assert snapshot.total_predictions == 1
    assert snapshot.successful_predictions == 1
    assert snapshot.failed_predictions == 0
    assert snapshot.success_rate == 1.0
    assert snapshot.failure_rate == 0.0
    assert snapshot.last_model_identifier == "xgb:1"
    assert snapshot.last_prediction == {"size": 1}


def test_snapshot_healthy_property():
    monitor = make_monitor()

    assert monitor.snapshot().healthy is False

    monitor.monitor([1])

    assert monitor.snapshot().healthy is True


def test_last_prediction_is_updated():
    monitor = make_monitor()

    monitor.monitor([1])
    monitor.monitor([1, 2, 3])

    snapshot = monitor.snapshot()

    assert snapshot.last_prediction == {
        "size": 3
    }


def test_model_identifier_is_preserved_after_failure():
    monitor = make_monitor()

    monitor.monitor([1])

    monitor.execution.binding.model.predict = (
        lambda data: (_ for _ in ()).throw(
            RuntimeError("failure")
        )
    )

    result = monitor.monitor([2])

    assert result.healthy is False
    assert result.model_identifier == "xgb:1"


def test_reset_clears_metrics():
    monitor = make_monitor()

    monitor.monitor([1])
    monitor.monitor(None)

    monitor.reset()

    assert monitor.total_predictions == 0
    assert monitor.successful_predictions == 0
    assert monitor.failed_predictions == 0
    assert monitor.success_rate == 0.0
    assert monitor.failure_rate == 0.0
    assert monitor.snapshot().last_model_identifier is None
    assert monitor.snapshot().last_prediction is None


def test_empty_batch_does_not_change_metrics():
    monitor = make_monitor()

    results = monitor.monitor_batch([])

    assert results == []
    assert monitor.total_predictions == 0


def test_failed_result_preserves_execution_errors():
    monitor = make_monitor()

    result = monitor.monitor(None)

    assert any(
        "must not be None" in error
        for error in result.errors
    )