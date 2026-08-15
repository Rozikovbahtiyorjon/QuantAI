from dataclasses import dataclass, field

import pytest

from src.quantai_production_model_registry_integration import (
    QuantAIProductionModelRegistryIntegration,
)
from src.quantai_production_model_runtime_binding import (
    QuantAIProductionModelRuntimeBinding,
)
from src.quantai_production_model_runtime_execution import (
    ProductionInferenceResult,
    QuantAIProductionModelRuntimeExecution,
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
        return {
            "size": len(data)
        }


class Registry:
    def __init__(self):
        self._champion = Model(
            "xgb",
            "1",
        )

    @property
    def champion(self):
        return self._champion

    def promote(
        self,
        identifier,
    ):
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


def make_gateway():
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

    return QuantAIProductionModelRuntimeExecution(
        binding
    )


def test_constructor_rejects_invalid_binding():
    with pytest.raises(TypeError):
        QuantAIProductionModelRuntimeExecution(
            object()
        )


def test_initial_gateway_is_not_ready():
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

    gateway = (
        QuantAIProductionModelRuntimeExecution(
            binding
        )
    )

    assert gateway.is_ready is False
    assert gateway.prediction_count == 0
    assert gateway.failure_count == 0


def test_execute_requires_active_model():
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

    gateway = (
        QuantAIProductionModelRuntimeExecution(
            binding
        )
    )

    result = gateway.execute(
        [1, 2]
    )

    assert isinstance(
        result,
        ProductionInferenceResult,
    )

    assert result.success is False
    assert gateway.failure_count == 1


def test_execute_rejects_none_input():
    gateway = make_gateway()

    result = gateway.execute(
        None
    )

    assert result.success is False

    assert (
        "must not be None"
        in result.errors[0]
    )

    assert gateway.failure_count == 1


def test_execute_returns_prediction():
    gateway = make_gateway()

    result = gateway.execute(
        [1, 2, 3]
    )

    assert result.success is True
    assert result.ready is True

    assert result.prediction == {
        "size": 3
    }

    assert result.model_identifier == "xgb:1"

    assert gateway.prediction_count == 1
    assert gateway.failure_count == 0


def test_execute_preserves_model_identifier():
    gateway = make_gateway()

    result = gateway.execute(
        [1]
    )

    assert (
        result.model_identifier
        == gateway.binding.identifier
    )


def test_execute_increments_prediction_counter_only_on_success():
    gateway = make_gateway()

    gateway.execute([1])
    gateway.execute([1, 2])

    assert gateway.prediction_count == 2
    assert gateway.failure_count == 0


def test_execute_batch_returns_one_result_per_item():
    gateway = make_gateway()

    results = gateway.execute_batch(
        [
            [1],
            [1, 2],
            [1, 2, 3],
        ]
    )

    assert len(results) == 3

    assert all(
        result.success
        for result in results
    )

    assert gateway.prediction_count == 3


def test_execute_batch_rejects_non_list():
    gateway = make_gateway()

    with pytest.raises(TypeError):
        gateway.execute_batch(
            (1, 2)
        )


def test_execute_batch_handles_none_item():
    gateway = make_gateway()

    results = gateway.execute_batch(
        [
            [1],
            None,
            [1, 2],
        ]
    )

    assert results[0].success is True
    assert results[1].success is False
    assert results[2].success is True

    assert gateway.prediction_count == 2
    assert gateway.failure_count == 1


def test_model_prediction_failure_is_propagated():
    gateway = make_gateway()

    def failing_predict(data):
        raise RuntimeError(
            "model failure"
        )

    gateway.binding.model.predict = (
        failing_predict
    )

    result = gateway.execute(
        [1]
    )

    assert result.success is False

    assert (
        "model failure"
        in result.errors[0]
    )

    assert gateway.failure_count == 1


def test_binding_prediction_failure_is_propagated():
    gateway = make_gateway()

    def failing_binding_predict(data):
        raise RuntimeError(
            "binding failure"
        )

    gateway.binding.predict = (
        failing_binding_predict
    )

    result = gateway.execute(
        [1]
    )

    assert result.success is False

    assert (
        "binding failure"
        in result.errors[0]
    )

    assert gateway.failure_count == 1


def test_none_prediction_is_rejected():
    gateway = make_gateway()

    gateway.binding.model.predict = (
        lambda data: None
    )

    result = gateway.execute(
        [1]
    )

    assert result.success is False

    assert (
        "no prediction"
        in result.errors[0]
    )

    assert gateway.failure_count == 1


def test_reset_counters():
    gateway = make_gateway()

    gateway.execute([1])
    gateway.execute(None)

    gateway.reset_counters()

    assert gateway.prediction_count == 0
    assert gateway.failure_count == 0


def test_ready_property_requires_success_and_no_errors():
    assert ProductionInferenceResult(
        success=True,
        prediction=1,
    ).ready is True

    assert ProductionInferenceResult(
        success=False,
        prediction=1,
    ).ready is False

    assert ProductionInferenceResult(
        success=True,
        prediction=1,
        errors=["error"],
    ).ready is False


def test_empty_input_collection_is_valid():
    gateway = make_gateway()

    result = gateway.execute(
        []
    )

    assert result.success is True

    assert result.prediction == {
        "size": 0
    }

    assert gateway.prediction_count == 1


def test_batch_empty_list_returns_empty_results():
    gateway = make_gateway()

    results = gateway.execute_batch(
        []
    )

    assert results == []

    assert gateway.prediction_count == 0
    assert gateway.failure_count == 0


def test_batch_failure_counts_are_aggregated():
    gateway = make_gateway()

    gateway.execute_batch(
        [
            [1],
            None,
            None,
            [1, 2],
        ]
    )

    assert gateway.prediction_count == 2
    assert gateway.failure_count == 2


def test_successful_result_contains_no_errors():
    gateway = make_gateway()

    result = gateway.execute(
        [1, 2]
    )

    assert result.errors == []
    assert result.warnings == []