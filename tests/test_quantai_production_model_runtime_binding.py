from dataclasses import dataclass, field

import pytest

from src.quantai_production_model_registry_integration import (
    QuantAIProductionModelRegistryIntegration,
)
from src.quantai_production_model_runtime_binding import (
    ProductionModelBindingResult,
    QuantAIProductionModelRuntimeBinding,
)


@dataclass
class Model:
    name: str
    version: str
    artifact: object = field(default_factory=object)

    @property
    def identifier(self):
        return f"{self.name}:{self.version}"

    def predict(self, data):
        return {"value": len(data)}


@dataclass
class RegistryResult:
    success: bool
    errors: list[str] = field(
        default_factory=list
    )
    warnings: list[str] = field(
        default_factory=list
    )


class Registry:
    def __init__(self):
        self._champion = Model(
            "xgb",
            "1",
        )

        self.previous = None

        self.challenger = Model(
            "xgb",
            "2",
        )

    @property
    def champion(self):
        return self._champion

    def select_best_challenger(self):
        return self.challenger

    def promote(
        self,
        identifier,
    ):
        if identifier != self.challenger.identifier:
            return RegistryResult(
                False,
                ["unknown challenger"],
            )

        self.previous = self._champion
        self._champion = self.challenger

        return RegistryResult(True)

    def rollback(self):
        if self.previous is None:
            return RegistryResult(
                False,
                ["no previous champion"],
            )

        current = self._champion
        self._champion = self.previous
        self.previous = current

        return RegistryResult(True)


def make_binding():
    integration = (
        QuantAIProductionModelRegistryIntegration(
            Registry()
        )
    )

    return QuantAIProductionModelRuntimeBinding(
        integration
    )


def test_constructor_rejects_invalid_integration():
    with pytest.raises(TypeError):
        QuantAIProductionModelRuntimeBinding(
            object()
        )


def test_initial_state_is_unbound():
    binding = make_binding()

    assert binding.is_bound is False
    assert binding.model is None
    assert binding.identifier is None


def test_bind_resolves_champion():
    binding = make_binding()

    result = binding.bind()

    assert isinstance(
        result,
        ProductionModelBindingResult,
    )

    assert result.success is True
    assert result.bound is True
    assert result.identifier == "xgb:1"

    assert binding.is_bound is True


def test_activate_and_bind():
    binding = make_binding()

    result = binding.activate_and_bind()

    assert result.success is True
    assert binding.identifier == "xgb:1"


def test_bind_fails_without_champion():
    binding = make_binding()

    binding.registry_integration.registry._champion = None

    result = binding.bind()

    assert result.success is False
    assert binding.is_bound is False


def test_predict_requires_binding():
    binding = make_binding()

    result = binding.predict(
        [1, 2, 3]
    )

    assert result.success is False

    assert (
        "No production model"
        in result.errors[0]
    )


def test_predict_executes_bound_model():
    binding = make_binding()

    binding.bind()

    result = binding.predict(
        [1, 2, 3]
    )

    assert result.success is True
    assert result.prediction == {
        "value": 3
    }

    assert result.identifier == "xgb:1"


def test_predict_rejects_model_without_predict():
    binding = make_binding()

    binding.registry_integration.registry._champion = (
        type(
            "NoPredictModel",
            (),
            {
                "name": "rf",
                "version": "1",
                "artifact": object(),
            },
        )()
    )

    binding.bind()

    result = binding.predict(
        [1]
    )

    assert result.success is False

    assert (
        "does not expose predict()"
        in result.errors[0]
    )


def test_prediction_exception_is_propagated():
    binding = make_binding()

    class FailingModel(Model):
        def predict(self, data):
            raise RuntimeError(
                "prediction failure"
            )

    binding.registry_integration.registry._champion = (
        FailingModel(
            "xgb",
            "3",
        )
    )

    binding.bind()

    result = binding.predict(
        [1]
    )

    assert result.success is False

    assert (
        "prediction failure"
        in result.errors[0]
    )


def test_rebind_after_promotion():
    binding = make_binding()

    binding.bind()

    result = binding.rebind_after_promotion()

    assert result.success is True
    assert result.identifier == "xgb:2"

    assert binding.identifier == "xgb:2"


def test_failed_promotion_keeps_current_binding():
    binding = make_binding()

    binding.bind()

    result = binding.rebind_after_promotion(
        "xgb:99"
    )

    assert result.success is False
    assert binding.identifier == "xgb:1"


def test_rebind_after_rollback():
    binding = make_binding()

    binding.bind()

    binding.rebind_after_promotion()

    result = binding.rebind_after_rollback()

    assert result.success is True
    assert result.identifier == "xgb:1"


def test_failed_rollback_keeps_current_binding():
    binding = make_binding()

    binding.bind()

    result = binding.rebind_after_rollback()

    assert result.success is False
    assert binding.identifier == "xgb:1"


def test_unbind():
    binding = make_binding()

    binding.bind()

    binding.unbind()

    assert binding.is_bound is False
    assert binding.model is None
    assert binding.identifier is None


def test_promotion_prediction_uses_new_model():
    binding = make_binding()

    binding.bind()

    binding.rebind_after_promotion()

    result = binding.predict(
        [1, 2]
    )

    assert result.success is True
    assert result.identifier == "xgb:2"
    assert result.prediction == {
        "value": 2
    }


def test_rollback_prediction_uses_restored_model():
    binding = make_binding()

    binding.bind()

    binding.rebind_after_promotion()

    binding.rebind_after_rollback()

    result = binding.predict(
        [1, 2, 3, 4]
    )

    assert result.success is True
    assert result.identifier == "xgb:1"
    assert result.prediction == {
        "value": 4
    }


def test_failed_promotion_preserves_model_object():
    binding = make_binding()

    binding.bind()

    original = binding.model

    binding.rebind_after_promotion(
        "invalid"
    )

    assert binding.model is original


def test_failed_rollback_preserves_model_object():
    binding = make_binding()

    binding.bind()

    original = binding.model

    binding.rebind_after_rollback()

    assert binding.model is original


def test_result_bound_requires_model():
    result = ProductionModelBindingResult(
        success=True,
        model=None,
    )

    assert result.bound is False