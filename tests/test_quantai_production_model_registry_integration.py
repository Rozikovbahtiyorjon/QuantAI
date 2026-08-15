from dataclasses import dataclass, field

import pytest

from src.quantai_production_model_registry_integration import (
    ProductionModelResolutionResult,
    QuantAIProductionModelRegistryIntegration,
)


@dataclass
class Model:
    name: str
    version: str
    artifact: object = object()

    @property
    def identifier(self):
        return f"{self.name}:{self.version}"


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

        self.promotions = []

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
        self.promotions.append(identifier)

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


def make():
    return QuantAIProductionModelRegistryIntegration(
        Registry()
    )


def test_constructor_rejects_none():
    with pytest.raises(TypeError):
        QuantAIProductionModelRegistryIntegration(
            None
        )


def test_constructor_rejects_invalid_registry():
    with pytest.raises(TypeError):
        QuantAIProductionModelRegistryIntegration(
            object()
        )


def test_constructor_rejects_invalid_artifact_flag():
    with pytest.raises(TypeError):
        QuantAIProductionModelRegistryIntegration(
            Registry(),
            require_artifact=1,
        )


def test_resolve_champion():
    result = make().resolve_champion()

    assert isinstance(
        result,
        ProductionModelResolutionResult,
    )

    assert result.success is True

    assert result.active_identifier == "xgb:1"


def test_activate_champion():
    integration = make()

    result = integration.activate_champion()

    assert result.ready is True

    assert integration.is_active is True

    assert integration.active_identifier == "xgb:1"


def test_clear_active_model():
    integration = make()

    integration.activate_champion()

    integration.clear_active_model()

    assert integration.is_active is False

    assert integration.active_identifier is None


def test_missing_champion_fails():
    integration = make()

    integration.registry._champion = None

    result = integration.resolve_champion()

    assert result.success is False

    assert "No champion" in result.errors[0]


def test_artifact_requirement():
    integration = (
        QuantAIProductionModelRegistryIntegration(
            Registry(),
            require_artifact=True,
        )
    )

    integration.registry._champion.artifact = None

    result = integration.resolve_champion()

    assert result.success is False

    assert "artifact" in result.errors[0]


def test_missing_identifier_fails():
    integration = make()

    integration.registry._champion = object()

    result = integration.resolve_champion()

    assert result.success is False

    assert "identifier" in result.errors[0]


def test_promote_best_challenger():
    integration = make()

    result = integration.promote_challenger()

    assert result.success is True

    assert result.active_identifier == "xgb:2"


def test_promote_explicit_challenger():
    integration = make()

    result = integration.promote_challenger(
        "xgb:2"
    )

    assert result.success is True

    assert integration.active_identifier == "xgb:2"


def test_invalid_challenger_identifier_fails():
    integration = make()

    result = integration.promote_challenger(
        "xgb:99"
    )

    assert result.success is False

    assert result.errors == [
        "unknown challenger"
    ]


def test_promotion_exception_is_propagated():
    integration = make()

    def failing_promote(identifier):
        raise RuntimeError(
            "promotion failure"
        )

    integration.registry.promote = (
        failing_promote
    )

    result = integration.promote_challenger(
        "xgb:2"
    )

    assert result.success is False

    assert (
        "promotion failure"
        in result.errors[0]
    )


def test_rollback_restores_previous_champion():
    integration = make()

    integration.promote_challenger(
        "xgb:2"
    )

    result = integration.rollback_champion()

    assert result.success is True

    assert result.active_identifier == "xgb:1"


def test_rollback_without_previous_fails():
    integration = make()

    result = integration.rollback_champion()

    assert result.success is False

    assert (
        "no previous champion"
        in result.errors[0]
    )


def test_rollback_exception_is_propagated():
    integration = make()

    def failing_rollback():
        raise RuntimeError(
            "rollback failure"
        )

    integration.registry.rollback = (
        failing_rollback
    )

    result = integration.rollback_champion()

    assert result.success is False

    assert (
        "rollback failure"
        in result.errors[0]
    )


def test_failed_registry_result_without_errors_gets_default_error():
    integration = make()

    integration.registry.promote = (
        lambda identifier: RegistryResult(
            False
        )
    )

    result = integration.promote_challenger(
        "xgb:2"
    )

    assert result.success is False

    assert result.errors == [
        "Challenger promotion was rejected."
    ]


def test_failed_rollback_result_without_errors_gets_default_error():
    integration = make()

    integration.registry.rollback = (
        lambda: RegistryResult(False)
    )

    result = integration.rollback_champion()

    assert result.success is False

    assert result.errors == [
        "Champion rollback was rejected."
    ]


def test_identifier_fallback_from_name_and_version():
    integration = make()

    integration.registry._champion = type(
        "ModelWithoutIdentifier",
        (),
        {
            "name": "rf",
            "version": "3",
            "artifact": object(),
        },
    )()

    result = integration.resolve_champion()

    assert result.success is True

    assert result.active_identifier == "rf:3"
