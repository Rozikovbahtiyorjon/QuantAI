from __future__ import annotations

import pytest

from experimental.src.quantai_production_model_registry import (
    ModelRegistryEvent,
    ModelRegistryResult,
    ModelVersion,
    QuantAIProductionModelRegistry,
    create_production_model_registry,
)


def make_model(
    name: str,
    version: str,
    performance: float,
    stability: float,
) -> ModelVersion:
    return ModelVersion(
        name=name,
        version=version,
        artifact=object(),
        performance_score=performance,
        stability_score=stability,
    )


def test_register_model():
    registry = QuantAIProductionModelRegistry()

    result = registry.register(
        make_model(
            "xgb",
            "1.0",
            0.80,
            0.90,
        )
    )

    assert isinstance(
        result,
        ModelRegistryResult,
    )

    assert result.success is True

    assert registry.get(
        "xgb:1.0"
    ) is not None

    assert len(
        registry.models
    ) == 1


def test_duplicate_registration_fails():
    registry = QuantAIProductionModelRegistry()

    model = make_model(
        "xgb",
        "1.0",
        0.80,
        0.90,
    )

    registry.register(model)

    result = registry.register(model)

    assert result.success is False
    assert result.errors


def test_promote_eligible_model():
    registry = QuantAIProductionModelRegistry(
        minimum_performance_score=0.70,
        minimum_stability_score=0.80,
    )

    registry.register(
        make_model(
            "xgb",
            "1.0",
            0.80,
            0.90,
        )
    )

    result = registry.promote(
        "xgb:1.0"
    )

    assert result.success is True

    assert (
        result.champion_identifier
        == "xgb:1.0"
    )


def test_promotion_rejects_below_threshold():
    registry = QuantAIProductionModelRegistry(
        minimum_performance_score=0.80,
        minimum_stability_score=0.80,
    )

    registry.register(
        make_model(
            "xgb",
            "1.0",
            0.79,
            0.90,
        )
    )

    result = registry.promote(
        "xgb:1.0"
    )

    assert result.success is False
    assert registry.champion is None


def test_previous_champion_is_preserved_on_promotion():
    registry = QuantAIProductionModelRegistry()

    registry.register(
        make_model(
            "xgb",
            "1.0",
            0.80,
            0.80,
        )
    )

    registry.register(
        make_model(
            "xgb",
            "2.0",
            0.90,
            0.90,
        )
    )

    registry.promote(
        "xgb:1.0"
    )

    registry.promote(
        "xgb:2.0"
    )

    assert (
        registry.champion.identifier
        == "xgb:2.0"
    )

    assert (
        registry.previous_champion.identifier
        == "xgb:1.0"
    )


def test_rollback_restores_previous_champion():
    registry = QuantAIProductionModelRegistry()

    registry.register(
        make_model(
            "xgb",
            "1.0",
            0.80,
            0.80,
        )
    )

    registry.register(
        make_model(
            "xgb",
            "2.0",
            0.90,
            0.90,
        )
    )

    registry.promote(
        "xgb:1.0"
    )

    registry.promote(
        "xgb:2.0"
    )

    result = registry.rollback()

    assert result.success is True

    assert (
        registry.champion.identifier
        == "xgb:1.0"
    )

    assert (
        registry.previous_champion.identifier
        == "xgb:2.0"
    )


def test_rollback_without_history_fails():
    registry = QuantAIProductionModelRegistry()

    result = registry.rollback()

    assert result.success is False


def test_demote_champion():
    registry = QuantAIProductionModelRegistry()

    registry.register(
        make_model(
            "xgb",
            "1.0",
            0.80,
            0.80,
        )
    )

    registry.promote(
        "xgb:1.0"
    )

    result = registry.demote(
        "xgb:1.0"
    )

    assert result.success is True
    assert registry.champion is None


def test_cannot_unregister_active_champion():
    registry = QuantAIProductionModelRegistry()

    registry.register(
        make_model(
            "xgb",
            "1.0",
            0.80,
            0.80,
        )
    )

    registry.promote(
        "xgb:1.0"
    )

    result = registry.unregister(
        "xgb:1.0"
    )

    assert result.success is False

    assert (
        registry.get(
            "xgb:1.0"
        )
        is not None
    )


def test_unregister_non_champion():
    registry = QuantAIProductionModelRegistry()

    registry.register(
        make_model(
            "xgb",
            "1.0",
            0.80,
            0.80,
        )
    )

    registry.register(
        make_model(
            "xgb",
            "2.0",
            0.70,
            0.70,
        )
    )

    result = registry.unregister(
        "xgb:2.0"
    )

    assert result.success is True

    assert registry.get(
        "xgb:2.0"
    ) is None


def test_best_challenger():
    registry = QuantAIProductionModelRegistry(
        minimum_performance_score=0.70,
        minimum_stability_score=0.70,
    )

    registry.register(
        make_model(
            "xgb",
            "1.0",
            0.80,
            0.80,
        )
    )

    registry.register(
        make_model(
            "xgb",
            "2.0",
            0.90,
            0.85,
        )
    )

    registry.register(
        make_model(
            "xgb",
            "3.0",
            0.85,
            0.95,
        )
    )

    registry.promote(
        "xgb:1.0"
    )

    challenger = (
        registry.select_best_challenger()
    )

    assert challenger is not None

    assert (
        challenger.identifier
        == "xgb:2.0"
    )


def test_best_challenger_returns_none_when_no_eligible_model():
    registry = QuantAIProductionModelRegistry(
        minimum_performance_score=0.90,
        minimum_stability_score=0.90,
    )

    registry.register(
        make_model(
            "xgb",
            "1.0",
            0.90,
            0.90,
        )
    )

    registry.promote(
        "xgb:1.0"
    )

    registry.register(
        make_model(
            "xgb",
            "2.0",
            0.70,
            0.70,
        )
    )

    assert (
        registry.select_best_challenger()
        is None
    )


def test_missing_model_promotion_fails():
    registry = QuantAIProductionModelRegistry()

    result = registry.promote(
        "missing:1.0"
    )

    assert result.success is False


def test_model_lookup():
    registry = QuantAIProductionModelRegistry()

    registry.register(
        make_model(
            "xgb",
            "1.0",
            0.80,
            0.80,
        )
    )

    model = registry.get(
        "xgb:1.0"
    )

    assert model is not None
    assert model.name == "xgb"


def test_events_are_recorded():
    registry = QuantAIProductionModelRegistry()

    registry.register(
        make_model(
            "xgb",
            "1.0",
            0.80,
            0.80,
        )
    )

    registry.promote(
        "xgb:1.0"
    )

    assert len(
        registry.events
    ) == 2

    assert all(
        isinstance(
            event,
            ModelRegistryEvent,
        )
        for event in registry.events
    )


def test_invalid_model_type():
    registry = QuantAIProductionModelRegistry()

    with pytest.raises(TypeError):
        registry.register(
            "invalid"
        )


def test_invalid_registry_threshold():
    with pytest.raises(ValueError):
        QuantAIProductionModelRegistry(
            minimum_performance_score=1.1
        )

    with pytest.raises(ValueError):
        QuantAIProductionModelRegistry(
            minimum_stability_score=-0.1
        )


def test_invalid_registry_threshold_type():
    with pytest.raises(TypeError):
        QuantAIProductionModelRegistry(
            minimum_performance_score=True
        )


def test_invalid_model_score():
    registry = QuantAIProductionModelRegistry()

    with pytest.raises(ValueError):
        registry.register(
            make_model(
                "xgb",
                "1.0",
                1.1,
                0.80,
            )
        )


def test_invalid_model_name():
    registry = QuantAIProductionModelRegistry()

    with pytest.raises(ValueError):
        registry.register(
            make_model(
                "",
                "1.0",
                0.80,
                0.80,
            )
        )


def test_invalid_model_version():
    registry = QuantAIProductionModelRegistry()

    with pytest.raises(ValueError):
        registry.register(
            make_model(
                "xgb",
                "",
                0.80,
                0.80,
            )
        )


def test_empty_champion_state():
    registry = QuantAIProductionModelRegistry()

    assert registry.champion is None
    assert registry.previous_champion is None


def test_factory():
    registry = create_production_model_registry(
        minimum_performance_score=0.70,
        minimum_stability_score=0.80,
    )

    assert (
        registry.minimum_performance_score
        == 0.70
    )

    assert (
        registry.minimum_stability_score
        == 0.80
    )