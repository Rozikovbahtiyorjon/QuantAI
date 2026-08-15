from __future__ import annotations

import pytest

from src.strategy_bank import (
    VALID_STATUSES,
    StrategyRecord,
    StrategyRegistry,
)
from src.strategy_genome import StrategyGenome


def make_genome(
    strategy_id: str = "strategy_001",
) -> StrategyGenome:
    return StrategyGenome(
        strategy_id=strategy_id,
        version="1.0.0",
        market="BTC/USDT",
        timeframes=("15m", "1h"),
        features=("returns", "rsi", "atr"),
        indicators=("EMA", "RSI", "ATR"),
        ml_model="XGBoostClassifier",
        regime_filters=("TREND_UP", "RANGE"),
        entry_logic={
            "ml_confirmation": True,
        },
        exit_logic={
            "take_profit": 0.02,
            "stop_loss": 0.01,
        },
        risk_profile="BALANCED",
        position_sizing={
            "method": "confidence_adjusted",
        },
        portfolio_constraints={
            "max_exposure": 0.40,
        },
        parameters={
            "rsi_period": 14,
        },
    )


def test_register_strategy() -> None:
    registry = StrategyRegistry()

    record = registry.register(make_genome())

    assert isinstance(record, StrategyRecord)
    assert record.status == "candidate"
    assert registry.count() == 1


def test_register_custom_status() -> None:
    registry = StrategyRegistry()

    record = registry.register(
        make_genome(),
        status="experimental",
    )

    assert record.status == "experimental"


def test_duplicate_registration_is_rejected() -> None:
    registry = StrategyRegistry()

    registry.register(make_genome())

    with pytest.raises(ValueError):
        registry.register(make_genome())


def test_get_strategy() -> None:
    registry = StrategyRegistry()
    genome = make_genome()

    registry.register(genome)

    result = registry.get("strategy_001")

    assert result.genome == genome


def test_get_unknown_strategy() -> None:
    registry = StrategyRegistry()

    with pytest.raises(KeyError):
        registry.get("missing")


def test_contains() -> None:
    registry = StrategyRegistry()

    assert registry.contains("strategy_001") is False

    registry.register(make_genome())

    assert registry.contains("strategy_001") is True


def test_update_status() -> None:
    registry = StrategyRegistry()

    registry.register(make_genome())

    result = registry.update_status(
        "strategy_001",
        "validated",
    )

    assert result.status == "validated"
    assert (
        registry.get("strategy_001").status
        == "validated"
    )


def test_invalid_status_is_rejected() -> None:
    registry = StrategyRegistry()

    with pytest.raises(ValueError):
        registry.register(
            make_genome(),
            status="invalid",
        )


def test_list_by_status() -> None:
    registry = StrategyRegistry()

    registry.register(
        make_genome("strategy_001")
    )

    registry.register(
        make_genome("strategy_002"),
        status="validated",
    )

    registry.register(
        make_genome("strategy_003"),
        status="validated",
    )

    validated = registry.list(
        status="validated"
    )

    assert len(validated) == 2

    assert all(
        record.status == "validated"
        for record in validated
    )


def test_champion_initially_none() -> None:
    registry = StrategyRegistry()

    assert registry.champion() is None


def test_set_champion() -> None:
    registry = StrategyRegistry()

    registry.register(
        make_genome("strategy_001")
    )

    registry.register(
        make_genome("strategy_002"),
        status="validated",
    )

    champion = registry.set_champion(
        "strategy_002"
    )

    assert champion.status == "champion"
    assert registry.champion() == champion


def test_replacing_champion_demotes_previous() -> None:
    registry = StrategyRegistry()

    registry.register(
        make_genome("strategy_001"),
        status="champion",
    )

    registry.register(
        make_genome("strategy_002"),
        status="validated",
    )

    new_champion = registry.set_champion(
        "strategy_002"
    )

    assert new_champion.status == "champion"

    assert (
        registry.get("strategy_001").status
        == "validated"
    )


def test_count_by_status() -> None:
    registry = StrategyRegistry()

    registry.register(
        make_genome("strategy_001")
    )

    registry.register(
        make_genome("strategy_002"),
        status="validated",
    )

    assert registry.count() == 2
    assert registry.count("candidate") == 1
    assert registry.count("validated") == 1


def test_serialization_round_trip() -> None:
    registry = StrategyRegistry()

    registry.register(
        make_genome("strategy_001")
    )

    registry.register(
        make_genome("strategy_002"),
        status="validated",
    )

    restored = StrategyRegistry.from_dict(
        registry.to_dict()
    )

    assert restored.to_dict() == registry.to_dict()


def test_remove_strategy() -> None:
    registry = StrategyRegistry()

    registry.register(make_genome())

    removed = registry.remove(
        "strategy_001"
    )

    assert (
        removed.genome.strategy_id
        == "strategy_001"
    )

    assert registry.count() == 0


def test_constructor_validation() -> None:
    with pytest.raises(TypeError):
        StrategyRecord(
            genome="invalid",
            status="candidate",
        )

    with pytest.raises(TypeError):
        StrategyRegistry().register(
            genome="invalid",
        )

    with pytest.raises(TypeError):
        StrategyRegistry().get(123)

    with pytest.raises(ValueError):
        StrategyRegistry().get("")

    with pytest.raises(TypeError):
        StrategyRegistry().list(status=123)

    with pytest.raises(ValueError):
        StrategyRegistry().list(
            status="invalid"
        )


def test_from_dict_validation() -> None:
    with pytest.raises(TypeError):
        StrategyRegistry.from_dict([])

    with pytest.raises(ValueError):
        StrategyRegistry.from_dict(
            {
                "strategy_001": {
                    "status": "candidate",
                }
            }
        )

    data = {
        "strategy_001": {
            "genome": make_genome(
                "different_id"
            ).to_dict(),
            "status": "candidate",
        }
    }

    with pytest.raises(ValueError):
        StrategyRegistry.from_dict(data)


def test_all_statuses_are_supported() -> None:
    registry = StrategyRegistry()

    for index, status in enumerate(
        sorted(VALID_STATUSES)
    ):
        registry.register(
            make_genome(
                f"strategy_{index:03d}"
            ),
            status=status,
        )

    assert (
        registry.count()
        == len(VALID_STATUSES)
    )