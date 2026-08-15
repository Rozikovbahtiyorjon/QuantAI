from __future__ import annotations

import pytest

from src.strategy_genome import StrategyGenome


def make_genome() -> StrategyGenome:
    return StrategyGenome(
        strategy_id="quantai_trend_v1",
        version="1.0.0",
        market="BTC/USDT",
        timeframes=("15m", "1h"),
        features=("returns", "rsi", "atr"),
        indicators=("EMA", "RSI", "ATR"),
        ml_model="XGBoostClassifier",
        regime_filters=("TREND_UP", "TREND_DOWN"),
        entry_logic={
            "technical_score": 0.6,
            "ml_confirmation": True,
        },
        exit_logic={
            "take_profit": 0.02,
            "stop_loss": 0.01,
        },
        risk_profile="BALANCED",
        position_sizing={
            "method": "confidence_adjusted",
            "max_position": 0.10,
        },
        portfolio_constraints={
            "max_exposure": 0.40,
            "max_drawdown": 0.15,
        },
        parameters={
            "rsi_period": 14,
            "ema_period": 20,
        },
    )


def test_genome_creation() -> None:
    genome = make_genome()

    assert genome.strategy_id == "quantai_trend_v1"
    assert genome.version == "1.0.0"
    assert genome.market == "BTC/USDT"
    assert genome.ml_model == "XGBoostClassifier"
    assert genome.risk_profile == "BALANCED"


def test_genome_is_immutable() -> None:
    genome = make_genome()

    with pytest.raises(AttributeError):
        genome.version = "2.0.0"


def test_to_dict() -> None:
    genome = make_genome()

    data = genome.to_dict()

    assert data["strategy_id"] == "quantai_trend_v1"
    assert data["timeframes"] == ["15m", "1h"]
    assert data["features"] == ["returns", "rsi", "atr"]
    assert data["parameters"]["rsi_period"] == 14


def test_from_dict_round_trip() -> None:
    genome = make_genome()

    restored = StrategyGenome.from_dict(
        genome.to_dict()
    )

    assert restored == genome


def test_from_dict_without_optional_parameters() -> None:
    data = make_genome().to_dict()
    data.pop("parameters")

    restored = StrategyGenome.from_dict(data)

    assert restored.parameters == {}


def test_evolve_changes_selected_fields() -> None:
    genome = make_genome()

    evolved = genome.evolve(
        version="1.1.0",
        features=("returns", "rsi", "atr", "volume"),
        parameters={
            "rsi_period": 21,
            "ema_period": 20,
        },
    )

    assert genome.version == "1.0.0"
    assert evolved.version == "1.1.0"
    assert evolved.features == (
        "returns",
        "rsi",
        "atr",
        "volume",
    )
    assert evolved.parameters["rsi_period"] == 21


def test_evolve_preserves_strategy_id() -> None:
    genome = make_genome()

    evolved = genome.evolve(
        version="2.0.0",
    )

    assert evolved.strategy_id == genome.strategy_id


def test_evolve_rejects_unknown_fields() -> None:
    genome = make_genome()

    with pytest.raises(ValueError):
        genome.evolve(
            unknown_field="invalid"
        )


def test_constructor_requires_non_empty_sequences() -> None:
    with pytest.raises(ValueError):
        StrategyGenome(
            strategy_id="test",
            version="1.0.0",
            market="BTC/USDT",
            timeframes=(),
            features=("rsi",),
            indicators=("RSI",),
            ml_model="XGBoost",
            regime_filters=("RANGE",),
            entry_logic={},
            exit_logic={},
            risk_profile="BALANCED",
            position_sizing={},
            portfolio_constraints={},
        )


def test_constructor_rejects_invalid_text() -> None:
    with pytest.raises(TypeError):
        StrategyGenome(
            strategy_id=123,
            version="1.0.0",
            market="BTC/USDT",
            timeframes=("15m",),
            features=("rsi",),
            indicators=("RSI",),
            ml_model="XGBoost",
            regime_filters=("RANGE",),
            entry_logic={},
            exit_logic={},
            risk_profile="BALANCED",
            position_sizing={},
            portfolio_constraints={},
        )


def test_constructor_rejects_invalid_sequence_type() -> None:
    with pytest.raises(TypeError):
        StrategyGenome(
            strategy_id="test",
            version="1.0.0",
            market="BTC/USDT",
            timeframes=["15m"],
            features=("rsi",),
            indicators=("RSI",),
            ml_model="XGBoost",
            regime_filters=("RANGE",),
            entry_logic={},
            exit_logic={},
            risk_profile="BALANCED",
            position_sizing={},
            portfolio_constraints={},
        )


def test_constructor_rejects_invalid_mapping() -> None:
    with pytest.raises(TypeError):
        StrategyGenome(
            strategy_id="test",
            version="1.0.0",
            market="BTC/USDT",
            timeframes=("15m",),
            features=("rsi",),
            indicators=("RSI",),
            ml_model="XGBoost",
            regime_filters=("RANGE",),
            entry_logic=[],
            exit_logic={},
            risk_profile="BALANCED",
            position_sizing={},
            portfolio_constraints={},
        )


def test_from_dict_validation() -> None:
    data = make_genome().to_dict()
    data.pop("market")

    with pytest.raises(ValueError):
        StrategyGenome.from_dict(data)


def test_from_dict_rejects_non_mapping() -> None:
    with pytest.raises(TypeError):
        StrategyGenome.from_dict("invalid")


def test_evolve_can_change_strategy_structure() -> None:
    genome = make_genome()

    evolved = genome.evolve(
        entry_logic={
            "technical_score": 0.7,
            "market_intelligence": 0.8,
        },
        exit_logic={
            "take_profit": 0.03,
            "stop_loss": 0.012,
        },
        risk_profile="PROTECTIVE",
    )

    assert evolved.entry_logic["market_intelligence"] == 0.8
    assert evolved.exit_logic["take_profit"] == 0.03
    assert evolved.risk_profile == "PROTECTIVE"