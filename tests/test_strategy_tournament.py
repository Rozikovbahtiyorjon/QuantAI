from __future__ import annotations

import pytest

from src.strategy_bank import StrategyRegistry
from src.strategy_tournament import (
    StrategyEvaluation,
    StrategyTournament,
    TournamentRanking,
    TournamentResult,
)
from src.strategy_genome import StrategyGenome


def make_evaluation(
    strategy_id: str,
    total_return: float = 0.20,
    sharpe_ratio: float = 1.5,
    max_drawdown: float = 0.10,
    win_rate: float = 0.60,
    profit_factor: float = 1.8,
    walk_forward_score: float = 0.80,
    robustness_score: float = 0.80,
    monte_carlo_score: float = 0.80,
    stress_score: float = 0.80,
) -> StrategyEvaluation:
    return StrategyEvaluation(
        strategy_id=strategy_id,
        total_return=total_return,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        profit_factor=profit_factor,
        walk_forward_score=walk_forward_score,
        robustness_score=robustness_score,
        monte_carlo_score=monte_carlo_score,
        stress_score=stress_score,
    )


def make_genome(
    strategy_id: str,
) -> StrategyGenome:
    return StrategyGenome(
        strategy_id=strategy_id,
        version="1.0.0",
        market="BTC/USDT",
        timeframes=("15m",),
        features=("returns", "rsi"),
        indicators=("EMA", "RSI"),
        ml_model="XGBoostClassifier",
        regime_filters=("TREND_UP",),
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


def test_default_weights_are_normalized() -> None:
    tournament = StrategyTournament()

    assert sum(tournament.weights.values()) == pytest.approx(
        1.0
    )


def test_evaluate_returns_score() -> None:
    tournament = StrategyTournament()

    evaluation = make_evaluation(
        "strategy_001"
    )

    score = tournament.evaluate(
        evaluation
    )

    assert 0.0 <= score <= 1.0


def test_better_strategy_gets_higher_score() -> None:
    tournament = StrategyTournament()

    weak = make_evaluation(
        "weak",
        total_return=0.05,
        sharpe_ratio=0.5,
        max_drawdown=0.30,
        win_rate=0.45,
        profit_factor=1.1,
        walk_forward_score=0.50,
        robustness_score=0.50,
        monte_carlo_score=0.50,
        stress_score=0.50,
    )

    strong = make_evaluation(
        "strong",
        total_return=0.40,
        sharpe_ratio=2.0,
        max_drawdown=0.05,
        win_rate=0.70,
        profit_factor=2.5,
        walk_forward_score=0.95,
        robustness_score=0.95,
        monte_carlo_score=0.95,
        stress_score=0.95,
    )

    assert tournament.evaluate(
        strong
    ) > tournament.evaluate(weak)


def test_rank_returns_sorted_results() -> None:
    tournament = StrategyTournament()

    weak = make_evaluation(
        "weak",
        total_return=0.05,
        sharpe_ratio=0.5,
        max_drawdown=0.30,
        win_rate=0.45,
        profit_factor=1.1,
        walk_forward_score=0.50,
        robustness_score=0.50,
        monte_carlo_score=0.50,
        stress_score=0.50,
    )

    strong = make_evaluation(
        "strong",
        total_return=0.40,
        sharpe_ratio=2.0,
        max_drawdown=0.05,
        win_rate=0.70,
        profit_factor=2.5,
        walk_forward_score=0.95,
        robustness_score=0.95,
        monte_carlo_score=0.95,
        stress_score=0.95,
    )

    ranking = tournament.rank(
        [weak, strong]
    )

    assert isinstance(
        ranking,
        TournamentRanking,
    )

    assert ranking.results[0].strategy_id == "strong"
    assert ranking.results[0].rank == 1
    assert ranking.results[1].rank == 2


def test_select_champion() -> None:
    tournament = StrategyTournament()

    first = make_evaluation(
        "strategy_a",
        total_return=0.10,
    )

    second = make_evaluation(
        "strategy_b",
        total_return=0.35,
        sharpe_ratio=2.0,
        max_drawdown=0.05,
    )

    champion = tournament.select_champion(
        [first, second]
    )

    assert isinstance(
        champion,
        TournamentResult,
    )

    assert champion.strategy_id == "strategy_b"
    assert champion.rank == 1


def test_compare_with_champion() -> None:
    tournament = StrategyTournament()

    champion = make_evaluation(
        "champion",
        total_return=0.20,
    )

    candidate = make_evaluation(
        "candidate",
        total_return=0.40,
        sharpe_ratio=2.0,
        max_drawdown=0.05,
        robustness_score=0.95,
    )

    assert tournament.compare_with_champion(
        candidate,
        champion,
    ) is True


def test_weaker_candidate_does_not_replace_champion() -> None:
    tournament = StrategyTournament()

    champion = make_evaluation(
        "champion",
        total_return=0.40,
        sharpe_ratio=2.0,
        max_drawdown=0.05,
        robustness_score=0.95,
    )

    candidate = make_evaluation(
        "candidate",
        total_return=0.10,
        sharpe_ratio=0.5,
        max_drawdown=0.30,
        robustness_score=0.50,
    )

    assert tournament.compare_with_champion(
        candidate,
        champion,
    ) is False


def test_promote_champion_updates_registry() -> None:
    registry = StrategyRegistry()

    registry.register(
        make_genome("strategy_a")
    )

    registry.register(
        make_genome("strategy_b")
    )

    tournament = StrategyTournament()

    champion = make_evaluation(
        "strategy_b",
        total_return=0.40,
        sharpe_ratio=2.0,
        max_drawdown=0.05,
        robustness_score=0.95,
    )

    candidate = make_evaluation(
        "strategy_a",
        total_return=0.10,
    )

    record = tournament.promote_champion(
        registry,
        [candidate, champion],
    )

    assert record.genome.strategy_id == "strategy_b"
    assert record.status == "champion"

    assert (
        registry.champion().genome.strategy_id
        == "strategy_b"
    )


def test_custom_weights_are_supported() -> None:
    weights = {
        "total_return": 1.0,
        "sharpe_ratio": 0.0,
        "max_drawdown": 0.0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "walk_forward_score": 0.0,
        "robustness_score": 0.0,
        "monte_carlo_score": 0.0,
        "stress_score": 0.0,
    }

    tournament = StrategyTournament(
        weights=weights
    )

    assert tournament.weights[
        "total_return"
    ] == pytest.approx(1.0)

    evaluation = make_evaluation(
        "strategy",
        total_return=0.50,
    )

    assert tournament.evaluate(
        evaluation
    ) == pytest.approx(0.75)


def test_empty_tournament_is_rejected() -> None:
    tournament = StrategyTournament()

    with pytest.raises(ValueError):
        tournament.rank([])


def test_duplicate_strategy_ids_are_rejected() -> None:
    tournament = StrategyTournament()

    evaluation = make_evaluation(
        "duplicate"
    )

    with pytest.raises(ValueError):
        tournament.rank(
            [evaluation, evaluation]
        )


def test_invalid_evaluation_type_is_rejected() -> None:
    tournament = StrategyTournament()

    with pytest.raises(TypeError):
        tournament.evaluate("invalid")


def test_invalid_evaluation_values_are_rejected() -> None:
    with pytest.raises(ValueError):
        make_evaluation(
            "strategy",
            max_drawdown=-0.10,
        )

    with pytest.raises(ValueError):
        make_evaluation(
            "strategy",
            win_rate=1.1,
        )

    with pytest.raises(ValueError):
        make_evaluation(
            "strategy",
            robustness_score=-0.1,
        )

    with pytest.raises(ValueError):
        make_evaluation(
            "strategy",
            profit_factor=-1.0,
        )


def test_constructor_validation() -> None:
    with pytest.raises(TypeError):
        make_evaluation(
            "strategy",
            total_return="invalid",
        )

    with pytest.raises(ValueError):
        make_evaluation("")

    with pytest.raises(TypeError):
        StrategyTournament(
            weights="invalid"
        )

    with pytest.raises(ValueError):
        StrategyTournament(
            weights={
                "invalid": 1.0,
            }
        )


def test_same_strategy_cannot_be_compared_with_itself() -> None:
    tournament = StrategyTournament()

    evaluation = make_evaluation(
        "strategy"
    )

    with pytest.raises(ValueError):
        tournament.compare_with_champion(
            evaluation,
            evaluation,
        )


def test_invalid_registry_is_rejected() -> None:
    tournament = StrategyTournament()

    evaluation = make_evaluation(
        "strategy"
    )

    with pytest.raises(TypeError):
        tournament.promote_champion(
            "invalid",
            [evaluation],
        )