import pytest

from src.champion_evolution import (
    ChampionDecision,
    ChampionEvaluation,
    ChampionEvolution,
)


def make_evaluation(
    strategy_id: str = "s1",
    **overrides,
) -> ChampionEvaluation:
    values = {
        "strategy_id": strategy_id,
        "performance_score": 0.8,
        "robustness_score": 0.8,
        "risk_adjusted_score": 0.8,
        "evidence_score": 0.8,
    }

    values.update(overrides)

    return ChampionEvaluation(**values)


def test_score_is_bounded() -> None:
    engine = ChampionEvolution()

    score = engine.score(
        make_evaluation()
    )

    assert 0.0 <= score <= 1.0


def test_candidate_is_accepted_when_it_beats_champion() -> None:
    engine = ChampionEvolution(
        minimum_improvement=0.01
    )

    result = engine.evaluate_candidate(
        make_evaluation(
            "champion",
            performance_score=0.70,
        ),
        make_evaluation(
            "candidate",
            performance_score=0.95,
        ),
    )

    assert result.decision is ChampionDecision.ACCEPT
    assert result.improvement > 0.01


def test_candidate_is_rejected_when_improvement_is_too_small() -> None:
    engine = ChampionEvolution(
        minimum_improvement=0.10
    )

    result = engine.evaluate_candidate(
        make_evaluation("champion"),
        make_evaluation(
            "candidate",
            performance_score=0.81,
        ),
    )

    assert result.decision is ChampionDecision.REJECT


def test_low_robustness_is_rejected() -> None:
    engine = ChampionEvolution(
        minimum_robustness=0.7
    )

    result = engine.evaluate_candidate(
        make_evaluation("champion"),
        make_evaluation(
            "candidate",
            performance_score=1.0,
            robustness_score=0.6,
        ),
    )

    assert result.decision is ChampionDecision.REJECT


def test_low_evidence_is_rejected() -> None:
    engine = ChampionEvolution(
        minimum_evidence=0.7
    )

    result = engine.evaluate_candidate(
        make_evaluation("champion"),
        make_evaluation(
            "candidate",
            performance_score=1.0,
            evidence_score=0.6,
        ),
    )

    assert result.decision is ChampionDecision.REJECT


def test_same_strategy_is_invalid() -> None:
    engine = ChampionEvolution()

    with pytest.raises(ValueError):
        engine.evaluate_candidate(
            make_evaluation("same"),
            make_evaluation("same"),
        )


def test_select_champion_uses_eligible_pool() -> None:
    engine = ChampionEvolution()

    result = engine.select_champion(
        [
            make_evaluation(
                "weak",
                performance_score=1.0,
                robustness_score=0.2,
            ),
            make_evaluation(
                "strong",
                performance_score=0.9,
                robustness_score=0.9,
            ),
        ]
    )

    assert result.strategy_id == "strong"


def test_select_champion_rejects_empty_collection() -> None:
    with pytest.raises(ValueError):
        ChampionEvolution().select_champion([])


def test_constructor_type_validation() -> None:
    with pytest.raises(TypeError):
        ChampionEvolution(
            minimum_improvement="bad"
        )


def test_constructor_value_validation() -> None:
    with pytest.raises(ValueError):
        ChampionEvolution(
            minimum_robustness=1.1
        )


def test_evaluation_type_validation() -> None:
    with pytest.raises(TypeError):
        ChampionEvolution().score("invalid")


def test_evaluation_range_validation() -> None:
    with pytest.raises(ValueError):
        make_evaluation(
            performance_score=1.1
        )


def test_empty_strategy_id_is_invalid() -> None:
    with pytest.raises(ValueError):
        make_evaluation("")


def test_candidate_result_contains_ids() -> None:
    result = ChampionEvolution().evaluate_candidate(
        make_evaluation("champion"),
        make_evaluation(
            "candidate",
            performance_score=1.0,
        ),
    )

    assert result.champion_id == "champion"
    assert result.candidate_id == "candidate"


def test_score_ordering() -> None:
    engine = ChampionEvolution()

    low = make_evaluation(
        "low",
        performance_score=0.4,
    )

    high = make_evaluation(
        "high",
        performance_score=0.9,
    )

    assert engine.score(high) > engine.score(low)