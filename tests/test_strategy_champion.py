import pytest

from src.champion_evolution import ChampionDecision, ChampionEvaluation
from src.strategy_champion import ChampionState, StrategyChampion


def make_evaluation(
    strategy_id: str,
    performance_score: float = 0.8,
    robustness_score: float = 0.8,
    risk_adjusted_score: float = 0.8,
    evidence_score: float = 0.8,
) -> ChampionEvaluation:
    return ChampionEvaluation(
        strategy_id=strategy_id,
        performance_score=performance_score,
        robustness_score=robustness_score,
        risk_adjusted_score=risk_adjusted_score,
        evidence_score=evidence_score,
    )


def test_initial_champion_can_be_set() -> None:
    engine = StrategyChampion()
    state = engine.set_initial(make_evaluation("champion"))

    assert isinstance(state, ChampionState)
    assert state.strategy_id == "champion"
    assert engine.champion.strategy_id == "champion"


def test_initial_champion_cannot_be_replaced_by_set_initial() -> None:
    engine = StrategyChampion()
    engine.set_initial(make_evaluation("champion"))

    with pytest.raises(RuntimeError):
        engine.set_initial(make_evaluation("other"))


def test_candidate_is_rejected_without_enough_improvement() -> None:
    engine = StrategyChampion(minimum_improvement=0.1)
    engine.set_initial(make_evaluation("champion"))

    result = engine.promote(
        make_evaluation(
            "candidate",
            performance_score=0.81,
        )
    )

    assert result.decision is ChampionDecision.REJECT
    assert engine.champion.strategy_id == "champion"


def test_candidate_is_promoted_when_accepted() -> None:
    engine = StrategyChampion(minimum_improvement=0.01)
    engine.set_initial(make_evaluation("champion"))

    result = engine.promote(
        make_evaluation(
            "candidate",
            performance_score=0.95,
        )
    )

    assert result.decision is ChampionDecision.ACCEPT
    assert engine.champion.strategy_id == "candidate"


def test_low_robustness_cannot_replace_champion() -> None:
    engine = StrategyChampion(minimum_robustness=0.7)
    engine.set_initial(make_evaluation("champion"))

    result = engine.promote(
        make_evaluation(
            "candidate",
            performance_score=1.0,
            robustness_score=0.6,
        )
    )

    assert result.decision is ChampionDecision.REJECT
    assert engine.champion.strategy_id == "champion"


def test_low_evidence_cannot_replace_champion() -> None:
    engine = StrategyChampion(minimum_evidence=0.7)
    engine.set_initial(make_evaluation("champion"))

    result = engine.promote(
        make_evaluation(
            "candidate",
            performance_score=1.0,
            evidence_score=0.6,
        )
    )

    assert result.decision is ChampionDecision.REJECT
    assert engine.champion.strategy_id == "champion"


def test_evaluate_does_not_change_champion() -> None:
    engine = StrategyChampion(minimum_improvement=0.01)
    engine.set_initial(make_evaluation("champion"))

    result = engine.evaluate(
        make_evaluation(
            "candidate",
            performance_score=0.95,
        )
    )

    assert result.decision is ChampionDecision.ACCEPT
    assert engine.champion.strategy_id == "champion"


def test_state_updates_after_promotion() -> None:
    engine = StrategyChampion(minimum_improvement=0.01)
    engine.set_initial(make_evaluation("champion"))

    engine.promote(
        make_evaluation(
            "candidate",
            performance_score=0.95,
        )
    )

    state = engine.state()

    assert state is not None
    assert state.strategy_id == "candidate"
    assert state.score == pytest.approx(
        engine._evolution.score(engine.champion)
    )


def test_no_champion_blocks_evaluation() -> None:
    engine = StrategyChampion()

    with pytest.raises(RuntimeError):
        engine.evaluate(
            make_evaluation("candidate")
        )


def test_invalid_evaluation_is_rejected() -> None:
    engine = StrategyChampion()

    with pytest.raises(TypeError):
        engine.set_initial("invalid")


def test_candidate_and_champion_must_differ() -> None:
    engine = StrategyChampion()
    engine.set_initial(make_evaluation("same"))

    with pytest.raises(ValueError):
        engine.evaluate(
            make_evaluation("same")
        )