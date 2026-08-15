import pytest

from src.champion_evolution import ChampionComparison, ChampionDecision
from src.champion_registry import ChampionRegistry, ChampionState


def make_comparison(
    candidate_id: str,
    champion_id: str,
    candidate_score: float,
    decision: ChampionDecision,
) -> ChampionComparison:
    return ChampionComparison(
        candidate_id=candidate_id,
        champion_id=champion_id,
        candidate_score=candidate_score,
        champion_score=candidate_score - 0.1,
        improvement=0.1,
        decision=decision,
        reason="test",
    )


def test_registry_starts_with_champion() -> None:
    registry = ChampionRegistry(ChampionState("v1", 0.70))

    assert registry.champion.strategy_id == "v1"
    assert registry.champion.score == pytest.approx(0.70)
    assert registry.champion.version == 1


def test_accepts_stronger_candidate() -> None:
    registry = ChampionRegistry(ChampionState("v1", 0.70))

    result = registry.consider(
        make_comparison("v2", "v1", 0.80, ChampionDecision.ACCEPT)
    )

    assert result is True
    assert registry.champion.strategy_id == "v2"
    assert registry.champion.score == pytest.approx(0.80)
    assert registry.champion.version == 2


def test_rejects_rejected_candidate() -> None:
    registry = ChampionRegistry(ChampionState("v1", 0.70))

    result = registry.consider(
        make_comparison("v2", "v1", 0.90, ChampionDecision.REJECT)
    )

    assert result is False
    assert registry.champion.strategy_id == "v1"


def test_rejects_equal_score() -> None:
    registry = ChampionRegistry(ChampionState("v1", 0.70))

    result = registry.consider(
        make_comparison("v2", "v1", 0.70, ChampionDecision.ACCEPT)
    )

    assert result is False
    assert registry.champion.strategy_id == "v1"


def test_rejects_weaker_candidate() -> None:
    registry = ChampionRegistry(ChampionState("v1", 0.70))

    result = registry.consider(
        make_comparison("v2", "v1", 0.60, ChampionDecision.ACCEPT)
    )

    assert result is False
    assert registry.champion.strategy_id == "v1"


def test_requires_matching_champion_id() -> None:
    registry = ChampionRegistry(ChampionState("v1", 0.70))

    with pytest.raises(ValueError):
        registry.consider(
            make_comparison("v2", "other", 0.80, ChampionDecision.ACCEPT)
        )


def test_requires_comparison_type() -> None:
    registry = ChampionRegistry(ChampionState("v1", 0.70))

    with pytest.raises(TypeError):
        registry.consider("invalid")


def test_requires_state_type() -> None:
    with pytest.raises(TypeError):
        ChampionRegistry("invalid")


def test_version_increments_only_on_promotion() -> None:
    registry = ChampionRegistry(ChampionState("v1", 0.70))

    registry.consider(
        make_comparison("v2", "v1", 0.60, ChampionDecision.ACCEPT)
    )
    assert registry.champion.version == 1

    registry.consider(
        make_comparison("v3", "v1", 0.80, ChampionDecision.ACCEPT)
    )
    assert registry.champion.version == 2


def test_snapshot() -> None:
    registry = ChampionRegistry(ChampionState("v1", 0.70))

    snapshot = registry.snapshot()

    assert snapshot["strategy_id"] == "v1"
    assert snapshot["score"] == pytest.approx(0.70)
    assert snapshot["version"] == 1