import pytest

from src.champion_rollback_guard import ChampionRollbackGuard


def test_keep_when_performance_is_stable():
    decision = ChampionRollbackGuard().evaluate(
        {"score": 0.95},
        {"score": 1.00},
        samples=30,
    )

    assert decision.action == "KEEP"
    assert decision.reason == "STABLE"


def test_rollback_when_degradation_reaches_threshold():
    decision = ChampionRollbackGuard(
        min_degradation=0.10
    ).evaluate(
        {"score": 0.89},
        {"score": 1.00},
        samples=30,
    )

    assert decision.action == "ROLLBACK"
    assert decision.reason == "PERFORMANCE_DEGRADATION"
    assert decision.degradation == pytest.approx(0.11)


def test_hold_without_baseline():
    decision = ChampionRollbackGuard().evaluate(
        {"score": 0.5},
        {},
        samples=30,
    )

    assert decision.action == "HOLD"
    assert decision.reason == "NO_BASELINE"


def test_hold_with_insufficient_samples():
    decision = ChampionRollbackGuard(
        min_samples=20
    ).evaluate(
        {"score": 0.5},
        {"score": 1.0},
        samples=19,
    )

    assert decision.action == "HOLD"
    assert decision.reason == "INSUFFICIENT_SAMPLES"


def test_zero_baseline_does_not_rollback_positive_score():
    decision = ChampionRollbackGuard(
        min_degradation=0.10
    ).evaluate(
        {"score": 0.2},
        {"score": 0.0},
        samples=20,
    )

    assert decision.action == "KEEP"


def test_negative_current_score_with_zero_baseline_rolls_back():
    decision = ChampionRollbackGuard(
        min_degradation=0.10
    ).evaluate(
        {"score": -0.1},
        {"score": 0.0},
        samples=20,
    )

    assert decision.action == "ROLLBACK"


def test_profitability_fallback_is_supported():
    decision = ChampionRollbackGuard().evaluate(
        {"profitability": 1.0},
        {"profitability": 1.0},
        samples=20,
    )

    assert decision.action == "KEEP"


def test_invalid_configuration():
    with pytest.raises(ValueError):
        ChampionRollbackGuard(min_degradation=-0.1)

    with pytest.raises(ValueError):
        ChampionRollbackGuard(min_samples=0)


def test_invalid_samples():
    with pytest.raises(ValueError):
        ChampionRollbackGuard().evaluate(
            {"score": 1},
            {"score": 1},
            samples=-1,
        )