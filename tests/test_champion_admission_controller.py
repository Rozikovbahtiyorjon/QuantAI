import pytest

from src.champion_admission_controller import ChampionAdmissionController


def test_admit_superior_candidate():
    controller = ChampionAdmissionController(
        min_improvement=0.05,
        min_samples=20,
    )

    decision = controller.evaluate(
        {"score": 1.10},
        {"score": 1.00},
        samples=20,
    )

    assert decision.action == "ADMIT"
    assert decision.reason == "CANDIDATE_OUTPERFORMS_CHAMPION"
    assert decision.improvement == pytest.approx(0.10)


def test_reject_candidate_without_required_improvement():
    controller = ChampionAdmissionController(
        min_improvement=0.05,
        min_samples=20,
    )

    decision = controller.evaluate(
        {"score": 1.02},
        {"score": 1.00},
        samples=20,
    )

    assert decision.action == "REJECT"
    assert decision.reason == "CANDIDATE_NOT_SUPERIOR"


def test_hold_with_insufficient_samples():
    controller = ChampionAdmissionController(min_samples=20)

    decision = controller.evaluate(
        {"score": 1.20},
        {"score": 1.00},
        samples=19,
    )

    assert decision.action == "HOLD"
    assert decision.reason == "INSUFFICIENT_SAMPLES"


def test_reject_without_candidate_metrics():
    decision = ChampionAdmissionController().evaluate(
        {},
        {"score": 1.00},
        samples=20,
    )

    assert decision.action == "REJECT"
    assert decision.reason == "NO_CANDIDATE_METRICS"


def test_admit_without_existing_champion():
    decision = ChampionAdmissionController(
        min_samples=20
    ).evaluate(
        {"score": 1.00},
        {},
        samples=20,
    )

    assert decision.action == "ADMIT"
    assert decision.reason == "NO_EXISTING_CHAMPION"


def test_hold_without_existing_champion_and_insufficient_samples():
    decision = ChampionAdmissionController(
        min_samples=20
    ).evaluate(
        {"score": 1.00},
        {},
        samples=19,
    )

    assert decision.action == "HOLD"
    assert decision.reason == "INSUFFICIENT_SAMPLES"


def test_profitability_fallback():
    decision = ChampionAdmissionController(
        min_improvement=0.05,
        min_samples=20,
    ).evaluate(
        {"profitability": 1.10},
        {"profitability": 1.00},
        samples=20,
    )

    assert decision.action == "ADMIT"


def test_zero_champion_score():
    decision = ChampionAdmissionController(
        min_improvement=0.05,
        min_samples=20,
    ).evaluate(
        {"score": 0.10},
        {"score": 0.00},
        samples=20,
    )

    assert decision.action == "ADMIT"


def test_invalid_configuration():
    with pytest.raises(ValueError):
        ChampionAdmissionController(min_improvement=-0.01)

    with pytest.raises(ValueError):
        ChampionAdmissionController(min_samples=0)


def test_invalid_samples():
    controller = ChampionAdmissionController()

    with pytest.raises(ValueError):
        controller.evaluate(
            {"score": 1.0},
            {"score": 1.0},
            samples=-1,
        )