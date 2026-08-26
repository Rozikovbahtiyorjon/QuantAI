from experimental.src.champion_transition_decision import (
    ChampionTransitionDecisionEngine,
)


def test_missing_candidate_is_rejected():
    result = ChampionTransitionDecisionEngine().decide({"score": 1}, None)

    assert result.action == "REJECT"


def test_empty_champion_allows_stable_candidate():
    result = ChampionTransitionDecisionEngine().decide(
        None,
        {"score": 2, "stability": 0.8},
    )

    assert result.action == "PROMOTE"


def test_unstable_candidate_is_held():
    result = ChampionTransitionDecisionEngine().decide(
        {"score": 5},
        {"score": 10, "stability": 0.2},
    )

    assert result.action == "HOLD"
    assert result.stable is False


def test_candidate_replaces_when_margin_is_met():
    engine = ChampionTransitionDecisionEngine(min_score_margin=1.0)

    result = engine.decide(
        {"score": 5},
        {"score": 6, "stability": 0.9},
    )

    assert result.action == "REPLACE"
    assert result.margin == 1.0


def test_champion_is_kept_when_margin_is_insufficient():
    engine = ChampionTransitionDecisionEngine(min_score_margin=1.0)

    result = engine.decide(
        {"score": 5},
        {"score": 5.5, "stability": 0.9},
    )

    assert result.action == "KEEP"


def test_score_can_be_derived_from_metrics():
    result = ChampionTransitionDecisionEngine().decide(
        {
            "profitability": 1,
            "return_rate": 1,
            "win_rate": 0.5,
        },
        {
            "profitability": 3,
            "return_rate": 2,
            "win_rate": 0.8,
            "stability": 0.9,
        },
    )

    assert result.action == "REPLACE"


def test_drawdown_reduces_score():
    result = ChampionTransitionDecisionEngine().decide(
        {"score": 5},
        {
            "profitability": 4,
            "return": 2,
            "win_rate": 0.5,
            "drawdown": 2,
            "stability": 0.9,
        },
    )

    assert result.score == 4.5


def test_stability_is_clamped():
    result = ChampionTransitionDecisionEngine().decide(
        {"score": 1},
        {"score": 2, "stability": 2.0},
    )

    assert result.stable is True


def test_invalid_configuration():
    try:
        ChampionTransitionDecisionEngine(min_score_margin=-1)
    except ValueError:
        pass
    else:
        raise AssertionError("negative margin must fail")

    try:
        ChampionTransitionDecisionEngine(min_stability_score=2)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid stability threshold must fail")


def test_margin_is_reported():
    result = ChampionTransitionDecisionEngine().decide(
        {"score": 4},
        {"score": 7, "stability": 0.9},
    )

    assert result.margin == 3.0