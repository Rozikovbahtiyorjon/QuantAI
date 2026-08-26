from experimental.src.champion_replacement_guard import (
    ChampionReplacementGuard,
    ReplacementGuardConfig,
)


def valid_champion():
    return {
        "return": 0.20,
        "profit_factor": 1.30,
        "drawdown": 0.10,
        "win_rate": 0.55,
        "trade_count": 100,
    }


def valid_challenger():
    return {
        "return": 0.25,
        "profit_factor": 1.40,
        "drawdown": 0.12,
        "win_rate": 0.58,
        "trade_count": 110,
    }


def test_strong_challenger_is_approved():
    decision = ChampionReplacementGuard().evaluate(
        valid_champion(),
        valid_challenger(),
    )

    assert decision.approved is True
    assert decision.reason == "challenger_approved"
    assert decision.failed_checks == ()


def test_return_must_improve():
    challenger = valid_challenger()
    challenger["return"] = 0.20

    decision = ChampionReplacementGuard().evaluate(
        valid_champion(),
        challenger,
    )

    assert decision.approved is False
    assert "return_not_improved" in decision.failed_checks


def test_profit_factor_minimum():
    challenger = valid_challenger()
    challenger["profit_factor"] = 0.99

    decision = ChampionReplacementGuard().evaluate(
        valid_champion(),
        challenger,
    )

    assert decision.approved is False
    assert "profit_factor_below_minimum" in decision.failed_checks


def test_drawdown_limit():
    challenger = valid_challenger()
    challenger["drawdown"] = 0.21

    decision = ChampionReplacementGuard().evaluate(
        valid_champion(),
        challenger,
    )

    assert decision.approved is False
    assert "drawdown_above_maximum" in decision.failed_checks


def test_win_rate_minimum():
    challenger = valid_challenger()
    challenger["win_rate"] = 0.44

    decision = ChampionReplacementGuard().evaluate(
        valid_champion(),
        challenger,
    )

    assert decision.approved is False
    assert "win_rate_below_minimum" in decision.failed_checks


def test_trade_count_minimum():
    challenger = valid_challenger()
    challenger["trade_count"] = 9

    decision = ChampionReplacementGuard().evaluate(
        valid_champion(),
        challenger,
    )

    assert decision.approved is False
    assert "trade_count_below_minimum" in decision.failed_checks


def test_multiple_failed_checks_are_reported():
    challenger = {
        "return": 0.10,
        "profit_factor": 0.8,
        "drawdown": 0.30,
        "win_rate": 0.30,
        "trade_count": 5,
    }

    decision = ChampionReplacementGuard().evaluate(
        valid_champion(),
        challenger,
    )

    assert decision.approved is False
    assert len(decision.failed_checks) == 5


def test_empty_metrics_are_rejected():
    decision = ChampionReplacementGuard().evaluate(
        {},
        valid_challenger(),
    )

    assert decision.approved is False
    assert decision.reason == "insufficient_data"


def test_missing_challenger_metrics_are_rejected():
    decision = ChampionReplacementGuard().evaluate(
        valid_champion(),
        {},
    )

    assert decision.approved is False
    assert decision.reason == "insufficient_data"


def test_should_replace_matches_evaluation():
    guard = ChampionReplacementGuard()

    assert guard.should_replace(
        valid_champion(),
        valid_challenger(),
    ) is True

    challenger = valid_challenger()
    challenger["return"] = 0.20

    assert guard.should_replace(
        valid_champion(),
        challenger,
    ) is False


def test_custom_improvement_threshold():
    guard = ChampionReplacementGuard(
        ReplacementGuardConfig(min_improvement=0.10)
    )

    challenger = valid_challenger()
    challenger["return"] = 0.29

    assert guard.should_replace(
        valid_champion(),
        challenger,
    ) is False

    challenger["return"] = 0.31

    assert guard.should_replace(
        valid_champion(),
        challenger,
    ) is True


def test_negative_drawdown_is_treated_as_absolute():
    challenger = valid_challenger()
    challenger["drawdown"] = -0.12

    assert ChampionReplacementGuard().should_replace(
        valid_champion(),
        challenger,
    ) is True