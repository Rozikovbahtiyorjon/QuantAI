from dataclasses import dataclass

import pytest

from experimental.src.champion_transition_executor import ChampionTransitionExecutor


@dataclass
class Decision:
    action: str
    reason: str = ""


def test_promote_replaces_empty_champion():
    executor = ChampionTransitionExecutor()

    result = executor.execute(
        Decision("PROMOTE", "no_current_champion"),
        {},
        {"id": "candidate"},
    )

    assert result.action == "PROMOTE"
    assert result.changed is True
    assert result.champion["id"] == "candidate"


def test_replace_updates_champion():
    executor = ChampionTransitionExecutor()

    result = executor.execute(
        Decision("REPLACE", "candidate_outperforms_champion"),
        {"id": "old"},
        {"id": "new"},
    )

    assert result.action == "REPLACE"
    assert result.changed is True
    assert result.champion["id"] == "new"


def test_keep_preserves_champion():
    executor = ChampionTransitionExecutor()

    result = executor.execute(
        Decision("KEEP", "champion_remains_superior"),
        {"id": "current"},
        {"id": "candidate"},
    )

    assert result.action == "KEEP"
    assert result.changed is False
    assert result.champion["id"] == "current"


def test_hold_preserves_champion():
    executor = ChampionTransitionExecutor()

    result = executor.execute(
        Decision("HOLD", "candidate_not_stable"),
        {"id": "current"},
        {"id": "candidate"},
    )

    assert result.action == "HOLD"
    assert result.changed is False
    assert result.champion["id"] == "current"


def test_reject_preserves_champion():
    executor = ChampionTransitionExecutor()

    result = executor.execute(
        Decision("REJECT", "candidate_missing"),
        {"id": "current"},
        {},
    )

    assert result.action == "REJECT"
    assert result.changed is False
    assert result.champion["id"] == "current"


def test_promote_without_candidate_is_rejected():
    executor = ChampionTransitionExecutor()

    result = executor.execute(
        Decision("PROMOTE"),
        {},
        {},
    )

    assert result.action == "REJECT"
    assert result.changed is False


def test_replace_without_candidate_is_rejected():
    executor = ChampionTransitionExecutor()

    result = executor.execute(
        Decision("REPLACE"),
        {"id": "current"},
        {},
    )

    assert result.action == "REJECT"
    assert result.changed is False
    assert result.champion["id"] == "current"


def test_invalid_action_raises():
    executor = ChampionTransitionExecutor()

    with pytest.raises(ValueError):
        executor.execute(
            Decision("UNKNOWN"),
            {"id": "current"},
            {"id": "candidate"},
        )


def test_original_mappings_are_not_modified():
    executor = ChampionTransitionExecutor()

    current = {"id": "current"}
    candidate = {"id": "candidate"}

    executor.execute(
        Decision("REPLACE"),
        current,
        candidate,
    )

    assert current == {"id": "current"}
    assert candidate == {"id": "candidate"}


def test_reason_is_preserved():
    executor = ChampionTransitionExecutor()

    result = executor.execute(
        Decision("HOLD", "stability_below_threshold"),
        {"id": "current"},
        {"id": "candidate"},
    )

    assert result.reason == "stability_below_threshold"