import pytest

from src.champion_evolution import (
    ChampionComparison,
    ChampionDecision,
)
from src.champion_history import (
    ChampionHistory,
    ChampionHistoryEntry,
)


def comparison(
    candidate_id: str,
    score: float,
    decision: ChampionDecision,
    improvement: float,
) -> ChampionComparison:
    return ChampionComparison(
        candidate_id=candidate_id,
        champion_id="champion",
        candidate_score=score,
        champion_score=score - improvement,
        improvement=improvement,
        decision=decision,
        reason="test",
    )


def test_history_starts_empty() -> None:
    history = ChampionHistory()

    assert history.entries == ()
    assert history.best() is None
    assert history.acceptance_rate() == 0.0


def test_record_adds_entry() -> None:
    history = ChampionHistory()

    entry = history.record(
        comparison(
            "candidate",
            0.9,
            ChampionDecision.ACCEPT,
            0.1,
        )
    )

    assert isinstance(entry, ChampionHistoryEntry)
    assert len(history.entries) == 1
    assert entry.strategy_id == "candidate"


def test_entries_are_read_only() -> None:
    history = ChampionHistory()

    history.record(
        comparison(
            "candidate",
            0.9,
            ChampionDecision.ACCEPT,
            0.1,
        )
    )

    assert isinstance(history.entries, tuple)


def test_accepted_returns_only_accepted_entries() -> None:
    history = ChampionHistory()

    history.record(
        comparison(
            "accepted",
            0.9,
            ChampionDecision.ACCEPT,
            0.1,
        )
    )
    history.record(
        comparison(
            "rejected",
            0.7,
            ChampionDecision.REJECT,
            -0.1,
        )
    )

    accepted = history.accepted()

    assert len(accepted) == 1
    assert accepted[0].strategy_id == "accepted"


def test_rejected_returns_only_rejected_entries() -> None:
    history = ChampionHistory()

    history.record(
        comparison(
            "accepted",
            0.9,
            ChampionDecision.ACCEPT,
            0.1,
        )
    )
    history.record(
        comparison(
            "rejected",
            0.7,
            ChampionDecision.REJECT,
            -0.1,
        )
    )

    rejected = history.rejected()

    assert len(rejected) == 1
    assert rejected[0].strategy_id == "rejected"


def test_best_returns_highest_score() -> None:
    history = ChampionHistory()

    history.record(
        comparison(
            "low",
            0.7,
            ChampionDecision.ACCEPT,
            0.05,
        )
    )
    history.record(
        comparison(
            "high",
            0.95,
            ChampionDecision.ACCEPT,
            0.15,
        )
    )

    best = history.best()

    assert best is not None
    assert best.strategy_id == "high"
    assert best.score == pytest.approx(0.95)


def test_acceptance_rate_is_calculated() -> None:
    history = ChampionHistory()

    history.extend(
        [
            comparison(
                "a",
                0.9,
                ChampionDecision.ACCEPT,
                0.1,
            ),
            comparison(
                "b",
                0.8,
                ChampionDecision.REJECT,
                -0.1,
            ),
            comparison(
                "c",
                0.92,
                ChampionDecision.ACCEPT,
                0.12,
            ),
        ]
    )

    assert history.acceptance_rate() == pytest.approx(2 / 3)


def test_summary_contains_key_metrics() -> None:
    history = ChampionHistory()

    history.record(
        comparison(
            "best",
            0.95,
            ChampionDecision.ACCEPT,
            0.15,
        )
    )

    summary = history.summary()

    assert summary["total_candidates"] == 1
    assert summary["accepted"] == 1
    assert summary["rejected"] == 0
    assert summary["acceptance_rate"] == 1.0
    assert summary["best_strategy_id"] == "best"
    assert summary["best_score"] == pytest.approx(0.95)


def test_extend_records_all_comparisons() -> None:
    history = ChampionHistory()

    history.extend(
        [
            comparison(
                "a",
                0.8,
                ChampionDecision.ACCEPT,
                0.05,
            ),
            comparison(
                "b",
                0.75,
                ChampionDecision.REJECT,
                -0.05,
            ),
        ]
    )

    assert len(history.entries) == 2


def test_invalid_comparison_is_rejected() -> None:
    history = ChampionHistory()

    with pytest.raises(TypeError):
        history.record("invalid")


def test_summary_for_empty_history() -> None:
    summary = ChampionHistory().summary()

    assert summary["total_candidates"] == 0
    assert summary["accepted"] == 0
    assert summary["rejected"] == 0
    assert summary["acceptance_rate"] == 0.0
    assert summary["best_strategy_id"] is None
    assert summary["best_score"] is None