from dataclasses import dataclass

from src.champion_improvement_cycle import (
    ChampionImprovementCycle,
    ChampionImprovementSnapshot,
)


@dataclass
class Feedback:
    status: str = "OK"
    reason: str = "performance acceptable"


@dataclass
class Lifecycle:
    status: str = "READY"
    decision: str = "PROMOTE"
    admitted: bool = True
    transitioned: bool = True
    stable: bool = True
    replacement_allowed: bool = True
    rollback_allowed: bool = False
    reason: str = "candidate passed lifecycle checks"

    def run(
        self,
        candidate_id=None,
        champion_id=None,
        feedback=None,
        **kwargs,
    ):
        return self


class FeedbackEngine:
    def __init__(self):
        self.calls = 0
        self.last_metrics = None

    def analyze(
        self,
        metrics=None,
        baseline=None,
        context=None,
    ):
        self.calls += 1
        self.last_metrics = dict(metrics or {})
        return Feedback()


def test_run_returns_snapshot():
    cycle = ChampionImprovementCycle(
        Lifecycle(),
        FeedbackEngine(),
    )

    snapshot = cycle.run(
        "candidate-1",
        "champion-1",
        {"return": 0.2},
    )

    assert isinstance(
        snapshot,
        ChampionImprovementSnapshot,
    )


def test_feedback_runs_before_lifecycle():
    events = []

    class OrderedFeedback:
        def analyze(self, **kwargs):
            events.append("feedback")
            return {"status": "OK"}

    class OrderedLifecycle:
        def run(self, **kwargs):
            events.append("lifecycle")
            return {"status": "READY"}

    cycle = ChampionImprovementCycle(
        OrderedLifecycle(),
        OrderedFeedback(),
    )

    cycle.run(
        metrics={"return": 0.1},
    )

    assert events == [
        "feedback",
        "lifecycle",
    ]


def test_candidate_and_champion_ids_are_preserved():
    cycle = ChampionImprovementCycle(
        Lifecycle(),
        FeedbackEngine(),
    )

    snapshot = cycle.run(
        "candidate-7",
        "champion-2",
    )

    assert snapshot.candidate_id == "candidate-7"
    assert snapshot.champion_id == "champion-2"


def test_feedback_status_is_preserved():
    class WarningFeedback:
        def analyze(self, **kwargs):
            return {
                "status": "WARNING",
                "reason": "insufficient data",
            }

    lifecycle = Lifecycle(
        status="EVALUATED",
        admitted=False,
        transitioned=False,
    )

    snapshot = ChampionImprovementCycle(
        lifecycle,
        WarningFeedback(),
    ).run()

    assert snapshot.feedback_status == "WARNING"


def test_promoted_status_requires_transition_and_stability():
    snapshot = ChampionImprovementCycle(
        Lifecycle(
            transitioned=True,
            stable=True,
            admitted=True,
            replacement_allowed=True,
        ),
        FeedbackEngine(),
    ).run()

    assert snapshot.status == "PROMOTED"


def test_ready_status_requires_admission_and_replacement_permission():
    snapshot = ChampionImprovementCycle(
        Lifecycle(
            transitioned=False,
            stable=False,
            admitted=True,
            replacement_allowed=True,
        ),
        FeedbackEngine(),
    ).run()

    assert snapshot.status == "READY"


def test_admitted_status_is_used_when_not_ready():
    snapshot = ChampionImprovementCycle(
        Lifecycle(
            transitioned=False,
            stable=False,
            admitted=True,
            replacement_allowed=False,
        ),
        FeedbackEngine(),
    ).run()

    assert snapshot.status == "ADMITTED"


def test_rollback_has_priority():
    snapshot = ChampionImprovementCycle(
        Lifecycle(
            transitioned=True,
            stable=True,
            admitted=True,
            replacement_allowed=True,
            rollback_allowed=True,
        ),
        FeedbackEngine(),
    ).run()

    assert snapshot.status == "ROLLBACK"


def test_missing_dependencies_methods_are_safe():
    snapshot = ChampionImprovementCycle(
        object(),
        object(),
    ).run()

    assert snapshot.status == "EVALUATED"
    assert snapshot.feedback_status == "UNKNOWN"
    assert snapshot.lifecycle_status == "UNKNOWN"


def test_evaluate_alias_matches_run():
    cycle = ChampionImprovementCycle(
        Lifecycle(),
        FeedbackEngine(),
    )

    first = cycle.run(
        "candidate",
        "champion",
    )

    second = cycle.evaluate(
        "candidate",
        "champion",
    )

    assert first == second


def test_reason_prefers_lifecycle_reason():
    snapshot = ChampionImprovementCycle(
        Lifecycle(
            reason="lifecycle reason",
        ),
        FeedbackEngine(),
    ).run()

    assert snapshot.reason == "lifecycle reason"