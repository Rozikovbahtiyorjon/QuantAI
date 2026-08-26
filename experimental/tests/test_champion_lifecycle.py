from __future__ import annotations

import py_compile
from pathlib import Path

import pytest

from experimental.src.champion_lifecycle import (
    ChampionLifecycle,
    ChampionLifecycleSnapshot,
)


class FakeEvaluator:
    def __init__(self, qualified: bool = True) -> None:
        self.qualified = qualified

    def evaluate(self, candidate, champion):
        return {
            "qualified": self.qualified,
            "candidate_score": 2.0 if self.qualified else 0.5,
        }


class FakeAdmission:
    def __init__(self, admitted: bool = True) -> None:
        self.admitted = admitted

    def admit(self, evaluation, context):
        return {
            "admitted": self.admitted,
            "reason": "test",
        }


class FakeDecision:
    def __init__(self, decision: str = "PROMOTE") -> None:
        self.decision = decision

    def decide(
        self,
        candidate_id,
        champion_id,
        evaluation,
        context,
    ):
        return {
            "decision": self.decision,
        }


class FakeExecutor:
    def __init__(self, transitioned: bool = True) -> None:
        self.transitioned = transitioned

    def execute(
        self,
        candidate_id,
        champion_id,
        context,
    ):
        return {
            "transitioned": self.transitioned,
        }


class FakeStability:
    def __init__(self, stable: bool = True) -> None:
        self.stable = stable

    def analyze(self, context):
        return {
            "stable": self.stable,
        }


class FakeReplacement:
    def __init__(self, allowed: bool = True) -> None:
        self.allowed = allowed

    def allow(
        self,
        evaluation,
        stability,
        context,
    ):
        return {
            "allowed": self.allowed,
        }


class FakeRollback:
    def __init__(self, allowed: bool = False) -> None:
        self.allowed = allowed

    def allow(self, stability, context):
        return {
            "allowed": self.allowed,
        }


def make_lifecycle(
    *,
    qualified=True,
    admitted=True,
    decision="PROMOTE",
    transitioned=True,
    stable=True,
    replacement=True,
    rollback=False,
):
    return ChampionLifecycle(
        evaluator=FakeEvaluator(qualified),
        admission_controller=FakeAdmission(admitted),
        transition_decision=FakeDecision(decision),
        transition_executor=FakeExecutor(transitioned),
        stability_monitor=FakeStability(stable),
        replacement_guard=FakeReplacement(replacement),
        rollback_guard=FakeRollback(rollback),
    )


def test_module_compiles():
    path = (
        Path(__file__).parents[1]
        / "src"
        / "champion_lifecycle.py"
    )

    py_compile.compile(
        str(path),
        doraise=True,
    )


def test_rejected_evaluation_stops_lifecycle():
    snapshot = make_lifecycle(
        qualified=False
    ).run(
        "candidate",
        "champion",
        {},
        {},
    )

    assert snapshot.status == "REJECTED"
    assert snapshot.admitted is False
    assert snapshot.transitioned is False


def test_admission_denial_stops_lifecycle():
    snapshot = make_lifecycle(
        admitted=False
    ).run(
        "candidate",
        "champion",
        {},
        {},
    )

    assert snapshot.status == "REJECTED"
    assert snapshot.admitted is False


def test_transition_decision_can_reject():
    snapshot = make_lifecycle(
        decision="REJECT"
    ).run(
        "candidate",
        "champion",
        {},
        {},
    )

    assert snapshot.status == "REJECTED"
    assert snapshot.admitted is True
    assert snapshot.transitioned is False


def test_successful_lifecycle_promotes_candidate():
    snapshot = make_lifecycle().run(
        "candidate",
        "champion",
        {},
        {},
    )

    assert snapshot.status == "PROMOTED"
    assert snapshot.admitted is True
    assert snapshot.transitioned is True
    assert snapshot.stable is True
    assert snapshot.replacement_allowed is True


def test_unstable_transition_is_degraded():
    snapshot = make_lifecycle(
        stable=False
    ).run(
        "candidate",
        "champion",
        {},
        {},
    )

    assert snapshot.status == "DEGRADED"
    assert snapshot.transitioned is True
    assert snapshot.stable is False


def test_failed_transition_is_rejected():
    snapshot = make_lifecycle(
        transitioned=False
    ).run(
        "candidate",
        "champion",
        {},
        {},
    )

    assert snapshot.status == "REJECTED"
    assert snapshot.transitioned is False


def test_rollback_flag_is_exposed():
    snapshot = make_lifecycle(
        rollback=True
    ).run(
        "candidate",
        "champion",
        {},
        {},
    )

    assert snapshot.rollback_allowed is True


def test_candidate_id_is_required():
    with pytest.raises(ValueError):
        make_lifecycle().run(
            "",
            "champion",
            {},
            {},
        )


def test_champion_id_is_required():
    with pytest.raises(ValueError):
        make_lifecycle().run(
            "candidate",
            "",
            {},
            {},
        )


def test_snapshot_is_immutable():
    snapshot = make_lifecycle().run(
        "candidate",
        "champion",
        {},
        {},
    )

    assert isinstance(
        snapshot,
        ChampionLifecycleSnapshot,
    )

    with pytest.raises(Exception):
        snapshot.status = "REJECTED"