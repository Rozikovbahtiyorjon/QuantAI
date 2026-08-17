import py_compile
from pathlib import Path

import pytest

from src.champion_admission_controller import (
    ChampionAdmissionController,
)
from src.champion_evaluator import ChampionEvaluator
from src.champion_governance_engine import (
    ChampionGovernanceEngine,
    ChampionGovernanceSnapshot,
)
from src.champion_performance_feedback import (
    ChampionPerformanceFeedback,
)
from src.champion_replacement_guard import (
    ChampionReplacementGuard,
)
from src.champion_rollback_guard import (
    ChampionRollbackGuard,
)
from src.champion_stability_monitor import (
    ChampionStabilityMonitor,
)
from src.champion_transition_decision import (
    ChampionTransitionDecisionEngine,
)
from src.champion_transition_executor import (
    ChampionTransitionExecutor,
)


def metrics(
    profit_factor=1.5,
    net_profit=100.0,
    win_rate=0.55,
    sharpe_ratio=1.0,
    max_drawdown=0.10,
    profitability=1.0,
    return_value=0.10,
    drawdown=0.10,
    trade_count=30,
    signal_quality=0.70,
    stability=0.80,
):
    return {
        "profit_factor": profit_factor,
        "net_profit": net_profit,
        "win_rate": win_rate,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "profitability": profitability,
        "return": return_value,
        "drawdown": drawdown,
        "trade_count": trade_count,
        "signal_quality": signal_quality,
        "stability": stability,
    }


def make_engine(feedback=None):
    return ChampionGovernanceEngine(
        evaluator=ChampionEvaluator(),
        admission_controller=ChampionAdmissionController(
            min_improvement=0.05,
            min_samples=20,
        ),
        transition_decision=ChampionTransitionDecisionEngine(
            min_score_margin=0.0,
            min_stability_score=0.5,
        ),
        transition_executor=ChampionTransitionExecutor(),
        stability_monitor=ChampionStabilityMonitor(),
        replacement_guard=ChampionReplacementGuard(),
        rollback_guard=ChampionRollbackGuard(
            min_degradation=0.10,
            min_samples=20,
        ),
        performance_feedback=feedback,
    )


def test_module_compiles():
    path = (
        Path(__file__).parents[1]
        / "src"
        / "champion_governance_engine.py"
    )

    py_compile.compile(
        str(path),
        doraise=True,
    )


def test_run_returns_expected_snapshot():
    engine = make_engine()

    snapshot = engine.run(
        "candidate",
        "champion",
        metrics(
            profit_factor=1.8,
            net_profit=120.0,
            win_rate=0.60,
            sharpe_ratio=1.2,
            max_drawdown=0.08,
            profitability=2.0,
            return_value=0.20,
            drawdown=0.08,
            stability=0.80,
        ),
        metrics(),
        samples=30,
    )

    assert isinstance(
        snapshot,
        ChampionGovernanceSnapshot,
    )


def test_superior_candidate_is_promoted():
    engine = make_engine()

    snapshot = engine.run(
        "candidate",
        "champion",
        metrics(
            profit_factor=2.0,
            net_profit=150.0,
            win_rate=0.60,
            sharpe_ratio=1.3,
            max_drawdown=0.08,
            profitability=2.0,
            return_value=0.20,
            drawdown=0.08,
            stability=0.80,
        ),
        metrics(),
        samples=30,
    )

    assert snapshot.status == "PROMOTED"
    assert snapshot.qualified is True
    assert snapshot.admission_action == "ADMIT"
    assert snapshot.transition_action == "REPLACE"
    assert snapshot.transition_changed is True
    assert snapshot.stable is True
    assert snapshot.replacement_allowed is True


def test_insufficient_samples_holds_candidate():
    engine = make_engine()

    snapshot = engine.run(
        "candidate",
        "champion",
        metrics(
            profit_factor=2.0,
            net_profit=150.0,
            win_rate=0.60,
            sharpe_ratio=1.3,
            max_drawdown=0.08,
            profitability=2.0,
            return_value=0.20,
            drawdown=0.08,
        ),
        metrics(),
        samples=19,
    )

    assert snapshot.status == "HOLD"
    assert snapshot.admission_action == "HOLD"
    assert snapshot.transition_changed is False


def test_equal_candidate_is_rejected_by_evaluator():
    engine = make_engine()

    snapshot = engine.run(
        "candidate",
        "champion",
        metrics(),
        metrics(),
        samples=30,
    )

    assert snapshot.status == "REJECTED"
    assert snapshot.qualified is False
    assert snapshot.transition_changed is False


def test_unstable_candidate_is_degraded():
    engine = make_engine()

    snapshot = engine.run(
        "candidate",
        "champion",
        metrics(
            profit_factor=2.0,
            net_profit=150.0,
            win_rate=0.60,
            sharpe_ratio=1.3,
            max_drawdown=0.08,
            profitability=2.0,
            return_value=0.20,
            win_rate=0.40,
            drawdown=0.25,
            profit_factor=0.9,
            stability=0.40,
        ),
        metrics(),
        samples=30,
    )

    assert snapshot.status == "DEGRADED"
    assert snapshot.stable is False
    assert snapshot.transition_changed is False


def test_replacement_guard_can_reject_candidate():
    engine = make_engine()

    snapshot = engine.run(
        "candidate",
        "champion",
        metrics(
            profit_factor=2.0,
            net_profit=150.0,
            win_rate=0.60,
            sharpe_ratio=1.3,
            max_drawdown=0.08,
            profitability=2.0,
            return_value=0.20,
            drawdown=0.08,
            profit_factor=0.8,
            stability=0.80,
        ),
        metrics(),
        samples=30,
    )

    assert snapshot.status == "REJECTED"
    assert snapshot.replacement_allowed is False
    assert snapshot.transition_changed is False


def test_rollback_has_priority_after_transition():
    engine = make_engine()

    snapshot = engine.run(
        "candidate",
        "champion",
        metrics(
            profit_factor=2.0,
            net_profit=110.0,
            win_rate=0.60,
            sharpe_ratio=1.2,
            max_drawdown=0.08,
            profitability=0.90,
            return_value=0.50,
            drawdown=0.01,
            trade_count=30,
            signal_quality=0.80,
            stability=0.90,
        ),
        metrics(
            profitability=1.0,
            return_value=0.10,
            win_rate=0.50,
            drawdown=0.10,
            profit_factor=1.2,
            net_profit=100.0,
            max_drawdown=0.10,
            stability=0.80,
        ),
        samples=30,
    )

    assert snapshot.status == "ROLLBACK"
    assert snapshot.rollback_action == "ROLLBACK"
    assert snapshot.rollback_allowed is True


def test_empty_champion_promotes_stable_candidate():
    engine = make_engine()

    snapshot = engine.run(
        "candidate",
        None,
        metrics(
            profit_factor=1.8,
            net_profit=120.0,
            win_rate=0.60,
            sharpe_ratio=1.2,
            max_drawdown=0.08,
            profitability=1.2,
            return_value=0.15,
            drawdown=0.08,
            stability=0.80,
        ),
        {},
        samples=30,
    )

    assert snapshot.status == "PROMOTED"
    assert snapshot.evaluated is False
    assert snapshot.admission_action == "ADMIT"
    assert snapshot.transition_action == "PROMOTE"


def test_performance_feedback_is_recorded():
    feedback = ChampionPerformanceFeedback()
    engine = make_engine(feedback)

    snapshot = engine.run(
        "candidate",
        "champion",
        metrics(
            profit_factor=1.8,
            net_profit=120.0,
            win_rate=0.60,
            sharpe_ratio=1.2,
            max_drawdown=0.08,
            profitability=2.0,
            return_value=0.20,
            drawdown=0.08,
            trade_count=30,
            signal_quality=0.80,
            stability=0.90,
        ),
        metrics(),
        samples=30,
    )

    assert snapshot.status == "PROMOTED"
    assert feedback.get("candidate") is not None
    assert feedback.get("champion") is not None
    assert feedback.get("candidate").net_profit == 120.0


def test_candidate_id_is_required():
    with pytest.raises(ValueError):
        make_engine().run(
            "",
            "champion",
            metrics(),
            metrics(),
            samples=30,
        )


def test_negative_samples_are_rejected():
    with pytest.raises(ValueError):
        make_engine().run(
            "candidate",
            "champion",
            metrics(),
            metrics(),
            samples=-1,
        )


def test_invalid_candidate_metrics_type_is_rejected():
    with pytest.raises(TypeError):
        make_engine().run(
            "candidate",
            "champion",
            [],
            metrics(),
            samples=30,
        )


def test_evaluate_alias_matches_run_for_rejection():
    engine = make_engine()

    candidate = metrics()
    champion = metrics()

    first = engine.run(
        "candidate",
        "champion",
        candidate,
        champion,
        samples=30,
    )

    second = engine.evaluate(
        "candidate",
        "champion",
        candidate,
        champion,
        samples=30,
    )

    assert first == second