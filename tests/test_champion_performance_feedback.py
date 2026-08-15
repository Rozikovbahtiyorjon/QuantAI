import py_compile
from pathlib import Path

import pytest

from src.champion_performance_feedback import (
    ChampionPerformanceFeedback,
    ChampionPerformanceSnapshot,
)


def metrics(**overrides):
    data = {
        "net_profit": 120.0,
        "win_rate": 58.0,
        "trade_count": 40,
        "max_drawdown": 35.0,
        "signal_quality": 0.72,
        "stability": 0.81,
    }

    data.update(overrides)

    return data


def test_module_compiles():
    path = (
        Path(__file__).parents[1]
        / "src"
        / "champion_performance_feedback.py"
    )

    py_compile.compile(
        str(path),
        doraise=True,
    )


def test_record_creates_snapshot():
    snapshot = ChampionPerformanceFeedback().record(
        "champion-a",
        metrics(),
    )

    assert isinstance(
        snapshot,
        ChampionPerformanceSnapshot,
    )

    assert snapshot.champion_id == "champion-a"
    assert snapshot.net_profit == 120.0


def test_record_preserves_optional_metrics():
    snapshot = ChampionPerformanceFeedback().record(
        "champion-a",
        metrics(
            profit_factor=1.8,
            expectancy=3.2,
        ),
    )

    assert snapshot.profit_factor == 1.8
    assert snapshot.expectancy == 3.2


def test_get_returns_latest_snapshot():
    feedback = ChampionPerformanceFeedback()

    feedback.record(
        "champion-a",
        metrics(net_profit=10.0),
    )

    feedback.update(
        "champion-a",
        metrics(net_profit=25.0),
    )

    snapshot = feedback.get("champion-a")

    assert snapshot is not None
    assert snapshot.net_profit == 25.0


def test_unknown_champion_returns_none():
    assert (
        ChampionPerformanceFeedback().get("missing")
        is None
    )


def test_snapshot_returns_all_champions():
    feedback = ChampionPerformanceFeedback()

    feedback.record(
        "a",
        metrics(),
    )

    feedback.record(
        "b",
        metrics(net_profit=50.0),
    )

    result = feedback.snapshot()

    assert set(result) == {
        "a",
        "b",
    }


def test_missing_metric_is_rejected():
    data = metrics()

    del data["stability"]

    with pytest.raises(ValueError):
        ChampionPerformanceFeedback().record(
            "a",
            data,
        )


def test_invalid_win_rate_is_rejected():
    with pytest.raises(ValueError):
        ChampionPerformanceFeedback().record(
            "a",
            metrics(win_rate=101.0),
        )


def test_invalid_drawdown_is_rejected():
    with pytest.raises(ValueError):
        ChampionPerformanceFeedback().record(
            "a",
            metrics(max_drawdown=-1.0),
        )


def test_invalid_signal_quality_is_rejected():
    with pytest.raises(ValueError):
        ChampionPerformanceFeedback().record(
            "a",
            metrics(signal_quality=1.1),
        )


def test_invalid_stability_is_rejected():
    with pytest.raises(ValueError):
        ChampionPerformanceFeedback().record(
            "a",
            metrics(stability=-0.1),
        )


def test_empty_champion_id_is_rejected():
    with pytest.raises(ValueError):
        ChampionPerformanceFeedback().record(
            "",
            metrics(),
        )


def test_compare_returns_objective_deltas():
    feedback = ChampionPerformanceFeedback()

    candidate = feedback.record(
        "candidate",
        metrics(
            net_profit=150.0,
            win_rate=62.0,
            trade_count=45,
            max_drawdown=30.0,
            signal_quality=0.80,
            stability=0.90,
        ),
    )

    champion = feedback.record(
        "champion",
        metrics(),
    )

    result = feedback.compare(
        candidate,
        champion,
    )

    assert result["net_profit_delta"] == 30.0
    assert result["win_rate_delta"] == 4.0
    assert result["trade_count_delta"] == 5.0
    assert result["max_drawdown_delta"] == -5.0
    assert result["signal_quality_delta"] == pytest.approx(0.08)
    assert result["stability_delta"] == pytest.approx(0.09)


def test_compare_accepts_mappings():
    result = ChampionPerformanceFeedback.compare(
        metrics(net_profit=130.0),
        metrics(net_profit=100.0),
    )

    assert result["net_profit_delta"] == 30.0