import pytest

from src.champion_stability_monitor import (
    ChampionStabilityMonitor,
    StabilitySnapshot,
    StabilityThresholds,
)


def test_default_thresholds():
    thresholds = StabilityThresholds()

    assert thresholds.min_win_rate == 0.45
    assert thresholds.max_drawdown == 0.20
    assert thresholds.min_profit_factor == 1.0
    assert thresholds.min_trade_count == 10
    assert thresholds.max_return_drop == 0.10


def test_stable_champion():
    monitor = ChampionStabilityMonitor()

    current = {
        "return": 0.20,
        "win_rate": 0.55,
        "drawdown": 0.10,
        "profit_factor": 1.30,
        "trade_count": 50,
    }

    snapshot = monitor.analyze(current, {"return": 0.18})

    assert isinstance(snapshot, StabilitySnapshot)
    assert snapshot.status == "STABLE"
    assert snapshot.reasons == ()
    assert monitor.is_stable(current)


def test_low_win_rate_warning():
    snapshot = ChampionStabilityMonitor().analyze(
        {
            "return": 0.20,
            "win_rate": 0.40,
            "drawdown": 0.10,
            "profit_factor": 1.30,
            "trade_count": 50,
        }
    )

    assert snapshot.status == "WARNING"
    assert "low_win_rate" in snapshot.reasons


def test_high_drawdown_degraded():
    snapshot = ChampionStabilityMonitor().analyze(
        {
            "return": 0.20,
            "win_rate": 0.55,
            "drawdown": 0.25,
            "profit_factor": 1.30,
            "trade_count": 50,
        }
    )

    assert snapshot.status == "DEGRADED"
    assert "high_drawdown" in snapshot.reasons


def test_low_profit_factor_degraded():
    snapshot = ChampionStabilityMonitor().analyze(
        {
            "return": 0.20,
            "win_rate": 0.55,
            "drawdown": 0.10,
            "profit_factor": 0.90,
            "trade_count": 50,
        }
    )

    assert snapshot.status == "DEGRADED"
    assert "low_profit_factor" in snapshot.reasons


def test_low_trade_count_warning():
    snapshot = ChampionStabilityMonitor().analyze(
        {
            "return": 0.20,
            "win_rate": 0.55,
            "drawdown": 0.10,
            "profit_factor": 1.30,
            "trade_count": 5,
        }
    )

    assert snapshot.status == "WARNING"
    assert "low_trade_count" in snapshot.reasons


def test_return_degradation_warning():
    snapshot = ChampionStabilityMonitor().analyze(
        {
            "return": 0.15,
            "win_rate": 0.55,
            "drawdown": 0.10,
            "profit_factor": 1.30,
            "trade_count": 50,
        },
        {"return": 0.20},
    )

    assert snapshot.status == "WARNING"
    assert "return_degradation" in snapshot.reasons


def test_multiple_reasons():
    snapshot = ChampionStabilityMonitor().analyze(
        {
            "return": 0.10,
            "win_rate": 0.40,
            "drawdown": 0.25,
            "profit_factor": 0.80,
            "trade_count": 5,
        },
        {"return": 0.20},
    )

    assert snapshot.status == "DEGRADED"

    assert {
        "low_win_rate",
        "high_drawdown",
        "low_profit_factor",
        "low_trade_count",
        "return_degradation",
    }.issubset(set(snapshot.reasons))


def test_total_return_alias():
    snapshot = ChampionStabilityMonitor().analyze(
        {
            "total_return": 0.20,
            "win_rate": 0.55,
            "drawdown": 0.10,
            "profit_factor": 1.30,
            "trade_count": 20,
        }
    )

    assert snapshot.return_value == pytest.approx(0.20)


def test_negative_drawdown_is_normalized():
    snapshot = ChampionStabilityMonitor().analyze(
        {
            "return": 0.20,
            "win_rate": 0.55,
            "drawdown": -0.25,
            "profit_factor": 1.30,
            "trade_count": 20,
        }
    )

    assert snapshot.drawdown == pytest.approx(0.25)
    assert snapshot.status == "DEGRADED"


def test_invalid_trade_count_is_safe():
    snapshot = ChampionStabilityMonitor().analyze(
        {
            "return": 0.20,
            "win_rate": 0.55,
            "drawdown": 0.10,
            "profit_factor": 1.30,
            "trade_count": "invalid",
        }
    )

    assert snapshot.trade_count == 0
    assert snapshot.status == "WARNING"


def test_custom_thresholds():
    monitor = ChampionStabilityMonitor(
        StabilityThresholds(
            min_win_rate=0.60,
            max_drawdown=0.10,
            min_profit_factor=1.20,
            min_trade_count=20,
            max_return_drop=0.05,
        )
    )

    snapshot = monitor.analyze(
        {
            "return": 0.18,
            "win_rate": 0.58,
            "drawdown": 0.11,
            "profit_factor": 1.10,
            "trade_count": 15,
        },
        {"return": 0.20},
    )

    assert snapshot.status == "DEGRADED"
    assert len(snapshot.reasons) == 5


def test_empty_metrics_are_not_stable():
    snapshot = ChampionStabilityMonitor().analyze({})

    assert snapshot.status == "WARNING"
    assert snapshot.trade_count == 0


def test_is_stable_false_for_degraded():
    monitor = ChampionStabilityMonitor()

    assert not monitor.is_stable(
        {
            "return": 0.05,
            "win_rate": 0.40,
            "drawdown": 0.25,
            "profit_factor": 0.80,
            "trade_count": 50,
        }
    )