from __future__ import annotations

import pytest

from src.trading_activity_optimizer import (
    ActivityAction,
    ActivitySnapshot,
    TradingActivityOptimizer,
)


def test_snapshot_accepts_valid_values() -> None:
    snapshot = ActivitySnapshot(
        trades=5,
        min_trades=3,
        max_trades=8,
        win_rate=0.6,
        average_quality=0.7,
    )

    assert snapshot.trades == 5


def test_snapshot_rejects_invalid_range() -> None:
    with pytest.raises(ValueError):
        ActivitySnapshot(
            trades=5,
            min_trades=9,
            max_trades=8,
        )


def test_optimizer_configuration() -> None:
    optimizer = TradingActivityOptimizer(
        max_target_step=3,
        min_confidence=0.7,
        quality_floor=0.5,
    )

    assert optimizer.max_target_step == 3
    assert optimizer.min_confidence == 0.7


def test_optimizer_rejects_invalid_step() -> None:
    with pytest.raises(ValueError):
        TradingActivityOptimizer(
            max_target_step=0
        )


def test_optimizer_rejects_invalid_confidence() -> None:
    with pytest.raises(ValueError):
        TradingActivityOptimizer(
            min_confidence=1.1
        )


def test_optimizer_rejects_invalid_quality_floor() -> None:
    with pytest.raises(ValueError):
        TradingActivityOptimizer(
            quality_floor=-0.1
        )


def test_low_activity_requests_increase() -> None:
    optimizer = TradingActivityOptimizer()

    result = optimizer.optimize(
        ActivitySnapshot(
            trades=1,
            min_trades=4,
            max_trades=8,
            average_quality=0.8,
        )
    )

    assert result.action is ActivityAction.INCREASE
    assert result.trade_target == 3
    assert result.adjustments["entry_threshold"] < 0.0


def test_low_activity_does_not_relax_low_quality() -> None:
    optimizer = TradingActivityOptimizer()

    result = optimizer.optimize(
        ActivitySnapshot(
            trades=1,
            min_trades=4,
            max_trades=8,
            average_quality=0.2,
        )
    )

    assert result.action is ActivityAction.HOLD
    assert result.trade_target == 1


def test_high_activity_requests_decrease() -> None:
    optimizer = TradingActivityOptimizer()

    result = optimizer.optimize(
        ActivitySnapshot(
            trades=10,
            min_trades=3,
            max_trades=8,
            average_quality=0.8,
        )
    )

    assert result.action is ActivityAction.DECREASE
    assert result.trade_target == 8
    assert result.adjustments["entry_threshold"] > 0.0


def test_activity_in_range_holds() -> None:
    optimizer = TradingActivityOptimizer()

    result = optimizer.optimize(
        ActivitySnapshot(
            trades=5,
            min_trades=3,
            max_trades=8,
            average_quality=0.8,
        )
    )

    assert result.action is ActivityAction.HOLD
    assert result.trade_target == 5


def test_in_range_low_quality_holds() -> None:
    optimizer = TradingActivityOptimizer()

    result = optimizer.optimize(
        ActivitySnapshot(
            trades=5,
            min_trades=3,
            max_trades=8,
            average_quality=0.2,
        )
    )

    assert result.action is ActivityAction.HOLD
    assert (
        result.adjustments["confidence_threshold"]
        == 0.0
    )


def test_from_diagnostics() -> None:
    optimizer = TradingActivityOptimizer()

    snapshot = optimizer.from_diagnostics(
        {
            "trades": 2,
            "min_trades": 4,
            "max_trades": 8,
            "win_rate": 0.55,
            "average_quality": 0.7,
        }
    )

    assert snapshot.trades == 2
    assert snapshot.win_rate == 0.55


def test_from_diagnostics_requires_core_fields() -> None:
    optimizer = TradingActivityOptimizer()

    with pytest.raises(ValueError):
        optimizer.from_diagnostics(
            {
                "trades": 2,
                "min_trades": 4,
            }
        )


def test_from_diagnostics_rejects_non_mapping() -> None:
    optimizer = TradingActivityOptimizer()

    with pytest.raises(TypeError):
        optimizer.from_diagnostics([])


def test_optimize_from_diagnostics() -> None:
    optimizer = TradingActivityOptimizer()

    result = optimizer.optimize_from_diagnostics(
        {
            "trades": 10,
            "min_trades": 3,
            "max_trades": 8,
            "average_quality": 0.8,
        }
    )

    assert result.action is ActivityAction.DECREASE


def test_target_step_is_bounded() -> None:
    optimizer = TradingActivityOptimizer(
        max_target_step=2
    )

    result = optimizer.optimize(
        ActivitySnapshot(
            trades=0,
            min_trades=10,
            max_trades=20,
            average_quality=0.8,
        )
    )

    assert result.trade_target == 2