from __future__ import annotations

from datetime import datetime

import pytest

from src.trading_activity_monitor import (
    TradingActivityMonitor,
    TradingActivityRecord,
)


def test_initial_snapshot():
    monitor = TradingActivityMonitor(min_trades=2, max_trades=5)

    snapshot = monitor.snapshot()

    assert snapshot.total_signals == 0
    assert snapshot.executed_trades == 0
    assert snapshot.activity_status == "UNDERACTIVE"
    assert "NO_SIGNALS" in snapshot.diagnostics


def test_signal_aggregation():
    monitor = TradingActivityMonitor(min_trades=1, max_trades=5)

    monitor.add_record(
        signal="BUY",
        executed=True,
        confidence=0.8,
    )

    monitor.add_record(
        signal="SELL",
        executed=True,
        confidence=0.7,
    )

    monitor.add_record(
        signal="HOLD",
        executed=False,
        confidence=0.6,
    )

    snapshot = monitor.snapshot()

    assert snapshot.total_signals == 3
    assert snapshot.executed_trades == 2
    assert snapshot.buy_signals == 1
    assert snapshot.sell_signals == 1
    assert snapshot.hold_signals == 1


def test_activity_normal():
    monitor = TradingActivityMonitor(
        min_trades=2,
        max_trades=4,
    )

    for _ in range(2):
        monitor.add_record(
            signal="BUY",
            executed=True,
            confidence=0.8,
        )

    assert monitor.snapshot().activity_status == "NORMAL"


def test_underactive():
    monitor = TradingActivityMonitor(
        min_trades=3,
        max_trades=5,
    )

    monitor.add_record(
        signal="BUY",
        executed=True,
    )

    assert monitor.snapshot().activity_status == "UNDERACTIVE"
    assert "TRADE_COUNT_BELOW_MINIMUM" in monitor.diagnostics()


def test_overactive():
    monitor = TradingActivityMonitor(
        min_trades=1,
        max_trades=2,
    )

    for _ in range(3):
        monitor.add_record(
            signal="SELL",
            executed=True,
        )

    assert monitor.snapshot().activity_status == "OVERACTIVE"
    assert "TRADE_COUNT_ABOVE_MAXIMUM" in monitor.diagnostics()


def test_rates_and_averages():
    monitor = TradingActivityMonitor()

    monitor.add_record(
        signal="BUY",
        executed=True,
        confidence=0.8,
        quality=0.6,
        pnl=10,
    )

    monitor.add_record(
        signal="HOLD",
        executed=False,
        confidence=0.4,
        quality=0.8,
        pnl=-2,
    )

    snapshot = monitor.snapshot()

    assert snapshot.execution_rate == pytest.approx(0.5)
    assert snapshot.trade_rate == pytest.approx(0.5)
    assert snapshot.average_confidence == pytest.approx(0.6)
    assert snapshot.average_quality == pytest.approx(0.7)
    assert snapshot.total_pnl == pytest.approx(8)


def test_low_confidence_diagnostic():
    monitor = TradingActivityMonitor(
        min_confidence=0.7,
    )

    monitor.add_record(
        signal="HOLD",
        confidence=0.5,
    )

    assert "LOW_CONFIDENCE_SIGNALS" in monitor.diagnostics()


def test_low_quality_diagnostic():
    monitor = TradingActivityMonitor(
        min_quality=0.7,
    )

    monitor.add_record(
        signal="HOLD",
        quality=0.5,
    )

    assert "LOW_SIGNAL_QUALITY" in monitor.diagnostics()


def test_invalid_signal():
    monitor = TradingActivityMonitor()

    with pytest.raises(ValueError):
        monitor.add_record(
            signal="INVALID",
        )


def test_invalid_configuration():
    with pytest.raises(ValueError):
        TradingActivityMonitor(
            min_trades=5,
            max_trades=2,
        )

    with pytest.raises(ValueError):
        TradingActivityMonitor(
            min_confidence=1.1,
        )


def test_window_analysis():
    monitor = TradingActivityMonitor(
        min_trades=1,
        max_trades=5,
    )

    monitor.add_record(
        timestamp=datetime(2026, 8, 1, 10, 0),
        signal="BUY",
        executed=True,
    )

    monitor.add_record(
        timestamp=datetime(2026, 8, 2, 10, 0),
        signal="SELL",
        executed=True,
    )

    snapshot = monitor.analyze_window(
        start=datetime(2026, 8, 2),
        end=datetime(2026, 8, 2, 23, 59),
    )

    assert snapshot.total_signals == 1
    assert snapshot.executed_trades == 1


def test_daily_snapshots():
    monitor = TradingActivityMonitor(
        min_trades=1,
        max_trades=5,
    )

    monitor.add_record(
        timestamp=datetime(2026, 8, 1, 10, 0),
        signal="BUY",
        executed=True,
    )

    monitor.add_record(
        timestamp=datetime(2026, 8, 1, 11, 0),
        signal="HOLD",
    )

    monitor.add_record(
        timestamp=datetime(2026, 8, 2, 10, 0),
        signal="SELL",
        executed=True,
    )

    daily = monitor.daily_snapshots()

    assert len(daily) == 2
    assert daily[datetime(2026, 8, 1).date()].executed_trades == 1


def test_reset_and_records_isolation():
    monitor = TradingActivityMonitor()

    monitor.add_record(signal="BUY")

    records = monitor.records
    records.clear()

    assert len(monitor.records) == 1

    monitor.reset()

    assert len(monitor.records) == 0


def test_analyze_result():
    monitor = TradingActivityMonitor(
        min_trades=1,
        max_trades=3,
    )

    monitor.add_record(
        signal="BUY",
        executed=True,
        confidence=0.9,
    )

    result = monitor.analyze()

    assert result["executed_trades"] == 1
    assert result["activity_status"] == "NORMAL"
    assert result["trade_range"] == {
        "min": 1,
        "max": 3,
    }