"""
=========================================================
QuantAI Professional v5
Paper Trading Monitor Tests
=========================================================
"""

from __future__ import annotations

import pytest

from src.paper_trading_engine import (
    PaperTradingEngine,
)
from src.paper_trading_session import (
    PaperTradingSession,
)
from src.paper_trading_monitor import (
    PaperTradingMonitor,
    PaperTradingMonitorSnapshot,
)
from src.strategy import (
    SignalResult,
)


# =========================================================
# HELPERS
# =========================================================

def make_session() -> PaperTradingSession:

    return PaperTradingSession(
        enable_risk_controls=False,
        initial_balance=1000.0,
        commission=0.0,
        quantity=1.0,
    )


def make_signal(
    signal: str,
    entry: float = 100.0,
) -> SignalResult:

    return SignalResult(
        signal=signal,
        score=10.0,
        confidence=80.0,
        entry=entry,
        stop_loss=95.0,
        take_profit=110.0,
    )


# =========================================================
# 1. INITIAL STATE
# =========================================================

def test_initial_state():

    session = make_session()

    monitor = PaperTradingMonitor(
        session
    )

    assert monitor.balance == 1000.0
    assert monitor.realized_profit == 0.0
    assert monitor.total_steps == 0

    assert monitor.opened_positions == 0
    assert monitor.closed_positions == 0

    assert monitor.total_trades == 0
    assert monitor.winning_trades == 0
    assert monitor.losing_trades == 0

    assert monitor.win_rate == 0.0
    assert monitor.return_percent == 0.0

    assert monitor.current_signal == "HOLD"

    assert monitor.has_position is False
    assert monitor.position_side is None
    assert monitor.entry_price is None
    assert monitor.quantity is None


# =========================================================
# 2. INVALID SESSION
# =========================================================

def test_invalid_session():

    with pytest.raises(TypeError):

        PaperTradingMonitor(
            None
        )


# =========================================================
# 3. OPEN LONG
# =========================================================

def test_open_long():

    session = make_session()

    result = session.runner.process_signal(
        make_signal(
            "BUY",
            entry=100.0,
        )
    )

    session._steps = [
        result
    ]

    monitor = PaperTradingMonitor(
        session
    )

    assert monitor.has_position is True
    assert monitor.position_side == "LONG"
    assert monitor.entry_price == 100.0
    assert monitor.quantity == 1.0

    assert monitor.opened_positions == 1
    assert monitor.closed_positions == 0

    assert monitor.current_signal == "BUY"


# =========================================================
# 4. OPEN SHORT
# =========================================================

def test_open_short():

    session = make_session()

    result = session.runner.process_signal(
        make_signal(
            "SELL",
            entry=100.0,
        )
    )

    session._steps = [
        result
    ]

    monitor = PaperTradingMonitor(
        session
    )

    assert monitor.has_position is True
    assert monitor.position_side == "SHORT"
    assert monitor.entry_price == 100.0
    assert monitor.quantity == 1.0

    assert monitor.opened_positions == 1
    assert monitor.current_signal == "SELL"


# =========================================================
# 5. CURRENT HOLD SIGNAL
# =========================================================

def test_current_hold_signal():

    session = make_session()

    result = session.runner.process_signal(
        make_signal(
            "HOLD",
            entry=100.0,
        )
    )

    session._steps = [
        result
    ]

    monitor = PaperTradingMonitor(
        session
    )

    assert monitor.current_signal == "HOLD"
    assert monitor.has_position is False


# =========================================================
# 6. CLOSE PROFITABLE TRADE
# =========================================================

def test_profitable_trade():

    session = make_session()

    opened = session.runner.process_signal(
        make_signal(
            "BUY",
            entry=100.0,
        )
    )

    closed = session.runner.close_position(
        price=120.0,
        signal=make_signal(
            "HOLD",
            entry=120.0,
        ),
    )

    session._steps = [
        opened,
        closed,
    ]

    monitor = PaperTradingMonitor(
        session
    )

    assert monitor.has_position is False

    assert monitor.opened_positions == 1
    assert monitor.closed_positions == 1

    assert monitor.total_trades == 1
    assert monitor.winning_trades == 1
    assert monitor.losing_trades == 0

    assert monitor.realized_profit == 20.0
    assert monitor.win_rate == 100.0
    assert monitor.return_percent == 2.0


# =========================================================
# 7. LOSING TRADE
# =========================================================

def test_losing_trade():

    session = make_session()

    opened = session.runner.process_signal(
        make_signal(
            "BUY",
            entry=100.0,
        )
    )

    closed = session.runner.close_position(
        price=90.0,
        signal=make_signal(
            "HOLD",
            entry=90.0,
        ),
    )

    session._steps = [
        opened,
        closed,
    ]

    monitor = PaperTradingMonitor(
        session
    )

    assert monitor.total_trades == 1
    assert monitor.winning_trades == 0
    assert monitor.losing_trades == 1

    assert monitor.realized_profit == -10.0
    assert monitor.win_rate == 0.0
    assert monitor.return_percent == -1.0


# =========================================================
# 8. MULTIPLE TRADES
# =========================================================

def test_multiple_trades():

    session = make_session()

    opened_1 = session.runner.process_signal(
        make_signal(
            "BUY",
            entry=100.0,
        )
    )

    closed_1 = session.runner.close_position(
        price=110.0,
    )

    opened_2 = session.runner.process_signal(
        make_signal(
            "SELL",
            entry=100.0,
        )
    )

    closed_2 = session.runner.close_position(
        price=90.0,
    )

    session._steps = [
        opened_1,
        closed_1,
        opened_2,
        closed_2,
    ]

    monitor = PaperTradingMonitor(
        session
    )

    assert monitor.total_trades == 2
    assert monitor.winning_trades == 2
    assert monitor.losing_trades == 0

    assert monitor.realized_profit == 20.0
    assert monitor.win_rate == 100.0
    assert monitor.return_percent == 2.0


# =========================================================
# 9. MIXED WIN / LOSS
# =========================================================

def test_mixed_win_loss():

    session = make_session()

    opened_1 = session.runner.process_signal(
        make_signal(
            "BUY",
            entry=100.0,
        )
    )

    closed_1 = session.runner.close_position(
        price=110.0,
    )

    opened_2 = session.runner.process_signal(
        make_signal(
            "BUY",
            entry=100.0,
        )
    )

    closed_2 = session.runner.close_position(
        price=90.0,
    )

    session._steps = [
        opened_1,
        closed_1,
        opened_2,
        closed_2,
    ]

    monitor = PaperTradingMonitor(
        session
    )

    assert monitor.total_trades == 2
    assert monitor.winning_trades == 1
    assert monitor.losing_trades == 1

    assert monitor.realized_profit == 0.0
    assert monitor.win_rate == 50.0
    assert monitor.return_percent == 0.0


# =========================================================
# 10. SNAPSHOT
# =========================================================

def test_snapshot():

    session = make_session()

    opened = session.runner.process_signal(
        make_signal(
            "BUY",
            entry=100.0,
        )
    )

    session._steps = [
        opened
    ]

    monitor = PaperTradingMonitor(
        session
    )

    snapshot = monitor.snapshot()

    assert isinstance(
        snapshot,
        PaperTradingMonitorSnapshot,
    )

    assert snapshot.balance == 1000.0
    assert snapshot.realized_profit == 0.0

    assert snapshot.total_steps == 1
    assert snapshot.opened_positions == 1
    assert snapshot.closed_positions == 0

    assert snapshot.current_signal == "BUY"

    assert snapshot.has_position is True
    assert snapshot.position_side == "LONG"

    assert snapshot.entry_price == 100.0
    assert snapshot.quantity == 1.0

    assert snapshot.total_trades == 0
    assert snapshot.winning_trades == 0
    assert snapshot.losing_trades == 0

    assert snapshot.win_rate == 0.0
    assert snapshot.return_percent == 0.0


# =========================================================
# 11. SNAPSHOT WITHOUT POSITION
# =========================================================

def test_snapshot_without_position():

    session = make_session()

    monitor = PaperTradingMonitor(
        session
    )

    snapshot = monitor.snapshot()

    assert snapshot.has_position is False
    assert snapshot.position_side is None
    assert snapshot.entry_price is None
    assert snapshot.quantity is None


# =========================================================
# 12. MONITOR IS READ ONLY
# =========================================================

def test_monitor_does_not_change_balance():

    session = make_session()

    monitor = PaperTradingMonitor(
        session
    )

    balance_before = session.balance

    _ = monitor.snapshot()
    _ = monitor.balance
    _ = monitor.realized_profit
    _ = monitor.total_steps
    _ = monitor.opened_positions
    _ = monitor.closed_positions
    _ = monitor.total_trades
    _ = monitor.winning_trades
    _ = monitor.losing_trades
    _ = monitor.win_rate
    _ = monitor.return_percent

    assert session.balance == balance_before


# =========================================================
# 13. MONITOR SEES SESSION RESET
# =========================================================

def test_monitor_sees_reset():

    session = make_session()

    opened = session.runner.process_signal(
        make_signal(
            "BUY",
            entry=100.0,
        )
    )

    session._steps = [
        opened
    ]

    monitor = PaperTradingMonitor(
        session
    )

    assert monitor.has_position is True

    session.reset()

    assert monitor.has_position is False
    assert monitor.balance == 1000.0
    assert monitor.realized_profit == 0.0


# =========================================================
# 14. CURRENT SIGNAL AFTER MULTIPLE STEPS
# =========================================================

def test_current_signal_uses_latest_step():

    session = make_session()

    first = session.runner.process_signal(
        make_signal(
            "BUY",
            entry=100.0,
        )
    )

    second = session.runner.process_signal(
        make_signal(
            "BUY",
            entry=101.0,
        )
    )

    session._steps = [
        first,
        second,
    ]

    monitor = PaperTradingMonitor(
        session
    )

    assert monitor.current_signal == "BUY"


# =========================================================
# 15. RETURN CALCULATION
# =========================================================

def test_return_percent():

    session = make_session()

    opened = session.runner.process_signal(
        make_signal(
            "BUY",
            entry=100.0,
        )
    )

    closed = session.runner.close_position(
        price=150.0,
    )

    session._steps = [
        opened,
        closed,
    ]

    monitor = PaperTradingMonitor(
        session
    )

    assert monitor.realized_profit == 50.0
    assert monitor.return_percent == 5.0


# =========================================================
# 16. SHORT PROFIT
# =========================================================

def test_short_profit():

    session = make_session()

    opened = session.runner.process_signal(
        make_signal(
            "SELL",
            entry=100.0,
        )
    )

    closed = session.runner.close_position(
        price=80.0,
    )

    session._steps = [
        opened,
        closed,
    ]

    monitor = PaperTradingMonitor(
        session
    )

    assert monitor.position_side is None

    assert monitor.total_trades == 1
    assert monitor.winning_trades == 1
    assert monitor.realized_profit == 20.0
    assert monitor.win_rate == 100.0


# =========================================================
# 17. NO TRADES
# =========================================================

def test_no_trades():

    session = make_session()

    monitor = PaperTradingMonitor(
        session
    )

    assert monitor.total_trades == 0
    assert monitor.winning_trades == 0
    assert monitor.losing_trades == 0
    assert monitor.win_rate == 0.0


# =========================================================
# 18. STEP COUNT
# =========================================================

def test_step_count():

    session = make_session()

    steps = []

    for _ in range(3):

        steps.append(
            session.runner.process_signal(
                make_signal(
                    "HOLD"
                )
            )
        )

    session._steps = steps

    monitor = PaperTradingMonitor(
        session
    )

    assert monitor.total_steps == 3
