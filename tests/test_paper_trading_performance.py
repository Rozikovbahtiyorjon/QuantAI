"""
=========================================================
QuantAI Professional v5
Paper Trading Performance Tests
=========================================================
"""

from __future__ import annotations

import pytest
from src.paper_trading_engine import PaperTrade
from src.paper_trading_runner import PaperTradingStepResult
from src.paper_trading_session import (
    PaperTradingSessionResult,
)
from src.paper_trading_performance import (
    PaperTradingPerformance,
)


# =========================================================
# HELPERS
# =========================================================

def make_trade(
    profit: float,
) -> PaperTrade:

    return PaperTrade(
        side="LONG",
        entry_price=100.0,
        exit_price=100.0 + profit,
        quantity=1.0,
        gross_profit=profit,
        fees=0.0,
        net_profit=profit,
    )


def make_session(
    trades: list[PaperTrade],
    initial: float = 1000.0,
) -> PaperTradingSessionResult:

    steps = [
        PaperTradingStepResult(
            signal=None,
            trade=trade,
            position_opened=False,
            position_closed=True,
        )
        for trade in trades
    ]

    final = (
        initial
        + sum(
            trade.net_profit
            for trade in trades
        )
    )

    return PaperTradingSessionResult(
        steps=steps,
        initial_balance=initial,
        final_balance=final,
        realized_profit=(
            final - initial
        ),
        total_steps=len(steps),
        opened_positions=0,
        closed_positions=len(trades),
    )


# =========================================================
# 1. EMPTY SESSION
# =========================================================

def test_empty_session():

    session = make_session([])

    performance = PaperTradingPerformance(
        session
    )

    assert performance.total_trades == 0
    assert performance.winning_trades == 0
    assert performance.losing_trades == 0
    assert performance.win_rate == 0.0
    assert performance.total_profit == 0.0
    assert performance.average_trade == 0.0
    assert performance.average_win == 0.0
    assert performance.average_loss == 0.0
    assert performance.profit_factor == 0.0
    assert performance.cumulative_return == 0.0
    assert performance.max_drawdown == 0.0
    assert performance.max_drawdown_percent == 0.0


# =========================================================
# 2. TRADE COUNTS
# =========================================================

def test_trade_counts():

    trades = [
        make_trade(10.0),
        make_trade(-5.0),
        make_trade(20.0),
    ]

    performance = PaperTradingPerformance(
        make_session(trades)
    )

    assert performance.total_trades == 3
    assert performance.winning_trades == 2
    assert performance.losing_trades == 1


# =========================================================
# 3. WIN RATE
# =========================================================

def test_win_rate():

    trades = [
        make_trade(10.0),
        make_trade(20.0),
        make_trade(-5.0),
    ]

    performance = PaperTradingPerformance(
        make_session(trades)
    )

    assert performance.win_rate == 66.67


# =========================================================
# 4. TOTAL PROFIT
# =========================================================

def test_total_profit():

    trades = [
        make_trade(10.25),
        make_trade(-5.25),
        make_trade(20.00),
    ]

    performance = PaperTradingPerformance(
        make_session(trades)
    )

    assert performance.total_profit == 25.0


# =========================================================
# 5. AVERAGE TRADE
# =========================================================

def test_average_trade():

    trades = [
        make_trade(10.0),
        make_trade(-5.0),
        make_trade(20.0),
    ]

    performance = PaperTradingPerformance(
        make_session(trades)
    )

    assert performance.average_trade == pytest.approx(
    25.0 / 3.0,
    abs=1e-8,
    )


# =========================================================
# 6. AVERAGE WIN
# =========================================================

def test_average_win():

    trades = [
        make_trade(10.0),
        make_trade(20.0),
        make_trade(-5.0),
    ]

    performance = PaperTradingPerformance(
        make_session(trades)
    )

    assert performance.average_win == 15.0


# =========================================================
# 7. AVERAGE LOSS
# =========================================================

def test_average_loss():

    trades = [
        make_trade(10.0),
        make_trade(-5.0),
        make_trade(-15.0),
    ]

    performance = PaperTradingPerformance(
        make_session(trades)
    )

    assert performance.average_loss == -10.0


# =========================================================
# 8. PROFIT FACTOR
# =========================================================

def test_profit_factor():

    trades = [
        make_trade(20.0),
        make_trade(10.0),
        make_trade(-5.0),
        make_trade(-5.0),
    ]

    performance = PaperTradingPerformance(
        make_session(trades)
    )

    assert performance.profit_factor == 3.0


# =========================================================
# 9. PROFIT FACTOR WITHOUT LOSSES
# =========================================================

def test_profit_factor_without_losses():

    trades = [
        make_trade(10.0),
        make_trade(20.0),
    ]

    performance = PaperTradingPerformance(
        make_session(trades)
    )

    assert performance.profit_factor == float("inf")


# =========================================================
# 10. CUMULATIVE RETURN
# =========================================================

def test_cumulative_return():

    trades = [
        make_trade(10.0),
        make_trade(20.0),
    ]

    performance = PaperTradingPerformance(
        make_session(
            trades,
            initial=1000.0,
        )
    )

    assert performance.cumulative_return == 3.0


# =========================================================
# 11. EQUITY CURVE
# =========================================================

def test_equity_curve():

    trades = [
        make_trade(20.0),
        make_trade(-10.0),
        make_trade(30.0),
    ]

    performance = PaperTradingPerformance(
        make_session(trades)
    )

    assert performance.equity_curve == [
        1000.0,
        1020.0,
        1010.0,
        1040.0,
    ]


# =========================================================
# 12. MAX DRAWDOWN
# =========================================================

def test_max_drawdown():

    trades = [
        make_trade(20.0),
        make_trade(-10.0),
        make_trade(-30.0),
        make_trade(40.0),
    ]

    performance = PaperTradingPerformance(
        make_session(trades)
    )

    assert performance.max_drawdown == 40.0


# =========================================================
# 13. MAX DRAWDOWN %
# =========================================================

def test_max_drawdown_percent():

    trades = [
        make_trade(20.0),
        make_trade(-40.0),
    ]

    performance = PaperTradingPerformance(
        make_session(trades)
    )

    assert performance.max_drawdown_percent == 3.92


# =========================================================
# 14. SUMMARY
# =========================================================

def test_summary():

    trades = [
        make_trade(10.0),
        make_trade(-5.0),
        make_trade(20.0),
    ]

    performance = PaperTradingPerformance(
        make_session(trades)
    )

    summary = performance.summarize()

    assert summary.total_trades == 3
    assert summary.winning_trades == 2
    assert summary.losing_trades == 1
    assert summary.win_rate == 66.67
    assert summary.total_profit == 25.0
    assert summary.average_trade == pytest.approx(
    25.0 / 3.0,
    abs=1e-8,
    )
    assert summary.average_win == 15.0
    assert summary.average_loss == -5.0
    assert summary.profit_factor == 6.0
    assert summary.cumulative_return == 2.5
    assert summary.max_drawdown == 5.0
    assert summary.max_drawdown_percent == 0.5


# =========================================================
# 15. TYPE VALIDATION
# =========================================================

def test_invalid_session_result():

    try:
        PaperTradingPerformance(None)
        assert False
    except TypeError:
        pass
