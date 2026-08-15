"""
=========================================================
QuantAI WalkForwardAnalyzer Tests
=========================================================
"""

from __future__ import annotations

from src.backtest_engine import BacktestResult
from src.walk_forward_analyzer import (
    WalkForwardAnalyzer,
)
from src.walk_forward_engine import (
    WalkForwardWindowResult,
)


# =========================================================
# HELPERS
# =========================================================

def make_result(
    window_number: int,
    profit: float,
    trades: int,
    wins: int,
    losses: int,
    initial: float,
    final: float,
) -> WalkForwardWindowResult:

    backtest = BacktestResult(
        initial_balance=initial,
        final_balance=final,
        net_profit=profit,
        total_trades=trades,
        winning_trades=wins,
        losing_trades=losses,
        win_rate=(
            wins / trades * 100
            if trades
            else 0.0
        ),
        trades=[],
    )

    train_start = (
        window_number - 1
    ) * 5

    train_end = train_start + 10

    test_start = train_end

    test_end = test_start + 5

    return WalkForwardWindowResult(
        window_id=window_number,

        train_start=train_start,
        train_end=train_end,

        test_start=test_start,
        test_end=test_end,

        train_size=(
            train_end
            - train_start
        ),

        test_size=(
            test_end
            - test_start
        ),

        backtest_result=backtest,
    )


# =========================================================
# 1. EMPTY RESULTS
# =========================================================

def test_empty_results():

    analyzer = WalkForwardAnalyzer([])

    assert analyzer.total_windows == 0
    assert analyzer.total_trades == 0
    assert analyzer.winning_trades == 0
    assert analyzer.losing_trades == 0
    assert analyzer.total_profit == 0.0
    assert analyzer.initial_balance == 0.0
    assert analyzer.final_balance == 0.0
    assert analyzer.profitable_windows == 0
    assert analyzer.losing_windows == 0
    assert analyzer.win_rate == 0.0


# =========================================================
# 2. WINDOW COUNT
# =========================================================

def test_total_windows():

    results = [
        make_result(
            1,
            10.0,
            2,
            1,
            1,
            1000.0,
            1010.0,
        ),
        make_result(
            2,
            20.0,
            3,
            2,
            1,
            1010.0,
            1030.0,
        ),
    ]

    analyzer = WalkForwardAnalyzer(
        results
    )

    assert analyzer.total_windows == 2


# =========================================================
# 3. TRADE AGGREGATION
# =========================================================

def test_trade_counts_are_aggregated():

    results = [
        make_result(
            1,
            10.0,
            2,
            1,
            1,
            1000.0,
            1010.0,
        ),
        make_result(
            2,
            20.0,
            3,
            2,
            1,
            1010.0,
            1030.0,
        ),
    ]

    analyzer = WalkForwardAnalyzer(
        results
    )

    assert analyzer.total_trades == 5
    assert analyzer.winning_trades == 3
    assert analyzer.losing_trades == 2


# =========================================================
# 4. PROFIT
# =========================================================

def test_total_profit():

    results = [
        make_result(
            1,
            10.25,
            2,
            1,
            1,
            1000.0,
            1010.25,
        ),
        make_result(
            2,
            -5.25,
            2,
            1,
            1,
            1010.25,
            1005.00,
        ),
        make_result(
            3,
            20.00,
            2,
            2,
            0,
            1005.00,
            1025.00,
        ),
    ]

    analyzer = WalkForwardAnalyzer(
        results
    )

    assert analyzer.total_profit == 25.00


# =========================================================
# 5. BALANCE
# =========================================================

def test_initial_and_final_balance():

    results = [
        make_result(
            1,
            10.0,
            1,
            1,
            0,
            1000.0,
            1010.0,
        ),
        make_result(
            2,
            15.0,
            1,
            1,
            0,
            1010.0,
            1025.0,
        ),
    ]

    analyzer = WalkForwardAnalyzer(
        results
    )

    assert analyzer.initial_balance == 1000.0
    assert analyzer.final_balance == 1025.0


# =========================================================
# 6. PROFITABLE WINDOWS
# =========================================================

def test_profitable_and_losing_windows():

    results = [
        make_result(
            1,
            10.0,
            1,
            1,
            0,
            1000.0,
            1010.0,
        ),
        make_result(
            2,
            -5.0,
            1,
            0,
            1,
            1010.0,
            1005.0,
        ),
        make_result(
            3,
            20.0,
            1,
            1,
            0,
            1005.0,
            1025.0,
        ),
        make_result(
            4,
            0.0,
            1,
            0,
            1,
            1025.0,
            1025.0,
        ),
    ]

    analyzer = WalkForwardAnalyzer(
        results
    )

    assert analyzer.profitable_windows == 2
    assert analyzer.losing_windows == 1


# =========================================================
# 7. WIN RATE
# =========================================================

def test_win_rate():

    results = [
        make_result(
            1,
            10.0,
            4,
            3,
            1,
            1000.0,
            1010.0,
        ),
        make_result(
            2,
            10.0,
            2,
            1,
            1,
            1010.0,
            1020.0,
        ),
    ]

    analyzer = WalkForwardAnalyzer(
        results
    )

    assert analyzer.win_rate == 66.67


# =========================================================
# 8. SUMMARY
# =========================================================

def test_summary():

    results = [
        make_result(
            1,
            10.0,
            2,
            1,
            1,
            1000.0,
            1010.0,
        ),
        make_result(
            2,
            20.0,
            3,
            2,
            1,
            1010.0,
            1030.0,
        ),
    ]

    analyzer = WalkForwardAnalyzer(
        results
    )

    summary = analyzer.summarize()

    assert summary.total_windows == 2
    assert summary.total_trades == 5
    assert summary.winning_trades == 3
    assert summary.losing_trades == 2
    assert summary.total_profit == 30.0
    assert summary.initial_balance == 1000.0
    assert summary.final_balance == 1030.0
    assert summary.profitable_windows == 2
    assert summary.losing_windows == 0
    assert summary.win_rate == 60.0


# =========================================================
# 9. CUMULATIVE RETURN
# =========================================================

def test_cumulative_return():

    results = [
        make_result(
            1,
            10.0,
            1,
            1,
            0,
            1000.0,
            1010.0,
        ),
        make_result(
            2,
            20.0,
            1,
            1,
            0,
            1010.0,
            1030.0,
        ),
    ]

    analyzer = WalkForwardAnalyzer(
        results
    )

    assert analyzer.cumulative_return == 3.0


# =========================================================
# 10. MAXIMUM DRAWDOWN
# =========================================================

def test_max_drawdown():

    results = [
        make_result(
            1,
            20.0,
            1,
            1,
            0,
            1000.0,
            1020.0,
        ),
        make_result(
            2,
            -10.0,
            1,
            0,
            1,
            1020.0,
            1010.0,
        ),
        make_result(
            3,
            -30.0,
            1,
            0,
            1,
            1010.0,
            980.0,
        ),
        make_result(
            4,
            40.0,
            1,
            1,
            0,
            980.0,
            1020.0,
        ),
    ]

    analyzer = WalkForwardAnalyzer(
        results
    )

    assert analyzer.max_drawdown == 40.0


# =========================================================
# 11. MAXIMUM DRAWDOWN PERCENT
# =========================================================

def test_max_drawdown_percent():

    results = [
        make_result(
            1,
            20.0,
            1,
            1,
            0,
            1000.0,
            1020.0,
        ),
        make_result(
            2,
            -40.0,
            1,
            0,
            1,
            1020.0,
            980.0,
        ),
    ]

    analyzer = WalkForwardAnalyzer(
        results
    )

    assert analyzer.max_drawdown_percent == 3.92


# =========================================================
# 12. EMPTY DRAWDOWN
# =========================================================

def test_empty_drawdown():

    analyzer = WalkForwardAnalyzer([])

    assert analyzer.cumulative_return == 0.0
    assert analyzer.max_drawdown == 0.0
    assert analyzer.max_drawdown_percent == 0.0
