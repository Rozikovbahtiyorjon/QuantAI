"""
=========================================================
QuantAI Professional v5
Walk-Forward Report Tests

Tests for:

1. Constructor validation
2. Window statistics
3. Balance statistics
4. Profit statistics
5. Trade statistics
6. Profit factor
7. Consistency score
8. Window summary
9. Best/worst windows
10. Complete summary
11. Report printing
12. Convenience function
=========================================================
"""

from __future__ import annotations

from math import inf

import pytest

from src.backtest_engine import BacktestResult

from src.walk_forward_engine import (
    WalkForwardResult,
    WalkForwardWindowResult,
)

from src.walk_forward_report import (
    WalkForwardReport,
    WalkForwardReportResult,
    create_walk_forward_report,
)


# =========================================================
# HELPERS
# =========================================================

def make_backtest_result(
    initial_balance: float,
    final_balance: float,
    net_profit: float,
    total_trades: int,
    winning_trades: int,
    losing_trades: int,
    win_rate: float,
    trades=None,
) -> BacktestResult:

    return BacktestResult(
        initial_balance=initial_balance,

        final_balance=final_balance,

        net_profit=net_profit,

        total_trades=total_trades,

        winning_trades=winning_trades,

        losing_trades=losing_trades,

        win_rate=win_rate,

        trades=[] if trades is None else trades,
    )


def make_window(
    window_id: int,
    train_start: int,
    train_end: int,
    test_start: int,
    test_end: int,
    backtest_result: BacktestResult,
) -> WalkForwardWindowResult:

    return WalkForwardWindowResult(
        window_id=window_id,

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

        backtest_result=backtest_result,
    )


def make_result() -> WalkForwardResult:

    window1 = make_window(
        window_id=1,
        train_start=0,
        train_end=10,
        test_start=10,
        test_end=15,
        backtest_result=make_backtest_result(
            initial_balance=1000.0,
            final_balance=1010.0,
            net_profit=10.0,
            total_trades=2,
            winning_trades=1,
            losing_trades=1,
            win_rate=50.0,
            trades=[
                {"net_profit": 15.0},
                {"net_profit": -5.0},
            ],
        ),
    )

    window2 = make_window(
        window_id=2,
        train_start=5,
        train_end=15,
        test_start=15,
        test_end=20,
        backtest_result=make_backtest_result(
            initial_balance=1010.0,
            final_balance=1005.0,
            net_profit=-5.0,
            total_trades=2,
            winning_trades=0,
            losing_trades=2,
            win_rate=0.0,
            trades=[
                {"net_profit": -2.0},
                {"net_profit": -3.0},
            ],
        ),
    )

    window3 = make_window(
        window_id=3,
        train_start=10,
        train_end=20,
        test_start=20,
        test_end=25,
        backtest_result=make_backtest_result(
            initial_balance=1005.0,
            final_balance=1020.0,
            net_profit=15.0,
            total_trades=3,
            winning_trades=2,
            losing_trades=1,
            win_rate=66.67,
            trades=[
                {"net_profit": 20.0},
                {"net_profit": 5.0},
                {"net_profit": -10.0},
            ],
        ),
    )

    return WalkForwardResult(
        initial_balance=1000.0,

        final_balance=1020.0,

        net_profit=20.0,

        total_trades=7,

        winning_trades=3,

        losing_trades=4,

        win_rate=42.86,

        windows=[
            window1,
            window2,
            window3,
        ],
    )


# =========================================================
# CONSTRUCTOR
# =========================================================

def test_constructor_requires_walk_forward_result():

    with pytest.raises(TypeError):

        WalkForwardReport(
            None
        )


def test_constructor_accepts_walk_forward_result():

    result = make_result()

    report = WalkForwardReport(
        result
    )

    assert report.result is result


# =========================================================
# WINDOW COUNTS
# =========================================================

def test_total_windows():

    report = WalkForwardReport(
        make_result()
    )

    assert report.total_windows == 3


def test_profitable_windows():

    report = WalkForwardReport(
        make_result()
    )

    assert report.profitable_windows == 2


def test_losing_windows():

    report = WalkForwardReport(
        make_result()
    )

    assert report.losing_windows == 1


def test_flat_windows():

    report = WalkForwardReport(
        make_result()
    )

    assert report.flat_windows == 0


def test_window_win_rate():

    report = WalkForwardReport(
        make_result()
    )

    assert report.window_win_rate == 66.67


# =========================================================
# BALANCE
# =========================================================

def test_initial_balance():

    report = WalkForwardReport(
        make_result()
    )

    assert report.initial_balance == 1000.0


def test_final_balance():

    report = WalkForwardReport(
        make_result()
    )

    assert report.final_balance == 1020.0


# =========================================================
# PROFIT
# =========================================================

def test_net_profit():

    report = WalkForwardReport(
        make_result()
    )

    assert report.net_profit == 20.0


def test_cumulative_return():

    report = WalkForwardReport(
        make_result()
    )

    assert report.cumulative_return == 2.0


def test_window_profits():

    report = WalkForwardReport(
        make_result()
    )

    assert report.window_profits == [
        10.0,
        -5.0,
        15.0,
    ]


def test_average_window_profit():

    report = WalkForwardReport(
        make_result()
    )

    assert report.average_window_profit == (
        20.0 / 3.0
    )


def test_best_window_profit():

    report = WalkForwardReport(
        make_result()
    )

    assert report.best_window_profit == 15.0


def test_worst_window_profit():

    report = WalkForwardReport(
        make_result()
    )

    assert report.worst_window_profit == -5.0


# =========================================================
# TRADE STATISTICS
# =========================================================

def test_total_trades():

    report = WalkForwardReport(
        make_result()
    )

    assert report.total_trades == 7


def test_winning_trades():

    report = WalkForwardReport(
        make_result()
    )

    assert report.winning_trades == 3


def test_losing_trades():

    report = WalkForwardReport(
        make_result()
    )

    assert report.losing_trades == 4


def test_win_rate():

    report = WalkForwardReport(
        make_result()
    )

    assert report.win_rate == 42.86


def test_average_trade_profit():

    report = WalkForwardReport(
        make_result()
    )

    assert report.average_trade_profit == (
        20.0 / 7.0
    )


# =========================================================
# PROFIT FACTOR
# =========================================================

def test_profit_factor():

    report = WalkForwardReport(
        make_result()
    )

    # Gross profit:
    #
    # 15 + 20 + 5 = 40
    #
    # Gross loss:
    #
    # 5 + 2 + 3 + 10 = 20
    #
    # Profit factor = 2.0

    assert report.profit_factor == 2.0


def test_profit_factor_returns_zero_without_profit():

    result = WalkForwardResult(
        initial_balance=1000.0,
        final_balance=990.0,
        net_profit=-10.0,
        total_trades=1,
        winning_trades=0,
        losing_trades=1,
        win_rate=0.0,
        windows=[
            make_window(
                window_id=1,
                train_start=0,
                train_end=10,
                test_start=10,
                test_end=15,
                backtest_result=make_backtest_result(
                    initial_balance=1000.0,
                    final_balance=990.0,
                    net_profit=-10.0,
                    total_trades=1,
                    winning_trades=0,
                    losing_trades=1,
                    win_rate=0.0,
                    trades=[
                        {
                            "net_profit": -10.0
                        }
                    ],
                ),
            )
        ],
    )

    report = WalkForwardReport(
        result
    )

    assert report.profit_factor == 0.0


def test_profit_factor_returns_infinity_without_losses():

    result = WalkForwardResult(
        initial_balance=1000.0,
        final_balance=1030.0,
        net_profit=30.0,
        total_trades=2,
        winning_trades=2,
        losing_trades=0,
        win_rate=100.0,
        windows=[
            make_window(
                window_id=1,
                train_start=0,
                train_end=10,
                test_start=10,
                test_end=15,
                backtest_result=make_backtest_result(
                    initial_balance=1000.0,
                    final_balance=1030.0,
                    net_profit=30.0,
                    total_trades=2,
                    winning_trades=2,
                    losing_trades=0,
                    win_rate=100.0,
                    trades=[
                        {
                            "net_profit": 10.0
                        },
                        {
                            "net_profit": 20.0
                        },
                    ],
                ),
            )
        ],
    )

    report = WalkForwardReport(
        result
    )

    assert report.profit_factor == inf


# =========================================================
# CONSISTENCY
# =========================================================

def test_consistency_score():

    report = WalkForwardReport(
        make_result()
    )

    assert report.consistency_score == 66.67


# =========================================================
# WINDOW SUMMARY
# =========================================================

def test_window_summary_returns_all_windows():

    report = WalkForwardReport(
        make_result()
    )

    summary = report.window_summary()

    assert len(summary) == 3


def test_window_summary_contains_expected_fields():

    report = WalkForwardReport(
        make_result()
    )

    summary = report.window_summary()

    row = summary[0]

    assert row["window_id"] == 1
    assert row["train_start"] == 0
    assert row["train_end"] == 10
    assert row["test_start"] == 10
    assert row["test_end"] == 15

    assert row["train_size"] == 10
    assert row["test_size"] == 5

    assert row["initial_balance"] == 1000.0
    assert row["final_balance"] == 1010.0
    assert row["net_profit"] == 10.0

    assert row["total_trades"] == 2
    assert row["winning_trades"] == 1
    assert row["losing_trades"] == 1


# =========================================================
# BEST / WORST WINDOW
# =========================================================

def test_best_window():

    report = WalkForwardReport(
        make_result()
    )

    best = report.best_window

    assert best is not None
    assert best.window_id == 3


def test_worst_window():

    report = WalkForwardReport(
        make_result()
    )

    worst = report.worst_window

    assert worst is not None
    assert worst.window_id == 2


# =========================================================
# SUMMARY
# =========================================================

def test_summary_type():

    report = WalkForwardReport(
        make_result()
    )

    summary = report.summarize()

    assert isinstance(
        summary,
        WalkForwardReportResult,
    )


def test_summary_values():

    report = WalkForwardReport(
        make_result()
    )

    summary = report.summarize()

    assert summary.total_windows == 3
    assert summary.profitable_windows == 2
    assert summary.losing_windows == 1
    assert summary.flat_windows == 0

    assert summary.initial_balance == 1000.0
    assert summary.final_balance == 1020.0
    assert summary.net_profit == 20.0

    assert summary.cumulative_return == 2.0

    assert summary.total_trades == 7
    assert summary.winning_trades == 3
    assert summary.losing_trades == 4

    assert summary.win_rate == 42.86

    assert summary.average_window_profit == (
        20.0 / 3.0
    )

    assert summary.best_window_profit == 15.0
    assert summary.worst_window_profit == -5.0

    assert summary.profit_factor == 2.0
    assert summary.consistency_score == 66.67


# =========================================================
# PRINT REPORT
# =========================================================

def test_print_report(
    capsys,
):

    report = WalkForwardReport(
        make_result()
    )

    summary = report.summarize()

    WalkForwardReport.print_report(
        summary
    )

    captured = capsys.readouterr()

    assert (
        "QUANTAI WALK-FORWARD PERFORMANCE REPORT"
        in captured.out
    )

    assert "Initial Balance" in captured.out
    assert "Final Balance" in captured.out
    assert "Net Profit" in captured.out
    assert "Consistency Score" in captured.out
    assert "Window" in captured.out


def test_print_report_rejects_invalid_type():

    with pytest.raises(TypeError):

        WalkForwardReport.print_report(
            None
        )


# =========================================================
# CONVENIENCE FUNCTION
# =========================================================

def test_create_walk_forward_report():

    result = make_result()

    summary = create_walk_forward_report(
        result
    )

    assert isinstance(
        summary,
        WalkForwardReportResult,
    )

    assert summary.total_windows == 3
    assert summary.net_profit == 20.0
    assert summary.consistency_score == 66.67


# =========================================================
# EMPTY WINDOWS
# =========================================================

def test_empty_windows_are_supported():

    result = WalkForwardResult(
        initial_balance=1000.0,
        final_balance=1000.0,
        net_profit=0.0,
        total_trades=0,
        winning_trades=0,
        losing_trades=0,
        win_rate=0.0,
        windows=[],
    )

    report = WalkForwardReport(
        result
    )

    assert report.total_windows == 0
    assert report.profitable_windows == 0
    assert report.losing_windows == 0
    assert report.flat_windows == 0

    assert report.window_win_rate == 0.0
    assert report.average_window_profit == 0.0
    assert report.best_window_profit == 0.0
    assert report.worst_window_profit == 0.0

    assert report.total_trades == 0
    assert report.win_rate == 0.0
    assert report.average_trade_profit == 0.0

    assert report.profit_factor == 0.0
    assert report.consistency_score == 0.0

    assert report.best_window is None
    assert report.worst_window is None
