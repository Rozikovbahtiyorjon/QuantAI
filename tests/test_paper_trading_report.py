"""
=========================================================
QuantAI Professional v5
Paper Trading Report Tests
=========================================================
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.paper_trading_engine import PaperTrade
from src.paper_trading_performance import (
    PaperTradingPerformance,
)
from src.paper_trading_report import (
    PaperTradingReport,
    PaperTradingReportResult,
)
from src.paper_trading_session import (
    PaperTradingSessionResult,
)
from src.paper_trading_runner import (
    PaperTradingStepResult,
)
from src.strategy import SignalResult


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


def make_signal() -> SignalResult:

    return SignalResult(
        signal="HOLD",
        entry=100.0,
    )


def make_session(
    trades: list[PaperTrade],
    initial_balance: float = 1000.0,
) -> PaperTradingSessionResult:

    steps = [
        PaperTradingStepResult(
            signal=make_signal(),
            trade=trade,
            position_opened=False,
            position_closed=True,
        )
        for trade in trades
    ]

    total_profit = sum(
        trade.net_profit
        for trade in trades
    )

    return PaperTradingSessionResult(
        steps=steps,
        initial_balance=initial_balance,
        final_balance=initial_balance + total_profit,
        realized_profit=total_profit,
        total_steps=len(steps),
        opened_positions=0,
        closed_positions=len(trades),
    )


def make_performance(
    trades: list[PaperTrade],
) -> PaperTradingPerformance:

    return PaperTradingPerformance(
        make_session(trades)
    )


# =========================================================
# 1. INITIALIZATION
# =========================================================

def test_initialization():

    performance = make_performance(
        [
            make_trade(10.0),
            make_trade(-5.0),
        ]
    )

    report = PaperTradingReport(
        performance
    )

    assert report.performance is performance


# =========================================================
# 2. INVALID PERFORMANCE
# =========================================================

def test_invalid_performance():

    with pytest.raises(TypeError):

        PaperTradingReport(
            "invalid"
        )


# =========================================================
# 3. PROFITABLE STATUS
# =========================================================

def test_profitable_status():

    report = PaperTradingReport(
        make_performance(
            [
                make_trade(10.0),
                make_trade(20.0),
            ]
        )
    )

    assert report.status == "PROFITABLE"


# =========================================================
# 4. LOSS STATUS
# =========================================================

def test_loss_status():

    report = PaperTradingReport(
        make_performance(
            [
                make_trade(-10.0),
                make_trade(-5.0),
            ]
        )
    )

    assert report.status == "LOSS"


# =========================================================
# 5. BREAK EVEN STATUS
# =========================================================

def test_break_even_status():

    report = PaperTradingReport(
        make_performance(
            [
                make_trade(10.0),
                make_trade(-10.0),
            ]
        )
    )

    assert report.status == "BREAK_EVEN"


# =========================================================
# 6. METRICS
# =========================================================

def test_metrics():

    report = PaperTradingReport(
        make_performance(
            [
                make_trade(10.0),
                make_trade(-5.0),
                make_trade(20.0),
            ]
        )
    )

    metrics = report.metrics

    assert metrics["total_trades"] == 3.0
    assert metrics["winning_trades"] == 2.0
    assert metrics["losing_trades"] == 1.0

    assert metrics["win_rate"] == 66.67
    assert metrics["total_profit"] == 25.0

    assert metrics["average_trade"] == (
        25.0 / 3.0
    )

    assert metrics["average_win"] == 15.0
    assert metrics["average_loss"] == -5.0

    assert metrics["profit_factor"] == 6.0
    assert metrics["cumulative_return"] == 2.5


# =========================================================
# 7. SUMMARY CONTENT
# =========================================================

def test_summary_contains_key_metrics():

    report = PaperTradingReport(
        make_performance(
            [
                make_trade(10.0),
                make_trade(-5.0),
                make_trade(20.0),
            ]
        )
    )

    summary = report.summary()

    assert "Paper Trading Report" in summary
    assert "Status: PROFITABLE" in summary
    assert "Total trades: 3" in summary
    assert "Winning trades: 2" in summary
    assert "Losing trades: 1" in summary
    assert "Win rate: 66.67%" in summary
    assert "Total profit: 25.00000000" in summary
    assert "Cumulative return: 2.50%" in summary


# =========================================================
# 8. GENERATE
# =========================================================

def test_generate_returns_report_result():

    report = PaperTradingReport(
        make_performance(
            [
                make_trade(10.0),
                make_trade(-5.0),
            ]
        )
    )

    result = report.generate()

    assert isinstance(
        result,
        PaperTradingReportResult,
    )

    assert result.performance == (
        report.performance.summarize()
    )

    assert result.status == report.status
    assert result.summary == report.summary()
    assert result.metrics == report.metrics


# =========================================================
# 9. GENERATE PERFORMANCE
# =========================================================

def test_generate_performance_values():

    report = PaperTradingReport(
        make_performance(
            [
                make_trade(10.0),
                make_trade(-5.0),
                make_trade(20.0),
            ]
        )
    )

    result = report.generate()

    assert result.performance.total_trades == 3
    assert result.performance.winning_trades == 2
    assert result.performance.losing_trades == 1
    assert result.performance.total_profit == 25.0


# =========================================================
# 10. TO TEXT
# =========================================================

def test_to_text():

    report = PaperTradingReport(
        make_performance(
            [
                make_trade(15.0),
            ]
        )
    )

    assert report.to_text() == report.summary()


# =========================================================
# 11. EMPTY PERFORMANCE
# =========================================================

def test_empty_performance():

    report = PaperTradingReport(
        make_performance([])
    )

    assert report.status == "BREAK_EVEN"

    assert report.metrics["total_trades"] == 0.0
    assert report.metrics["winning_trades"] == 0.0
    assert report.metrics["losing_trades"] == 0.0
    assert report.metrics["total_profit"] == 0.0
    assert report.metrics["win_rate"] == 0.0


# =========================================================
# 12. LOSS REPORT
# =========================================================

def test_loss_report():

    report = PaperTradingReport(
        make_performance(
            [
                make_trade(-10.0),
                make_trade(-20.0),
            ]
        )
    )

    result = report.generate()

    assert result.status == "LOSS"
    assert result.performance.total_profit == -30.0
    assert result.performance.winning_trades == 0
    assert result.performance.losing_trades == 2


# =========================================================
# 13. METRICS ARE NUMERIC
# =========================================================

def test_metrics_are_numeric():

    report = PaperTradingReport(
        make_performance(
            [
                make_trade(10.0),
            ]
        )
    )

    for value in report.metrics.values():

        assert isinstance(
            value,
            float,
        )


# =========================================================
# 14. SUMMARY IS STRING
# =========================================================

def test_summary_is_string():

    report = PaperTradingReport(
        make_performance(
            [
                make_trade(10.0),
            ]
        )
    )

    assert isinstance(
        report.summary(),
        str,
    )


# =========================================================
# 15. REPORT DOES NOT MODIFY PERFORMANCE
# =========================================================

def test_report_does_not_modify_performance():

    performance = make_performance(
        [
            make_trade(10.0),
            make_trade(-5.0),
        ]
    )

    before = performance.summarize()

    report = PaperTradingReport(
        performance
    )

    report.generate()
    report.to_text()

    after = performance.summarize()

    assert before == after
