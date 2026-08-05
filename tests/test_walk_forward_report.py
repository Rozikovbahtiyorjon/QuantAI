"""
=========================================================
QuantAI WalkForwardReport Tests
=========================================================
"""

from __future__ import annotations

from src.walk_forward_analyzer import (
    WalkForwardSummary,
)

from src.walk_forward_report import (
    WalkForwardReport,
    create_walk_forward_report,
)


# =========================================================
# HELPER
# =========================================================

def make_summary():

    return WalkForwardSummary(
        total_windows=4,
        total_trades=10,
        winning_trades=6,
        losing_trades=4,
        total_profit=75.50,
        initial_balance=1000.0,
        final_balance=1075.50,
        profitable_windows=3,
        losing_windows=1,
        win_rate=60.0,
        cumulative_return=7.55,
        max_drawdown=25.0,
        max_drawdown_percent=2.5,
    )


# =========================================================
# 1. CREATION
# =========================================================

def test_report_creation():

    summary = make_summary()

    report = WalkForwardReport(
        summary=summary
    )

    assert report.summary is summary


# =========================================================
# 2. TEXT
# =========================================================

def test_to_text_contains_main_metrics():

    summary = make_summary()

    report = WalkForwardReport(
        summary=summary
    )

    text = report.to_text()

    assert "QUANTAI WALK-FORWARD REPORT" in text

    assert "Total Windows" in text
    assert "Total Trades" in text
    assert "Winning Trades" in text
    assert "Losing Trades" in text
    assert "Win Rate" in text

    assert "Initial Balance" in text
    assert "Final Balance" in text
    assert "Total Profit" in text

    assert "Cumulative Return" in text
    assert "Max Drawdown" in text
    assert "Max Drawdown %" in text


# =========================================================
# 3. VALUES
# =========================================================

def test_to_text_contains_values():

    summary = make_summary()

    report = WalkForwardReport(
        summary=summary
    )

    text = report.to_text()

    assert "4" in text
    assert "10" in text
    assert "75.50" in text
    assert "1000.00" in text
    assert "1075.50" in text
    assert "7.55%" in text
    assert "2.50%" in text


# =========================================================
# 4. PRINT
# =========================================================

def test_print_report(capsys):

    summary = make_summary()

    report = WalkForwardReport(
        summary=summary
    )

    report.print_report()

    captured = capsys.readouterr()

    assert "QUANTAI WALK-FORWARD REPORT" in captured.out
    assert "1075.50" in captured.out


# =========================================================
# 5. CONVENIENCE FUNCTION
# =========================================================

def test_create_walk_forward_report():

    summary = make_summary()

    report = create_walk_forward_report(
        summary
    )

    assert isinstance(
        report,
        WalkForwardReport,
    )

    assert report.summary is summary


# =========================================================
# 6. EMPTY SUMMARY
# =========================================================

def test_empty_summary_report():

    summary = WalkForwardSummary(
        total_windows=0,
        total_trades=0,
        winning_trades=0,
        losing_trades=0,
        total_profit=0.0,
        initial_balance=0.0,
        final_balance=0.0,
        profitable_windows=0,
        losing_windows=0,
        win_rate=0.0,
        cumulative_return=0.0,
        max_drawdown=0.0,
        max_drawdown_percent=0.0,
    )

    report = WalkForwardReport(
        summary=summary
    )

    text = report.to_text()

    assert "Total Windows       : 0" in text
    assert "Total Profit        : 0.00" in text
    assert "Cumulative Return   : 0.00%" in text
    assert "Max Drawdown        : 0.00" in text


# =========================================================
# 7. SUMMARY REFERENCE
# =========================================================

def test_report_keeps_summary_reference():

    summary = make_summary()

    report = create_walk_forward_report(
        summary
    )

    assert report.summary.total_windows == 4
    assert report.summary.total_trades == 10
    assert report.summary.total_profit == 75.50


# =========================================================
# 8. REPORT FORMAT
# =========================================================

def test_report_format():

    summary = make_summary()

    report = WalkForwardReport(
        summary=summary
    )

    text = report.to_text()

    assert text.startswith(
        "\n============================================================"
    )

    assert text.endswith(
        "============================================================"
    )