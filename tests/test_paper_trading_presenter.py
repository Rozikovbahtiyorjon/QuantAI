import pytest
from src.paper_trading_presenter import PaperTradingPresenter
from src.paper_trading_session import PaperTradingSessionResult


def test_format_summary():
    result = PaperTradingSessionResult(
        steps=[],
        initial_balance=1000.0,
        final_balance=1050.0,
        realized_profit=50.0,
        total_steps=100,
        opened_positions=2,
        closed_positions=2,
    )
    summary = PaperTradingPresenter.format_summary(result)

    assert summary["initial_balance"] == 1000.0
    assert summary["final_balance"] == 1050.0
    assert summary["realized_profit"] == 50.0
    assert summary["roi_pct"] == 5.0
    assert summary["closed_positions"] == 2


def test_render_text_report():
    result = PaperTradingSessionResult(
        steps=[],
        initial_balance=1000.0,
        final_balance=1000.0,
        realized_profit=0.0,
        total_steps=10,
        opened_positions=0,
        closed_positions=0,
    )
    report = PaperTradingPresenter.render_text_report(result)
    assert "QUANTAI PAPER TRADING REPORT" in report
    assert "Initial Balance:  $1,000.00" in report


def test_invalid_type_raises():
    with pytest.raises(TypeError):
        PaperTradingPresenter.format_summary("invalid_type")