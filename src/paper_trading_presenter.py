"""
=========================================================
QuantAI Professional v5
Paper Trading Presenter

Formats paper trading session results for presentation, 
logging, and CLI summary outputs.
=========================================================
"""

from __future__ import annotations

from typing import Any, Dict
from src.paper_trading_session import PaperTradingSessionResult


class PaperTradingPresenter:
    """
    Formats PaperTradingSessionResult into structured 
    metrics dictionaries and human-readable text summaries.
    """

    @staticmethod
    def format_summary(result: PaperTradingSessionResult) -> Dict[str, Any]:
        if not isinstance(result, PaperTradingSessionResult):
            raise TypeError("result must be an instance of PaperTradingSessionResult.")

        roi_pct = 0.0
        if result.initial_balance > 0:
            roi_pct = (result.realized_profit / result.initial_balance) * 100.0

        return {
            "initial_balance": round(result.initial_balance, 2),
            "final_balance": round(result.final_balance, 2),
            "realized_profit": round(result.realized_profit, 2),
            "roi_pct": round(roi_pct, 2),
            "total_steps": result.total_steps,
            "opened_positions": result.opened_positions,
            "closed_positions": result.closed_positions,
        }

    @staticmethod
    def render_text_report(result: PaperTradingSessionResult) -> str:
        summary = PaperTradingPresenter.format_summary(result)
        lines = [
            "=========================================================",
            "             QUANTAI PAPER TRADING REPORT                ",
            "=========================================================",
            f" Initial Balance:  ${summary['initial_balance']:,.2f}",
            f" Final Balance:    ${summary['final_balance']:,.2f}",
            f" Realized Profit:  ${summary['realized_profit']:,.2f} ({summary['roi_pct']:.2f}%)",
            "---------------------------------------------------------",
            f" Total Steps:      {summary['total_steps']}",
            f" Opened Positions: {summary['opened_positions']}",
            f" Closed Positions: {summary['closed_positions']}",
            "=========================================================",
        ]
        return "\n".join(lines)


__all__ = ["PaperTradingPresenter"]