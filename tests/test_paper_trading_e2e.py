"""
=========================================================
QuantAI Professional v5
Paper Trading End-to-End Integration Test
=========================================================
"""

import pandas as pd
import pytest

from src.paper_trading_presenter import PaperTradingPresenter
from src.paper_trading_session import PaperTradingSession
from src.strategy import SignalResult


def create_sample_ohlcv_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [100.0, 102.0, 101.0, 105.0, 104.0],
            "high": [103.0, 104.0, 106.0, 107.0, 105.0],
            "low": [99.0, 101.0, 100.0, 103.0, 102.0],
            "close": [102.0, 101.0, 105.0, 104.0, 103.0],
            "volume": [1000.0, 1200.0, 1500.0, 1100.0, 900.0],
        }
    )


def test_e2e_paper_trading_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    # Имитация цепочки сигналов: HOLD -> BUY -> HOLD -> SELL -> HOLD
    signals = ["HOLD", "BUY", "HOLD", "SELL", "HOLD"]
    step_idx = 0

    def fake_signal(df: pd.DataFrame) -> SignalResult:
        nonlocal step_idx
        sig = signals[step_idx]
        step_idx += 1
        return SignalResult(signal=sig, entry=float(df["close"].iloc[-1]))

    monkeypatch.setattr(
        "src.paper_trading_session.generate_signal_result",
        fake_signal,
    )

    # 1. Инициализация сессии
    session = PaperTradingSession(
        enable_risk_controls=False,
        initial_balance=1000.0,
        commission=0.001,
        quantity=1.0,
    )

    # 2. Запуск торгового прогона
    df = create_sample_ohlcv_data()
    session_result = session.run(df)

    # 3. Форматирование результатов через Presenter
    summary = PaperTradingPresenter.format_summary(session_result)
    report = PaperTradingPresenter.render_text_report(session_result)

    # Проверка целостности сквозного потока
    # BUY открывает LONG (1), SELL закрывает LONG и открывает SHORT (2)
    assert session_result.total_steps == 5
    assert session_result.opened_positions == 2
    assert session_result.closed_positions == 1
    assert summary["total_steps"] == 5
    assert "QUANTAI PAPER TRADING REPORT" in report
    assert "Closed Positions: 1" in report