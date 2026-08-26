import pandas as pd
import pytest

from src.paper_trading_session import (
    PaperTradingSession,
)
from src.strategy import SignalResult


def create_market_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [1000.0, 1100.0, 1200.0],
        }
    )


def test_initial_state() -> None:
    session = PaperTradingSession(
        enable_risk_controls=False,
        initial_balance=1000.0,
        commission=0.0,
        quantity=1.0,
    )

    assert session.balance == 1000.0
    assert session.has_position is False
    assert session.realized_profit == 0.0
    assert session.steps == []
    assert session.market_data is None


def test_run_processes_all_market_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_signal(
        df: pd.DataFrame,
    ) -> SignalResult:
        return SignalResult(
            signal="HOLD",
            entry=float(df["close"].iloc[-1]),
        )

    monkeypatch.setattr(
        "src.paper_trading_session.generate_signal_result",
        fake_signal,
    )

    session = PaperTradingSession(
        enable_risk_controls=False,
        initial_balance=1000.0,
        commission=0.0,
        quantity=1.0,
    )

    result = session.run(
        create_market_data()
    )

    assert result.total_steps == 3
    assert len(result.steps) == 3
    assert result.opened_positions == 0
    assert result.closed_positions == 0

    assert session.market_data is not None
    assert session.market_data.position == 3
    assert session.market_data.finished is True


def test_session_uses_market_data_sequentially(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    windows: list[int] = []

    def fake_signal(
        df: pd.DataFrame,
    ) -> SignalResult:
        windows.append(len(df))

        return SignalResult(
            signal="HOLD",
            entry=float(df["close"].iloc[-1]),
        )

    monkeypatch.setattr(
        "src.paper_trading_session.generate_signal_result",
        fake_signal,
    )

    session = PaperTradingSession(
        enable_risk_controls=False,
        initial_balance=1000.0,
        commission=0.0,
        quantity=1.0,
    )

    session.run(
        create_market_data()
    )

    assert windows == [1, 2, 3]


def test_buy_signal_opens_position(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_signal(
        df: pd.DataFrame,
    ) -> SignalResult:
        return SignalResult(
            signal="BUY",
            entry=float(df["close"].iloc[-1]),
        )

    monkeypatch.setattr(
        "src.paper_trading_session.generate_signal_result",
        fake_signal,
    )

    session = PaperTradingSession(
        enable_risk_controls=False,
        initial_balance=1000.0,
        commission=0.0,
        quantity=1.0,
    )

    result = session.run(
        create_market_data()
    )

    assert result.total_steps == 3
    assert result.opened_positions == 1
    assert result.closed_positions == 0

    assert session.has_position is True
    assert session.runner.engine.position is not None
    assert session.runner.engine.position.side == "LONG"
    assert session.runner.engine.position.entry_price == 100.5


def test_result_property() -> None:
    session = PaperTradingSession(
        enable_risk_controls=False,
        initial_balance=1000.0,
        commission=0.0,
        quantity=1.0,
    )

    result = session.result

    assert result.total_steps == 0
    assert result.initial_balance == 1000.0
    assert result.final_balance == 1000.0
    assert result.realized_profit == 0.0
    assert result.opened_positions == 0
    assert result.closed_positions == 0


def test_steps_returns_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_signal(
        df: pd.DataFrame,
    ) -> SignalResult:
        return SignalResult(
            signal="HOLD",
            entry=float(df["close"].iloc[-1]),
        )

    monkeypatch.setattr(
        "src.paper_trading_session.generate_signal_result",
        fake_signal,
    )

    session = PaperTradingSession(
        enable_risk_controls=False,
        initial_balance=1000.0,
        commission=0.0,
        quantity=1.0,
    )

    session.run(
        create_market_data()
    )

    steps = session.steps
    steps.clear()

    assert len(session.steps) == 3


def test_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_signal(
        df: pd.DataFrame,
    ) -> SignalResult:
        return SignalResult(
            signal="BUY",
            entry=float(df["close"].iloc[-1]),
        )

    monkeypatch.setattr(
        "src.paper_trading_session.generate_signal_result",
        fake_signal,
    )

    session = PaperTradingSession(
        enable_risk_controls=False,
        initial_balance=1000.0,
        commission=0.0,
        quantity=1.0,
    )

    session.run(
        create_market_data()
    )

    assert session.has_position is True
    assert len(session.steps) == 3
    assert session.market_data is not None
    assert session.market_data.position == 3

    session.reset()

    assert session.balance == 1000.0
    assert session.has_position is False
    assert session.realized_profit == 0.0
    assert session.steps == []

    assert session.market_data is not None
    assert session.market_data.position == 0
    assert session.market_data.finished is False


def test_invalid_dataframe_type() -> None:
    session = PaperTradingSession(
        enable_risk_controls=False,)

    with pytest.raises(TypeError):
        session.run([1, 2, 3])


def test_empty_dataframe() -> None:
    session = PaperTradingSession(
        enable_risk_controls=False,)

    with pytest.raises(ValueError):
        session.run(
            pd.DataFrame()
        )