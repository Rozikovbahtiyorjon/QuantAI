from __future__ import annotations

import pandas as pd
import pytest

from src.paper_trading_market_session import (
    PaperTradingMarketSession,
    PaperTradingMarketSessionResult,
)
from src.strategy import SignalResult


def make_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [101.0, 102.0, 103.0, 104.0, 105.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [100.0, 101.0, 102.0, 103.0, 104.0],
            "volume": [
                1000.0,
                1100.0,
                1200.0,
                1300.0,
                1400.0,
            ],
        }
    )


def hold_signal(
    df: pd.DataFrame,
) -> SignalResult:
    return SignalResult(
        signal="HOLD",
        entry=float(
            df["close"].iloc[-1]
        ),
    )


def test_run_integrates_market_data_and_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.paper_trading_runner.generate_signal_result",
        hold_signal,
    )

    engine = PaperTradingMarketSession()

    result = engine.run(
        make_data()
    )

    assert isinstance(
        result,
        PaperTradingMarketSessionResult,
    )

    assert result.market_rows == 5

    assert result.session.total_steps == 5

    assert result.session.initial_balance == 1000.0
    assert result.session.final_balance == 1000.0
    assert result.session.realized_profit == 0.0

    assert result.session.opened_positions == 0
    assert result.session.closed_positions == 0


def test_market_data_is_processed_sequentially(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received_lengths: list[int] = []

    def tracking_signal(
        df: pd.DataFrame,
    ) -> SignalResult:
        received_lengths.append(
            len(df)
        )

        return SignalResult(
            signal="HOLD",
            entry=float(
                df["close"].iloc[-1]
            ),
        )

    monkeypatch.setattr(
        "src.paper_trading_runner.generate_signal_result",
        tracking_signal,
    )

    engine = PaperTradingMarketSession()

    result = engine.run(
        make_data()
    )

    assert received_lengths == [
        1,
        2,
        3,
        4,
        5,
    ]

    assert result.market_rows == 5
    assert result.session.total_steps == 5


def test_invalid_data_type() -> None:
    engine = PaperTradingMarketSession()

    with pytest.raises(TypeError):
        engine.run(
            [1, 2, 3]
        )


def test_empty_data() -> None:
    engine = PaperTradingMarketSession()

    with pytest.raises(ValueError):
        engine.run(
            pd.DataFrame()
        )


def test_result_state_properties(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.paper_trading_runner.generate_signal_result",
        hold_signal,
    )

    engine = PaperTradingMarketSession(
        initial_balance=2000.0,
        commission=0.0,
        quantity=1.0,
    )

    result = engine.run(
        make_data()
    )

    assert engine.result is result

    assert engine.balance == 2000.0
    assert engine.has_position is False
    assert engine.realized_profit == 0.0

    assert len(
        engine.steps
    ) == 5


def test_reset_clears_integration_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.paper_trading_runner.generate_signal_result",
        hold_signal,
    )

    engine = PaperTradingMarketSession()

    engine.run(
        make_data()
    )

    assert engine.result is not None

    assert len(
        engine.steps
    ) == 5

    engine.reset()

    assert engine.result is None

    assert engine.balance == 1000.0
    assert engine.has_position is False
    assert engine.realized_profit == 0.0

    assert engine.steps == []


def test_run_preserves_input_dataframe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.paper_trading_runner.generate_signal_result",
        hold_signal,
    )

    data = make_data()

    original = data.copy(
        deep=True
    )

    engine = PaperTradingMarketSession()

    engine.run(
        data
    )

    pd.testing.assert_frame_equal(
        data,
        original,
    )


def test_market_rows_match_input_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.paper_trading_runner.generate_signal_result",
        hold_signal,
    )

    data = make_data().iloc[:3].copy()

    engine = PaperTradingMarketSession()

    result = engine.run(
        data
    )

    assert result.market_rows == len(data)

    assert (
        result.session.total_steps
        == len(data)
    )