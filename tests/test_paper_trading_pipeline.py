"""
=========================================================
QuantAI Professional v5
Paper Trading Pipeline Tests
=========================================================
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.paper_trading_pipeline import (
    PaperTradingPipeline,
    PaperTradingPipelineResult,
)
from src.paper_trading_session import (
    PaperTradingSessionResult,
)


# =========================================================
# HELPERS
# =========================================================

def make_dataframe(rows: int = 5) -> pd.DataFrame:

    return pd.DataFrame(
        {
            "timestamp": range(rows),
            "open": [
                100.0 + i
                for i in range(rows)
            ],
            "high": [
                101.0 + i
                for i in range(rows)
            ],
            "low": [
                99.0 + i
                for i in range(rows)
            ],
            "close": [
                100.0 + i
                for i in range(rows)
            ],
            "volume": [1000.0] * rows,
        }
    )


def make_session_result(
    initial_balance: float = 1000.0,
    final_balance: float = 1050.0,
    realized_profit: float = 50.0,
    total_steps: int = 5,
    opened_positions: int = 2,
    closed_positions: int = 1,
) -> PaperTradingSessionResult:

    return PaperTradingSessionResult(
        steps=[],

        initial_balance=initial_balance,

        final_balance=final_balance,

        realized_profit=realized_profit,

        total_steps=total_steps,

        opened_positions=opened_positions,

        closed_positions=closed_positions,
    )


# =========================================================
# 1. INITIALIZATION
# =========================================================

def test_initialization():

    pipeline = PaperTradingPipeline()

    assert pipeline.result is None
    assert pipeline.balance == 1000.0
    assert pipeline.has_position is False
    assert pipeline.realized_profit == 0.0


# =========================================================
# 2. CUSTOM PARAMETERS
# =========================================================

def test_custom_parameters():

    pipeline = PaperTradingPipeline(
        initial_balance=5000.0,
        commission=0.001,
        quantity=2.0,
    )

    assert pipeline.balance == 5000.0

    assert (
        pipeline.session.runner.quantity
        == 2.0
    )

    assert (
        pipeline.session.runner.engine.commission
        == 0.001
    )


# =========================================================
# 3. INVALID INITIAL BALANCE
# =========================================================

def test_invalid_initial_balance():

    with pytest.raises(ValueError):

        PaperTradingPipeline(
            initial_balance=0
        )


# =========================================================
# 4. INVALID COMMISSION
# =========================================================

def test_invalid_commission():

    with pytest.raises(ValueError):

        PaperTradingPipeline(
            commission=-0.001
        )


# =========================================================
# 5. INVALID QUANTITY
# =========================================================

def test_invalid_quantity():

    with pytest.raises(ValueError):

        PaperTradingPipeline(
            quantity=0
        )


# =========================================================
# 6. INVALID DATA TYPE
# =========================================================

def test_run_invalid_data_type(
    monkeypatch,
):

    pipeline = PaperTradingPipeline()

    with pytest.raises(TypeError):

        pipeline.run(
            [1, 2, 3]
        )


# =========================================================
# 7. EMPTY DATAFRAME
# =========================================================

def test_run_empty_dataframe():

    pipeline = PaperTradingPipeline()

    with pytest.raises(ValueError):

        pipeline.run(
            pd.DataFrame()
        )


# =========================================================
# 8. RUN
# =========================================================

def test_run_returns_pipeline_result(
    monkeypatch,
):

    pipeline = PaperTradingPipeline()

    fake_result = make_session_result()

    def fake_run(df):

        return fake_result

    monkeypatch.setattr(
        pipeline.session,
        "run",
        fake_run,
    )

    result = pipeline.run(
        make_dataframe()
    )

    assert isinstance(
        result,
        PaperTradingPipelineResult,
    )

    assert result.session_result is fake_result


# =========================================================
# 9. RESULT IS STORED
# =========================================================

def test_result_is_stored(
    monkeypatch,
):

    pipeline = PaperTradingPipeline()

    fake_result = make_session_result(
        final_balance=1100.0,
        realized_profit=100.0,
    )

    monkeypatch.setattr(
        pipeline.session,
        "run",
        lambda df: fake_result,
    )

    result = pipeline.run(
        make_dataframe()
    )

    assert pipeline.result is result


# =========================================================
# 10. BALANCE METRICS
# =========================================================

def test_result_metrics(
    monkeypatch,
):

    pipeline = PaperTradingPipeline()

    fake_result = make_session_result(
        initial_balance=1000.0,
        final_balance=1075.0,
        realized_profit=75.0,
        total_steps=20,
        opened_positions=4,
        closed_positions=3,
    )

    monkeypatch.setattr(
        pipeline.session,
        "run",
        lambda df: fake_result,
    )

    result = pipeline.run(
        make_dataframe()
    )

    assert result.initial_balance == 1000.0
    assert result.final_balance == 1075.0
    assert result.realized_profit == 75.0
    assert result.total_steps == 20
    assert result.opened_positions == 4
    assert result.closed_positions == 3


# =========================================================
# 11. RETURN PERCENT
# =========================================================

def test_return_percent(
    monkeypatch,
):

    pipeline = PaperTradingPipeline()

    fake_result = make_session_result(
        initial_balance=1000.0,
        final_balance=1050.0,
    )

    monkeypatch.setattr(
        pipeline.session,
        "run",
        lambda df: fake_result,
    )

    result = pipeline.run(
        make_dataframe()
    )

    assert result.return_percent == 5.0


# =========================================================
# 12. NEGATIVE RETURN
# =========================================================

def test_negative_return(
    monkeypatch,
):

    pipeline = PaperTradingPipeline()

    fake_result = make_session_result(
        initial_balance=1000.0,
        final_balance=950.0,
    )

    monkeypatch.setattr(
        pipeline.session,
        "run",
        lambda df: fake_result,
    )

    result = pipeline.run(
        make_dataframe()
    )

    assert result.return_percent == -5.0


# =========================================================
# 13. SESSION IS CALLED
# =========================================================

def test_session_run_is_called(
    monkeypatch,
):

    pipeline = PaperTradingPipeline()

    calls = []

    fake_result = make_session_result()

    def fake_run(df):

        calls.append(df)

        return fake_result

    monkeypatch.setattr(
        pipeline.session,
        "run",
        fake_run,
    )

    df = make_dataframe()

    pipeline.run(df)

    assert len(calls) == 1
    assert calls[0] is df


# =========================================================
# 14. RESET
# =========================================================

def test_reset(
    monkeypatch,
):

    pipeline = PaperTradingPipeline()

    fake_result = make_session_result()

    monkeypatch.setattr(
        pipeline.session,
        "run",
        lambda df: fake_result,
    )

    pipeline.run(
        make_dataframe()
    )

    assert pipeline.result is not None

    pipeline.reset()

    assert pipeline.result is None


# =========================================================
# 15. RUN REPLACES PREVIOUS RESULT
# =========================================================

def test_run_replaces_previous_result(
    monkeypatch,
):

    pipeline = PaperTradingPipeline()

    first = make_session_result(
        final_balance=1050.0,
    )

    second = make_session_result(
        final_balance=1200.0,
    )

    results = [
        first,
        second,
    ]

    monkeypatch.setattr(
        pipeline.session,
        "run",
        lambda df: results.pop(0),
    )

    first_result = pipeline.run(
        make_dataframe()
    )

    second_result = pipeline.run(
        make_dataframe()
    )

    assert (
        first_result.final_balance
        == 1050.0
    )

    assert (
        second_result.final_balance
        == 1200.0
    )

    assert (
        pipeline.result
        is second_result
    )


# =========================================================
# 16. ZERO RETURN
# =========================================================

def test_zero_return(
    monkeypatch,
):

    pipeline = PaperTradingPipeline()

    fake_result = make_session_result(
        initial_balance=1000.0,
        final_balance=1000.0,
    )

    monkeypatch.setattr(
        pipeline.session,
        "run",
        lambda df: fake_result,
    )

    result = pipeline.run(
        make_dataframe()
    )

    assert result.return_percent == 0.0
