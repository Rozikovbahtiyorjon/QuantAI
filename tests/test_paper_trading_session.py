"""
=========================================================
QuantAI Professional v5
Paper Trading Session Tests
=========================================================
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.paper_trading_session import (
    PaperTradingSession,
    PaperTradingSessionResult,
)
from src.paper_trading_runner import (
    PaperTradingStepResult,
)


# =========================================================
# HELPERS
# =========================================================

def make_dataframe(rows: int = 5) -> pd.DataFrame:

    return pd.DataFrame(
        {
            "timestamp": [
                f"2026-01-{i + 1:02d}"
                for i in range(rows)
            ],

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

            "volume": [
                1000.0
            ] * rows,
        }
    )


# =========================================================
# 1. INITIAL STATE
# =========================================================

def test_initial_state():

    session = PaperTradingSession(
        initial_balance=1000.0,
        commission=0.0004,
        quantity=1.0,
    )

    assert session.balance == 1000.0
    assert session.has_position is False
    assert session.realized_profit == 0.0
    assert session.steps == []


# =========================================================
# 2. INVALID DATA TYPE
# =========================================================

def test_run_requires_dataframe():

    session = PaperTradingSession()

    with pytest.raises(TypeError):

        session.run(
            [1, 2, 3]
        )


# =========================================================
# 3. EMPTY DATAFRAME
# =========================================================

def test_run_rejects_empty_dataframe():

    session = PaperTradingSession()

    with pytest.raises(ValueError):

        session.run(
            pd.DataFrame()
        )


# =========================================================
# 4. RUN RETURNS SESSION RESULT
# =========================================================

def test_run_returns_session_result(
    monkeypatch,
):

    session = PaperTradingSession()

    fake_steps = [
        PaperTradingStepResult(
            signal="HOLD",
            position_opened=False,
            position_closed=False,
            trade=None,
        ),

        PaperTradingStepResult(
            signal="HOLD",
            position_opened=False,
            position_closed=False,
            trade=None,
        ),
    ]

    monkeypatch.setattr(
        session.runner,
        "process_dataframe",
        lambda df: fake_steps,
    )

    result = session.run(
        make_dataframe(2)
    )

    assert isinstance(
        result,
        PaperTradingSessionResult,
    )

    assert result.total_steps == 2
    assert result.opened_positions == 0
    assert result.closed_positions == 0


# =========================================================
# 5. STEPS ARE STORED
# =========================================================

def test_steps_are_stored(
    monkeypatch,
):

    session = PaperTradingSession()

    fake_steps = [
        PaperTradingStepResult(
            signal="BUY",
            position_opened=True,
            position_closed=False,
            trade=None,
        ),
    ]

    monkeypatch.setattr(
        session.runner,
        "process_dataframe",
        lambda df: fake_steps,
    )

    session.run(
        make_dataframe(1)
    )

    assert len(
        session.steps
    ) == 1

    assert (
        session.steps[0].position_opened
        is True
    )

    assert (
        session.steps[0].signal
        == "BUY"
    )


# =========================================================
# 6. OPENED POSITION COUNT
# =========================================================

def test_opened_position_count(
    monkeypatch,
):

    session = PaperTradingSession()

    fake_steps = [
        PaperTradingStepResult(
            signal="BUY",
            position_opened=True,
            position_closed=False,
            trade=None,
        ),

        PaperTradingStepResult(
            signal="HOLD",
            position_opened=False,
            position_closed=False,
            trade=None,
        ),

        PaperTradingStepResult(
            signal="BUY",
            position_opened=True,
            position_closed=False,
            trade=None,
        ),
    ]

    monkeypatch.setattr(
        session.runner,
        "process_dataframe",
        lambda df: fake_steps,
    )

    result = session.run(
        make_dataframe(3)
    )

    assert result.opened_positions == 2


# =========================================================
# 7. CLOSED POSITION COUNT
# =========================================================

def test_closed_position_count(
    monkeypatch,
):

    session = PaperTradingSession()

    fake_steps = [
        PaperTradingStepResult(
            signal="SELL",
            position_opened=False,
            position_closed=True,
            trade=None,
        ),

        PaperTradingStepResult(
            signal="HOLD",
            position_opened=False,
            position_closed=False,
            trade=None,
        ),

        PaperTradingStepResult(
            signal="SELL",
            position_opened=False,
            position_closed=True,
            trade=None,
        ),
    ]

    monkeypatch.setattr(
        session.runner,
        "process_dataframe",
        lambda df: fake_steps,
    )

    result = session.run(
        make_dataframe(3)
    )

    assert result.closed_positions == 2


# =========================================================
# 8. RESULT PROPERTY
# =========================================================

def test_result_property():

    session = PaperTradingSession()

    result = session.result

    assert isinstance(
        result,
        PaperTradingSessionResult,
    )

    assert result.total_steps == 0
    assert result.opened_positions == 0
    assert result.closed_positions == 0

    assert result.initial_balance == 1000.0
    assert result.final_balance == 1000.0
    assert result.realized_profit == 0.0


# =========================================================
# 9. RESET
# =========================================================

def test_reset(
    monkeypatch,
):

    session = PaperTradingSession()

    fake_steps = [
        PaperTradingStepResult(
            signal="BUY",
            position_opened=True,
            position_closed=False,
            trade=None,
        ),
    ]

    monkeypatch.setattr(
        session.runner,
        "process_dataframe",
        lambda df: fake_steps,
    )

    session.run(
        make_dataframe(1)
    )

    assert len(
        session.steps
    ) == 1

    session.reset()

    assert session.steps == []
    assert session.balance == 1000.0
    assert session.has_position is False
    assert session.realized_profit == 0.0


# =========================================================
# 10. RUN CAN BE CALLED AGAIN
# =========================================================

def test_run_replaces_previous_steps(
    monkeypatch,
):

    session = PaperTradingSession()

    first_steps = [
        PaperTradingStepResult(
            signal="HOLD",
            position_opened=False,
            position_closed=False,
            trade=None,
        ),

        PaperTradingStepResult(
            signal="HOLD",
            position_opened=False,
            position_closed=False,
            trade=None,
        ),
    ]

    second_steps = [
        PaperTradingStepResult(
            signal="BUY",
            position_opened=True,
            position_closed=False,
            trade=None,
        ),
    ]

    monkeypatch.setattr(
        session.runner,
        "process_dataframe",
        lambda df: first_steps,
    )

    session.run(
        make_dataframe(2)
    )

    assert len(
        session.steps
    ) == 2

    monkeypatch.setattr(
        session.runner,
        "process_dataframe",
        lambda df: second_steps,
    )

    session.run(
        make_dataframe(1)
    )

    assert len(
        session.steps
    ) == 1

    assert (
        session.steps[0].position_opened
        is True
    )

    assert (
        session.steps[0].signal
        == "BUY"
    )


# =========================================================
# 11. CUSTOM INITIAL BALANCE
# =========================================================

def test_custom_initial_balance():

    session = PaperTradingSession(
        initial_balance=2500.0,
    )

    assert session.balance == 2500.0

    result = session.result

    assert result.initial_balance == 2500.0
    assert result.final_balance == 2500.0


# =========================================================
# 12. CUSTOM QUANTITY
# =========================================================

def test_custom_quantity():

    session = PaperTradingSession(
        quantity=2.5,
    )

    assert (
        session.runner.engine.commission
        == 0.0004
    )

    assert (
        session.runner.quantity
        == 2.5
    )
