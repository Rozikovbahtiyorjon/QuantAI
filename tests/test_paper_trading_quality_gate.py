from __future__ import annotations

import pytest

from src.paper_trading_pipeline import (
    PaperTradingPipelineResult,
)
from src.paper_trading_session import (
    PaperTradingSessionResult,
)
from src.paper_trading_quality_gate import (
    PaperTradingQualityGate,
    PaperTradingQualityGateResult,
    evaluate_paper_trading_quality,
)


def make_result(
    initial_balance: float = 1000.0,
    final_balance: float = 1005.0,
    realized_profit: float = 5.0,
    total_steps: int = 2,
    opened_positions: int = 1,
    closed_positions: int = 1,
) -> PaperTradingPipelineResult:

    session = PaperTradingSessionResult(
        steps=[
            {
                "net_profit": 5.0,
            },
        ],
        initial_balance=initial_balance,
        final_balance=final_balance,
        realized_profit=realized_profit,
        total_steps=total_steps,
        opened_positions=opened_positions,
        closed_positions=closed_positions,
    )

    return PaperTradingPipelineResult(
        session_result=session,
        initial_balance=initial_balance,
        final_balance=final_balance,
        realized_profit=realized_profit,
        total_steps=total_steps,
        opened_positions=opened_positions,
        closed_positions=closed_positions,
        return_percent=0.5,
    )


def test_valid_quality_gate():

    result = PaperTradingQualityGate(
        initial_balance=1000.0,
        minimum_win_rate=0.0,
        minimum_profit_factor=0.0,
    ).evaluate(
        make_result()
    )

    assert isinstance(
        result,
        PaperTradingQualityGateResult,
    )

    assert result.passed is True

    assert result.validation.valid is True

    assert result.performance.total_trades == 1


def test_threshold_can_fail():

    result_data = make_result(
        final_balance=995.0,
        realized_profit=-5.0,
    )

    result_data.session_result.steps = [
        {
            "net_profit": -5.0,
        },
    ]

    result = PaperTradingQualityGate(
        initial_balance=1000.0,
        minimum_win_rate=100.0,
    ).evaluate(
        result_data
    )

    assert result.passed is False

    assert any(
        "Win rate"
        in error
        for error in result.errors
    )


def test_drawdown_threshold_can_fail():

    session = PaperTradingSessionResult(
        steps=[
            {
                "net_profit": 10.0,
            },
            {
                "net_profit": -50.0,
            },
        ],
        initial_balance=1000.0,
        final_balance=960.0,
        realized_profit=-40.0,
        total_steps=2,
        opened_positions=2,
        closed_positions=2,
    )

    pipeline = PaperTradingPipelineResult(
        session_result=session,
        initial_balance=1000.0,
        final_balance=960.0,
        realized_profit=-40.0,
        total_steps=2,
        opened_positions=2,
        closed_positions=2,
        return_percent=-4.0,
    )

    result = PaperTradingQualityGate(
        maximum_drawdown_percent=1.0,
    ).evaluate(
        pipeline
    )

    assert result.passed is False

    assert any(
        "drawdown"
        in error.lower()
        for error in result.errors
    )


def test_invalid_result_type():

    with pytest.raises(TypeError):

        PaperTradingQualityGate().evaluate(
            None
        )


def test_invalid_configuration():

    with pytest.raises(ValueError):

        PaperTradingQualityGate(
            initial_balance=0.0
        )

    with pytest.raises(ValueError):

        PaperTradingQualityGate(
            minimum_win_rate=101.0
        )

    with pytest.raises(ValueError):

        PaperTradingQualityGate(
            minimum_profit_factor=-1.0
        )

    with pytest.raises(ValueError):

        PaperTradingQualityGate(
            maximum_drawdown_percent=-1.0
        )


def test_convenience_function():

    result = evaluate_paper_trading_quality(
        make_result()
    )

    assert isinstance(
        result,
        PaperTradingQualityGateResult,
    )

    assert result.passed is True


def test_zero_trade_case_is_valid_but_warns():

    session = PaperTradingSessionResult(
        steps=[],
        initial_balance=1000.0,
        final_balance=1000.0,
        realized_profit=0.0,
        total_steps=0,
        opened_positions=0,
        closed_positions=0,
    )

    pipeline = PaperTradingPipelineResult(
        session_result=session,
        initial_balance=1000.0,
        final_balance=1000.0,
        realized_profit=0.0,
        total_steps=0,
        opened_positions=0,
        closed_positions=0,
        return_percent=0.0,
    )

    result = PaperTradingQualityGate().evaluate(
        pipeline
    )

    assert result.passed is True

    assert result.performance.total_trades == 0

    assert result.warnings


def test_validation_errors_are_propagated():

    pipeline = make_result(
        initial_balance=-100.0,
        final_balance=50.0,
        realized_profit=150.0,
    )

    result = PaperTradingQualityGate().evaluate(
        pipeline
    )

    assert result.passed is False

    assert any(
        "Initial balance"
        in error
        for error in result.errors
    )


def test_trade_dataframe_is_independent():

    result = make_result()

    gate = PaperTradingQualityGate()

    trades = gate._trades_from_result(
        result
    )

    trades.loc[
        0,
        "net_profit",
    ] = 999.0

    assert (
        result.session_result.steps[0]["net_profit"]
        == 5.0
    )