from __future__ import annotations

import pytest

from experimental.src.portfolio_stress_monte_carlo import (
    PortfolioStressMonteCarloEngine,
    PortfolioStressMonteCarloResult,
)


def test_initialization() -> None:
    engine = PortfolioStressMonteCarloEngine(
        initial_balance=1000.0,
        simulations=100,
        seed=42,
        drawdown_limit=0.20,
    )

    assert engine.initial_balance == 1000.0
    assert engine.simulations == 100
    assert engine.seed == 42
    assert engine.drawdown_limit == 0.20
    assert engine.result is None


def test_invalid_initial_balance() -> None:
    with pytest.raises(ValueError):
        PortfolioStressMonteCarloEngine(
            initial_balance=0.0,
        )


def test_invalid_simulations() -> None:
    with pytest.raises(ValueError):
        PortfolioStressMonteCarloEngine(
            simulations=0,
        )


def test_negative_drawdown_limit_rejected() -> None:
    with pytest.raises(ValueError):
        PortfolioStressMonteCarloEngine(
            drawdown_limit=-0.01,
        )


def test_empty_equity_curve_rejected() -> None:
    engine = PortfolioStressMonteCarloEngine(
        simulations=20,
        seed=42,
    )

    with pytest.raises(ValueError):
        engine.run(
            equity_curve=[],
            trade_pnls=[10.0],
        )


def test_empty_trade_pnls_rejected() -> None:
    engine = PortfolioStressMonteCarloEngine(
        simulations=20,
        seed=42,
    )

    with pytest.raises(ValueError):
        engine.run(
            equity_curve=[1000.0, 1010.0],
            trade_pnls=[],
        )


def test_invalid_equity_curve_values_rejected() -> None:
    engine = PortfolioStressMonteCarloEngine(
        simulations=20,
        seed=42,
    )

    with pytest.raises(ValueError):
        engine.run(
            equity_curve=[1000.0, 0.0, 1010.0],
            trade_pnls=[10.0],
        )


def test_run_returns_integrated_result() -> None:
    engine = PortfolioStressMonteCarloEngine(
        initial_balance=1000.0,
        simulations=50,
        seed=42,
    )

    result = engine.run(
        equity_curve=[
            1000.0,
            1010.0,
            1005.0,
            1025.0,
        ],
        trade_pnls=[
            10.0,
            -5.0,
            20.0,
        ],
    )

    assert isinstance(
        result,
        PortfolioStressMonteCarloResult,
    )

    assert result.monte_carlo is not None
    assert result.stress_test is not None

    assert result.monte_carlo.simulations == 50
    assert result.monte_carlo.periods == 3

    assert result.stress_test.initial_balance == 1000.0
    assert result.stress_test.final_balance == 1025.0


def test_result_property() -> None:
    engine = PortfolioStressMonteCarloEngine(
        initial_balance=1000.0,
        simulations=25,
        seed=42,
    )

    result = engine.run(
        equity_curve=[
            1000.0,
            1010.0,
            1020.0,
        ],
        trade_pnls=[
            10.0,
            20.0,
        ],
    )

    assert engine.result is result


def test_reset() -> None:
    engine = PortfolioStressMonteCarloEngine(
        initial_balance=1000.0,
        simulations=25,
        seed=42,
    )

    engine.run(
        equity_curve=[
            1000.0,
            1010.0,
            1020.0,
        ],
        trade_pnls=[
            10.0,
            20.0,
        ],
    )

    assert engine.result is not None

    engine.reset()

    assert engine.result is None


def test_custom_stress_parameters() -> None:
    engine = PortfolioStressMonteCarloEngine(
        initial_balance=1000.0,
        simulations=25,
        seed=42,
    )

    result = engine.run(
        equity_curve=[
            1000.0,
            1010.0,
        ],
        trade_pnls=[
            10.0,
        ],
        scenario="EXTREME",
        slippage_cost_per_trade=1.0,
        commission_cost_per_trade=1.0,
        slippage_multiplier=3.0,
        commission_multiplier=2.0,
    )

    assert result.stress_test.scenario == "EXTREME"
    assert result.stress_test.final_balance == 1003.0