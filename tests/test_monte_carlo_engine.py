from __future__ import annotations

import pytest

from src.monte_carlo_engine import (
    MonteCarloEngine,
    MonteCarloResult,
)


def test_initialization() -> None:
    engine = MonteCarloEngine(
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


def test_default_initial_balance() -> None:
    engine = MonteCarloEngine()

    assert engine.initial_balance == 1000.0
    assert engine.simulations == 1000
    assert engine.seed == 42
    assert engine.drawdown_limit == 0.20


def test_invalid_initial_balance() -> None:
    with pytest.raises(ValueError):
        MonteCarloEngine(
            initial_balance=0.0
        )


def test_invalid_simulations() -> None:
    with pytest.raises(ValueError):
        MonteCarloEngine(
            simulations=0
        )


def test_invalid_drawdown_limit_negative() -> None:
    with pytest.raises(ValueError):
        MonteCarloEngine(
            drawdown_limit=-0.01
        )


def test_invalid_drawdown_limit_one() -> None:
    with pytest.raises(ValueError):
        MonteCarloEngine(
            drawdown_limit=1.0
        )


def test_run_returns_result() -> None:
    engine = MonteCarloEngine(
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
    )

    assert isinstance(
        result,
        MonteCarloResult,
    )

    assert result.simulations == 50
    assert result.periods == 3
    assert result.initial_balance == 1000.0


def test_run_uses_engine_initial_balance_by_default() -> None:
    engine = MonteCarloEngine(
        initial_balance=2000.0,
        simulations=10,
        seed=42,
    )

    result = engine.run(
        equity_curve=[
            1000.0,
            1010.0,
            1020.0,
        ],
    )

    assert result.initial_balance == 2000.0


def test_run_accepts_explicit_initial_balance() -> None:
    engine = MonteCarloEngine(
        initial_balance=1000.0,
        simulations=10,
        seed=42,
    )

    result = engine.run(
        equity_curve=[
            1000.0,
            1010.0,
            1020.0,
        ],
        initial_balance=1500.0,
    )

    assert result.initial_balance == 1500.0


def test_run_invalid_explicit_initial_balance() -> None:
    engine = MonteCarloEngine()

    with pytest.raises(ValueError):
        engine.run(
            equity_curve=[
                1000.0,
                1010.0,
            ],
            initial_balance=0.0,
        )


def test_distributions_have_expected_length() -> None:
    engine = MonteCarloEngine(
        initial_balance=1000.0,
        simulations=25,
        seed=42,
    )

    result = engine.run(
        equity_curve=[
            1000.0,
            1010.0,
            1005.0,
            1025.0,
        ],
    )

    assert len(
        result.final_balance_distribution
    ) == 25

    assert len(
        result.max_drawdown_distribution
    ) == 25


def test_probabilities_are_percentages() -> None:
    engine = MonteCarloEngine(
        simulations=100,
        seed=42,
    )

    result = engine.run(
        equity_curve=[
            1000.0,
            1010.0,
            1005.0,
            1025.0,
        ],
    )

    assert (
        0.0
        <= result.probability_of_profit
        <= 100.0
    )

    assert (
        0.0
        <= result.probability_of_loss
        <= 100.0
    )

    assert (
        0.0
        <= result.probability_of_drawdown_exceeding_limit
        <= 100.0
    )


def test_drawdown_values_are_non_negative() -> None:
    engine = MonteCarloEngine(
        simulations=50,
        seed=42,
    )

    result = engine.run(
        equity_curve=[
            1000.0,
            1100.0,
            1050.0,
            1200.0,
        ],
    )

    assert result.mean_max_drawdown >= 0.0
    assert result.median_max_drawdown >= 0.0
    assert result.percentile_95_max_drawdown >= 0.0

    assert all(
        value >= 0.0
        for value in result.max_drawdown_distribution
    )


def test_percentile_ordering() -> None:
    engine = MonteCarloEngine(
        simulations=100,
        seed=42,
    )

    result = engine.run(
        equity_curve=[
            1000.0,
            1010.0,
            990.0,
            1030.0,
        ],
    )

    assert (
        result.percentile_5_final_balance
        <= result.percentile_25_final_balance
        <= result.median_final_balance
        <= result.percentile_75_final_balance
        <= result.percentile_95_final_balance
    )


def test_reproducibility_with_same_seed() -> None:
    equity_curve = [
        1000.0,
        1010.0,
        1005.0,
        1025.0,
    ]

    first = MonteCarloEngine(
        initial_balance=1000.0,
        simulations=50,
        seed=42,
    ).run(
        equity_curve
    )

    second = MonteCarloEngine(
        initial_balance=1000.0,
        simulations=50,
        seed=42,
    ).run(
        equity_curve
    )

    assert (
        first.final_balance_distribution
        == second.final_balance_distribution
    )

    assert (
        first.max_drawdown_distribution
        == second.max_drawdown_distribution
    )


def test_result_property() -> None:
    engine = MonteCarloEngine(
        simulations=25,
        seed=42,
    )

    assert engine.result is None

    result = engine.run(
        equity_curve=[
            1000.0,
            1010.0,
            1020.0,
        ],
    )

    assert engine.result is result


def test_reset() -> None:
    engine = MonteCarloEngine(
        simulations=25,
        seed=42,
    )

    engine.run(
        equity_curve=[
            1000.0,
            1010.0,
            1020.0,
        ],
    )

    assert engine.result is not None

    engine.reset()

    assert engine.result is None


def test_empty_equity_curve() -> None:
    engine = MonteCarloEngine()

    with pytest.raises(ValueError):
        engine.run([])


def test_single_value_equity_curve() -> None:
    engine = MonteCarloEngine()

    with pytest.raises(ValueError):
        engine.run(
            [1000.0]
        )


def test_non_positive_equity_curve() -> None:
    engine = MonteCarloEngine()

    with pytest.raises(ValueError):
        engine.run(
            [
                1000.0,
                0.0,
                1010.0,
            ]
        )


def test_non_finite_equity_curve() -> None:
    engine = MonteCarloEngine()

    with pytest.raises(ValueError):
        engine.run(
            [
                1000.0,
                float("nan"),
                1010.0,
            ]
        )


def test_string_equity_curve() -> None:
    engine = MonteCarloEngine()

    with pytest.raises(TypeError):
        engine.run(
            "1000,1010"
        )


def test_non_numeric_equity_curve() -> None:
    engine = MonteCarloEngine()

    with pytest.raises(TypeError):
        engine.run(
            [
                1000.0,
                "invalid",
                1010.0,
            ]
        )


def test_engine_interface_is_compatible_with_portfolio_stress() -> None:
    engine = MonteCarloEngine(
        initial_balance=1000.0,
        simulations=50,
        seed=42,
        drawdown_limit=0.20,
    )

    result = engine.run(
        equity_curve=[
            1000.0,
            1010.0,
            1005.0,
            1025.0,
        ],
    )

    assert result.initial_balance == 1000.0
    assert result.simulations == 50

    assert len(
        result.final_balance_distribution
    ) == 50


def test_frozen_result() -> None:
    engine = MonteCarloEngine(
        simulations=10,
        seed=42,
    )

    result = engine.run(
        equity_curve=[
            1000.0,
            1010.0,
        ],
    )

    with pytest.raises(AttributeError):
        result.initial_balance = 2000.0