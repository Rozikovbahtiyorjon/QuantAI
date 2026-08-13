from __future__ import annotations

import pytest

from src.stress_test_engine import (
    StressTestEngine,
    StressTestResult,
)


def test_baseline_scenario_without_costs() -> None:
    engine = StressTestEngine(
        initial_balance=1000.0,
    )

    result = engine.run(
        [10.0, -5.0, 20.0],
        scenario="BASELINE",
    )

    assert isinstance(
        result,
        StressTestResult,
    )

    assert result.initial_balance == 1000.0
    assert result.final_balance == 1025.0
    assert result.total_profit == 25.0
    assert result.total_return == 2.5
    assert result.total_trades == 3
    assert result.winning_trades == 2
    assert result.losing_trades == 1

    assert result.win_rate == pytest.approx(
        66.66666667
    )

    assert result.max_drawdown == 5.0
    assert result.scenario == "BASELINE"


def test_slippage_and_commission_reduce_result() -> None:
    engine = StressTestEngine(
        initial_balance=1000.0,
        slippage_multiplier=1.0,
        commission_multiplier=1.0,
    )

    result = engine.run(
        [10.0, 20.0],
        slippage_cost_per_trade=1.0,
        commission_cost_per_trade=0.5,
        scenario="STRESS",
    )

    assert result.final_balance == 1027.0
    assert result.total_profit == 27.0
    assert result.total_trades == 2


def test_multipliers_are_applied() -> None:
    engine = StressTestEngine(
        initial_balance=1000.0,
        slippage_multiplier=2.0,
        commission_multiplier=3.0,
    )

    result = engine.run(
        [10.0],
        slippage_cost_per_trade=1.0,
        commission_cost_per_trade=1.0,
    )

    assert result.final_balance == 1005.0


def test_custom_scenario_multipliers() -> None:
    engine = StressTestEngine(
        initial_balance=1000.0,
    )

    result = engine.run_scenario(
        [10.0],
        slippage_cost_per_trade=1.0,
        commission_cost_per_trade=1.0,
        slippage_multiplier=3.0,
        commission_multiplier=2.0,
        scenario="EXTREME",
    )

    assert result.final_balance == 1005.0
    assert result.scenario == "EXTREME"


def test_drawdown_is_calculated_from_equity_curve() -> None:
    engine = StressTestEngine(
        initial_balance=1000.0,
    )

    result = engine.run(
        [100.0, -50.0, -25.0]
    )

    assert result.final_balance == 1025.0
    assert result.max_drawdown == 75.0

    assert result.max_drawdown_percent == pytest.approx(
        6.81818182
    )


def test_all_losing_trades() -> None:
    engine = StressTestEngine(
        initial_balance=1000.0,
    )

    result = engine.run(
        [-10.0, -20.0, -5.0]
    )

    assert result.final_balance == 965.0
    assert result.winning_trades == 0
    assert result.losing_trades == 3
    assert result.win_rate == 0.0


def test_zero_pnl_trade_is_not_win_or_loss() -> None:
    engine = StressTestEngine()

    result = engine.run([0.0])

    assert result.total_trades == 1
    assert result.winning_trades == 0
    assert result.losing_trades == 0
    assert result.win_rate == 0.0


def test_scenario_name_is_trimmed() -> None:
    engine = StressTestEngine()

    result = engine.run(
        [1.0],
        scenario="  TEST  ",
    )

    assert result.scenario == "TEST"


@pytest.mark.parametrize(
    "value",
    [0.0, -1.0],
)
def test_invalid_initial_balance(
    value: float,
) -> None:
    with pytest.raises(ValueError):
        StressTestEngine(
            initial_balance=value,
        )


@pytest.mark.parametrize(
    "value",
    [-0.1, -1.0],
)
def test_negative_slippage_multiplier_rejected(
    value: float,
) -> None:
    with pytest.raises(ValueError):
        StressTestEngine(
            slippage_multiplier=value,
        )


def test_negative_commission_multiplier_rejected() -> None:
    with pytest.raises(ValueError):
        StressTestEngine(
            commission_multiplier=-0.1,
        )


def test_empty_trades_rejected() -> None:
    engine = StressTestEngine()

    with pytest.raises(ValueError):
        engine.run([])


def test_invalid_trade_value_rejected() -> None:
    engine = StressTestEngine()

    with pytest.raises(ValueError):
        engine.run(
            [10.0, float("nan")]
        )


def test_invalid_cost_rejected() -> None:
    engine = StressTestEngine()

    with pytest.raises(ValueError):
        engine.run(
            [10.0],
            slippage_cost_per_trade=-1.0,
        )


def test_empty_scenario_name_rejected() -> None:
    engine = StressTestEngine()

    with pytest.raises(ValueError):
        engine.run(
            [1.0],
            scenario="   ",
        )


def test_compare_scenarios() -> None:
    engine = StressTestEngine(
        initial_balance=1000.0,
    )

    results = engine.compare(
        [10.0, 10.0],
        [
            {
                "scenario": "NORMAL",
                "slippage_cost_per_trade": 0.0,
                "commission_cost_per_trade": 0.0,
            },
            {
                "scenario": "STRESS",
                "slippage_cost_per_trade": 1.0,
                "commission_cost_per_trade": 0.5,
            },
        ],
    )

    assert len(results) == 2
    assert results[0].final_balance == 1020.0
    assert results[1].final_balance == 1017.0
    assert results[0].scenario == "NORMAL"
    assert results[1].scenario == "STRESS"


def test_compare_empty_scenarios_rejected() -> None:
    engine = StressTestEngine()

    with pytest.raises(ValueError):
        engine.compare(
            [1.0],
            [],
        )


def test_compare_non_dict_scenario_rejected() -> None:
    engine = StressTestEngine()

    with pytest.raises(TypeError):
        engine.compare(
            [1.0],
            ["invalid"],  # type: ignore[list-item]
        )


def test_deterministic_results() -> None:
    engine = StressTestEngine(
        initial_balance=1000.0,
        slippage_multiplier=1.5,
        commission_multiplier=2.0,
    )

    first = engine.run(
        [12.5, -3.5, 8.0],
        slippage_cost_per_trade=0.25,
        commission_cost_per_trade=0.1,
        scenario="DETERMINISTIC",
    )

    second = engine.run(
        [12.5, -3.5, 8.0],
        slippage_cost_per_trade=0.25,
        commission_cost_per_trade=0.1,
        scenario="DETERMINISTIC",
    )

    assert first == second