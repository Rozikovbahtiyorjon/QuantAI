from __future__ import annotations

import pytest

from experimental.src.execution_simulator import (
    ExecutionResult,
    ExecutionSimulator,
)


def test_buy_execution_without_slippage() -> None:
    simulator = ExecutionSimulator(
        commission=0.001,
        slippage=0.0,
    )

    result = simulator.execute(
        side="BUY",
        price=100.0,
        quantity=2.0,
    )

    assert isinstance(result, ExecutionResult)
    assert result.side == "BUY"
    assert result.requested_price == 100.0
    assert result.execution_price == 100.0
    assert result.quantity == 2.0
    assert result.notional == 200.0
    assert result.commission == 0.2
    assert result.slippage == 0.0


def test_buy_execution_with_slippage() -> None:
    simulator = ExecutionSimulator(
        commission=0.001,
        slippage=0.01,
    )

    result = simulator.execute(
        side="BUY",
        price=100.0,
        quantity=2.0,
    )

    assert result.execution_price == pytest.approx(
        101.0
    )
    assert result.slippage == pytest.approx(
        1.0
    )
    assert result.notional == pytest.approx(
        202.0
    )
    assert result.commission == pytest.approx(
        0.202
    )


def test_sell_execution_with_slippage() -> None:
    simulator = ExecutionSimulator(
        commission=0.001,
        slippage=0.01,
    )

    result = simulator.execute(
        side="SELL",
        price=100.0,
        quantity=2.0,
    )

    assert result.execution_price == pytest.approx(
        99.0
    )
    assert result.slippage == pytest.approx(
        1.0
    )
    assert result.notional == pytest.approx(
        198.0
    )
    assert result.commission == pytest.approx(
        0.198
    )


def test_open_and_close_long() -> None:
    simulator = ExecutionSimulator(
        commission=0.001,
        slippage=0.0,
    )

    entry = simulator.open_long(
        price=100.0,
        quantity=2.0,
    )

    exit = simulator.close_long(
        price=110.0,
        quantity=2.0,
    )

    pnl = simulator.calculate_long_pnl(
        entry,
        exit,
    )

    assert entry.side == "BUY"
    assert exit.side == "SELL"
    assert pnl == pytest.approx(
        19.58
    )


def test_open_and_close_short() -> None:
    simulator = ExecutionSimulator(
        commission=0.001,
        slippage=0.0,
    )

    entry = simulator.open_short(
        price=110.0,
        quantity=2.0,
    )

    exit = simulator.close_short(
        price=100.0,
        quantity=2.0,
    )

    pnl = simulator.calculate_short_pnl(
        entry,
        exit,
    )

    assert entry.side == "SELL"
    assert exit.side == "BUY"
    assert pnl == pytest.approx(
        19.58
    )


def test_long_loss() -> None:
    simulator = ExecutionSimulator(
        commission=0.0,
        slippage=0.0,
    )

    entry = simulator.open_long(
        price=100.0,
        quantity=1.0,
    )

    exit = simulator.close_long(
        price=90.0,
        quantity=1.0,
    )

    pnl = simulator.calculate_long_pnl(
        entry,
        exit,
    )

    assert pnl == pytest.approx(
        -10.0
    )


def test_short_loss() -> None:
    simulator = ExecutionSimulator(
        commission=0.0,
        slippage=0.0,
    )

    entry = simulator.open_short(
        price=100.0,
        quantity=1.0,
    )

    exit = simulator.close_short(
        price=110.0,
        quantity=1.0,
    )

    pnl = simulator.calculate_short_pnl(
        entry,
        exit,
    )

    assert pnl == pytest.approx(
        -10.0
    )


@pytest.mark.parametrize(
    "side",
    ["INVALID", "", None],
)
def test_invalid_side(
    side: str,
) -> None:
    simulator = ExecutionSimulator()

    with pytest.raises(ValueError):
        simulator.execute(
            side=side,
            price=100.0,
            quantity=1.0,
        )


@pytest.mark.parametrize(
    "price",
    [0.0, -1.0],
)
def test_invalid_price(
    price: float,
) -> None:
    simulator = ExecutionSimulator()

    with pytest.raises(ValueError):
        simulator.execute(
            side="BUY",
            price=price,
            quantity=1.0,
        )


@pytest.mark.parametrize(
    "quantity",
    [0.0, -1.0],
)
def test_invalid_quantity(
    quantity: float,
) -> None:
    simulator = ExecutionSimulator()

    with pytest.raises(ValueError):
        simulator.execute(
            side="BUY",
            price=100.0,
            quantity=quantity,
        )


def test_negative_commission_rejected() -> None:
    with pytest.raises(ValueError):
        ExecutionSimulator(
            commission=-0.001
        )


def test_negative_slippage_rejected() -> None:
    with pytest.raises(ValueError):
        ExecutionSimulator(
            slippage=-0.001
        )


def test_long_quantity_mismatch_rejected() -> None:
    simulator = ExecutionSimulator()

    entry = simulator.open_long(
        price=100.0,
        quantity=1.0,
    )

    exit = simulator.close_long(
        price=110.0,
        quantity=2.0,
    )

    with pytest.raises(ValueError):
        simulator.calculate_long_pnl(
            entry,
            exit,
        )


def test_short_quantity_mismatch_rejected() -> None:
    simulator = ExecutionSimulator()

    entry = simulator.open_short(
        price=100.0,
        quantity=1.0,
    )

    exit = simulator.close_short(
        price=90.0,
        quantity=2.0,
    )

    with pytest.raises(ValueError):
        simulator.calculate_short_pnl(
            entry,
            exit,
        )


def test_wrong_long_execution_sides_rejected() -> None:
    simulator = ExecutionSimulator()

    entry = simulator.open_short(
        price=100.0,
        quantity=1.0,
    )

    exit = simulator.close_long(
        price=110.0,
        quantity=1.0,
    )

    with pytest.raises(ValueError):
        simulator.calculate_long_pnl(
            entry,
            exit,
        )


def test_wrong_short_execution_sides_rejected() -> None:
    simulator = ExecutionSimulator()

    entry = simulator.open_long(
        price=100.0,
        quantity=1.0,
    )

    exit = simulator.close_short(
        price=90.0,
        quantity=1.0,
    )

    with pytest.raises(ValueError):
        simulator.calculate_short_pnl(
            entry,
            exit,
        )


def test_execution_is_deterministic() -> None:
    simulator = ExecutionSimulator(
        commission=0.0004,
        slippage=0.002,
    )

    first = simulator.execute(
        side="BUY",
        price=123.45,
        quantity=3.0,
    )

    second = simulator.execute(
        side="BUY",
        price=123.45,
        quantity=3.0,
    )

    assert first == second