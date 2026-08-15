"""
QuantAI TradeEngine mathematical tests.
"""

from types import SimpleNamespace

import pytest

from config.settings import (
    COMMISSION,
    INITIAL_BALANCE,
    SLIPPAGE,
)

from src.trade_engine import (
    CloseReason,
    Position,
    PositionSide,
    PositionStatus,
    TradeEngine,
)


# ============================================================
# BASIC ENGINE
# ============================================================

def test_trade_engine_initial_balance():
    engine = TradeEngine()

    assert engine.balance == INITIAL_BALANCE
    assert engine.equity == INITIAL_BALANCE


# ============================================================
# COMMISSION
# ============================================================

def test_commission_calculation():
    engine = TradeEngine()

    quantity = 1.0
    price = 100000.0

    expected = quantity * price * COMMISSION

    assert engine.calculate_commission(
        quantity,
        price,
    ) == pytest.approx(expected)


# ============================================================
# SLIPPAGE
# ============================================================

def test_long_slippage():
    engine = TradeEngine()

    price = 100000.0

    result = engine.apply_slippage(
        PositionSide.LONG,
        price,
    )

    expected = price * (1 + SLIPPAGE)

    assert result == pytest.approx(expected)


def test_short_slippage():
    engine = TradeEngine()

    price = 100000.0

    result = engine.apply_slippage(
        PositionSide.SHORT,
        price,
    )

    expected = price * (1 - SLIPPAGE)

    assert result == pytest.approx(expected)


# ============================================================
# POSITION HELPERS
# ============================================================

def make_position(
    side: PositionSide,
    entry_price: float,
    quantity: float,
) -> Position:

    engine = TradeEngine()

    entry_commission = engine.calculate_commission(
        quantity,
        entry_price,
    )

    return Position(
        id=1,
        side=side,
        status=PositionStatus.OPEN,
        entry_time="2026-01-01",
        entry_price=entry_price,
        stop_loss=0.0,
        take_profit=0.0,
        quantity=quantity,
        commission=entry_commission,
    )


# ============================================================
# LONG PNL
# ============================================================

def test_long_profit_without_slippage():

    engine = TradeEngine()

    position = make_position(
        PositionSide.LONG,
        entry_price=100.0,
        quantity=1.0,
    )

    candle = {
        "timestamp": "2026-01-02",
        "high": 110.0,
        "low": 109.0,
        "close": 110.0,
        "atr": 1.0,
    }

    engine.close_position(
        position,
        candle,
        110.0,
        CloseReason.TAKE_PROFIT,
    )

    expected_exit = 110.0 * (1 + SLIPPAGE)

    expected_gross = (
        expected_exit - 100.0
    )

    expected_commission = (
        100.0 * COMMISSION
        + expected_exit * COMMISSION
    )

    expected_net = (
        expected_gross
        - expected_commission
    )

    assert position.gross_profit == pytest.approx(
        round(expected_gross, 2)
    )

    assert position.commission == pytest.approx(
        round(expected_commission, 4)
    )

    assert position.net_profit == pytest.approx(
        round(expected_net, 2)
    )


# ============================================================
# SHORT PNL
# ============================================================

def test_short_profit_without_slippage():

    engine = TradeEngine()

    position = make_position(
        PositionSide.SHORT,
        entry_price=100.0,
        quantity=1.0,
    )

    candle = {
        "timestamp": "2026-01-02",
        "high": 91.0,
        "low": 90.0,
        "close": 90.0,
        "atr": 1.0,
    }

    engine.close_position(
        position,
        candle,
        90.0,
        CloseReason.TAKE_PROFIT,
    )

    expected_exit = 90.0 * (1 - SLIPPAGE)

    expected_gross = (
        100.0 - expected_exit
    )

    expected_commission = (
        100.0 * COMMISSION
        + expected_exit * COMMISSION
    )

    expected_net = (
        expected_gross
        - expected_commission
    )

    assert position.gross_profit == pytest.approx(
        round(expected_gross, 2)
    )

    assert position.commission == pytest.approx(
        round(expected_commission, 4)
    )

    assert position.net_profit == pytest.approx(
        round(expected_net, 2)
    )


# ============================================================
# BALANCE UPDATE
# ============================================================

def test_balance_updates_after_close():

    engine = TradeEngine()

    starting_balance = engine.balance

    position = make_position(
        PositionSide.LONG,
        entry_price=100.0,
        quantity=1.0,
    )

    candle = {
        "timestamp": "2026-01-02",
        "high": 110.0,
        "low": 109.0,
        "close": 110.0,
        "atr": 1.0,
    }

    engine.close_position(
        position,
        candle,
        110.0,
        CloseReason.TAKE_PROFIT,
    )

    assert engine.balance == pytest.approx(
        starting_balance
        + position.net_profit
    )

    assert engine.equity == pytest.approx(
        engine.balance
    )


# ============================================================
# ARCHIVING
# ============================================================

def test_closed_position_is_archived():

    engine = TradeEngine()

    position = make_position(
        PositionSide.LONG,
        entry_price=100.0,
        quantity=1.0,
    )

    engine.positions.append(position)

    candle = {
        "timestamp": "2026-01-02",
        "high": 110.0,
        "low": 109.0,
        "close": 110.0,
        "atr": 1.0,
    }

    engine.close_position(
        position,
        candle,
        110.0,
        CloseReason.TAKE_PROFIT,
    )

    assert position in engine.closed_positions

    assert position not in engine.positions

    assert position.status == PositionStatus.CLOSED