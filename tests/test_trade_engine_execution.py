"""
====================================================
QuantAI Professional v5
Trade Engine Execution Tests
====================================================

Tests:
    - LONG position opening
    - SHORT position opening
    - execution price with slippage
    - position size calculation
    - SL / TP assignment
    - commission on entry
    - maximum open positions
    - position state after opening
    - balance / equity consistency
"""

from __future__ import annotations

import pytest

from config.settings import (
    INITIAL_BALANCE,
    COMMISSION,
    SLIPPAGE,
    RISK_PERCENT,
    MAX_OPEN_POSITIONS,
)

from src.trade_engine import (
    TradeEngine,
    PositionSide,
    PositionStatus,
)

from src.risk_manager import (
    calculate_position_size,
    calculate_sl_tp,
)


# ============================================================
# HELPERS
# ============================================================

def make_candle(
    timestamp="2026-01-01 00:00:00",
    open_price=100.0,
    high=105.0,
    low=95.0,
    close=100.0,
    atr=2.0,
):
    """Create a minimal valid candle."""

    return {
        "timestamp": timestamp,
        "open": float(open_price),
        "high": float(high),
        "low": float(low),
        "close": float(close),
        "volume": 1000.0,
        "atr": float(atr),
    }


def make_engine():
    """Create a fresh TradeEngine."""

    return TradeEngine()


# ============================================================
# BASIC CONFIGURATION
# ============================================================

def test_execution_configuration_is_valid():
    """Execution-related configuration must be valid."""

    assert INITIAL_BALANCE > 0
    assert COMMISSION >= 0
    assert SLIPPAGE >= 0
    assert RISK_PERCENT > 0
    assert MAX_OPEN_POSITIONS > 0


# ============================================================
# LONG EXECUTION PRICE
# ============================================================

def test_long_execution_price_includes_slippage():
    """LONG entry price must include upward slippage."""

    engine = make_engine()

    market_price = 100.0

    execution_price = engine.apply_slippage(
        PositionSide.LONG,
        market_price,
    )

    expected_price = (
        market_price
        * (1.0 + SLIPPAGE)
    )

    assert execution_price == pytest.approx(
        expected_price,
        rel=1e-10,
    )

    assert execution_price > market_price


# ============================================================
# SHORT EXECUTION PRICE
# ============================================================

def test_short_execution_price_includes_slippage():
    """SHORT entry price must include downward slippage."""

    engine = make_engine()

    market_price = 100.0

    execution_price = engine.apply_slippage(
        PositionSide.SHORT,
        market_price,
    )

    expected_price = (
        market_price
        * (1.0 - SLIPPAGE)
    )

    assert execution_price == pytest.approx(
        expected_price,
        rel=1e-10,
    )

    assert execution_price < market_price


# ============================================================
# POSITION SIZE
# ============================================================

def test_long_position_size_is_positive():
    """LONG position size must be positive."""

    quantity = calculate_position_size(
        balance=INITIAL_BALANCE,
        risk_percent=RISK_PERCENT,
        entry_price=100.0,
        stop_loss=95.0,
    )

    assert quantity > 0


def test_short_position_size_is_positive():
    """SHORT position size must be positive."""

    quantity = calculate_position_size(
        balance=INITIAL_BALANCE,
        risk_percent=RISK_PERCENT,
        entry_price=100.0,
        stop_loss=105.0,
    )

    assert quantity > 0


# ============================================================
# SL / TP
# ============================================================

def test_long_sl_tp_are_valid():
    """LONG SL must be below entry and TP above entry."""

    entry_price = 100.0
    atr = 2.0

    stop_loss, take_profit = calculate_sl_tp(
        entry_price=entry_price,
        atr=atr,
    )

    assert stop_loss < entry_price
    assert take_profit > entry_price


def test_sl_tp_distance_is_positive():
    """SL and TP must have non-zero distances from entry."""

    entry_price = 100.0
    atr = 2.0

    stop_loss, take_profit = calculate_sl_tp(
        entry_price=entry_price,
        atr=atr,
    )

    assert abs(entry_price - stop_loss) > 0
    assert abs(take_profit - entry_price) > 0


# ============================================================
# COMMISSION
# ============================================================

def test_entry_commission_is_positive_when_commission_enabled():
    """Entry commission must be positive when configured."""

    engine = make_engine()

    quantity = 0.5
    price = 100.0

    commission = engine.calculate_commission(
        quantity,
        price,
    )

    if COMMISSION > 0:
        assert commission > 0
    else:
        assert commission == 0


def test_entry_commission_matches_configuration():
    """Commission calculation must match configuration."""

    engine = make_engine()

    quantity = 0.5
    price = 100.0

    expected = (
        quantity
        * price
        * COMMISSION
    )

    actual = engine.calculate_commission(
        quantity,
        price,
    )

    assert actual == pytest.approx(
        expected,
        rel=1e-10,
    )


# ============================================================
# POSITION CAPACITY
# ============================================================

def test_new_engine_can_open_position():
    """Fresh engine must allow opening a position."""

    engine = make_engine()

    assert engine.can_open_position() is True


def test_position_capacity_respects_maximum():
    """
    Engine must not allow more than MAX_OPEN_POSITIONS
    simultaneously.
    """

    engine = make_engine()

    assert engine.can_open_position() is True

    # Fill the position slots with simple open positions.
    from src.trade_engine import Position

    for position_id in range(
        1,
        MAX_OPEN_POSITIONS + 1,
    ):
        position = Position(
            id=position_id,
            side=PositionSide.LONG,
            status=PositionStatus.OPEN,
            entry_time="2026-01-01 00:00:00",
            entry_price=100.0,
            quantity=0.1,
        )

        engine.positions.append(position)

    assert engine.can_open_position() is False


# ============================================================
# POSITION OBJECT
# ============================================================

def test_open_position_object_has_correct_state():
    """A newly created position must have OPEN status."""

    from src.trade_engine import Position

    position = Position(
        id=1,
        side=PositionSide.LONG,
        status=PositionStatus.OPEN,
        entry_time="2026-01-01 00:00:00",
        entry_price=100.0,
        quantity=0.5,
    )

    assert position.status == PositionStatus.OPEN
    assert position.side == PositionSide.LONG
    assert position.entry_price == 100.0
    assert position.quantity == 0.5


def test_short_position_object_has_correct_state():
    """SHORT position must preserve SHORT side."""

    from src.trade_engine import Position

    position = Position(
        id=1,
        side=PositionSide.SHORT,
        status=PositionStatus.OPEN,
        entry_time="2026-01-01 00:00:00",
        entry_price=100.0,
        quantity=0.5,
    )

    assert position.status == PositionStatus.OPEN
    assert position.side == PositionSide.SHORT
    assert position.entry_price == 100.0
    assert position.quantity == 0.5


# ============================================================
# BALANCE / EQUITY
# ============================================================

def test_fresh_engine_balance_matches_initial_balance():
    """Fresh TradeEngine must start with configured balance."""

    engine = make_engine()

    assert engine.balance == pytest.approx(
        INITIAL_BALANCE,
        abs=1e-10,
    )


def test_fresh_engine_equity_matches_initial_balance():
    """Fresh TradeEngine equity must equal initial balance."""

    engine = make_engine()

    assert engine.equity == pytest.approx(
        INITIAL_BALANCE,
        abs=1e-10,
    )


# ============================================================
# RISK CONSISTENCY
# ============================================================

def test_position_size_matches_risk_amount():
    """
    Position size must correspond to the configured
    percentage risk when the maximum position cap
    is not reached.
    """

    balance = 1000.0
    risk_percent = 1.0

    entry_price = 100.0
    stop_loss = 80.0

    quantity = calculate_position_size(
        balance=balance,
        risk_percent=risk_percent,
        entry_price=entry_price,
        stop_loss=stop_loss,
    )

    risk_amount = (
        balance
        * risk_percent
        / 100.0
    )

    expected_quantity = (
        risk_amount
        / abs(entry_price - stop_loss)
    )

    assert quantity == pytest.approx(
        expected_quantity,
        abs=1e-6,
    )


def test_position_size_for_short_uses_stop_distance():
    """SHORT sizing must use the absolute stop distance."""

    balance = 1000.0
    risk_percent = 1.0

    entry_price = 100.0
    stop_loss = 120.0

    quantity = calculate_position_size(
        balance=balance,
        risk_percent=risk_percent,
        entry_price=entry_price,
        stop_loss=stop_loss,
    )

    expected_quantity = (
        balance
        * risk_percent
        / 100.0
        / abs(entry_price - stop_loss)
    )

    assert quantity == pytest.approx(
        expected_quantity,
        abs=1e-6,
    )


# ============================================================
# END
# ============================================================

def test_execution_module_smoke():
    """Basic execution test module smoke check."""

    engine = make_engine()

    candle = make_candle()

    assert candle["close"] > 0
    assert engine.balance > 0
    assert engine.can_open_position() is True
