"""
====================================================
QuantAI Professional v5
Trade Engine Financial Tests
====================================================

Tests:
    - commission calculation
    - slippage calculation
    - LONG PnL
    - SHORT PnL
    - position sizing
    - risk consistency
    - commission impact
    - balance consistency
"""

from __future__ import annotations

import math

import pytest

from config.settings import (
    COMMISSION,
    SLIPPAGE,
    INITIAL_BALANCE,
    RISK_PERCENT,
)

from src.trade_engine import (
    TradeEngine,
    PositionSide,
    CloseReason,
)

from src.risk_manager import (
    calculate_position_size,
)


# ============================================================
# HELPERS
# ============================================================

def make_candle(
    timestamp="2026-01-01 00:00:00",
    open_price=100.0,
    high=101.0,
    low=99.0,
    close=100.0,
    atr=1.0,
):
    """Create a minimal candle."""

    return {
        "timestamp": timestamp,
        "open": float(open_price),
        "high": float(high),
        "low": float(low),
        "close": float(close),
        "volume": 1000.0,
        "atr": float(atr),
    }


# ============================================================
# COMMISSION
# ============================================================

def test_commission_calculation():
    """Commission must equal quantity * price * COMMISSION."""

    engine = TradeEngine()

    quantity = 2.0
    price = 1000.0

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


def test_commission_increases_with_quantity():
    """Larger quantity must produce larger commission."""

    engine = TradeEngine()

    commission_small = engine.calculate_commission(
        1.0,
        1000.0,
    )

    commission_large = engine.calculate_commission(
        2.0,
        1000.0,
    )

    assert commission_large > commission_small

    assert commission_large == pytest.approx(
        commission_small * 2,
        rel=1e-10,
    )


def test_commission_increases_with_price():
    """Higher price must produce higher commission."""

    engine = TradeEngine()

    commission_low = engine.calculate_commission(
        1.0,
        1000.0,
    )

    commission_high = engine.calculate_commission(
        1.0,
        2000.0,
    )

    assert commission_high > commission_low

    assert commission_high == pytest.approx(
        commission_low * 2,
        rel=1e-10,
    )


# ============================================================
# SLIPPAGE
# ============================================================

def test_long_slippage_increases_entry_price():
    """LONG execution price must move upward with slippage."""

    engine = TradeEngine()

    price = 100.0

    actual = engine.apply_slippage(
        PositionSide.LONG,
        price,
    )

    expected = (
        price
        * (1 + SLIPPAGE)
    )

    assert actual == pytest.approx(
        expected,
        rel=1e-10,
    )


def test_short_slippage_decreases_execution_price():
    """SHORT execution price must move downward with slippage."""

    engine = TradeEngine()

    price = 100.0

    actual = engine.apply_slippage(
        PositionSide.SHORT,
        price,
    )

    expected = (
        price
        * (1 - SLIPPAGE)
    )

    assert actual == pytest.approx(
        expected,
        rel=1e-10,
    )


def test_slippage_is_deterministic():
    """Same input must always produce same execution price."""

    engine = TradeEngine()

    price = 50000.0

    first = engine.apply_slippage(
        PositionSide.LONG,
        price,
    )

    second = engine.apply_slippage(
        PositionSide.LONG,
        price,
    )

    assert first == second


# ============================================================
# POSITION SIZE
# ============================================================

def test_position_size_is_positive():
    """Valid risk configuration must produce positive quantity."""

    quantity = calculate_position_size(
        balance=1000.0,
        risk_percent=RISK_PERCENT,
        entry_price=100.0,
        stop_loss=98.0,
    )

    assert quantity > 0


def test_position_size_decreases_when_stop_is_farther():
    """
    Wider stop distance should reduce position size
    when risk percentage remains constant and
    MAX_POSITION_SIZE is not reached.
    """

    quantity_close_stop = calculate_position_size(
        balance=1000.0,
        risk_percent=RISK_PERCENT,
        entry_price=100.0,
        stop_loss=80.0,
    )

    quantity_far_stop = calculate_position_size(
        balance=1000.0,
        risk_percent=RISK_PERCENT,
        entry_price=100.0,
        stop_loss=60.0,
    )

    assert quantity_close_stop > quantity_far_stop

    assert quantity_close_stop == pytest.approx(
        0.5,
        abs=1e-6,
    )

    assert quantity_far_stop == pytest.approx(
        0.25,
        abs=1e-6,
    )


def test_position_size_increases_with_balance():
    """
    Higher account balance should increase position size
    when MAX_POSITION_SIZE is not reached.
    """

    quantity_small = calculate_position_size(
        balance=1000.0,
        risk_percent=RISK_PERCENT,
        entry_price=100.0,
        stop_loss=60.0,
    )

    quantity_large = calculate_position_size(
        balance=2000.0,
        risk_percent=RISK_PERCENT,
        entry_price=100.0,
        stop_loss=60.0,
    )

    assert quantity_large > quantity_small

    assert quantity_small == pytest.approx(
        0.25,
        abs=1e-6,
    )

    assert quantity_large == pytest.approx(
        0.5,
        abs=1e-6,
    )


# ============================================================
# LONG PNL
# ============================================================

def test_long_profitable_trade_has_positive_pnl():
    """LONG trade with higher exit price must be profitable."""

    engine = TradeEngine()

    entry_price = 100.0
    exit_price = 110.0
    quantity = 1.0

    candle = make_candle(
        close=exit_price,
    )

    # Construct position directly so the PnL test
    # isolates TradeEngine.close_position().
    from src.trade_engine import (
        Position,
        PositionStatus,
    )

    position = Position(
        id=1,
        side=PositionSide.LONG,
        status=PositionStatus.OPEN,
        entry_time=candle["timestamp"],
        entry_price=entry_price,
        quantity=quantity,
    )

    engine.positions.append(position)

    engine.close_position(
        position,
        candle,
        exit_price,
        CloseReason.MANUAL,
    )

    expected_gross = (
        exit_price
        - entry_price
    ) * quantity

    assert position.gross_profit == pytest.approx(
        expected_gross,
        abs=0.02,
    )

    assert position.net_profit < position.gross_profit

    assert position.net_profit > 0


# ============================================================
# LONG LOSS
# ============================================================

def test_long_losing_trade_has_negative_pnl():
    """LONG trade with lower exit price must lose money."""

    engine = TradeEngine()

    entry_price = 100.0
    exit_price = 90.0
    quantity = 1.0

    candle = make_candle(
        close=exit_price,
    )

    from src.trade_engine import (
        Position,
        PositionStatus,
    )

    position = Position(
        id=1,
        side=PositionSide.LONG,
        status=PositionStatus.OPEN,
        entry_time=candle["timestamp"],
        entry_price=entry_price,
        quantity=quantity,
    )

    engine.positions.append(position)

    engine.close_position(
        position,
        candle,
        exit_price,
        CloseReason.STOP_LOSS,
    )

    assert position.gross_profit < 0

    assert position.net_profit < 0


# ============================================================
# SHORT PNL
# ============================================================

def test_short_profitable_trade_has_positive_pnl():
    """SHORT trade with lower exit price must be profitable."""

    engine = TradeEngine()

    entry_price = 100.0
    exit_price = 90.0
    quantity = 1.0

    candle = make_candle(
        close=exit_price,
    )

    from src.trade_engine import (
        Position,
        PositionStatus,
    )

    position = Position(
        id=1,
        side=PositionSide.SHORT,
        status=PositionStatus.OPEN,
        entry_time=candle["timestamp"],
        entry_price=entry_price,
        quantity=quantity,
    )

    engine.positions.append(position)

    engine.close_position(
        position,
        candle,
        exit_price,
        CloseReason.TAKE_PROFIT,
    )

    expected_gross = (
        entry_price
        - exit_price
    ) * quantity

    assert position.gross_profit == pytest.approx(
        expected_gross,
        abs=0.02,
    )

    assert position.net_profit > 0


# ============================================================
# SHORT LOSS
# ============================================================

def test_short_losing_trade_has_negative_pnl():
    """SHORT trade with higher exit price must lose money."""

    engine = TradeEngine()

    entry_price = 100.0
    exit_price = 110.0
    quantity = 1.0

    candle = make_candle(
        close=exit_price,
    )

    from src.trade_engine import (
        Position,
        PositionStatus,
    )

    position = Position(
        id=1,
        side=PositionSide.SHORT,
        status=PositionStatus.OPEN,
        entry_time=candle["timestamp"],
        entry_price=entry_price,
        quantity=quantity,
    )

    engine.positions.append(position)

    engine.close_position(
        position,
        candle,
        exit_price,
        CloseReason.STOP_LOSS,
    )

    assert position.gross_profit < 0

    assert position.net_profit < 0


# ============================================================
# COMMISSION REDUCES PROFIT
# ============================================================

def test_commission_reduces_net_profit():
    """
    Net profit must be lower than gross profit
    whenever commission is greater than zero.
    """

    engine = TradeEngine()

    entry_price = 100.0
    exit_price = 110.0
    quantity = 1.0

    candle = make_candle(
        close=exit_price,
    )

    from src.trade_engine import (
        Position,
        PositionStatus,
    )

    position = Position(
        id=1,
        side=PositionSide.LONG,
        status=PositionStatus.OPEN,
        entry_time=candle["timestamp"],
        entry_price=entry_price,
        quantity=quantity,
    )

    engine.positions.append(position)

    engine.close_position(
        position,
        candle,
        exit_price,
        CloseReason.TAKE_PROFIT,
    )

    if COMMISSION > 0:
        assert position.net_profit < position.gross_profit

    assert position.commission >= 0


# ============================================================
# BALANCE CONSISTENCY
# ============================================================

def test_balance_changes_by_net_profit():
    """Balance must change exactly by the position net profit."""

    engine = TradeEngine()

    initial_balance = engine.balance

    entry_price = 100.0
    exit_price = 110.0
    quantity = 1.0

    candle = make_candle(
        close=exit_price,
    )

    from src.trade_engine import (
        Position,
        PositionStatus,
    )

    position = Position(
        id=1,
        side=PositionSide.LONG,
        status=PositionStatus.OPEN,
        entry_time=candle["timestamp"],
        entry_price=entry_price,
        quantity=quantity,
    )

    engine.positions.append(position)

    engine.close_position(
        position,
        candle,
        exit_price,
        CloseReason.TAKE_PROFIT,
    )

    expected_balance = round(
        initial_balance
        + position.net_profit,
        2,
    )

    assert engine.balance == pytest.approx(
        expected_balance,
        abs=0.01,
    )


# ============================================================
# BALANCE AFTER TWO TRADES
# ============================================================

def test_balance_is_consistent_after_multiple_trades():
    """
    Final balance must equal initial balance plus
    the sum of all closed trade net profits.
    """

    engine = TradeEngine()

    initial_balance = engine.balance

    from src.trade_engine import (
        Position,
        PositionStatus,
    )

    # --------------------------------------------------------
    # Trade 1: LONG profit
    # --------------------------------------------------------

    candle_1 = make_candle(
        timestamp="2026-01-01 00:00:00",
        close=110.0,
    )

    position_1 = Position(
        id=1,
        side=PositionSide.LONG,
        status=PositionStatus.OPEN,
        entry_time=candle_1["timestamp"],
        entry_price=100.0,
        quantity=1.0,
    )

    engine.positions.append(position_1)

    engine.close_position(
        position_1,
        candle_1,
        110.0,
        CloseReason.TAKE_PROFIT,
    )

    # --------------------------------------------------------
    # Trade 2: SHORT loss
    # --------------------------------------------------------

    candle_2 = make_candle(
        timestamp="2026-01-01 00:15:00",
        close=105.0,
    )

    position_2 = Position(
        id=2,
        side=PositionSide.SHORT,
        status=PositionStatus.OPEN,
        entry_time=candle_2["timestamp"],
        entry_price=100.0,
        quantity=1.0,
    )

    engine.positions.append(position_2)

    engine.close_position(
        position_2,
        candle_2,
        105.0,
        CloseReason.STOP_LOSS,
    )

    expected_profit = sum(
        position.net_profit
        for position in engine.closed_positions
    )

    expected_balance = round(
        initial_balance
        + expected_profit,
        2,
    )

    assert engine.total_trades == 2

    assert engine.balance == pytest.approx(
        expected_balance,
        abs=0.01,
    )

    assert engine.total_profit == pytest.approx(
        expected_profit,
        abs=0.01,
    )


# ============================================================
# FINANCIAL SANITY
# ============================================================

def test_initial_balance_is_positive():
    """Initial trading balance must be positive."""

    assert INITIAL_BALANCE > 0


def test_risk_percent_is_positive():
    """Configured risk percentage must be positive."""

    assert RISK_PERCENT > 0


def test_commission_is_non_negative():
    """Commission configuration cannot be negative."""

    assert COMMISSION >= 0


def test_slippage_is_non_negative():
    """Slippage configuration cannot be negative."""

    assert SLIPPAGE >= 0