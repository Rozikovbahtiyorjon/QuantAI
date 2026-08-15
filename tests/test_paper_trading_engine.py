"""
=========================================================
QuantAI PaperTradingEngine Tests
=========================================================
"""

from __future__ import annotations

import pytest

from src.paper_trading_engine import (
    PaperPosition,
    PaperTrade,
    PaperTradingEngine,
)


# =========================================================
# 1. INITIALIZATION
# =========================================================

def test_initialization():

    engine = PaperTradingEngine(
        initial_balance=1000.0,
        commission=0.0004,
    )

    assert engine.initial_balance == 1000.0
    assert engine.balance == 1000.0
    assert engine.commission == 0.0004
    assert engine.has_position is False
    assert engine.trade_history == []


# =========================================================
# 2. INVALID BALANCE
# =========================================================

def test_invalid_initial_balance():

    with pytest.raises(ValueError):

        PaperTradingEngine(
            initial_balance=0
        )


# =========================================================
# 3. INVALID COMMISSION
# =========================================================

def test_invalid_commission():

    with pytest.raises(ValueError):

        PaperTradingEngine(
            commission=-0.001
        )


# =========================================================
# 4. OPEN LONG
# =========================================================

def test_open_long():

    engine = PaperTradingEngine(
        initial_balance=1000.0,
        commission=0.0004,
    )

    position = engine.open_position(
        side="LONG",
        price=100.0,
        quantity=1.0,
    )

    assert isinstance(
        position,
        PaperPosition,
    )

    assert position.side == "LONG"
    assert position.entry_price == 100.0
    assert position.quantity == 1.0

    assert engine.has_position is True


# =========================================================
# 5. OPEN SHORT
# =========================================================

def test_open_short():

    engine = PaperTradingEngine()

    position = engine.open_position(
        side="SHORT",
        price=100.0,
        quantity=1.0,
    )

    assert position.side == "SHORT"
    assert engine.has_position is True


# =========================================================
# 6. INVALID SIDE
# =========================================================

def test_invalid_side():

    engine = PaperTradingEngine()

    with pytest.raises(ValueError):

        engine.open_position(
            side="BUY",
            price=100.0,
            quantity=1.0,
        )


# =========================================================
# 7. SECOND POSITION
# =========================================================

def test_cannot_open_second_position():

    engine = PaperTradingEngine()

    engine.open_position(
        side="LONG",
        price=100.0,
        quantity=1.0,
    )

    with pytest.raises(RuntimeError):

        engine.open_position(
            side="SHORT",
            price=100.0,
            quantity=1.0,
        )


# =========================================================
# 8. CLOSE LONG
# =========================================================

def test_close_long():

    engine = PaperTradingEngine(
        initial_balance=1000.0,
        commission=0.0,
    )

    engine.open_position(
        side="LONG",
        price=100.0,
        quantity=1.0,
    )

    trade = engine.close_position(
        price=110.0
    )

    assert isinstance(
        trade,
        PaperTrade,
    )

    assert trade.side == "LONG"
    assert trade.gross_profit == 10.0
    assert trade.fees == 0.0
    assert trade.net_profit == 10.0

    assert engine.balance == 1010.0
    assert engine.has_position is False


# =========================================================
# 9. CLOSE SHORT
# =========================================================

def test_close_short():

    engine = PaperTradingEngine(
        initial_balance=1000.0,
        commission=0.0,
    )

    engine.open_position(
        side="SHORT",
        price=100.0,
        quantity=1.0,
    )

    trade = engine.close_position(
        price=90.0
    )

    assert trade.side == "SHORT"
    assert trade.gross_profit == 10.0
    assert trade.net_profit == 10.0

    assert engine.balance == 1010.0


# =========================================================
# 10. LOSING LONG
# =========================================================

def test_losing_long():

    engine = PaperTradingEngine(
        initial_balance=1000.0,
        commission=0.0,
    )

    engine.open_position(
        side="LONG",
        price=100.0,
        quantity=1.0,
    )

    trade = engine.close_position(
        price=90.0
    )

    assert trade.gross_profit == -10.0
    assert trade.net_profit == -10.0

    assert engine.balance == 990.0


# =========================================================
# 11. COMMISSION
# =========================================================

def test_commission_is_applied():

    engine = PaperTradingEngine(
        initial_balance=1000.0,
        commission=0.001,
    )

    engine.open_position(
        side="LONG",
        price=100.0,
        quantity=1.0,
    )

    trade = engine.close_position(
        price=110.0
    )

    assert trade.entry_price == 100.0
    assert trade.exit_price == 110.0

    assert trade.fees == pytest.approx(
        0.21
    )

    assert trade.net_profit == pytest.approx(
        9.79
    )


# =========================================================
# 12. CLOSE WITHOUT POSITION
# =========================================================

def test_close_without_position():

    engine = PaperTradingEngine()

    with pytest.raises(RuntimeError):

        engine.close_position(
            price=100.0
        )


# =========================================================
# 13. TRADE HISTORY
# =========================================================

def test_trade_history():

    engine = PaperTradingEngine(
        initial_balance=1000.0,
        commission=0.0,
    )

    engine.open_position(
        side="LONG",
        price=100.0,
        quantity=1.0,
    )

    engine.close_position(
        price=110.0
    )

    engine.open_position(
        side="SHORT",
        price=100.0,
        quantity=1.0,
    )

    engine.close_position(
        price=90.0
    )

    assert len(
        engine.trade_history
    ) == 2

    assert (
        engine.realized_profit
        == 20.0
    )


# =========================================================
# 14. RESET
# =========================================================

def test_reset():

    engine = PaperTradingEngine(
        initial_balance=1000.0,
        commission=0.0,
    )

    engine.open_position(
        side="LONG",
        price=100.0,
        quantity=1.0,
    )

    engine.close_position(
        price=110.0
    )

    engine.reset()

    assert engine.balance == 1000.0
    assert engine.has_position is False
    assert engine.trade_history == []


# =========================================================
# 15. INVALID PRICE
# =========================================================

def test_invalid_open_price():

    engine = PaperTradingEngine()

    with pytest.raises(ValueError):

        engine.open_position(
            side="LONG",
            price=0,
            quantity=1.0,
        )


# =========================================================
# 16. INVALID QUANTITY
# =========================================================

def test_invalid_quantity():

    engine = PaperTradingEngine()

    with pytest.raises(ValueError):

        engine.open_position(
            side="LONG",
            price=100.0,
            quantity=0,
        )