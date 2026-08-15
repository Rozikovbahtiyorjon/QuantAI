"""
=========================================================
QuantAI Professional v5
Paper Trading Runner Tests
=========================================================
"""

from __future__ import annotations

from src.paper_trading_runner import (
    PaperTradingRunner,
    PaperTradingStepResult,
)
from src.strategy import SignalResult


# =========================================================
# 1. INITIAL STATE
# =========================================================

def test_runner_initial_state():

    runner = PaperTradingRunner(
        initial_balance=1000.0,
        commission=0.0004,
        quantity=1.0,
    )

    assert runner.balance == 1000.0
    assert runner.has_position is False
    assert runner.realized_profit == 0.0


# =========================================================
# 2. INVALID QUANTITY
# =========================================================

def test_invalid_quantity():

    try:

        PaperTradingRunner(
            quantity=0,
        )

        assert False

    except ValueError:
        pass


# =========================================================
# 3. HOLD DOES NOTHING
# =========================================================

def test_hold_does_nothing():

    runner = PaperTradingRunner()

    signal = SignalResult(
        signal="HOLD",
        entry=100.0,
    )

    result = runner.process_signal(
        signal
    )

    assert isinstance(
        result,
        PaperTradingStepResult,
    )

    assert result.position_opened is False
    assert result.position_closed is False
    assert result.trade is None

    assert runner.has_position is False


# =========================================================
# 4. BUY OPENS LONG
# =========================================================

def test_buy_opens_long():

    runner = PaperTradingRunner(
        initial_balance=1000.0,
        commission=0.0004,
        quantity=1.0,
    )

    signal = SignalResult(
        signal="BUY",
        entry=100.0,
        confidence=80.0,
    )

    result = runner.process_signal(
        signal
    )

    assert result.position_opened is True
    assert result.position_closed is False
    assert result.trade is None

    assert runner.has_position is True

    assert runner.engine.position is not None
    assert runner.engine.position.side == "LONG"
    assert runner.engine.position.entry_price == 100.0
    assert runner.engine.position.quantity == 1.0


# =========================================================
# 5. SELL OPENS SHORT
# =========================================================

def test_sell_opens_short():

    runner = PaperTradingRunner(
        initial_balance=1000.0,
        commission=0.0004,
        quantity=1.0,
    )

    signal = SignalResult(
        signal="SELL",
        entry=100.0,
        confidence=80.0,
    )

    result = runner.process_signal(
        signal
    )

    assert result.position_opened is True
    assert result.position_closed is False
    assert result.trade is None

    assert runner.has_position is True

    assert runner.engine.position is not None
    assert runner.engine.position.side == "SHORT"
    assert runner.engine.position.entry_price == 100.0


# =========================================================
# 6. SECOND BUY DOES NOT OPEN ANOTHER POSITION
# =========================================================

def test_second_buy_does_not_open_position():

    runner = PaperTradingRunner(
        initial_balance=1000.0,
        commission=0.0004,
        quantity=1.0,
    )

    first_signal = SignalResult(
        signal="BUY",
        entry=100.0,
    )

    second_signal = SignalResult(
        signal="BUY",
        entry=105.0,
    )

    first = runner.process_signal(
        first_signal
    )

    second = runner.process_signal(
        second_signal
    )

    assert first.position_opened is True
    assert second.position_opened is False

    assert runner.engine.position is not None
    assert runner.engine.position.entry_price == 100.0


# =========================================================
# 7. CLOSE LONG
# =========================================================

def test_close_long():

    runner = PaperTradingRunner(
        initial_balance=1000.0,
        commission=0.0,
        quantity=1.0,
    )

    signal = SignalResult(
        signal="BUY",
        entry=100.0,
    )

    runner.process_signal(
        signal
    )

    result = runner.close_position(
        price=110.0
    )

    assert result.position_closed is True
    assert result.trade is not None

    assert result.trade.side == "LONG"
    assert result.trade.entry_price == 100.0
    assert result.trade.exit_price == 110.0
    assert result.trade.net_profit == 10.0

    assert runner.has_position is False
    assert runner.realized_profit == 10.0


# =========================================================
# 8. CLOSE SHORT
# =========================================================

def test_close_short():

    runner = PaperTradingRunner(
        initial_balance=1000.0,
        commission=0.0,
        quantity=1.0,
    )

    signal = SignalResult(
        signal="SELL",
        entry=100.0,
    )

    runner.process_signal(
        signal
    )

    result = runner.close_position(
        price=90.0
    )

    assert result.position_closed is True
    assert result.trade is not None

    assert result.trade.side == "SHORT"
    assert result.trade.entry_price == 100.0
    assert result.trade.exit_price == 90.0
    assert result.trade.net_profit == 10.0

    assert runner.has_position is False
    assert runner.realized_profit == 10.0


# =========================================================
# 9. CLOSE WITHOUT POSITION
# =========================================================

def test_close_without_position():

    runner = PaperTradingRunner()

    try:

        runner.close_position(
            price=100.0
        )

        assert False

    except RuntimeError:
        pass


# =========================================================
# 10. INVALID SIGNAL TYPE
# =========================================================

def test_invalid_signal_type():

    runner = PaperTradingRunner()

    try:

        runner.process_signal(
            "BUY"
        )

        assert False

    except TypeError:
        pass


# =========================================================
# 11. UNKNOWN SIGNAL
# =========================================================

def test_unknown_signal():

    runner = PaperTradingRunner()

    signal = SignalResult(
        signal="UNKNOWN",
        entry=100.0,
    )

    try:

        runner.process_signal(
            signal
        )

        assert False

    except ValueError:
        pass


# =========================================================
# 12. INVALID CLOSE PRICE
# =========================================================

def test_invalid_close_price():

    runner = PaperTradingRunner()

    try:

        runner.close_position(
            price=0,
        )

        assert False

    except ValueError:
        pass


# =========================================================
# 13. RESET
# =========================================================

def test_reset():

    runner = PaperTradingRunner(
        initial_balance=1000.0,
        commission=0.0,
        quantity=1.0,
    )

    signal = SignalResult(
        signal="BUY",
        entry=100.0,
    )

    runner.process_signal(
        signal
    )

    assert runner.has_position is True

    runner.reset()

    assert runner.balance == 1000.0
    assert runner.has_position is False
    assert runner.realized_profit == 0.0


# =========================================================
# 14. CUSTOM QUANTITY
# =========================================================

def test_custom_quantity():

    runner = PaperTradingRunner(
        initial_balance=1000.0,
        commission=0.0,
        quantity=2.5,
    )

    signal = SignalResult(
        signal="BUY",
        entry=100.0,
    )

    runner.process_signal(
        signal
    )

    assert runner.engine.position is not None
    assert runner.engine.position.quantity == 2.5


# =========================================================
# 15. LONG PROFIT
# =========================================================

def test_long_profit():

    runner = PaperTradingRunner(
        initial_balance=1000.0,
        commission=0.0,
        quantity=2.0,
    )

    runner.process_signal(
        SignalResult(
            signal="BUY",
            entry=100.0,
        )
    )

    result = runner.close_position(
        price=105.0
    )

    assert result.trade is not None
    assert result.trade.gross_profit == 10.0
    assert result.trade.net_profit == 10.0


# =========================================================
# 16. SHORT PROFIT
# =========================================================

def test_short_profit():

    runner = PaperTradingRunner(
        initial_balance=1000.0,
        commission=0.0,
        quantity=2.0,
    )

    runner.process_signal(
        SignalResult(
            signal="SELL",
            entry=100.0,
        )
    )

    result = runner.close_position(
        price=95.0
    )

    assert result.trade is not None
    assert result.trade.gross_profit == 10.0
    assert result.trade.net_profit == 10.0