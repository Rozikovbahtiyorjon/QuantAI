"""
R2.2: PaperAccountState strict ledger tests.

Guarantees:
    identity  cash == initial + realized_gross - fees_paid
    mirror    engine.balance == account_state.cash after every op
    equity    flat == initial + net; marked adds unrealized
"""

from __future__ import annotations

import pytest

from src.paper_trading_engine import (
    PaperAccountState,
    PaperTradingEngine,
)


class TestIdentity:
    def test_initial_state(self) -> None:
        s = PaperAccountState(initial_cash=1000.0)

        assert s.cash == pytest.approx(1000.0)
        assert s.identity_gap < 1e-9
        assert s.equity() == pytest.approx(1000.0)

    def test_identity_after_long_win(self) -> None:
        eng = PaperTradingEngine(initial_balance=1000.0, commission=0.001)

        eng.open_position(side="LONG", price=100.0, quantity=1.0)
        trade = eng.close_position(price=110.0)

        st = eng.account_state

        assert st.identity_gap < 1e-6
        assert st.cash == pytest.approx(eng.balance)
        assert st.fees_paid == pytest.approx(trade.fees)
        assert st.realized_gross == pytest.approx(trade.gross_profit)
        assert st.equity() == pytest.approx(1000.0 + trade.net_profit)

    def test_identity_after_short_loss(self) -> None:
        eng = PaperTradingEngine(initial_balance=500.0, commission=0.002)

        eng.open_position(side="SHORT", price=50.0, quantity=2.0)
        trade = eng.close_position(price=52.0)   # adverse move for SHORT

        st = eng.account_state
        assert st.identity_gap < 1e-6
        assert st.cash == pytest.approx(eng.balance)
        assert st.equity() == pytest.approx(500.0 + trade.net_profit)
        assert trade.net_profit < 0


class TestSequences:
    def test_flip_sequence_reconciles(self) -> None:
        eng = PaperTradingEngine(initial_balance=2000.0, commission=0.0004)

        eng.open_position("LONG", 100.0, 3.0)
        t1 = eng.close_position(104.0)
        eng.open_position("SHORT", 104.0, 2.5)
        t2 = eng.close_position(101.0)
        eng.open_position("LONG", 101.0, 1.0)
        t3 = eng.close_position(99.0)

        st = eng.account_state

        assert st.identity_gap < 1e-6
        assert st.cash == pytest.approx(eng.balance)

        expected = 2000.0 + sum(t.net_profit for t in (t1, t2, t3))
        assert st.equity() == pytest.approx(expected, abs=1e-6)
        assert eng.balance == pytest.approx(expected, abs=1e-6)

    def test_many_trades_no_drift(self) -> None:
        eng = PaperTradingEngine(initial_balance=10000.0, commission=0.0004)

        price = 100.0
        for i in range(50):
            side = "LONG" if i % 2 == 0 else "SHORT"
            eng.open_position(side, price, 0.1)
            price += 0.7 if i % 3 else -0.5
            eng.close_position(price)

        assert eng.account_state.identity_gap < 1e-6
        assert eng.account_state.cash == pytest.approx(eng.balance)


class TestMarkToMarket:
    def test_equity_marked_includes_unrealized(self) -> None:
        s = PaperAccountState(initial_cash=1000.0)
        s.apply_open("LONG", price=100.0, quantity=2.0, entry_fee=0.2)

        # cash already reflects entry fee deduction.
        assert s.cash == pytest.approx(999.8)

        eq_flat = s.initial_cash + (s.realized_gross - s.fees_paid)
        assert s.equity() == pytest.approx(eq_flat)

        # LONG +2 @100, last=103 -> unrealized +6
        assert s.unrealized(103.0) == pytest.approx(6.0)
        assert s.equity(last_price=103.0) == pytest.approx(eq_flat + 6.0)

    def test_reset_restores_everything(self) -> None:
        eng = PaperTradingEngine(initial_balance=800.0, commission=0.001)
        eng.open_position("LONG", 90.0, 1.0)
        eng.close_position(95.0)

        eng.reset()

        st = eng.account_state
        assert st.cash == pytest.approx(800.0)
        assert st.fees_paid == 0.0
        assert st.realized_gross == 0.0
        assert st.position_side is None
        assert st.identity_gap < 1e-9