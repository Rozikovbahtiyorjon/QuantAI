"""
R0.1 E2E risk gates.

Checklist from the consolidation report (category A / critical):
    - FLIP ordering with projected exposure baseline
    - rejected risk  =>  engine state unchanged
    - drawdown block
    - exposure block
    - position sizing respects risk_percent
    - NaN / Inf signal rejection
    - zero balance rejection
    - same-side duplicate rejection
    - safe default: risk controls ON for PaperTradingRunner()
"""

from __future__ import annotations

import math

import pytest

from src.paper_trading_runner import PaperTradingRunner
from src.strategy.signal_generator import SignalResult


def make_signal(
    signal: str = "BUY",
    entry: float = 100.0,
    stop_loss: float = 98.0,
    take_profit: float | None = 106.0,
) -> SignalResult:
    return SignalResult(
        signal=signal,
        entry=entry,
        stop_loss=stop_loss,
        take_profit=take_profit if take_profit is not None else 0.0,
        confidence=70.0,
    )


class TestSafeDefaults:
    def test_risk_controls_on_by_default(self) -> None:
        runner = PaperTradingRunner()
        assert runner.enable_risk_controls is True

    def test_legacy_mode_available(self) -> None:
        runner = PaperTradingRunner(enable_risk_controls=False)
        assert runner.enable_risk_controls is False


class TestFlipOrdering:
    def test_flip_approved_against_projected_baseline(self) -> None:
        """
        Exposure limit would reject old+new simultaneously but fits
        after close. R0.1: flip must be APPROVED (projected baseline).
        """
        runner = PaperTradingRunner(
            initial_balance=1000.0,
            commission=0.0,
            enable_risk_controls=True,
            max_total_exposure_percent=7.0,
            max_position_exposure_percent=5.0,
        )

        assert runner.process_signal(make_signal("BUY")).position_opened
        assert runner.engine.position.side == "LONG"

        res = runner.process_signal(make_signal("SELL"))

        assert res.risk_approved is True
        assert res.position_closed and res.position_opened
        assert runner.engine.position.side == "SHORT"

    def test_flip_rejected_leaves_state_unchanged(self) -> None:
        """
        Drawdown-blocked flip must NOT close the existing position.
        Atomicity invariant.
        """
        runner = PaperTradingRunner(
            initial_balance=1000.0,
            commission=0.0,
            enable_risk_controls=True,
            max_drawdown_percent=0.1,   # blocks everything after open
        )

        assert runner.process_signal(make_signal("BUY")).position_opened

        # Force drawdown breach by equity drop.
        runner.engine.balance = 950.0
        runner.drawdown_guard._peak_equity = 1000.0

        res = runner.process_signal(make_signal("SELL"))

        assert res.risk_approved is False
        assert res.position_closed is False
        assert res.position_opened is False

        assert runner.has_position is True
        assert runner.engine.position.side == "LONG"
        assert runner.engine.balance == pytest.approx(950.0)


class TestRejectionGates:
    def test_exposure_block_fresh_entry(self) -> None:
        runner = PaperTradingRunner(
            initial_balance=1000.0,
            commission=0.0,
            enable_risk_controls=True,
            max_total_exposure_percent=3.0,
            max_position_exposure_percent=5.0,
        )

        res = runner.process_signal(make_signal("BUY"))

        assert res.risk_approved is False
        assert "exposure" in res.risk_reason.lower()

        assert not runner.has_position
        assert runner.current_exposure == pytest.approx(0.0)

    def test_nan_entry_rejected(self) -> None:
        runner = PaperTradingRunner(commission=0.0)

        res = runner.process_signal(
            make_signal(entry=float("nan"), stop_loss=98.0)
        )

        assert res.risk_approved is False
        assert "NaN" in res.risk_reason or "Inf" in res.risk_reason
        assert not runner.has_position

    def test_inf_stop_rejected(self) -> None:
        runner = PaperTradingRunner(commission=0.0)

        res = runner.process_signal(
            make_signal(stop_loss=float("inf"))
        )

        assert res.risk_approved is False
        assert not runner.has_position

    def test_zero_balance_rejected(self) -> None:
        runner = PaperTradingRunner(commission=0.0)
        runner.engine.balance = 0.0

        res = runner.process_signal(make_signal())

        assert res.risk_approved is False
        assert "balance" in res.risk_reason.lower()
        assert not runner.has_position

    def test_zero_stop_rejected(self) -> None:
        runner = PaperTradingRunner(commission=0.0)

        res = runner.process_signal(make_signal(stop_loss=0.0))

        assert res.risk_approved is False
        assert not runner.has_position


class TestSizingAndDuplicates:
    def test_sizing_respects_risk_percent_and_position_cap(self) -> None:
        """
        qty <= min( risk_amount/stop_distance , position_cap_notional/entry )
        risk 1% of 1000 = $10; stop distance $2 -> risk qty 5;
        position cap 5% -> $50 -> 0.5 @ entry 100.
        """
        runner = PaperTradingRunner(
            initial_balance=1000.0,
            commission=0.0,
            enable_risk_controls=True,
            risk_percent=1.0,
            max_total_exposure_percent=60.0,
            max_position_exposure_percent=5.0,
        )

        res = runner.process_signal(make_signal("BUY", entry=100.0, stop_loss=98.0))

        assert res.risk_approved is True
        qty = runner.engine.position.quantity

        expected_cap_qty = 50.0 / 100.0
        assert qty == pytest.approx(expected_cap_qty)
        assert qty * 100.0 <= 1000.0 * 0.05 + 1e-6

    def test_same_side_duplicate_rejected_state_unchanged(self) -> None:
        runner = PaperTradingRunner(commission=0.0)

        first = runner.process_signal(make_signal("BUY"))
        assert first.position_opened is True

        qty_before = runner.engine.position.quantity
        balance_before = runner.engine.balance

        second = runner.process_signal(make_signal("BUY"))

        assert second.position_opened is False
        assert "same side" in second.risk_reason.lower()

        assert runner.engine.position.quantity == pytest.approx(qty_before)
        assert runner.engine.balance == pytest.approx(balance_before)


class TestRiskContextContract:
    def test_orchestrator_uses_projected_exposure(self) -> None:
        from src.risk.risk_context import RiskContext
        from src.risk.risk_orchestrator import RiskOrchestrator
        from src.exposure_manager import ExposureManager
        from src.drawdown_guard import DrawdownGuard
        from src.position_sizer import PositionSizer

        orch = RiskOrchestrator(
            drawdown_guard=DrawdownGuard(max_drawdown_percent=10.0),
            exposure_manager=ExposureManager(
                max_total_exposure_percent=7.0,
                max_position_exposure_percent=5.0,
            ),
            position_sizer=PositionSizer(min_leverage=1.0, max_leverage=50.0),
            default_risk_percent=1.0,
        )

        sig = make_signal("SELL", entry=100.0, stop_loss=102.0)

        # Without context: current exposure counts -> blocked.
        d_old = orch.evaluate(signal=sig, equity=1000.0, current_exposure=50.0)
        assert d_old.allowed is False

        # With flip context: projected baseline zero -> approved.
        ctx = RiskContext(
            equity=1000.0,
            balance=1000.0,
            current_exposure=50.0,
            projected_exposure=0.0,
            requested_side="SHORT",
            is_flip=True,
            entry=sig.entry,
            stop_loss=sig.stop_loss,
            take_profit=sig.take_profit,
        )
        d_new = orch.evaluate(signal=sig, equity=1000.0, current_exposure=50.0,
                              context=ctx)
        assert d_new.allowed is True
        assert d_new.metadata.get("is_flip") is True

    def test_risk_context_validates_inputs(self) -> None:
        from src.risk.risk_context import RiskContext

        with pytest.raises(ValueError):
            RiskContext(equity=float("nan"), balance=0.0,
                        requested_side="LONG")

        with pytest.raises(ValueError):
            RiskContext(equity=100.0, balance=100.0,
                        requested_side="SIDEWAYS")
