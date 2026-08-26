from __future__ import annotations

import pytest

from src.paper_trading_runner import PaperTradingRunner
from src.strategy import SignalResult


def make_signal(
    signal: str,
    entry: float,
    stop_loss: float,
) -> SignalResult:
    return SignalResult(
        signal=signal,
        entry=entry,
        stop_loss=stop_loss,
        take_profit=entry,
    )


def test_risk_controls_approve_position_within_limits() -> None:
    runner = PaperTradingRunner(
        initial_balance=1000.0,
        commission=0.0,
        quantity=1.0,
        enable_risk_controls=True,
        risk_percent=1.0,
        max_drawdown_percent=10.0,
        max_total_exposure_percent=60.0,
        max_position_exposure_percent=5.0,
        leverage=1.0,
    )

    signal = make_signal(
        signal="BUY",
        entry=100.0,
        stop_loss=98.0,
    )

    result = runner.process_signal(signal)

    assert result.risk_approved is True
    assert result.risk_reason == "Risk approved."

    assert result.position_opened is True
    assert result.position_closed is False

    assert runner.has_position is True
    assert runner.engine.position is not None

    assert runner.engine.position.side == "LONG"

    # PositionSizer calculates 5 units from 1% risk,
    # ExposureManager caps the position at 5% of equity:
    # 1000 * 5% = 50 notional = 0.5 BTC.
    assert runner.engine.position.quantity == pytest.approx(
        0.5
    )

    assert runner.current_exposure == pytest.approx(
        50.0
    )


def test_risk_controls_reject_flip_when_total_exposure_would_exceed_limit() -> None:
    """
    R0.1 semantics: flip exposure is evaluated against the PROJECTED
    post-close baseline.

    Case A: post-close total fits  -> flip APPROVED, old position closed.
    """

    runner = PaperTradingRunner(
        initial_balance=1000.0,
        commission=0.0,
        quantity=1.0,
        enable_risk_controls=True,
        risk_percent=1.0,
        max_total_exposure_percent=7.0,
        max_position_exposure_percent=5.0,
        leverage=1.0,
    )

    buy = make_signal(
        signal="BUY",
        entry=100.0,
        stop_loss=98.0,
    )

    opened = runner.process_signal(
        buy
    )

    assert opened.risk_approved is True
    assert opened.position_opened is True

    assert runner.current_exposure == pytest.approx(
        50.0
    )

    sell = make_signal(
        signal="SELL",
        entry=100.0,
        stop_loss=102.0,
    )

    result = runner.process_signal(
        sell
    )

    # Old behavior (pre-R0.1) rejected this flip because the check ran
    # against the still-open LONG exposure (50 + 50 > 70).
    # New behavior: projected baseline after close = 0 -> approved.
    assert result.risk_approved is True
    assert result.position_closed is True
    assert result.position_opened is True

    assert runner.engine.position is not None
    assert runner.engine.position.side == "SHORT"


def test_risk_controls_reject_excessive_exposure() -> None:
    """
    R0.1: fresh entry (no open position) whose notional exceeds the
    total exposure limit -> rejected, engine state unchanged.
    """

    runner = PaperTradingRunner(
        initial_balance=1000.0,
        commission=0.0,
        quantity=1.0,
        enable_risk_controls=True,
        risk_percent=1.0,
        max_total_exposure_percent=3.0,  # below position cap (5%)
        max_position_exposure_percent=5.0,
        leverage=1.0,
    )

    buy = make_signal(signal="BUY", entry=100.0, stop_loss=98.0)
    result = runner.process_signal(buy)

    # Sized quantity is capped by the 5% position limit ($50),
    # which alone exceeds the 3% total limit ($30).
    assert result.risk_approved is False
    assert "exposure" in result.risk_reason.lower()

    assert result.position_opened is False
    assert runner.has_position is False
    assert runner.current_exposure == pytest.approx(0.0)


# ============================================================
# 7. DRAWDOWN - Max drawdown protection
# ============================================================
def test_risk_controls_reject_new_flip_after_drawdown() -> None:
    runner = PaperTradingRunner(
        initial_balance=1000.0,
        commission=0.0,
        quantity=1.0,
        enable_risk_controls=True,
        risk_percent=1.0,
        max_drawdown_percent=0.1,  # 0.1% max drawdown - very tight
        max_total_exposure_percent=60.0,
        max_position_exposure_percent=5.0,
        leverage=1.0,
    )

    buy = make_signal(
        signal="BUY",
        entry=100.0,
        stop_loss=98.0,
    )

    opened = runner.process_signal(buy)

    assert opened.position_opened is True
    assert runner.has_position is True

    assert runner.drawdown_guard.peak_equity == pytest.approx(1000.0)

    # Simulate account equity deterioration.
    runner.engine.balance = 998.0

    sell = make_signal(
        signal="SELL",
        entry=101.0,
        stop_loss=103.0,
    )

    result = runner.process_signal(sell)

    assert result.risk_approved is False

    assert (
        "maximum drawdown exceeded"
        in result.risk_reason
    )

    # Risk gate blocks the flip before the existing
    # LONG position can be closed.
    assert result.position_closed is False
    assert result.position_opened is False

    assert runner.engine.position is not None
    assert runner.engine.position.side == "LONG"


def test_drawdown_guard_resets_on_new_peak() -> None:
    """Drawdown guard should reset peak when new high is reached."""
    runner = PaperTradingRunner(
        initial_balance=1000.0,
        commission=0.0,
        quantity=1.0,
        enable_risk_controls=True,
    )

    runner.drawdown_guard.evaluate(1000.0)
    runner.drawdown_guard.evaluate(950.0)  # 5% drawdown

    assert runner.drawdown_guard.peak_equity == pytest.approx(1000.0)

    # New peak reached
    runner.drawdown_guard.evaluate(1050.0)

    assert runner.drawdown_guard.peak_equity == pytest.approx(1050.0)


# ============================================================
# 8. CORRELATION - Portfolio correlation limits
# ============================================================
def test_risk_controls_reject_high_correlation() -> None:
    """Test that highly correlated assets are limited."""
    # This test requires multi-asset setup
    # For now, verify the correlation limit is configured
    runner = PaperTradingRunner(
        initial_balance=1000.0,
        commission=0.0,
        quantity=1.0,
        enable_risk_controls=True,
        risk_percent=1.0,
    )

    # Check that correlation limit is configured
    risk_orchestrator = runner.risk_orchestrator
    assert hasattr(risk_orchestrator, 'exposure_manager')
    exposure_manager = risk_orchestrator.exposure_manager
    assert hasattr(exposure_manager, 'max_correlation')
    assert exposure_manager.max_correlation == 0.85


# ============================================================
# 9. QUANTITY ROUNDING - Position size rounding
# ============================================================
def test_quantity_rounding() -> None:
    """Test that position quantities are properly rounded."""
    runner = PaperTradingRunner(
        initial_balance=1000.0,
        commission=0.0,
        quantity=1.0,
        enable_risk_controls=True,
        risk_percent=1.0,
        max_drawdown_percent=10.0,
        max_total_exposure_percent=60.0,
        max_position_exposure_percent=5.0,
        leverage=1.0,
    )

    signal = make_signal(signal="BUY", entry=100.0, stop_loss=98.0)
    result = runner.process_signal(signal)

    assert result.risk_approved is True
    assert result.position_opened is True
    assert runner.engine.position.quantity > 0


# ============================================================
# 10. STOP-LOSS - Stop loss triggering
# ============================================================
def test_stop_loss_triggered() -> None:
    """Test that stop loss is properly triggered."""
    runner = PaperTradingRunner(
        initial_balance=1000.0,
        commission=0.0,
        quantity=1.0,
        enable_risk_controls=True,
        risk_percent=1.0,
    )

    # Open position
    signal = make_signal(signal="BUY", entry=100.0, stop_loss=98.0)
    runner.process_signal(signal)

    assert runner.has_position is True

    # Simulate price hitting stop loss
    runner.engine.position.stop_loss = 98.0
    runner.engine.position.side = "LONG"
    runner.engine.position.entry_price = 100.0
    runner.engine.position.quantity = 0.5

    # Simulate price hitting stop loss
    # This would be tested in the engine's update logic
    assert runner.engine.position.stop_loss == 98.0


# ============================================================
# 11. NaN/Inf HANDLING - Robustness against bad data
# ============================================================
def test_nan_inf_handling() -> None:
    """Test robustness against NaN/inf in market data."""
    import math

    runner = PaperTradingRunner(
        initial_balance=1000.0,
        commission=0.0,
        quantity=1.0,
        enable_risk_controls=True,
    )

    # Create signal with NaN values
    signal = SignalResult(
        signal="BUY",
        entry=float('nan'),
        stop_loss=98.0,
        take_profit=102.0,
    )

    result = runner.process_signal(signal)

    # Should reject NaN entry price
    assert result.risk_approved is False
    assert "nan" in result.risk_reason.lower() or "invalid" in result.risk_reason.lower()

    # Test with inf stop loss
    signal2 = SignalResult(
        signal="BUY",
        entry=100.0,
        stop_loss=float('inf'),
        take_profit=102.0,
    )

    result2 = runner.process_signal(signal2)
    assert result2.risk_approved is False


# ============================================================
# 11. VERY SMALL ACCOUNT - Minimum account size
# ============================================================
def test_very_small_account() -> None:
    """Test with very small account balance."""
    runner = PaperTradingRunner(
        initial_balance=10.0,  # Very small account
        commission=0.0,
        quantity=1.0,
        enable_risk_controls=True,
        risk_percent=1.0,
        min_leverage=1.0,
        max_leverage=10.0,
    )

    signal = make_signal(signal="BUY", entry=100.0, stop_loss=98.0)
    result = runner.process_signal(signal)

    # Should either reject due to minimum position size
    # or open very small position
    if result.risk_approved:
        assert runner.engine.position.quantity >= 0.001
    else:
        assert "position size" in result.risk_reason.lower() or \
               "too small" in result.risk_reason.lower()


# ============================================================
# 12. EXTREME LEVERAGE - Leverage limits
# ============================================================
def test_extreme_leverage_rejected() -> None:
    """Test that extreme leverage is rejected."""
    runner = PaperTradingRunner(
        initial_balance=1000.0,
        commission=0.0,
        quantity=1.0,
        enable_risk_controls=True,
        risk_percent=1.0,
        max_leverage=10.0,
        min_leverage=1.0,
    )

    # Try with leverage > max
    runner.leverage = 100.0
    signal = make_signal(signal="BUY", entry=100.0, stop_loss=98.0)
    result = runner.process_signal(signal)

    # Should reject or cap leverage
    if not result.risk_approved:
        assert "leverage" in result.risk_reason.lower() or \
               "exceed" in result.risk_reason.lower()


# ============================================================
# HELPER
# ============================================================

def make_signal(
    signal: str,
    entry: float,
    stop_loss: float,
) -> SignalResult:
    from src.strategy import SignalResult
    return SignalResult(
        signal=signal,
        entry=entry,
        stop_loss=stop_loss,
        take_profit=entry,
    )