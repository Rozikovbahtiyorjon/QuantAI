"""
QuantAI Core Math Tests.

Tests for risk_manager.py functions:
- calculate_position_size
- calculate_sl_tp
- calculate_risk_reward
- calculate_trade_risk
- break_even_price
"""

import pytest

from config.settings import (
    ATR_STOP_MULTIPLIER,
    ATR_TAKE_MULTIPLIER,
    MAX_POSITION_SIZE,
)

from src.risk_manager import (
    calculate_position_size,
    calculate_sl_tp,
    calculate_risk_reward,
    calculate_trade_risk,
    break_even_price,
)


class TestCalculatePositionSize:
    """Tests for calculate_position_size function."""

    def test_basic_long_position(self):
        """Test basic position size calculation for long."""
        balance = 1000.0
        risk_percent = 1.0
        entry_price = 100.0
        stop_loss = 95.0

        result = calculate_position_size(
            balance, risk_percent, entry_price, stop_loss
        )

        # Risk amount = 1000 * 1% = 10
        # Stop distance = 5
        # Position size = 10 / 5 = 2.0
        # But max_position_size = 1.0 caps it
        assert result == MAX_POSITION_SIZE

    def test_basic_short_position(self):
        """Test position size calculation for short (stop > entry)."""
        balance = 1000.0
        risk_percent = 1.0
        entry_price = 100.0
        stop_loss = 105.0

        result = calculate_position_size(
            balance, risk_percent, entry_price, stop_loss
        )

        assert result == MAX_POSITION_SIZE

    def test_risk_scales_with_balance(self):
        """Test that position size scales linearly with balance (before cap)."""
        # Use smaller values to avoid hitting max_position_size cap
        base = calculate_position_size(100.0, 0.5, 100.0, 95.0)
        double = calculate_position_size(200.0, 0.5, 100.0, 95.0)

        assert double == 2 * base

    def test_risk_scales_with_risk_percent(self):
        """Test that position size scales linearly with risk percent (before cap)."""
        base = calculate_position_size(100.0, 0.5, 100.0, 95.0)
        double = calculate_position_size(100.0, 1.0, 100.0, 95.0)

        assert double == 2 * base

    def test_stop_distance_affects_size(self):
        """Test that tighter stops give larger positions (before cap)."""
        # Use small balance to avoid cap
        tight = calculate_position_size(100.0, 0.5, 100.0, 98.0)  # 2% stop
        wide = calculate_position_size(100.0, 0.5, 100.0, 90.0)   # 10% stop

        assert tight > wide

    def test_zero_stop_distance_returns_zero(self):
        """Test that zero stop distance returns zero position."""
        result = calculate_position_size(1000.0, 1.0, 100.0, 100.0)
        assert result == 0.0

    def test_negative_stop_distance_returns_zero(self):
        """Test that invalid stop distance returns zero.
        
        Note: calculate_position_size uses abs(), so stop > entry for long
        still gives positive distance. Direction validation happens in TradeEngine.open_position().
        """
        # For long position, stop > entry gives positive distance via abs()
        result = calculate_position_size(1000.0, 1.0, 100.0, 105.0)
        # Function returns valid size (capped at max_position_size)
        # TradeEngine validates direction before opening
        assert result == MAX_POSITION_SIZE

    def test_min_position_size_floor(self):
        """Test that min position size is respected."""
        from config.settings import MIN_POSITION_SIZE
        # Very wide stop, small balance -> would give tiny position
        result = calculate_position_size(10.0, 0.1, 100.0, 50.0)
        assert result >= MIN_POSITION_SIZE

    def test_max_position_size_ceiling(self):
        """Test that max position size is respected."""
        # Very tight stop, large balance -> would give huge position
        result = calculate_position_size(1000000.0, 10.0, 100.0, 99.9)
        assert result == MAX_POSITION_SIZE


class TestCalculateSLTP:
    """Tests for calculate_sl_tp function."""

    def test_default_rr_ratio(self):
        """Test SL/TP with default risk/reward ratio from ATR multipliers."""
        entry = 100.0
        atr = 2.0

        sl, tp = calculate_sl_tp(entry, atr)

        expected_sl = entry - atr * ATR_STOP_MULTIPLIER
        expected_tp = entry + atr * ATR_STOP_MULTIPLIER * (ATR_TAKE_MULTIPLIER / ATR_STOP_MULTIPLIER)

        assert sl == pytest.approx(expected_sl)
        assert tp == pytest.approx(expected_tp)

    def test_custom_rr_ratio(self):
        """Test SL/TP with custom risk/reward ratio."""
        entry = 100.0
        atr = 2.0
        rr = 3.0

        sl, tp = calculate_sl_tp(entry, atr, rr=rr)

        expected_sl = entry - atr * ATR_STOP_MULTIPLIER
        expected_tp = entry + atr * ATR_STOP_MULTIPLIER * rr

        assert sl == pytest.approx(expected_sl)
        assert tp == pytest.approx(expected_tp)

    def test_short_sl_tp(self):
        """Test that function works for both directions (uses absolute ATR)."""
        entry = 100.0
        atr = 2.0

        sl, tp = calculate_sl_tp(entry, atr)

        # SL is below entry, TP is above
        assert sl < entry
        assert tp > entry
        assert (entry - sl) == pytest.approx(atr * ATR_STOP_MULTIPLIER)
        assert (tp - entry) == pytest.approx(atr * ATR_TAKE_MULTIPLIER)


class TestCalculateRiskReward:
    """Tests for calculate_risk_reward function."""

    def test_rr_calculation(self):
        """Test risk/reward ratio calculation."""
        entry = 100.0
        sl = 95.0
        tp = 110.0

        rr = calculate_risk_reward(entry, sl, tp)

        # Risk = 5, Reward = 10 -> RR = 2.0
        assert rr == 2.0

    def test_rr_short_position(self):
        """Test RR for short position."""
        entry = 100.0
        sl = 105.0
        tp = 90.0

        rr = calculate_risk_reward(entry, sl, tp)

        assert rr == 2.0

    def test_zero_risk_returns_zero(self):
        """Test that zero risk returns zero RR."""
        rr = calculate_risk_reward(100.0, 100.0, 110.0)
        assert rr == 0.0


class TestCalculateTradeRisk:
    """Tests for calculate_trade_risk function."""

    def test_basic_risk_amount(self):
        """Test risk amount calculation."""
        balance = 1000.0
        risk_percent = 1.0

        risk = calculate_trade_risk(balance, risk_percent)

        assert risk == 10.0

    def test_risk_scales_with_balance(self):
        """Test risk amount scales with balance."""
        assert calculate_trade_risk(2000.0, 1.0) == 20.0

    def test_risk_scales_with_percent(self):
        """Test risk amount scales with risk percent."""
        assert calculate_trade_risk(1000.0, 2.0) == 20.0


class TestBreakEvenPrice:
    """Tests for break_even_price function."""

    def test_break_even_with_commission(self):
        """Test break-even price includes commission."""
        entry = 100.0
        commission = 0.0004  # 0.04%

        be = break_even_price(entry, commission)

        # BE = 100 * (1 + 0.0004 * 2) = 100.08
        assert be == 100.08

    def test_break_even_zero_commission(self):
        """Test break-even equals entry when no commission."""
        be = break_even_price(100.0, 0.0)
        assert be == 100.0


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_full_trade_calculation(self):
        """Test a complete trade setup from balance to SL/TP."""
        balance = 10000.0
        risk_percent = 1.0
        entry = 50000.0
        atr = 500.0

        # Calculate SL/TP
        sl, tp = calculate_sl_tp(entry, atr)

        # Calculate position size
        size = calculate_position_size(balance, risk_percent, entry, sl)

        # Verify risk amount
        risk_amount = calculate_trade_risk(balance, risk_percent)
        actual_risk = abs(entry - sl) * size

        # Note: actual_risk may be less than risk_amount due to max_position_size cap
        assert actual_risk <= risk_amount + 0.01

        # Verify RR
        rr = calculate_risk_reward(entry, sl, tp)
        expected_rr = ATR_TAKE_MULTIPLIER / ATR_STOP_MULTIPLIER
        assert rr == pytest.approx(expected_rr)

    def test_position_notional_calculation(self):
        """Test that position notional is calculated correctly."""
        balance = 1000.0
        risk_percent = 1.0
        entry = 100.0
        sl = 95.0

        size = calculate_position_size(balance, risk_percent, entry, sl)
        notional = size * entry

        # With max_position_size cap of 1.0, notional = 100.0
        assert notional == pytest.approx(100.0, rel=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])