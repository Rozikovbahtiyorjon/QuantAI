"""
====================================================
QuantAI Professional
Kelly Criterion Position Sizing
====================================================

Kelly Criterion position sizing for optimal capital growth.
Maximizes expected log wealth (log utility).

Features:
- Full Kelly, Half Kelly, Quarter Kelly fractions
- Win rate / payoff ratio based calculation
- Volatility-adjusted Kelly
- Drawdown-aware Kelly scaling
- Multi-asset Kelly optimization (future)

====================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


class KellyFraction(Enum):
    """Kelly fraction presets."""
    FULL = 1.0      # Full Kelly (maximum growth, high volatility)
    HALF = 0.5      # Half Kelly (recommended balance)
    QUARTER = 0.25  # Quarter Kelly (conservative)
    TENTH = 0.1     # Tenth Kelly (very conservative)


@dataclass(frozen=True)
class KellyResult:
    """Result of Kelly Criterion calculation."""
    kelly_fraction: float          # Optimal fraction of capital to risk
    suggested_fraction: float      # Fraction after applying safety multiplier
    risk_amount: float             # Dollar amount to risk
    position_size: float           # Position size in contracts/coins
    position_notional: float       # Position notional value
    expected_growth_rate: float    # Expected log growth rate per trade
    risk_of_ruin: float            # Probability of ruin (approximate)
    max_drawdown_estimate: float   # Estimated max drawdown (approx)
    metadata: dict = None


class KellySizer:
    """
    Kelly Criterion Position Sizer.
    
    The Kelly Criterion maximizes the expected logarithm of wealth,
    which is equivalent to maximizing the geometric mean of returns.
    
    Basic formula: f* = (bp - q) / b = (p * b - q) / b
    Where:
    - f* = optimal fraction of capital to bet
    - b = net odds received on the bet (payoff ratio)
    - p = probability of winning
    - q = probability of losing = 1 - p
    
    For trading:
    - b = avg_win / avg_loss (payoff ratio)
    - p = win_rate
    - q = 1 - win_rate
    """

    def __init__(
        self,
        kelly_fraction: KellyFraction = KellyFraction.HALF,
        min_win_rate: float = 0.30,      # Minimum win rate for Kelly to be positive
        max_kelly_fraction: float = 0.5,  # Cap on Kelly fraction (safety)
        min_payoff_ratio: float = 1.5,    # Minimum payoff ratio (avg_win/avg_loss)
        max_position_pct: float = 0.25,   # Max position as % of equity (safety cap)
    ):
        """
        Args:
            kelly_fraction: Kelly fraction to apply (Full/Half/Quarter/Tenth)
            min_win_rate: Minimum win rate for Kelly to be valid
            max_kelly_fraction: Hard cap on Kelly fraction
            min_payoff_ratio: Minimum payoff ratio for Kelly to apply
            max_position_pct: Maximum position size as fraction of equity
        """
        if not 0 < kelly_fraction.value <= 1:
            raise ValueError("kelly_fraction must be in (0, 1]")
        if not 0 < min_win_rate < 1:
            raise ValueError("min_win_rate must be in (0, 1)")
        if max_kelly_fraction <= 0 or max_kelly_fraction > 1:
            raise ValueError("max_kelly_fraction must be in (0, 1]")
        if min_payoff_ratio <= 1:
            raise ValueError("min_payoff_ratio must be > 1")
        if not 0 < max_position_pct <= 1:
            raise ValueError("max_position_pct must be in (0, 1]")

        self.kelly_fraction = kelly_fraction
        self.min_win_rate = min_win_rate
        self.max_kelly_fraction = max_kelly_fraction
        self.min_payoff_ratio = min_payoff_ratio
        self.max_position_pct = max_position_pct

    def calculate(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        equity: float,
        entry_price: float,
        stop_loss: float,
        leverage: float = 1.0,
        volatility_adjustment: float = 1.0,
    ) -> KellyResult:
        """
        Calculate Kelly-optimal position size.
        
        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average winning trade profit (absolute)
            avg_loss: Average losing trade loss (absolute)
            equity: Current account equity
            entry_price: Planned entry price
            stop_loss: Stop loss price
            leverage: Leverage to use
            volatility_adjustment: Volatility adjustment factor (<1 reduces size)
            
        Returns:
            KellyResult with optimal position size and diagnostics
        """
        # Validate inputs
        if not 0 < win_rate < 1:
            return self._invalid_result(equity, "Win rate must be in (0, 1)")
        
        if avg_loss <= 0:
            return self._invalid_result(equity, "Average loss must be positive")
        
        if entry_price <= 0 or stop_loss <= 0:
            return self._invalid_result(equity, "Entry and stop loss must be positive")
        
        if leverage <= 0:
            return self._invalid_result(equity, "Leverage must be positive")

        # Calculate payoff ratio (b)
        payoff_ratio = avg_win / avg_loss
        
        if payoff_ratio < self.min_payoff_ratio:
            return self._invalid_result(
                equity, 
                f"Payoff ratio {payoff_ratio:.2f} below minimum {self.min_payoff_ratio}"
            )

        # Kelly formula: f* = (p * b - q) / b
        # where b = payoff_ratio, p = win_rate, q = 1 - win_rate
        p = win_rate
        q = 1 - win_rate
        b = payoff_ratio
        
        kelly_raw = (p * b - (1 - p)) / b
        
        # Kelly is only valid if positive
        if kelly_raw <= 0:
            return self._invalid_result(
                equity,
                f"Kelly fraction <= 0 (win_rate={win_rate:.2%}, payoff={payoff_ratio:.2f})"
            )

        # Apply Kelly fraction (Half Kelly = 0.5, etc.)
        kelly_fraction = kelly_raw * self.kelly_fraction.value
        
        # Cap at maximum
        kelly_fraction = min(kelly_fraction, self.max_kelly_fraction)
        
        # Apply volatility adjustment
        kelly_fraction *= volatility_adjustment
        
        # Cap at maximum position percentage
        kelly_fraction = min(kelly_fraction, self.max_position_pct)

        # Calculate position size
        risk_amount = equity * kelly_fraction
        stop_distance = abs(entry_price - stop_loss)
        
        if stop_distance <= 0:
            return self._invalid_result(equity, "Stop distance must be positive")

        position_size = risk_amount / stop_distance
        position_notional = position_size * entry_price

        # Apply leverage
        if leverage != 1.0:
            # With leverage, we can control more notional with same margin
            # But risk amount stays the same (we risk same dollar amount)
            margin_required = position_notional / leverage
        else:
            margin_required = position_notional

        # Expected growth rate (Kelly criterion)
        # g = p * log(1 + b*f) + q * log(1 - f)
        f = kelly_fraction
        expected_growth = (
            win_rate * math.log(1 + payoff_ratio * kelly_fraction) +
            (1 - win_rate) * math.log(1 - kelly_fraction)
        )

        # Risk of ruin approximation (for fractional Kelly)
        # For full Kelly: ruin probability = 1
        # For half Kelly: ruin probability ~ 0.5
        # Approximation: P(ruin) ≈ (1 - f)^n for small f
        if kelly_fraction >= 1.0:
            risk_of_ruin = 1.0
        else:
            risk_of_ruin = (1 - kelly_fraction) ** 100  # Rough approximation

        # Max drawdown estimate
        # Approximate: max drawdown ≈ 2 * f * σ (simplified)
        # Using win/loss stats
        avg_trade = win_rate * avg_win - (1 - win_rate) * avg_loss
        trade_std = math.sqrt(
            win_rate * (avg_win - avg_win * win_rate) ** 2 +
            (1 - win_rate) * (avg_loss + avg_loss * (1 - win_rate)) ** 2
        )
        # Rough approximation
        max_drawdown_estimate = min(1.0, 2 * kelly_fraction * 2.0)  # Conservative estimate

        return KellyResult(
            kelly_fraction=kelly_raw,
            suggested_fraction=kelly_fraction,
            risk_amount=equity * kelly_fraction,
            position_size=round(risk_amount / max(0.0001, abs(entry_price - (entry_price - stop_loss))), 8),
            position_notional=position_size * entry_price,
            expected_growth_rate=expected_growth,
            risk_of_ruin=risk_of_ruin,
            max_drawdown_estimate=max_drawdown_estimate,
            metadata={
                "win_rate": win_rate,
                "payoff_ratio": payoff_ratio,
                "kelly_raw": kelly_raw,
                "kelly_fraction_applied": self.kelly_fraction.value,
                "volatility_adjustment": 1.0,
                "leverage": 1.0,
            }
        )

    def _invalid_result(self, equity: float, reason: str) -> KellyResult:
        """Return a zero-result for invalid inputs."""
        return KellyResult(
            kelly_fraction=0.0,
            suggested_fraction=0.0,
            risk_amount=0.0,
            position_size=0.0,
            position_notional=0.0,
            expected_growth_rate=0.0,
            risk_of_ruin=1.0,
            max_drawdown_estimate=1.0,
            metadata={"error": reason, "valid": False},
        )

    def calculate_from_stats(
        self,
        trades: list[dict],  # List of {"pnl": float, "side": "long"/"short"}
        equity: float,
        entry_price: float,
        stop_loss: float,
        leverage: float = 1.0,
    ) -> KellyResult:
        """
        Calculate Kelly from trade history.
        
        Args:
            trades: List of trade dicts with 'pnl' and 'side'
            equity: Current equity
            entry_price: Current entry price
            stop_loss: Stop loss price
            
        Returns:
            KellyResult
        """
        if not trades:
            return self._invalid_result(0, "No trades provided")

        wins = [t for t in trades if t.get("pnl", 0) > 0]
        losses = [t for t in trades if t.get("pnl", 0) <= 0]

        if not wins or not losses:
            return self._invalid_result(0, "Need both winning and losing trades")

        win_rate = len(wins) / len(trades)
        avg_win = np.mean([t["pnl"] for t in wins])
        avg_loss = abs(np.mean([t["pnl"] for t in losses]))

        # Estimate equity from trade history if not provided
        equity = max(equity, sum(t["pnl"] for t in trades) + 1000)

        return self.calculate(
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            equity=equity,
            entry_price=0,  # Will be set by caller
            stop_loss=0,
            leverage=1.0,
        )


class VolatilityAdjustedKellySizer(KellySizer):
    """
    Kelly Sizer with volatility-based adjustment.
    
    Reduces position size when volatility is high,
    increases when volatility is low.
    """

    def __init__(
        self,
        kelly_fraction: KellyFraction = KellyFraction.HALF,
        vol_lookback: int = 20,
        target_vol: float = 0.15,  # 15% annualized target volatility
    ):
        super().__init__(kelly_fraction=kelly_fraction)
        self.vol_lookback = vol_lookback
        self.target_vol = target_vol

    def calculate_with_volatility(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        equity: float,
        entry_price: float,
        stop_loss: float,
        current_vol: float,  # Current annualized volatility
        leverage: float = 1.0,
    ) -> KellyResult:
        """
        Calculate Kelly with volatility adjustment.
        
        Volatility adjustment = target_vol / current_vol
        (capped at reasonable bounds)
        """
        if current_vol <= 0:
            vol_adjustment = 1.0
        else:
            vol_adjustment = min(2.0, max(0.25, self.target_vol / current_vol))
        
        return self.calculate(
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            equity=equity,
            entry_price=entry_price,
            stop_loss=stop_loss,
            leverage=1.0,
            volatility_adjustment=vol_adjustment,
        )


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "KellyFraction",
    "KellyResult",
    "KellySizer",
    "VolatilityAdjustedKellySizer",
]