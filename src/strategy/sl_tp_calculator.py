"""
QuantAI SL/TP Calculator

Regime-adaptive Stop Loss and Take Profit calculation.

Adapts multipliers based on:
- Volatility regime (high/normal/low)
- Trend alignment (bullish/bearish/neutral)
- Market microstructure (VPIN, liquidation levels)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from config.settings import settings


@dataclass
class SLTPConfig:
    """Configuration for SL/TP calculation."""
    
    # Base ATR multipliers
    base_sl_multiplier: float = 1.5
    base_tp_multiplier: float = 3.0
    base_trailing_multiplier: float = 2.0
    
    # Volatility regime adjustments
    high_vol_sl_mult: float = 1.2   # Wider stops in high vol
    high_vol_tp_mult: float = 0.8   # Closer targets in high vol
    low_vol_sl_mult: float = 0.8    # Tighter stops in low vol
    low_vol_tp_mult: float = 1.2    # Further targets in low vol
    
    # Trend alignment adjustments
    trend_aligned_sl_mult: float = 0.9    # Tighter when aligned
    trend_aligned_tp_mult: float = 1.1    # Further when aligned
    trend_counter_sl_mult: float = 1.2    # Wider against trend
    trend_counter_tp_mult: float = 0.9    # Closer against trend
    
    # Microstructure
    vpin_toxic_sl_mult: float = 1.3
    near_liquidation_sl_mult: float = 1.5
    
    # Min/Max bounds
    min_sl_atr_mult: float = 0.5
    max_sl_atr_mult: float = 5.0
    min_tp_atr_mult: float = 1.0
    max_tp_atr_mult: float = 10.0
    
    @classmethod
    def from_settings(cls) -> "SLTPConfig":
        return cls(
            base_sl_multiplier=getattr(settings.indicators, "atr_stop_multiplier", 1.5),
            base_tp_multiplier=getattr(settings.indicators, "atr_take_multiplier", 3.0),
            base_trailing_multiplier=getattr(settings.indicators, "trailing_stop_multiplier", 2.0),
        )


@dataclass
class SLTPResult:
    """Result of SL/TP calculation."""
    
    stop_loss: float
    take_profit: float
    trailing_stop: float
    
    # Multipliers used
    sl_atr_multiplier: float
    tp_atr_multiplier: float
    trailing_atr_multiplier: float
    
    # Method
    method: str = "atr_adaptive"
    reason: str = ""


class SLTPCalculator:
    """
    Calculates adaptive Stop Loss and Take Profit levels.
    
    Adapts based on:
    1. Volatility regime (high/normal/low via ATR percentile)
    2. Trend alignment (bullish/bearish/neutral)
    3. Microstructure (VPIN, liquidation proximity)
    """
    
    def __init__(self, config: SLTPConfig | None = None):
        self.config = config or SLTPConfig.from_settings()
        
    def calculate(
        self,
        entry_price: float,
        atr: float,
        signal: Literal["BUY", "SELL"],
        volatility_regime: Literal["high", "normal", "low"] = "normal",
        trend_alignment: Literal["bullish", "bearish", "neutral"] = "neutral",
        vpin: float = 0.0,
        near_resistance: bool = False,
        near_support: bool = False,
        liquidation_signal = None,
    ) -> SLTPResult:
        """
        Calculate adaptive SL/TP levels.
        
        Args:
            entry_price: Entry price
            atr: Current ATR value
            signal: BUY or SELL
            volatility_regime: high/normal/low
            trend_alignment: bullish/bearish/neutral
            vpin: VPIN toxicity (0-1)
            near_resistance: True if near resistance cluster
            near_support: True if near support cluster
        """
        # Start with base multipliers
        sl_mult = self.config.base_sl_multiplier
        tp_mult = self.config.base_tp_multiplier
        trail_mult = self.config.base_trailing_multiplier
        
        reasons = []
        
        # --- Volatility Regime Adjustment ---
        if volatility_regime == "high":
            sl_mult *= self.config.high_vol_sl_mult
            tp_mult *= self.config.high_vol_tp_mult
            reasons.append(f"high_vol(SL×{self.config.high_vol_sl_mult}, TP×{self.config.high_vol_tp_mult})")
        elif volatility_regime == "low":
            sl_mult *= self.config.low_vol_sl_mult
            tp_mult *= self.config.low_vol_tp_mult
            reasons.append(f"low_vol(SL×{self.config.low_vol_sl_mult}, TP×{self.config.low_vol_tp_mult})")
        
        # --- Trend Alignment Adjustment ---
        is_aligned = (
            (signal == "BUY" and trend_alignment == "bullish") or
            (signal == "SELL" and trend_alignment == "bearish")
        )
        is_counter = (
            (signal == "BUY" and trend_alignment == "bearish") or
            (signal == "SELL" and trend_alignment == "bullish")
        )
        
        if is_aligned:
            sl_mult *= self.config.trend_aligned_sl_mult
            tp_mult *= self.config.trend_aligned_tp_mult
            reasons.append(f"trend_aligned(SL×{self.config.trend_aligned_sl_mult}, TP×{self.config.trend_aligned_tp_mult})")
        elif is_counter:
            sl_mult *= self.config.trend_counter_sl_mult
            tp_mult *= self.config.trend_counter_tp_mult
            reasons.append(f"trend_counter(SL×{self.config.trend_counter_sl_mult}, TP×{self.config.trend_counter_tp_mult})")
        
        # --- Microstructure Adjustments ---
        if vpin >= 0.8:  # Toxic
            sl_mult *= self.config.vpin_toxic_sl_mult
            reasons.append(f"vpin_toxic(SL×{self.config.vpin_toxic_sl_mult})")
            
        # --- Liquidation-based Adjustments ---
        if liquidation_signal is not None:
            # Get liquidation data
            liq_imbalance = liquidation_signal.imbalance  # -1 to 1, positive = short liq dominant
            liq_intensity = liquidation_signal.intensity  # volume relative to baseline
            liq_context = liquidation_signal.context
            
            # High intensity liquidations increase risk
            if liq_intensity > 2.0:
                sl_mult *= 1.3
                reasons.append(f"liq_high_intensity(SL×1.3, intensity={liq_intensity:.1f})")
            elif liq_intensity > 1.5:
                sl_mult *= 1.15
                reasons.append(f"liq_elevated_intensity(SL×1.15, intensity={liq_intensity:.1f})")
            
            # Adjust based on imbalance direction relative to signal
            if signal == "BUY":
                # Long liquidations (imbalance < 0) are supportive for longs
                if liq_imbalance < -0.3:
                    tp_mult *= 1.1  # Extend target - liquidations provide fuel
                    reasons.append(f"liq_long_support(TP×1.1, imbalance={liq_imbalance:.2f})")
                # Short liquidations (imbalance > 0) are risky for longs
                elif liq_imbalance > 0.3:
                    sl_mult *= 1.2  # Widen stop - short liqs can cause cascade down
                    reasons.append(f"liq_short_risk(SL×1.2, imbalance={liq_imbalance:.2f})")
            elif signal == "SELL":
                # Short liquidations (imbalance > 0) are supportive for shorts
                if liq_imbalance > 0.3:
                    tp_mult *= 1.1
                    reasons.append(f"liq_short_support(TP×1.1, imbalance={liq_imbalance:.2f})")
                # Long liquidations (imbalance < 0) are risky for shorts
                elif liq_imbalance < -0.3:
                    sl_mult *= 1.2
                    reasons.append(f"liq_long_risk(SL×1.2, imbalance={liq_imbalance:.2f})")
            
            # Context-specific adjustments
            if liq_context == "LONG_LIQUIDATION_DOMINANT":
                if signal == "SELL":
                    sl_mult *= 1.15
                    reasons.append(f"liq_long_dom_short_risk(SL×1.15)")
            elif liq_context == "SHORT_LIQUIDATION_DOMINANT":
                if signal == "BUY":
                    sl_mult *= 1.15
                    reasons.append(f"liq_short_dom_long_risk(SL×1.15)")
        
        # --- Existing Microstructure Adjustments ---
        if vpin >= 0.8:  # Toxic
            sl_mult *= self.config.vpin_toxic_sl_mult
            reasons.append(f"vpin_toxic(SL×{self.config.vpin_toxic_sl_mult})")
        
        # --- Apply Bounds ---
        sl_mult = max(self.config.min_sl_atr_mult, min(self.config.max_sl_atr_mult, sl_mult))
        tp_mult = max(self.config.min_tp_atr_mult, min(self.config.max_tp_atr_mult, tp_mult))
        
        # --- Calculate Levels ---
        if signal == "BUY":
            stop_loss = entry_price - atr * sl_mult
            take_profit = entry_price + atr * tp_mult
            trailing_stop = entry_price - atr * trail_mult
        else:  # SELL
            stop_loss = entry_price + atr * sl_mult
            take_profit = entry_price - atr * tp_mult
            trailing_stop = entry_price + atr * trail_mult
        
        # Round to reasonable precision
        stop_loss = round(stop_loss, 2)
        take_profit = round(take_profit, 2)
        trailing_stop = round(trailing_stop, 2)
        
        reason_str = " | ".join(reasons) if reasons else "base"
        
        return SLTPResult(
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_stop=trailing_stop,
            sl_atr_multiplier=round(sl_mult, 2),
            tp_atr_multiplier=round(tp_mult, 2),
            trailing_atr_multiplier=round(trail_mult, 2),
            method="atr_adaptive",
            reason=reason_str,
        )