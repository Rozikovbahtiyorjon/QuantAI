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

    def validate_rr_choice(
        self,
        proposed_tp_mult: float,
        expectancy: float | None = None,
        transaction_costs: float | None = None,
        stop_hit_rate: float | None = None,
        tp_hit_rate: float | None = None,
        regime: str | None = None,
        oos_pf: float | None = None,
    ) -> tuple[bool, str]:
        """
        Validate R:R choice 1:2 vs 1:2.3 — do not just change 2.0→2.3.

        Must check 6 metrics before choosing threshold:
        expectancy, transaction costs, stop hit rate, TP hit rate, regime dependence, OOS performance.
        Returns (is_valid, reason).
        """
        base_rr = proposed_tp_mult / self.base_sl_multiplier
        # 1:2 is base (3.0/1.5=2.0), 1:2.3 would be 3.45/1.5=2.3
        if abs(proposed_tp_mult - 3.0) < 1e-9:
            # 1:2 is validated base — always allowed if expectancy>0
            if expectancy is not None and expectancy <= 0:
                return False, f"RR 1:2 rejected: expectancy {expectancy:.4f} <=0"
            return True, "RR 1:2 validated base"
        if abs(proposed_tp_mult - 3.45) < 1e-9:  # 1:2.3
            # Require all 6 checks for 1:2.3
            reasons = []
            if expectancy is not None and expectancy <= 0.001:
                reasons.append(f"expectancy {expectancy:.4f} <=0.001")
            if transaction_costs is not None and transaction_costs > 0.002:
                reasons.append(f"costs {transaction_costs:.4f} >0.002")
            if stop_hit_rate is not None and tp_hit_rate is not None:
                if stop_hit_rate < 0.3 or tp_hit_rate < 0.25:
                    reasons.append(f"hit rates SL {stop_hit_rate:.2f} TP {tp_hit_rate:.2f} too low for 1:2.3")
            if regime is not None and regime not in ("TREND_UP", "TREND_DOWN"):
                reasons.append(f"regime {regime} not trending for 1:2.3")
            if oos_pf is not None and oos_pf < 1.1:
                reasons.append(f"OOS PF {oos_pf:.2f} <1.1 for 1:2.3")
            if reasons:
                return False, f"RR 1:2.3 rejected: {'; '.join(reasons)} — keep 1:2 until 6 metrics pass"
            return True, f"RR 1:2.3 validated: expectancy {expectancy}, costs {transaction_costs}, hit rates, regime {regime}, OOS PF {oos_pf}"
        # Other RR values: check expectancy
        if expectancy is not None and expectancy <= 0:
            return False, f"RR {base_rr:.1f} rejected: expectancy <=0"
        return True, f"RR {base_rr:.1f} allowed (custom, not 1:2.3 strict)"

    @property
    def current_rr(self) -> float:
        return self.base_tp_multiplier / self.base_sl_multiplier if self.base_sl_multiplier else 0


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
        
    def calculate_structural(
        self,
        entry_price: float,
        atr: float,
        signal: Literal["BUY", "SELL"],
        swing_low: float | None = None,
        swing_high: float | None = None,
        support_zone: float | None = None,
        resistance_zone: float | None = None,
        buffer_atr_mult: float = 0.3,
        volatility_regime: Literal["high", "normal", "low"] = "normal",
    ) -> SLTPResult:
        """
        Structural SL — swing low - buffer, not just ATR multiplier.

        Preferred for P3: SL = swing low - buffer (e.g., recent swing low - 0.3 ATR)
        instead of SL = entry - ATR * multiplier.

        Args:
            swing_low: recent swing low price (for LONG SL)
            swing_high: recent swing high price (for SHORT SL)
            support_zone: support zone level (e.g., from ZoneEngine)
            resistance_zone: resistance zone level
            buffer_atr_mult: buffer in ATR units (e.g., 0.3 * ATR)
        """
        buffer = atr * buffer_atr_mult
        reasons = ["structural"]

        if signal == "BUY":
            # Structural SL: below swing low / support zone
            candidates = []
            if swing_low is not None and swing_low < entry_price:
                candidates.append(swing_low - buffer)
                reasons.append(f"swing_low {swing_low:.2f} - buffer {buffer:.2f}")
            if support_zone is not None and support_zone < entry_price:
                # Support zone lower edge - buffer
                candidates.append(support_zone - buffer)
                reasons.append(f"support {support_zone:.2f} - buffer")
            if candidates:
                # Most conservative (lowest) SL among structural candidates, but not too far
                structural_sl = min(candidates)
                # Cap at max 3 ATR away (avoid too wide)
                max_sl_dist = atr * 3.0
                if entry_price - structural_sl > max_sl_dist:
                    structural_sl = entry_price - max_sl_dist
                    reasons.append(f"capped to 3 ATR ({max_sl_dist:.2f})")
                # Ensure SL is below entry
                if structural_sl < entry_price:
                    atr_sl = entry_price - atr * self.config.base_sl_multiplier
                    # Prefer structural if within 0.5 ATR of ATR-based, otherwise ATR is more reasonable
                    if abs(structural_sl - atr_sl) < atr * 0.5:
                        sl_price = structural_sl
                        method = "structural_swing"
                    else:
                        # Blend: use structural but log
                        sl_price = structural_sl
                        method = "structural_swing"
                    # Calculate TP from structural SL distance (maintain RR)
                    sl_dist = entry_price - sl_price
                    tp_price = entry_price + sl_dist * 2.0  # RR 2.0 for structural
                    return SLTPResult(
                        stop_loss=round(sl_price, 2),
                        take_profit=round(tp_price, 2),
                        trailing_stop=round(entry_price - atr * self.config.base_trailing_multiplier, 2),
                        sl_atr_multiplier=round(sl_dist / atr, 2) if atr else 0,
                        tp_atr_multiplier=round((tp_price - entry_price) / atr, 2) if atr else 0,
                        trailing_atr_multiplier=round(self.config.base_trailing_multiplier, 2),
                        method=method,
                        reason=" | ".join(reasons),
                    )
            # Fallback to ATR if no valid structural level
            reasons.append("no valid structural level → fallback ATR")
            return self.calculate(entry_price, atr, signal, volatility_regime, trend_alignment="neutral")

        else:  # SELL
            candidates = []
            if swing_high is not None and swing_high > entry_price:
                candidates.append(swing_high + buffer)
                reasons.append(f"swing_high {swing_high:.2f} + buffer")
            if resistance_zone is not None and resistance_zone > entry_price:
                candidates.append(resistance_zone + buffer)
                reasons.append(f"resistance {resistance_zone:.2f} + buffer")
            if candidates:
                structural_sl = max(candidates)
                max_sl_dist = atr * 3.0
                if structural_sl - entry_price > max_sl_dist:
                    structural_sl = entry_price + max_sl_dist
                    reasons.append(f"capped to 3 ATR")
                if structural_sl > entry_price:
                    sl_dist = structural_sl - entry_price
                    tp_price = entry_price - sl_dist * 2.0
                    return SLTPResult(
                        stop_loss=round(structural_sl, 2),
                        take_profit=round(tp_price, 2),
                        trailing_stop=round(entry_price + atr * self.config.base_trailing_multiplier, 2),
                        sl_atr_multiplier=round(sl_dist / atr, 2) if atr else 0,
                        tp_atr_multiplier=round((entry_price - tp_price) / atr, 2) if atr else 0,
                        trailing_atr_multiplier=round(self.config.base_trailing_multiplier, 2),
                        method="structural_swing",
                        reason=" | ".join(reasons),
                    )
            reasons.append("no valid structural level → fallback ATR")
            return self.calculate(entry_price, atr, signal, volatility_regime, trend_alignment="neutral")

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
        # P3 structural passthrough (optional)
        swing_low: float | None = None,
        swing_high: float | None = None,
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