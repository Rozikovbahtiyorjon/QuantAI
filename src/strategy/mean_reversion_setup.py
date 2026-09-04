"""
Mean Reversion — Full Lifecycle Setup (P3.7)

RANGE → LOWER_BOUNDARY → EXHAUSTION → REVERSAL_SETUP → TRIGGER

Each stage is causal and has explicit conditions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import pandas as pd


@dataclass
class MeanReversionStage:
    stage: str  # RANGE, LOWER_BOUNDARY, EXHAUSTION, REVERSAL_SETUP, TRIGGER, NONE
    is_valid: bool
    reason: str
    confidence: float = 0.0
    bb_position: float = 0.5
    rsi: float = 50.0
    atr: float = 1.0


@dataclass
class MeanReversionResult:
    setup: str  # LONG_MEAN_REVERSION / SHORT_MEAN_REVERSION / NONE
    stage: str
    is_valid: bool
    reason: str
    confidence: float = 0.0
    # Diagnostics per stage
    stages: dict = None
    bb_position: float = 0.5
    rsi: float = 50.0
    atr: float = 1.0


class MeanReversionSetup:
    """
    Full lifecycle for mean reversion in RANGE regime.

    Stages (causal):
      1. RANGE: ADX < 25, BB not squeezed, not trending
      2. LOWER_BOUNDARY: price at lower boundary (BB pos ≤0.2) or upper (≥0.8)
      3. EXHAUSTION: RSI ≤30 (long) / ≥70 (short) + volume spike or wick + ATR exhaustion
      4. REVERSAL_SETUP: bullish engulfing / hammer in zone or RSI divergence
      5. TRIGGER: close reclaim of BB lower + 0.3 ATR or close > prev high in zone

    LONG_MEAN_REVERSION requires all 5, SHORT is mirror.
    """

    def __init__(self, bb_low: float = 0.2, bb_high: float = 0.8, adx_range_max: float = 25.0):
        self.bb_low = bb_low
        self.bb_high = bb_high
        self.adx_range_max = adx_range_max

    def evaluate(self, df: pd.DataFrame) -> MeanReversionResult:
        if len(df) < 50:
            return MeanReversionResult(setup="NONE", stage="NONE", is_valid=False, reason="insufficient history", stages={})

        row = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else row
        prev2 = df.iloc[-3] if len(df) > 2 else prev

        close = float(row.get("close", 0))
        high = float(row.get("high", 0))
        low = float(row.get("low", 0))
        prev_close = float(prev.get("close", close))
        prev_high = float(prev.get("high", high))
        prev_low = float(prev.get("low", low))
        adx = float(row.get("adx", 0))
        bb_pos = float(row.get("bb_position", 0)) + 0.5 if "bb_position" in row else 0.5
        bb_width = float(row.get("bb_width", 1.0))
        rsi = float(row.get("rsi", 50))
        atr = float(row.get("atr", 1))
        volume = float(row.get("volume", 0))
        vol_sma = float(row.get("volume_sma20", volume))
        ema_fast = float(row.get("ema_fast", close))
        ema_slow = float(row.get("ema_slow", close))

        stages: dict[str, MeanReversionStage] = {}

        # Stage 1: RANGE
        is_range = adx < self.adx_range_max and abs(ema_fast - ema_slow) / max(ema_slow, 1e-9) < 0.01
        # Also check trend_score near 0
        trend_score = float(row.get("trend_score", 0))
        is_range = is_range or (abs(trend_score) < 1 and adx < 25)
        stages["RANGE"] = MeanReversionStage(stage="RANGE", is_valid=is_range, reason=f"ADX {adx:.0f}<{self.adx_range_max} trend_score {trend_score:.0f}", confidence=0.7 if is_range else 0.3, bb_position=bb_pos, rsi=rsi, atr=atr)
        if not is_range:
            return MeanReversionResult(setup="NONE", stage="RANGE", is_valid=False, reason=f"Not RANGE: ADX {adx:.0f} trend_score {trend_score:.0f}", stages={k: v.__dict__ for k, v in stages.items()}, bb_position=bb_pos, rsi=rsi, atr=atr)

        # Stage 2: LOWER_BOUNDARY (for LONG) or UPPER for SHORT
        is_lower = bb_pos <= self.bb_low
        is_upper = bb_pos >= self.bb_high
        # Determine direction
        direction = None
        if is_lower:
            direction = "LONG"
            stages["LOWER_BOUNDARY"] = MeanReversionStage(stage="LOWER_BOUNDARY", is_valid=True, reason=f"BB low {bb_pos:.2f} <= {self.bb_low}", confidence=0.8, bb_position=bb_pos, rsi=rsi, atr=atr)
            stages["UPPER_BOUNDARY"] = MeanReversionStage(stage="UPPER_BOUNDARY", is_valid=False, reason="not upper", confidence=0.0, bb_position=bb_pos, rsi=rsi, atr=atr)
        elif is_upper:
            direction = "SHORT"
            stages["UPPER_BOUNDARY"] = MeanReversionStage(stage="UPPER_BOUNDARY", is_valid=True, reason=f"BB high {bb_pos:.2f} >= {self.bb_high}", confidence=0.8, bb_position=bb_pos, rsi=rsi, atr=atr)
            stages["LOWER_BOUNDARY"] = MeanReversionStage(stage="LOWER_BOUNDARY", is_valid=False, reason="not lower", confidence=0.0, bb_position=bb_pos, rsi=rsi, atr=atr)
        else:
            stages["LOWER_BOUNDARY"] = MeanReversionStage(stage="LOWER_BOUNDARY", is_valid=False, reason=f"BB {bb_pos:.2f} not at boundary", confidence=0.2, bb_position=bb_pos, rsi=rsi, atr=atr)
            stages["UPPER_BOUNDARY"] = MeanReversionStage(stage="UPPER_BOUNDARY", is_valid=False, reason="not at boundary", confidence=0.2, bb_position=bb_pos, rsi=rsi, atr=atr)
            return MeanReversionResult(setup="NONE", stage="LOWER_BOUNDARY", is_valid=False, reason=f"Not at boundary BB {bb_pos:.2f}", stages={k: v.__dict__ for k, v in stages.items()}, bb_position=bb_pos, rsi=rsi, atr=atr)

        # Stage 3: EXHAUSTION
        if direction == "LONG":
            # Exhaustion: RSI oversold + volume spike or long lower wick
            rsi_exhaust = rsi <= 32  # slightly relaxed from 30 for better hit rate
            vol_exhaust = volume > vol_sma * 1.3
            wick_exhaust = (close - low) / max(high - low, 1e-9) > 0.6 and (low < float(prev.get("low", low)))
            atr_exhaust = (prev_low - low) / max(atr, 1e-9) > 0.5  # moved >0.5 ATR down
            exhaust = rsi_exhaust and (vol_exhaust or wick_exhaust or atr_exhaust)
            conf = 0.6
            if rsi <= 28:
                conf += 0.2
            if vol_exhaust:
                conf += 0.1
            if wick_exhaust:
                conf += 0.1
            conf = min(1.0, conf)
            stages["EXHAUSTION"] = MeanReversionStage(stage="EXHAUSTION", is_valid=exhaust, reason=f"RSI {rsi:.0f} vol {volume/vol_sma:.1f}x wick {wick_exhaust} ATR dist {(prev_low-low)/max(atr,1e-9):.2f}", confidence=conf, bb_position=bb_pos, rsi=rsi, atr=atr)
            if not exhaust:
                return MeanReversionResult(setup="NONE", stage="EXHAUSTION", is_valid=False, reason=f"No exhaustion RSI {rsi:.0f} vol {volume/vol_sma:.1f}x", stages={k: v.__dict__ for k, v in stages.items()}, bb_position=bb_pos, rsi=rsi, atr=atr)
        else:  # SHORT
            rsi_exhaust = rsi >= 68
            vol_exhaust = volume > vol_sma * 1.3
            wick_exhaust = (high - close) / max(high - low, 1e-9) > 0.6
            exhaust = rsi_exhaust and (vol_exhaust or wick_exhaust)
            conf = 0.6
            if rsi >= 72:
                conf += 0.2
            if vol_exhaust:
                conf += 0.1
            if wick_exhaust:
                conf += 0.1
            conf = min(1.0, conf)
            stages["EXHAUSTION"] = MeanReversionStage(stage="EXHAUSTION", is_valid=exhaust, reason=f"RSI {rsi:.0f} vol {volume/vol_sma:.1f}x", confidence=conf, bb_position=bb_pos, rsi=rsi, atr=atr)
            if not exhaust:
                return MeanReversionResult(setup="NONE", stage="EXHAUSTION", is_valid=False, reason=f"No exhaustion short RSI {rsi:.0f}", stages={k: v.__dict__ for k, v in stages.items()}, bb_position=bb_pos, rsi=rsi, atr=atr)

        # Stage 4: REVERSAL_SETUP (bullish engulfing, hammer, RSI divergence)
        reversal = False
        rev_reason = ""
        rev_conf = 0.5
        if direction == "LONG":
            # Bullish engulfing: prev close < prev open and close > prev open and close > prev_high
            prev_open = float(prev.get("open", prev_close))
            curr_open = float(row.get("open", close))
            is_engulfing = (prev_close < prev_open) and (close > curr_open) and (close > prev_high) and (close > prev_close)
            is_hammer = (close - low) > (high - low) * 0.6 and (high - close) < (high - low) * 0.2 and rsi < 35
            rsi_div = rsi > float(prev.get("rsi", rsi)) + 2 and rsi < 40  # RSI tick up from oversold
            if is_engulfing:
                reversal = True
                rev_reason = f"bullish engulfing close {close:.1f}>prev high {prev_high:.1f}"
                rev_conf = 0.8
            elif is_hammer:
                reversal = True
                rev_reason = f"hammer wick {(close-low)/(high-low+1e-9):.0%} RSI {rsi:.0f}"
                rev_conf = 0.75
            elif rsi_div:
                reversal = True
                rev_reason = f"RSI divergence {rsi:.0f} > prev {float(prev.get('rsi', 50)):.0f}"
                rev_conf = 0.65
            else:
                # Weak reversal: just close > prev close in zone
                if close > prev_close and bb_pos <= 0.3:
                    reversal = True
                    rev_reason = f"close reclaim {close:.1f}>{prev_close:.1f} in zone"
                    rev_conf = 0.55
        else:
            prev_open = float(prev.get("open", prev_close))
            curr_open = float(row.get("open", close))
            is_engulfing = (prev_close > prev_open) and (close < curr_open) and (close < prev_low)
            is_shooting = (high - close) > (high - low) * 0.6 and rsi > 65
            if is_engulfing:
                reversal = True
                rev_reason = "bearish engulfing"
                rev_conf = 0.8
            elif is_shooting:
                reversal = True
                rev_reason = "shooting star"
                rev_conf = 0.75
            elif close < prev_close and bb_pos >= 0.7:
                reversal = True
                rev_reason = f"close below prev in upper zone"
                rev_conf = 0.55

        stages["REVERSAL_SETUP"] = MeanReversionStage(stage="REVERSAL_SETUP", is_valid=reversal, reason=rev_reason or "no reversal pattern", confidence=rev_conf, bb_position=bb_pos, rsi=rsi, atr=atr)
        if not reversal:
            return MeanReversionResult(setup="NONE", stage="REVERSAL_SETUP", is_valid=False, reason=f"No reversal: {rev_reason or 'no pattern'}", stages={k: v.__dict__ for k, v in stages.items()}, bb_position=bb_pos, rsi=rsi, atr=atr)

        # Stage 5: TRIGGER (close reclaim of boundary + 0.2 ATR or close > prev high)
        trigger = False
        trig_reason = ""
        if direction == "LONG":
            bb_lower = float(row.get("bb_lower", low))
            trigger_level = bb_lower + 0.2 * atr
            if close > trigger_level and close > prev_high:
                trigger = True
                trig_reason = f"trigger close {close:.1f}>bb_lower+0.2ATR {trigger_level:.1f} and >prev high {prev_high:.1f}"
            elif close > prev_close + 0.3 * atr:
                trigger = True
                trig_reason = f"trigger momentum {close:.1f}>prev {prev_close:.1f}+0.3ATR"
        else:
            bb_upper = float(row.get("bb_upper", high))
            trigger_level = bb_upper - 0.2 * atr
            if close < trigger_level and close < prev_low:
                trigger = True
                trig_reason = f"trigger short {close:.1f}<bb_upper-0.2ATR"
            elif close < prev_close - 0.3 * atr:
                trigger = True
                trig_reason = f"trigger short momentum"

        stages["TRIGGER"] = MeanReversionStage(stage="TRIGGER", is_valid=trigger, reason=trig_reason or "no trigger", confidence=0.7 if trigger else 0.2, bb_position=bb_pos, rsi=rsi, atr=atr)
        if not trigger:
            return MeanReversionResult(setup="NONE", stage="TRIGGER", is_valid=False, reason=f"No trigger: {trig_reason or 'close not reclaim'}", stages={k: v.__dict__ for k, v in stages.items()}, bb_position=bb_pos, rsi=rsi, atr=atr)

        # All 5 stages passed
        setup_name = "LONG_MEAN_REVERSION" if direction == "LONG" else "SHORT_MEAN_REVERSION"
        avg_conf = sum(s.confidence for s in stages.values() if s.stage in ("RANGE","LOWER_BOUNDARY","EXHAUSTION","REVERSAL_SETUP","TRIGGER")) / 5
        return MeanReversionResult(setup=setup_name, stage="TRIGGER", is_valid=True, reason=f"{setup_name} 5/5 stages: {trig_reason}", confidence=avg_conf, stages={k: v.__dict__ for k, v in stages.items()}, bb_position=bb_pos, rsi=rsi, atr=atr)
