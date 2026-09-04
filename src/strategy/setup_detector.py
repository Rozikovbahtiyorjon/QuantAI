"""
Setup Detector — Universal SETUP Layer (P3)

Fundamental distinction:
  SETUP ≠ BUY/SELL

SETUP answers: "Why should an entry opportunity exist now?"
  e.g., LONG_PULLBACK, LONG_BREAKOUT, LONG_MEAN_REVERSION, LONG_LIQUIDITY_SWEEP

Only after SETUP is confirmed does pipeline proceed to:
  Trigger → Confidence → ML → Risk → Execution

This is separate from:
  - BUY/SELL (directional signal)
  - Strategy (Breakout is one setup, not the engine)

Current pipeline before: AI → Confidence → ML → Gate → Order Flow → final signal
New pipeline: SETUP → Trigger → AI → Confidence → ML → Risk → ...

Setups are regime-aware and causal (only past data).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional
import pandas as pd


SetupType = Literal[
    "NONE",
    "LONG_PULLBACK", "SHORT_PULLBACK",
    "LONG_BREAKOUT", "SHORT_BREAKOUT",
    "LONG_MEAN_REVERSION", "SHORT_MEAN_REVERSION",
    "LONG_LIQUIDITY_SWEEP", "SHORT_LIQUIDITY_SWEEP",
    "LONG_TREND_CONTINUATION", "SHORT_TREND_CONTINUATION",
]


@dataclass
class SetupResult:
    setup: SetupType = "NONE"
    confidence: float = 0.0
    reason: str = ""
    is_valid: bool = False

    # Context
    regime: str = "UNKNOWN"
    adx: float = 0.0
    bb_position: float = 0.5
    rsi: float = 50.0


class SetupDetector:
    """
    Universal Setup Engine — why an entry should exist.

    Detects setups causally from last closed bar only.
    Each setup has specific market conditions:

    - LONG_PULLBACK: TREND_UP + price pullback to ema_fast/slow (within ATR) + RSI 40-60
    - LONG_BREAKOUT: Donchian 20-bar high breakout + EMA alignment + ADX>20
    - LONG_MEAN_REVERSION: RANGE + BB low (≤0.2) + RSI ≤30
    - LONG_LIQUIDITY_SWEEP: RANGE + sweep of recent low + volume spike
    - LONG_TREND_CONTINUATION: TREND_UP + trend_score ≥2 + no BB squeeze
    (mirrored for SHORT)
    """

    def __init__(self, min_adx_trend: float = 20.0, min_adx_range: float = 60.0):
        self.min_adx_trend = min_adx_trend
        self.min_adx_range = min_adx_range

    def detect(self, df: pd.DataFrame, regime: str = "Sideways") -> SetupResult:
        if len(df) < 50:
            return SetupResult(setup="NONE", reason="insufficient history")

        row = df.iloc[-1]
        close = float(row.get("close", 0))
        high = float(row.get("high", 0))
        low = float(row.get("low", 0))
        ema_fast = float(row.get("ema_fast", close))
        ema_slow = float(row.get("ema_slow", close))
        ema_trend = float(row.get("ema_trend", close))
        adx = float(row.get("adx", 0))
        atr = float(row.get("atr", 1))
        rsi = float(row.get("rsi", 50))
        bb_pos = float(row.get("bb_position", 0)) + 0.5 if "bb_position" in row else 0.5
        bb_width = float(row.get("bb_width", 1.0))
        volume = float(row.get("volume", 0))
        vol_sma = float(row.get("volume_sma20", volume))
        # Donchian
        recent_high = float(df["high"].iloc[-21:-1].max()) if len(df) > 21 else high
        recent_low = float(df["low"].iloc[-21:-1].min()) if len(df) > 21 else low

        # Helper: pullback distance in ATR units
        pullback_dist = abs(close - ema_fast) / max(atr, 1e-9)

        # TREND_UP setups
        if regime == "TREND_UP":
            # LONG_PULLBACK: uptrend + pullback to EMA
            if ema_fast > ema_slow > ema_trend and adx > self.min_adx_trend and 0.5 < pullback_dist < 2.0 and 40 <= rsi <= 65:
                return SetupResult(setup="LONG_PULLBACK", confidence=0.75, reason=f"TREND_UP pullback to EMA dist {pullback_dist:.2f} ATR RSI {rsi:.0f}", is_valid=True, regime=regime, adx=adx, bb_position=bb_pos, rsi=rsi)
            # LONG_TREND_CONTINUATION: strong trend, no squeeze, EMA aligned
            if ema_fast > ema_slow > ema_trend and adx > 25 and bb_width > 0.02:
                trend_score = float(row.get("trend_score", 0))
                if trend_score >= 2:
                    return SetupResult(setup="LONG_TREND_CONTINUATION", confidence=0.70, reason=f"TREND_UP continuation trend_score {trend_score:.0f} ADX {adx:.0f}", is_valid=True, regime=regime, adx=adx, bb_position=bb_pos, rsi=rsi)
            # LONG_BREAKOUT: Donchian breakout in uptrend
            if close > recent_high and adx > 20 and ema_fast > ema_slow:
                return SetupResult(setup="LONG_BREAKOUT", confidence=0.65, reason=f"TREND_UP breakout {close:.2f}>{recent_high:.2f} ADX {adx:.0f}", is_valid=True, regime=regime, adx=adx, bb_position=bb_pos, rsi=rsi)

        elif regime == "TREND_DOWN":
            # SHORT mirror
            if ema_fast < ema_slow < ema_trend and adx > self.min_adx_trend and 0.5 < pullback_dist < 2.0 and 35 <= rsi <= 60:
                return SetupResult(setup="SHORT_PULLBACK", confidence=0.75, reason=f"TREND_DOWN pullback", is_valid=True, regime=regime, adx=adx, bb_position=bb_pos, rsi=rsi)
            if ema_fast < ema_slow < ema_trend and adx > 25 and bb_width > 0.02:
                trend_score = float(row.get("trend_score", 0))
                if trend_score <= -2:
                    return SetupResult(setup="SHORT_TREND_CONTINUATION", confidence=0.70, reason=f"TREND_DOWN continuation", is_valid=True, regime=regime, adx=adx, bb_position=bb_pos, rsi=rsi)
            if close < recent_low and adx > 20 and ema_fast < ema_slow:
                return SetupResult(setup="SHORT_BREAKOUT", confidence=0.65, reason=f"TREND_DOWN breakout {close:.2f}<{recent_low:.2f}", is_valid=True, regime=regime, adx=adx, bb_position=bb_pos, rsi=rsi)

        else:  # RANGE / Sideways
            # LONG_MEAN_REVERSION: BB low + oversold
            if bb_pos <= 0.2 and rsi <= 30 and volume > 0:
                return SetupResult(setup="LONG_MEAN_REVERSION", confidence=0.70, reason=f"RANGE BB low {bb_pos:.2f} RSI {rsi:.0f}", is_valid=True, regime=regime, adx=adx, bb_position=bb_pos, rsi=rsi)
            if bb_pos >= 0.8 and rsi >= 70:
                return SetupResult(setup="SHORT_MEAN_REVERSION", confidence=0.70, reason=f"RANGE BB high {bb_pos:.2f} RSI {rsi:.0f}", is_valid=True, regime=regime, adx=adx, bb_position=bb_pos, rsi=rsi)
            # LIQUIDITY SWEEP: sweep recent low/high + volume spike
            if low < recent_low and volume > vol_sma * 1.5 and adx < 25:
                # Long sweep: price swept low then recovered
                if close > low + (high - low) * 0.5:  # wicked
                    return SetupResult(setup="LONG_LIQUIDITY_SWEEP", confidence=0.65, reason=f"RANGE liquidity sweep low {low:.2f}<{recent_low:.2f} vol {volume/vol_sma:.1f}x", is_valid=True, regime=regime, adx=adx, bb_position=bb_pos, rsi=rsi)
            if high > recent_high and volume > vol_sma * 1.5 and adx < 25:
                if close < high - (high - low) * 0.5:
                    return SetupResult(setup="SHORT_LIQUIDITY_SWEEP", confidence=0.65, reason=f"RANGE liquidity sweep high", is_valid=True, regime=regime, adx=adx, bb_position=bb_pos, rsi=rsi)
            # BREAKOUT in range still valid but with squeeze check
            if close > recent_high and adx > 15 and bb_width < 0.02:
                # Tight squeeze breakout
                return SetupResult(setup="LONG_BREAKOUT", confidence=0.60, reason=f"RANGE squeeze breakout long", is_valid=True, regime=regime, adx=adx, bb_position=bb_pos, rsi=rsi)
            if close < recent_low and adx > 15 and bb_width < 0.02:
                return SetupResult(setup="SHORT_BREAKOUT", confidence=0.60, reason=f"RANGE squeeze breakout short", is_valid=True, regime=regime, adx=adx, bb_position=bb_pos, rsi=rsi)

        return SetupResult(setup="NONE", confidence=0.0, reason=f"No setup in {regime} (ADX {adx:.0f} BB {bb_pos:.2f} RSI {rsi:.0f})", is_valid=False, regime=regime, adx=adx, bb_position=bb_pos, rsi=rsi)
