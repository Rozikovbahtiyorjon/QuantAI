"""
Trend Pullback Setup — Universal Layer (P3.6)

Separate object for trend pullback lifecycle:
  TREND_UP → PULLBACK_ZONE → QUALITY → TRIGGER → INVALIDATED

Returns detailed object: setup, quality, zone, invalidated
Existing EMA/ADX/ATR data allows this, but lifecycle was not formalized.

Causal: only past closed bars.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import pandas as pd


@dataclass
class PullbackZone:
    """Pullback zone boundaries in price."""
    upper: float  # e.g., ema_fast + 0.5 ATR
    lower: float  # e.g., ema_slow - 0.5 ATR
    ema_fast: float
    ema_slow: float
    ema_trend: float
    atr: float

    def contains(self, price: float) -> bool:
        return self.lower <= price <= self.upper

    def distance_pct(self, price: float) -> float:
        mid = (self.upper + self.lower) / 2
        return abs(price - mid) / mid * 100 if mid else 0


@dataclass
class TrendPullbackResult:
    setup: str  # LONG_PULLBACK / SHORT_PULLBACK / NONE
    quality: float  # 0.0 - 1.0
    zone: Optional[PullbackZone]
    invalidated: bool
    reason: str
    is_valid: bool

    # Diagnostics
    pullback_dist_atr: float = 0.0
    rsi: float = 50.0
    adx: float = 0.0
    trend_score: float = 0.0


class TrendPullbackSetup:
    """
    Universal Trend Pullback lifecycle formalization.

    Lifecycle:
      1. TREND detection: EMA stack + ADX
      2. PULLBACK_ZONE: price within EMA zone (0.5-2.0 ATR from ema_fast)
      3. QUALITY: scoring based on RSI, volume, ADX, trend_score
      4. TRIGGER: close reclaim of ema_fast or bullish engulfing in zone
      5. INVALIDATED: price breaks ema_trend opposite or ADX < threshold

    Uses only closed bar data, no look-ahead.
    """

    def __init__(self, min_adx: float = 20.0, atr_mult_zone: float = 1.5):
        self.min_adx = min_adx
        self.atr_mult_zone = atr_mult_zone

    def evaluate(self, df: pd.DataFrame) -> TrendPullbackResult:
        if len(df) < 50:
            return TrendPullbackResult(setup="NONE", quality=0.0, zone=None, invalidated=False, reason="insufficient history", is_valid=False)

        row = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else row

        close = float(row.get("close", 0))
        ema_fast = float(row.get("ema_fast", close))
        ema_slow = float(row.get("ema_slow", close))
        ema_trend = float(row.get("ema_trend", close))
        adx = float(row.get("adx", 0))
        atr = float(row.get("atr", 1))
        rsi = float(row.get("rsi", 50))
        trend_score = float(row.get("trend_score", 0))
        volume = float(row.get("volume", 0))
        vol_sma = float(row.get("volume_sma20", volume))

        # Zone: between ema_fast and ema_slow +/- 0.5 ATR
        zone_upper = max(ema_fast, ema_slow) + 0.5 * atr
        zone_lower = min(ema_fast, ema_slow) - 0.5 * atr
        # For uptrend, zone is below ema_fast
        if ema_fast > ema_slow > ema_trend:
            zone_upper = ema_fast + 0.3 * atr
            zone_lower = ema_slow - 0.5 * atr
        elif ema_fast < ema_slow < ema_trend:
            zone_upper = ema_slow + 0.5 * atr
            zone_lower = ema_fast - 0.3 * atr

        zone = PullbackZone(upper=zone_upper, lower=zone_lower, ema_fast=ema_fast, ema_slow=ema_slow, ema_trend=ema_trend, atr=atr)
        pullback_dist = abs(close - ema_fast) / max(atr, 1e-9)
        prev_close = float(prev.get("close", close))

        # Invalidated check first
        invalidated = False
        invalid_reason = ""
        # Invalidated if trend breaks: ema_fast crosses ema_trend opposite
        if ema_fast > ema_slow > ema_trend:  # was uptrend
            if ema_fast < ema_trend or adx < self.min_adx * 0.7:
                invalidated = True
                invalid_reason = f"uptrend invalidated ema_fast {ema_fast:.1f}<ema_trend {ema_trend:.1f} or adx {adx:.0f}<{self.min_adx*0.7:.0f}"
        elif ema_fast < ema_slow < ema_trend:  # was downtrend
            if ema_fast > ema_trend or adx < self.min_adx * 0.7:
                invalidated = True
                invalid_reason = f"downtrend invalidated"
        if invalidated:
            return TrendPullbackResult(setup="NONE", quality=0.0, zone=zone, invalidated=True, reason=invalid_reason, is_valid=False, pullback_dist_atr=pullback_dist, rsi=rsi, adx=adx, trend_score=trend_score)

        # Detect LONG_PULLBACK in TREND_UP
        if ema_fast > ema_slow > ema_trend and adx > self.min_adx:
            if zone.contains(close) and 40 <= rsi <= 65 and 0.5 < pullback_dist < 2.5:
                # Quality scoring
                quality = 0.5
                # RSI ideal 45-55
                if 45 <= rsi <= 55:
                    quality += 0.2
                elif 40 <= rsi <= 65:
                    quality += 0.1
                # ADX strong
                if adx > 25:
                    quality += 0.1
                if trend_score >= 2:
                    quality += 0.1
                # Volume not excessive (pullback should be low volume)
                if volume < vol_sma * 1.2:
                    quality += 0.1
                quality = min(1.0, quality)
                # Trigger: close reclaim ema_fast or bullish close
                prev_in_zone = zone.contains(prev_close)
                trigger = zone.contains(close) and (close > ema_fast or close > prev_close)
                if trigger or quality >= 0.7:
                    return TrendPullbackResult(setup="LONG_PULLBACK", quality=quality, zone=zone, invalidated=False, reason=f"TREND_UP pullback zone {zone.lower:.1f}-{zone.upper:.1f} dist {pullback_dist:.2f}ATR RSI {rsi:.0f} ADX {adx:.0f} quality {quality:.2f}", is_valid=True, pullback_dist_atr=pullback_dist, rsi=rsi, adx=adx, trend_score=trend_score)
                else:
                    return TrendPullbackResult(setup="NONE", quality=quality*0.6, zone=zone, invalidated=False, reason=f"pullback in zone but no trigger (quality {quality:.2f} no reclaim)", is_valid=False, pullback_dist_atr=pullback_dist, rsi=rsi, adx=adx, trend_score=trend_score)

        # Detect SHORT_PULLBACK in TREND_DOWN (mirror)
        if ema_fast < ema_slow < ema_trend and adx > self.min_adx:
            if zone.contains(close) and 35 <= rsi <= 60 and 0.5 < pullback_dist < 2.5:
                quality = 0.5
                if 45 <= rsi <= 55:
                    quality += 0.2
                if adx > 25:
                    quality += 0.1
                if trend_score <= -2:
                    quality += 0.1
                if volume < vol_sma * 1.2:
                    quality += 0.1
                quality = min(1.0, quality)
                trigger = zone.contains(close) and (close < ema_fast or close < prev_close)
                if trigger or quality >= 0.7:
                    return TrendPullbackResult(setup="SHORT_PULLBACK", quality=quality, zone=zone, invalidated=False, reason=f"TREND_DOWN pullback", is_valid=True, pullback_dist_atr=pullback_dist, rsi=rsi, adx=adx, trend_score=trend_score)

        return TrendPullbackResult(setup="NONE", quality=0.0, zone=zone, invalidated=False, reason=f"No pullback setup (regime not TREND or zone miss) ADX {adx:.0f} dist {pullback_dist:.2f}", is_valid=False, pullback_dist_atr=pullback_dist, rsi=rsi, adx=adx, trend_score=trend_score)
