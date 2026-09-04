"""
POI / Zone Engine — Universal Zone Engine (P3.8)

Types:
  support, resistance, liquidity pool, FVG (Fair Value Gap), imbalance,
  previous session POC, VAH, VAL, swing liquidity

Currently Feature Engine has:
  liquidation levels, support/resistance distances, strengths
But no universal Zone Engine / POI Engine.

This engine aggregates all zone types causally and provides:
  nearest zone, distance, strength, type

Causal: only past closed bars.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional
import pandas as pd
import numpy as np


ZoneType = Literal["support", "resistance", "liquidity_pool", "FVG", "imbalance", "POC", "VAH", "VAL", "swing_high", "swing_low"]


@dataclass
class Zone:
    zone_type: ZoneType
    price: float
    upper: float
    lower: float
    strength: float  # 0.0-1.0
    distance_pct: float  # distance from current close in %
    touches: int = 1
    volume: float = 0.0
    timestamp: Optional[pd.Timestamp] = None

    def contains(self, price: float) -> bool:
        return self.lower <= price <= self.upper

    def is_near(self, price: float, threshold_pct: float = 0.5) -> bool:
        return abs(self.distance_pct) <= threshold_pct


@dataclass
class POIResult:
    nearest_support: Optional[Zone] = None
    nearest_resistance: Optional[Zone] = None
    nearest_liquidity_pool: Optional[Zone] = None
    nearest_fvg: Optional[Zone] = None
    nearest_poc: Optional[Zone] = None
    vah: Optional[Zone] = None
    val: Optional[Zone] = None
    all_zones: List[Zone] = field(default_factory=list)

    @property
    def support_distance_pct(self) -> float:
        return self.nearest_support.distance_pct if self.nearest_support else 100.0

    @property
    def resistance_distance_pct(self) -> float:
        return self.nearest_resistance.distance_pct if self.nearest_resistance else 100.0

    @property
    def support_strength(self) -> float:
        return self.nearest_support.strength if self.nearest_support else 0.0

    @property
    def resistance_strength(self) -> float:
        return self.nearest_resistance.strength if self.nearest_resistance else 0.0


class ZoneEngine:
    """
    Universal Zone Engine — 9 POI types.

    Causally builds zones from past bars only:
      - support/resistance: swing highs/lows with multiple touches
      - liquidity pool: high volume nodes
      - FVG: 3-bar gaps (high[1] < low[3] for bullish FVG)
      - imbalance: volume profile imbalances
      - POC/VAH/VAL: volume profile from past 100 bars
      - swing: recent swing highs/lows
    """

    def __init__(self, lookback: int = 100, min_touches: int = 2, atr_mult: float = 0.5):
        self.lookback = lookback
        self.min_touches = min_touches
        self.atr_mult = atr_mult

    def _find_swings(self, df: pd.DataFrame, window: int = 5) -> tuple[list[float], list[float]]:
        """Find swing highs/lows causally (only past, no future)."""
        highs = []
        lows = []
        closes = df["close"].astype(float).values
        high_vals = df["high"].astype(float).values
        low_vals = df["low"].astype(float).values
        for i in range(window, len(df) - window):
            # swing high: high[i] is max in window*2+1
            if high_vals[i] == np.max(high_vals[i-window:i+window+1]):
                highs.append(high_vals[i])
            if low_vals[i] == np.min(low_vals[i-window:i+window+1]):
                lows.append(low_vals[i])
        return highs, lows

    def _volume_profile(self, df: pd.DataFrame, bins: int = 20) -> tuple[float, float, float]:
        """Volume profile: POC, VAH, VAL from last 100 bars."""
        if len(df) < 20:
            close = float(df["close"].iloc[-1])
            return close, close*1.01, close*0.99
        closes = df["close"].astype(float).values[-100:]
        volumes = df["volume"].astype(float).values[-100:]
        # Histogram by price
        hist, edges = np.histogram(closes, bins=bins, weights=volumes)
        poc_idx = int(np.argmax(hist))
        poc = (edges[poc_idx] + edges[poc_idx+1]) / 2
        # VAH/VAL: 70% of volume around POC
        total_vol = hist.sum()
        sorted_idx = np.argsort(hist)[::-1]
        cum_vol = 0
        included = set()
        for idx in sorted_idx:
            cum_vol += hist[idx]
            included.add(idx)
            if cum_vol >= total_vol * 0.7:
                break
        included = sorted(included)
        if included:
            vah = edges[max(included)+1]
            val = edges[min(included)]
        else:
            vah = poc * 1.01
            val = poc * 0.99
        return float(poc), float(vah), float(val)

    def _find_fvg(self, df: pd.DataFrame) -> List[Zone]:
        """Find Fair Value Gaps: 3-bar pattern high[1] < low[3] (bullish) or low[1] > high[3] (bearish)."""
        zones: List[Zone] = []
        if len(df) < 5:
            return zones
        for i in range(2, len(df)-1):
            high1 = float(df["high"].iloc[i-1])
            low1 = float(df["low"].iloc[i-1])
            high3 = float(df["high"].iloc[i+1]) if i+1 < len(df) else float(df["high"].iloc[i])
            low3 = float(df["low"].iloc[i+1]) if i+1 < len(df) else float(df["low"].iloc[i])
            close = float(df["close"].iloc[i])
            atr = float(df["atr"].iloc[i]) if "atr" in df.columns else 1.0
            # Bullish FVG: high1 < low3
            if high1 < low3:
                mid = (high1 + low3) / 2
                zones.append(Zone(zone_type="FVG", price=mid, upper=low3, lower=high1, strength=0.6, distance_pct=(mid-close)/close*100, volume=0.0))
            # Bearish FVG: low1 > high3
            if low1 > high3:
                mid = (low1 + high3) / 2
                zones.append(Zone(zone_type="FVG", price=mid, upper=low1, lower=high3, strength=0.6, distance_pct=(mid-close)/close*100, volume=0.0))
            # Limit to recent
            if len(zones) > 10:
                zones = zones[-10:]
        return zones

    def build_zones(self, df: pd.DataFrame) -> POIResult:
        if len(df) < 20:
            close = float(df["close"].iloc[-1]) if len(df) else 0
            return POIResult(all_zones=[])

        close = float(df["close"].iloc[-1])
        atr = float(df["atr"].iloc[-1]) if "atr" in df.columns else close * 0.01

        all_zones: List[Zone] = []

        # 1. Swing highs/lows -> support/resistance
        highs, lows = self._find_swings(df.tail(self.lookback), window=5)
        # Cluster swings by price proximity (0.5 ATR)
        for price in highs[-5:]:
            # Count touches within 0.5 ATR
            touches = sum(1 for h in highs if abs(h - price) / max(price,1e-9) < 0.005)
            if touches >= self.min_touches:
                strength = min(1.0, touches / 3.0)
                all_zones.append(Zone(zone_type="resistance", price=price, upper=price+0.3*atr, lower=price-0.3*atr, strength=strength, distance_pct=(price-close)/close*100, touches=touches))
            else:
                all_zones.append(Zone(zone_type="swing_high", price=price, upper=price+0.2*atr, lower=price-0.2*atr, strength=0.3, distance_pct=(price-close)/close*100, touches=1))
        for price in lows[-5:]:
            touches = sum(1 for l in lows if abs(l - price) / max(price,1e-9) < 0.005)
            if touches >= self.min_touches:
                all_zones.append(Zone(zone_type="support", price=price, upper=price+0.3*atr, lower=price-0.3*atr, strength=min(1.0, touches/3.0), distance_pct=(price-close)/close*100, touches=touches))
            else:
                all_zones.append(Zone(zone_type="swing_low", price=price, upper=price+0.2*atr, lower=price-0.2*atr, strength=0.3, distance_pct=(price-close)/close*100, touches=1))

        # 2. Volume profile POC/VAH/VAL
        poc, vah, val = self._volume_profile(df.tail(self.lookback))
        all_zones.append(Zone(zone_type="POC", price=poc, upper=poc+0.2*atr, lower=poc-0.2*atr, strength=0.8, distance_pct=(poc-close)/close*100))
        all_zones.append(Zone(zone_type="VAH", price=vah, upper=vah+0.2*atr, lower=vah-0.2*atr, strength=0.6, distance_pct=(vah-close)/close*100))
        all_zones.append(Zone(zone_type="VAL", price=val, upper=val+0.2*atr, lower=val-0.2*atr, strength=0.6, distance_pct=(val-close)/close*100))

        # 3. FVG
        all_zones.extend(self._find_fvg(df.tail(50)))

        # 4. Imbalance: high volume nodes
        if len(df) >= 20:
            vol = df["volume"].astype(float).values[-20:]
            closes = df["close"].astype(float).values[-20:]
            avg_vol = float(np.mean(vol))
            for i, (c, v) in enumerate(zip(closes, vol)):
                if v > avg_vol * 1.8:
                    all_zones.append(Zone(zone_type="imbalance", price=c, upper=c+0.2*atr, lower=c-0.2*atr, strength=min(1.0, v/avg_vol/3), distance_pct=(c-close)/close*100, volume=v))

        # 5. Liquidity pool: clusters of recent highs/lows with high volume
        # Approximate as swing with high volume
        for z in list(all_zones):
            if z.zone_type in ("resistance", "support") and z.touches >= 3:
                # Promote to liquidity pool if high touches
                pool = Zone(zone_type="liquidity_pool", price=z.price, upper=z.upper, lower=z.lower, strength=min(1.0, z.strength+0.2), distance_pct=z.distance_pct, touches=z.touches, volume=z.volume)
                all_zones.append(pool)

        # Sort by distance
        all_zones.sort(key=lambda z: abs(z.distance_pct))

        # Find nearest per type
        def nearest_of(types: List[ZoneType]) -> Optional[Zone]:
            for z in all_zones:
                if z.zone_type in types:
                    return z
            return None

        # Support: support, swing_low, VAL
        nearest_support = nearest_of(["support", "swing_low", "VAL", "liquidity_pool"])
        # Filter to below close for support
        supports = [z for z in all_zones if z.zone_type in ("support", "swing_low", "VAL", "liquidity_pool") and z.price < close]
        if supports:
            nearest_support = min(supports, key=lambda z: abs(z.distance_pct))

        resistances = [z for z in all_zones if z.zone_type in ("resistance", "swing_high", "VAH", "liquidity_pool") and z.price > close]
        nearest_resistance = min(resistances, key=lambda z: abs(z.distance_pct)) if resistances else nearest_of(["resistance", "swing_high", "VAH"])

        # Liquidity pool nearest
        pools = [z for z in all_zones if z.zone_type == "liquidity_pool"]
        nearest_pool = min(pools, key=lambda z: abs(z.distance_pct)) if pools else None

        # FVG nearest
        fvgs = [z for z in all_zones if z.zone_type == "FVG"]
        nearest_fvg = min(fvgs, key=lambda z: abs(z.distance_pct)) if fvgs else None

        # POC nearest
        pocs = [z for z in all_zones if z.zone_type == "POC"]
        nearest_poc = pocs[0] if pocs else None

        vah_zone = next((z for z in all_zones if z.zone_type == "VAH"), None)
        val_zone = next((z for z in all_zones if z.zone_type == "VAL"), None)

        return POIResult(
            nearest_support=nearest_support,
            nearest_resistance=nearest_resistance,
            nearest_liquidity_pool=nearest_pool,
            nearest_fvg=nearest_fvg,
            nearest_poc=nearest_poc,
            vah=vah_zone,
            val=val_zone,
            all_zones=all_zones[:20],  # top 20 nearest
        )
