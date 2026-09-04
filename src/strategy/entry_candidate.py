"""
EntryCandidate — P3.18 Entry Zone model

SignalResult currently has:
  entry, stop_loss, take_profit
but not:
  entry_zone_low, entry_zone_high, ideal_entry, max_chase_distance

This is candidate for new EntryCandidate model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import math


@dataclass
class EntryCandidate:
    """Candidate entry with zone, not just single price."""

    # Core
    signal: str  # BUY/SELL/HOLD
    entry: float  # current close or trigger price
    stop_loss: float
    take_profit: float

    # Zone (new)
    entry_zone_low: float
    entry_zone_high: float
    ideal_entry: float
    max_chase_distance: float  # max distance from ideal to still enter (e.g., 0.5 ATR)

    # Diagnostics
    zone_width_atr: float = 0.0
    distance_to_ideal_atr: float = 0.0
    is_chasing: bool = False
    reason: str = ""

    @property
    def is_valid(self) -> bool:
        return self.signal in ("BUY", "SELL") and self.entry_zone_low < self.entry_zone_high

    @property
    def is_in_zone(self) -> bool:
        return self.entry_zone_low <= self.entry <= self.entry_zone_high

    def should_enter(self, current_price: float) -> tuple[bool, str]:
        """Check if current price is still within chase distance."""
        dist = abs(current_price - self.ideal_entry)
        chase_atr = dist / max(self.zone_width_atr * 2.0, 1e-9) if self.zone_width_atr else 0
        # Actually distance_to_ideal_atr already
        dist_atr = abs(current_price - self.ideal_entry) / max(self.zone_width_atr, 1e-9) if self.zone_width_atr else 0
        if current_price < self.entry_zone_low or current_price > self.entry_zone_high:
            if dist > self.max_chase_distance:
                return False, f"price {current_price:.2f} outside zone [{self.entry_zone_low:.2f}, {self.entry_zone_high:.2f}] and chase {dist:.2f} > max {self.max_chase_distance:.2f}"
            else:
                return True, f"price outside zone but within chase {dist:.2f} <= {self.max_chase_distance:.2f} (chasing)"
        return True, f"price in zone"

    @classmethod
    def from_setup(
        cls,
        signal: str,
        entry: float,
        stop_loss: float,
        take_profit: float,
        atr: float,
        setup: str = "NONE",
        zone_atr_mult: float = 0.4,
        max_chase_atr_mult: float = 0.8,
    ) -> "EntryCandidate":
        """
        Create EntryCandidate from setup.

        Zone width depends on setup:
          - BREAKOUT: tight zone 0.3 ATR around entry (breakout level)
          - PULLBACK: wider 0.8 ATR around ema zone
          - MEAN_REVERSION: 0.6 ATR around BB boundary
        """
        if setup in ("LONG_BREAKOUT", "SHORT_BREAKOUT"):
            half_width = atr * 0.15
            max_chase = atr * 0.5
            ideal = entry
        elif "PULLBACK" in setup:
            half_width = atr * 0.4
            max_chase = atr * 0.8
            ideal = entry  # ideal is EMA level, but we use entry as proxy
        elif "MEAN_REVERSION" in setup:
            half_width = atr * 0.3
            max_chase = atr * 0.6
            ideal = entry
        elif "LIQUIDITY_SWEEP" in setup:
            half_width = atr * 0.5
            max_chase = atr * 1.0
            ideal = entry
        else:
            half_width = atr * zone_atr_mult
            max_chase = atr * max_chase_atr_mult
            ideal = entry

        zone_low = ideal - half_width
        zone_high = ideal + half_width

        # For BUY, zone is below entry (pullback), for SELL above
        # Adjust ideal to be center of zone
        if signal == "BUY":
            # For buy, ideal is lower edge + 30% up
            ideal = zone_low + (zone_high - zone_low) * 0.4
        elif signal == "SELL":
            ideal = zone_high - (zone_high - zone_low) * 0.4

        return cls(
            signal=signal,
            entry=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_zone_low=round(zone_low, 2),
            entry_zone_high=round(zone_high, 2),
            ideal_entry=round(ideal, 2),
            max_chase_distance=round(max_chase, 2),
            zone_width_atr=round((zone_high - zone_low) / max(atr, 1e-9), 2),
            distance_to_ideal_atr=round(abs(entry - ideal) / max(atr, 1e-9), 2),
            is_chasing=abs(entry - ideal) > half_width * 0.5,
            reason=f"zone {zone_low:.2f}-{zone_high:.2f} ideal {ideal:.2f} max chase {max_chase:.2f} ATR {atr:.2f} setup {setup}",
        )
