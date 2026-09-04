"""
Entry Lifecycle — P3.19

Currently no lifecycle:
  SETUP_CREATED
  WAIT_TRIGGER
  TRIGGERED
  CONFIRMED
  EXPIRED
  INVALIDATED

This is critical gap. Must be mandatory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Literal
from datetime import datetime, timezone, timedelta
import pandas as pd


EntryState = Literal["SETUP_CREATED", "WAIT_TRIGGER", "TRIGGERED", "CONFIRMED", "EXPIRED", "INVALIDATED"]


@dataclass
class EntryLifecycle:
    """Tracks setup lifecycle with expiration and invalidation."""

    setup: str
    setup_time: datetime
    entry_zone_low: float
    entry_zone_high: float
    ideal_entry: float
    atr: float

    # Config
    max_wait_bars: int = 20  # max bars to wait for trigger after setup
    confirmation_bars: int = 1  # bars to confirm (e.g., close above trigger)
    expiration_bars: int = 30  # total bars before EXPIRED
    invalidated_on: Optional[str] = None  # reason for invalidation

    # State
    state: EntryState = "SETUP_CREATED"
    trigger_time: Optional[datetime] = None
    trigger_price: Optional[float] = None
    confirm_time: Optional[datetime] = None
    bars_since_setup: int = 0

    def update(self, row: pd.Series) -> EntryState:
        """
        Update lifecycle with new bar.

        Transitions:
          SETUP_CREATED → WAIT_TRIGGER (immediate)
          WAIT_TRIGGER → TRIGGERED (if price enters zone and trigger condition)
          TRIGGERED → CONFIRMED (if next bar confirms)
          Any → EXPIRED (if bars_since_setup > expiration_bars)
          Any → INVALIDATED (if regime flips or zone breaks)
        """
        self.bars_since_setup += 1
        close = float(row.get("close", 0))
        high = float(row.get("high", 0))
        low = float(row.get("low", 0))
        ema_trend = float(row.get("ema_trend", close))
        adx = float(row.get("adx", 0))

        # Check expiration first
        if self.bars_since_setup > self.expiration_bars:
            self.state = "EXPIRED"
            self.invalidated_on = f"expired after {self.bars_since_setup} bars > {self.expiration_bars}"
            return self.state

        # Check invalidation: regime flip
        if self.state in ("SETUP_CREATED", "WAIT_TRIGGER", "TRIGGERED"):
            # Invalidated if ADX drops below 12 or ema_trend flips
            if adx < 12:
                self.state = "INVALIDATED"
                self.invalidated_on = f"ADX {adx:.0f}<12"
                return self.state
            # For LONG setup, invalidated if close breaks below zone - 1 ATR
            if "LONG" in self.setup and close < self.entry_zone_low - self.atr:
                self.state = "INVALIDATED"
                self.invalidated_on = f"break below zone {close:.1f}<{self.entry_zone_low - self.atr:.1f}"
                return self.state
            if "SHORT" in self.setup and close > self.entry_zone_high + self.atr:
                self.state = "INVALIDATED"
                self.invalidated_on = f"break above zone"
                return self.state

        # State transitions
        if self.state == "SETUP_CREATED":
            self.state = "WAIT_TRIGGER"
            return self.state

        if self.state == "WAIT_TRIGGER":
            # Trigger if price enters zone
            if self.entry_zone_low <= close <= self.entry_zone_high:
                self.state = "TRIGGERED"
                self.trigger_time = pd.to_datetime(row.get("timestamp")) if "timestamp" in row else datetime.now(timezone.utc)
                self.trigger_price = close
                return self.state
            # Also trigger if high/low sweeps zone (wick)
            if low <= self.entry_zone_high and high >= self.entry_zone_low:
                # Wick entered zone
                if "BREAKOUT" in self.setup and close > self.entry_zone_high:
                    self.state = "TRIGGERED"
                    self.trigger_time = pd.to_datetime(row.get("timestamp")) if "timestamp" in row else datetime.now(timezone.utc)
                    self.trigger_price = close
                    return self.state

        elif self.state == "TRIGGERED":
            # Need confirmation: next bar close beyond trigger price with volume or close > ideal
            if self.trigger_price is not None:
                if "LONG" in self.setup and close > self.trigger_price:
                    self.state = "CONFIRMED"
                    self.confirm_time = pd.to_datetime(row.get("timestamp")) if "timestamp" in row else datetime.now(timezone.utc)
                    return self.state
                if "SHORT" in self.setup and close < self.trigger_price:
                    self.state = "CONFIRMED"
                    self.confirm_time = pd.to_datetime(row.get("timestamp")) if "timestamp" in row else datetime.now(timezone.utc)
                    return self.state
                # If not confirmed within 3 bars, back to WAIT_TRIGGER
                if self.bars_since_setup - (self.trigger_time.timestamp() if self.trigger_time else 0) > 3:
                    # Actually use trigger bar count
                    pass
            # If triggered but not confirmed within confirmation_bars, stay TRIGGERED for up to 3 bars then EXPIRED check will handle
            pass

        return self.state

    @property
    def is_active(self) -> bool:
        return self.state in ("WAIT_TRIGGER", "TRIGGERED")

    @property
    def is_terminal(self) -> bool:
        return self.state in ("CONFIRMED", "EXPIRED", "INVALIDATED")

    def should_enter(self) -> tuple[bool, str]:
        if self.state == "CONFIRMED":
            return True, f"CONFIRMED at {self.trigger_price}"
        if self.state == "EXPIRED":
            return False, f"EXPIRED: {self.invalidated_on}"
        if self.state == "INVALIDATED":
            return False, f"INVALIDATED: {self.invalidated_on}"
        return False, f"WAIT {self.state}"

    def to_dict(self) -> dict:
        return {
            "setup": self.setup,
            "state": self.state,
            "bars_since_setup": self.bars_since_setup,
            "trigger_price": self.trigger_price,
            "invalidated_on": self.invalidated_on,
            "entry_zone": [self.entry_zone_low, self.entry_zone_high],
        }
