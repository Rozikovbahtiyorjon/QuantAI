"""
ENTRY-10 — Zone (PHASE 3 POI)

Zone object must contain:
  zone_type, lower, upper, center, strength, freshness, timestamp, source, confidence
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal, Optional
from enum import Enum


class ZoneType(str, Enum):
    SUPPORT = "support"
    RESISTANCE = "resistance"
    LIQUIDITY_POOL = "liquidity_pool"
    FVG = "FVG"
    IMBALANCE = "imbalance"
    POC = "POC"
    VAH = "VAH"
    VAL = "VAL"
    SWING_HIGH = "swing_high"
    SWING_LOW = "swing_low"
    PREVIOUS_HIGH = "previous_high"
    PREVIOUS_LOW = "previous_low"
    PREVIOUS_CLOSE = "previous_close"
    SESSION_POC = "session_poc"
    HIGH_VOLUME = "high_volume_zone"
    LOW_VOLUME = "low_volume_zone"


@dataclass(frozen=True)
class Zone:
    """ENTRY-10 Zone — immutable snapshot."""

    zone_type: ZoneType
    lower: float
    upper: float
    center: float
    strength: float  # 0.0-1.0
    freshness: float  # 0.0-1.0 (1.0 = fresh, 0.0 = stale, many touches)
    timestamp: datetime
    source: str  # e.g., "swing", "volume_profile", "fvg", "liquidation"
    confidence: float  # 0.0-1.0

    @property
    def width(self) -> float:
        return self.upper - self.lower

    def contains(self, price: float) -> bool:
        return self.lower <= price <= self.upper

    def distance_pct(self, price: float) -> float:
        return abs(price - self.center) / self.center * 100 if self.center else 0

    def is_near(self, price: float, threshold_pct: float = 0.5) -> bool:
        return self.distance_pct(price) <= threshold_pct

    @classmethod
    def from_swing(
        cls, price: float, atr: float, zone_type: ZoneType, touches: int, timestamp: datetime | None = None
    ) -> "Zone":
        half_width = atr * 0.3
        strength = min(1.0, touches / 3.0)
        freshness = max(0.0, 1.0 - (touches - 1) * 0.2)
        ts = timestamp or datetime.now(timezone.utc)
        return cls(
            zone_type=zone_type,
            lower=price - half_width,
            upper=price + half_width,
            center=price,
            strength=strength,
            freshness=freshness,
            timestamp=ts,
            source="swing",
            confidence=strength * freshness,
        )
