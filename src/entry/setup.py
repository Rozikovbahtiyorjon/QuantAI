"""
ENTRY-06/66 — Setup (PHASE 4)
SetupCandidate + SetupDetectorInterface
Each setup ONLY creates SetupCandidate.
Then: Trigger → Confirmation → ML → Quality → EV → Risk
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from abc import ABC, abstractmethod


class SetupType(str, Enum):
    TREND_PULLBACK = "TREND_PULLBACK"
    BREAKOUT = "BREAKOUT"
    MEAN_REVERSION = "MEAN_REVERSION"
    MOMENTUM_CONTINUATION = "MOMENTUM_CONTINUATION"
    LIQUIDITY_REVERSAL = "LIQUIDITY_REVERSAL"


class SetupDirection(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class SetupCandidate:
    """
    SetupCandidate — OUTPUT of a setup detector.
    Does NOT make trading decisions.
    Only describes: "Market is in this setup configuration."
    """
    setup_type: SetupType
    direction: SetupDirection
    quality: float  # 0.0-1.0 (setup quality, not confidence)
    confidence: float  # 0.0-1.0 (detector confidence)
    
    # Entry zone from POI
    entry_zone_low: float
    entry_zone_high: float
    ideal_entry: float
    max_chase_atr: float  # max chase in ATR units
    
    # Structural SL/TP (preliminary, will be refined by SL/TP engine)
    sl_candidate: float
    tp_candidate: float
    
    # Invalidation
    invalidated: bool = False
    invalidation_reason: str = ""
    valid_until_bars: int = 20  # max bars this setup stays valid
    
    # Context
    regime: str = ""
    htf_context: dict | None = None  # HTF direction, trend_strength, etc.
    poi: dict | None = None  # Zone info: type, strength, distance_pct
    
    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict = field(default_factory=dict)
    
    @property
    def is_valid(self) -> bool:
        return not self.invalidated and self.quality > 0.0
    
    @property
    def entry_zone_width(self) -> float:
        return self.entry_zone_high - self.entry_zone_low
    
    @property
    def risk_distance(self) -> float:
        if self.direction == SetupDirection.LONG:
            return self.ideal_entry - self.sl_candidate
        return self.sl_candidate - self.ideal_entry


class SetupDetectorInterface(ABC):
    """
    Single interface for ALL setup strategies.
    Each setup implements this.
    """
    
    @property
    @abstractmethod
    def setup_type(self) -> SetupType:
        pass
    
    @abstractmethod
    def detect(
        self,
        market_context: Any,  # MarketContext from entry_engine
        zones: list[Any],  # list[Zone]
    ) -> SetupCandidate | None:
        """
        Detect setup given market context and zones.
        Returns SetupCandidate if setup exists, None otherwise.
        NEVER returns a trading decision.
        """
        pass


# Concrete implementations (imported from strategy to avoid circular deps)
# from src.strategy.trend_pullback_setup import TrendPullbackSetup
# from src.strategy.breakout_setup import BreakoutSetup
# from src.strategy.mean_reversion_setup import MeanReversionSetup
# from src.strategy.momentum_continuation_setup import MomentumContinuationSetup
# from src.strategy.liquidity_reversal_setup import LiquidityReversalSetup