"""
ENTRY-18 — Setup Detector Interface (PHASE 4)

All strategies must implement same contract:
  detect(context, zones) -> SetupCandidate | None
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional
from dataclasses import dataclass

from src.entry.models import SetupCandidate, MarketContext
from src.entry.zone import Zone


class SetupDetectorInterface(ABC):
    """ENTRY-18: All strategies implement same contract."""

    @abstractmethod
    def detect(self, context: MarketContext, zones: list[Zone]) -> Optional[SetupCandidate]:
        """Detect setup given market context and zones. Return None if no setup."""
        pass


@dataclass
class SetupContext:
    """Context passed to setup detectors."""
    market_context: MarketContext
    zones: list[Zone]
    df: any  # DataFrame with indicators
