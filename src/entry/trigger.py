"""
ENTRY-25 — Trigger Interface (PHASE 5)

detect(setup, market_context) -> TriggerEvent | None
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from src.entry.models import TriggerEvent, SetupCandidate, MarketContext


class TriggerInterface(ABC):
    @abstractmethod
    def detect(self, setup: SetupCandidate, context: MarketContext) -> Optional[TriggerEvent]:
        pass
