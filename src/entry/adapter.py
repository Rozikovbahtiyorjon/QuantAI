"""
ENTRY-04 / ENTRY-62 — ExistingSignalAdapter / LegacySignalAdapter

Adapts current SignalGenerator SignalResult → LegacyStrategyCandidate
for shadow mode comparison.

Existing QuantAI already contains AI/Confidence/ML/Order Flow/SLTP pipeline.
Adapter converts SignalResult to legacy candidate without modifying SignalGenerator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import pandas as pd

from src.strategy.signal_generator import SignalResult


@dataclass
class LegacyStrategyCandidate:
    """Legacy candidate for shadow comparison."""
    signal: str  # BUY/SELL/HOLD
    confidence: float
    entry: float
    stop_loss: float
    take_profit: float
    reasons: list[str]
    raw_result: SignalResult

    @property
    def is_valid(self) -> bool:
        return self.signal in ("BUY", "SELL")


class ExistingSignalAdapter:
    """ENTRY-04 / ENTRY-62: Adapter to current SignalGenerator without modifying it."""

    def __init__(self, signal_generator=None):
        if signal_generator is None:
            from src.strategy.signal_generator import SignalGenerator
            self.generator = SignalGenerator()
        else:
            self.generator = signal_generator

    def generate_legacy_candidate(self, df: pd.DataFrame, order_flow_signal=None) -> LegacyStrategyCandidate:
        """Convert SignalResult → LegacyStrategyCandidate."""
        result: SignalResult = self.generator.generate(df, order_flow_signal=order_flow_signal)
        return LegacyStrategyCandidate(
            signal=result.signal,
            confidence=result.confidence,
            entry=result.entry,
            stop_loss=result.stop_loss,
            take_profit=result.take_profit,
            reasons=list(result.reasons),
            raw_result=result,
        )

    def to_new_entry_decision(self, candidate: LegacyStrategyCandidate, df: pd.DataFrame) -> dict:
        """For shadow mode: map legacy candidate to new EntryDecision-like dict for comparison."""
        # This is a lightweight mapping, not full EntryEngine
        return {
            "signal": candidate.signal,
            "entry": candidate.entry,
            "sl": candidate.stop_loss,
            "tp": candidate.take_profit,
            "confidence": candidate.confidence,
            "reasons": candidate.reasons,
            "source": "LEGACY",
        }


class LegacySignalAdapter(ExistingSignalAdapter):
    """ENTRY-62: Legacy Adapter — Old SignalGenerator becomes LegacySignalAdapter."""
    pass
