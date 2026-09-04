"""
PHASE 15 — STRATEGY BANK
ENTRY-66 — Plug-in Setup Bank
ENTRY-67 — Candidate evaluation
ENTRY-68 — Ablation tests
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import pandas as pd
from abc import ABC, abstractmethod


class SetupType(str, Enum):
    TREND_PULLBACK = "TREND_PULLBACK"
    BREAKOUT = "BREAKOUT"
    MEAN_REVERSION = "MEAN_REVERSION"
    MOMENTUM_CONTINUATION = "MOMENTUM_CONTINUATION"
    LIQUIDITY_REVERSAL = "LIQUIDITY_REVERSAL"


@dataclass
class SetupCandidate:
    """Universal setup candidate from any strategy."""
    setup_type: SetupType
    direction: str  # LONG / SHORT
    confidence: float
    quality: float
    entry_zone_low: float
    entry_zone_high: float
    ideal_entry: float
    stop_loss: float
    take_profit: float
    invalidated: bool
    invalidation_reason: str
    valid_until_bars: int
    metadata: dict = field(default_factory=dict)


class SetupDetectorInterface(ABC):
    """ENTRY-66: Single interface for all setup strategies."""

    @abstractmethod
    def detect(self, context: Any, zones: list[Any]) -> SetupCandidate | None:
        """Detect setup given market context and zones."""
        pass

    @property
    @abstractmethod
    def setup_type(self) -> SetupType:
        pass


class SetupBank:
    """Registry of all available setup strategies."""

    def __init__(self):
        self._detectors: dict[SetupType, SetupDetectorInterface] = {}

    def register(self, detector: SetupDetectorInterface):
        self._detectors[detector.setup_type] = detector

    def get(self, setup_type: SetupType) -> SetupDetectorInterface | None:
        return self._detectors.get(setup_type)

    def all(self) -> list[SetupDetectorInterface]:
        return list(self._detectors.values())

    def detect_all(self, context: Any, zones: list[Any]) -> list[SetupCandidate]:
        """Run all detectors, return valid candidates."""
        candidates = []
        for detector in self._detectors.values():
            candidate = detector.detect(context, zones)
            if candidate and not candidate.invalidated:
                candidates.append(candidate)
        return candidates


# Default bank instance
DEFAULT_SETUP_BANK = SetupBank()


@dataclass
class CandidateEvaluation:
    """ENTRY-67: Evaluation metrics for a setup candidate."""
    setup_type: SetupType
    frequency: int  # number of setups detected
    sample_size: int  # number of trades
    expectancy: float  # R-multiple expectancy
    profit_factor: float
    sharpe: float
    max_drawdown: float
    regime_performance: dict[str, float]  # regime -> expectancy
    long_short_performance: dict[str, float]  # LONG/SHORT -> expectancy
    cost_sensitivity: dict[str, float]  # 1x/1.5x/2x/3x -> net expectancy

    def to_dict(self) -> dict:
        return {
            "setup_type": self.setup_type.value,
            "frequency": self.frequency,
            "sample_size": self.sample_size,
            "expectancy": self.expectancy,
            "profit_factor": self.profit_factor,
            "sharpe": self.sharpe,
            "max_drawdown": self.max_drawdown,
            "regime_performance": self.regime_performance,
            "long_short_performance": self.long_short_performance,
            "cost_sensitivity": self.cost_sensitivity,
        }


class AblationTester:
    """ENTRY-68: Ablation tests - which layer adds value."""

    LAYERS = [
        "base",           # Strategy alone
        "mtf",            # + MTF
        "orderflow",      # + OrderFlow
        "ml",             # + ML
        "quality",        # + EntryQuality
        "ev",             # + EV Gate
    ]

    def __init__(self, entry_engine: Any):
        self.engine = entry_engine

    def run_ablation(self, df: pd.DataFrame, setup_type: SetupType) -> dict[str, CandidateEvaluation]:
        """
        Compare: Strategy alone -> +MTF -> +OrderFlow -> +ML -> +Quality -> +EV
        Each layer adds one component, measure marginal value.
        """
        results = {}

        # This is a framework - actual implementation runs in backtest
        # Returns dict of layer_name -> CandidateEvaluation
        # Real implementation would run backtest with progressively enabled layers
        for layer in self.LAYERS:
            results[layer] = CandidateEvaluation(
                setup_type=setup_type,
                frequency=0,
                sample_size=0,
                expectancy=0.0,
                profit_factor=0.0,
                sharpe=0.0,
                max_drawdown=0.0,
                regime_performance={},
                long_short_performance={},
                cost_sensitivity={},
            )
        return results