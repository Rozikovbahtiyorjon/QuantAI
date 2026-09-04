"""
ENTRY-05 — Shadow Mode

ENTRY_ENGINE_MODE = LEGACY / SHADOW / ACTIVE

In SHADOW:
  old engine makes decision
  new Entry Engine makes decision
  both are logged
  trades not changed

Compare:
  legacy BUY vs new BUY
  legacy HOLD vs new HOLD
  legacy BUY vs new HOLD

Only after successful shadow-validation:
  Old SignalGenerator → deprecated adapter → New Entry Engine
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Any
import pandas as pd
from enum import Enum


class EntryEngineMode(str, Enum):
    LEGACY = "LEGACY"  # only old SignalGenerator
    SHADOW = "SHADOW"  # both, compare, old decides trades
    ACTIVE = "ACTIVE"  # only new EntryEngine


@dataclass
class ShadowComparison:
    bar_timestamp: Any
    legacy_signal: str
    new_signal: str
    legacy_entry: float
    new_entry: float
    match: bool  # legacy == new?
    divergence_type: str  # BOTH_HOLD, BOTH_BUY, DIVERGENCE
    legacy_reasons: list[str] = field(default_factory=list)
    new_reasons: list[str] = field(default_factory=list)


@dataclass
class ShadowStats:
    total_bars: int = 0
    both_hold: int = 0
    both_buy: int = 0
    both_sell: int = 0
    divergence: int = 0
    legacy_buy_new_hold: int = 0
    legacy_hold_new_buy: int = 0

    @property
    def agreement_rate(self) -> float:
        if self.total_bars == 0:
            return 0.0
        return (self.both_hold + self.both_buy + self.both_sell) / self.total_bars

    def to_dict(self) -> dict:
        return {
            "total_bars": self.total_bars,
            "both_hold": self.both_hold,
            "both_buy": self.both_buy,
            "both_sell": self.both_sell,
            "divergence": self.divergence,
            "legacy_buy_new_hold": self.legacy_buy_new_hold,
            "legacy_hold_new_buy": self.legacy_hold_new_buy,
            "agreement_rate": self.agreement_rate,
        }


class ShadowEngine:
    """Runs both engines in SHADOW mode and compares."""

    def __init__(self, mode: EntryEngineMode = EntryEngineMode.SHADOW):
        self.mode = mode
        self.comparisons: list[ShadowComparison] = []
        self.stats = ShadowStats()

        # Lazy init engines
        self._legacy_adapter = None
        self._new_engine = None

    def _get_legacy(self):
        if self._legacy_adapter is None:
            from src.entry.adapter import ExistingSignalAdapter
            self._legacy_adapter = ExistingSignalAdapter()
        return self._legacy_adapter

    def _get_new(self):
        if self._new_engine is None:
            from src.entry.engine import EntryDecisionEngine
            self._new_engine = EntryDecisionEngine()
        return self._new_engine

    def process_bar(self, df: pd.DataFrame, order_flow_signal=None) -> dict:
        """
        Process one bar in SHADOW mode.

        Returns dict with legacy/new decisions and comparison.
        Trades are decided by legacy in SHADOW, by new in ACTIVE.
        """
        legacy_adapter = self._get_legacy()
        new_engine = self._get_new()

        legacy_candidate = legacy_adapter.generate_legacy_candidate(df, order_flow_signal=order_flow_signal)
        new_decision = new_engine.evaluate(df)

        legacy_signal = legacy_candidate.signal
        new_signal = new_decision.signal

        # Determine divergence
        if legacy_signal == new_signal:
            if legacy_signal == "HOLD":
                div_type = "BOTH_HOLD"
                self.stats.both_hold += 1
            elif legacy_signal == "BUY":
                div_type = "BOTH_BUY"
                self.stats.both_buy += 1
            elif legacy_signal == "SELL":
                div_type = "BOTH_SELL"
                self.stats.both_sell += 1
            else:
                div_type = "BOTH_HOLD"
            match = True
        else:
            div_type = "DIVERGENCE"
            self.stats.divergence += 1
            match = False
            if legacy_signal == "BUY" and new_signal == "HOLD":
                self.stats.legacy_buy_new_hold += 1
            elif legacy_signal == "HOLD" and new_signal in ("BUY", "SELL"):
                self.stats.legacy_hold_new_buy += 1

        self.stats.total_bars += 1

        ts = df.iloc[-1].get("timestamp") if "timestamp" in df.columns else None
        comp = ShadowComparison(
            bar_timestamp=ts,
            legacy_signal=legacy_signal,
            new_signal=new_signal,
            legacy_entry=legacy_candidate.entry,
            new_entry=new_decision.entry_zone.ideal_entry if new_decision.entry_zone else 0.0,
            match=match,
            divergence_type=div_type,
            legacy_reasons=legacy_candidate.reasons[:2],
            new_reasons=list(new_decision.reason_codes)[:2],
        )
        self.comparisons.append(comp)

        # Decide which engine's decision to use for actual trading
        if self.mode == EntryEngineMode.LEGACY:
            final_signal = legacy_signal
            final_source = "LEGACY"
        elif self.mode == EntryEngineMode.ACTIVE:
            final_signal = new_signal
            final_source = "NEW"
        else:  # SHADOW: legacy decides, new is shadow
            final_signal = legacy_signal
            final_source = "LEGACY_SHADOW"

        return {
            "final_signal": final_signal,
            "final_source": final_source,
            "legacy": legacy_candidate,
            "new": new_decision,
            "comparison": comp,
            "stats": self.stats.to_dict(),
            "match": match,
            "divergence_type": div_type,
        }

    def get_stats(self) -> dict:
        return self.stats.to_dict()

    def should_promote(self, min_bars: int = 500, min_agreement: float = 0.7, max_divergence_buy_hold: float = 0.15) -> tuple[bool, str]:
        """
        Check if new engine is ready to promote from SHADOW to ACTIVE.

        Criteria:
        - At least min_bars processed
        - Agreement rate >= min_agreement
        - legacy BUY → new HOLD divergence not too high (would miss too many trades)
        """
        if self.stats.total_bars < min_bars:
            return False, f"need {min_bars} bars, have {self.stats.total_bars}"
        if self.stats.agreement_rate < min_agreement:
            return False, f"agreement {self.stats.agreement_rate:.2%} < {min_agreement:.0%}"
        # Check divergence is not too high in one direction
        div_rate = self.stats.legacy_buy_new_hold / max(1, self.stats.total_bars)
        if div_rate > max_divergence_buy_hold:
            return False, f"legacy BUY→new HOLD divergence {div_rate:.2%} > {max_divergence_buy_hold:.0%} (new too strict)"
        return True, f"ready to promote: agreement {self.stats.agreement_rate:.2%}, bars {self.stats.total_bars}"
