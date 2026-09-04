"""
PHASE 18 — SELF-IMPROVEMENT
ENTRY-77 — Feedback loop (Paper → Results → Error Analysis → Hypothesis → Candidate → Backtest → WF → Paper)
ENTRY-78 — Entry error taxonomy
ENTRY-79 — Autonomous candidate generation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from collections import Counter


class EntryErrorType(str, Enum):
    """ENTRY-78: Taxonomy of entry errors for Supervisor feedback."""
    FALSE_BREAKOUT = "FALSE_BREAKOUT"
    LATE_TRIGGER = "LATE_TRIGGER"
    BAD_ZONE = "BAD_ZONE"
    ML_FALSE_POSITIVE = "ML_FALSE_POSITIVE"
    ORDERFLOW_FAILURE = "ORDERFLOW_FAILURE"
    BAD_REGIME = "BAD_REGIME"
    LOW_EV = "LOW_EV"
    SL_TOO_TIGHT = "SL_TOO_TIGHT"
    MISSED_ENTRY = "MISSED_ENTRY"
    EXECUTION_MISS = "EXECUTION_MISS"
    INVALIDATED_SETUP = "INVALIDATED_SETUP"
    TRIGGER_EXPIRED = "TRIGGER_EXPIRED"
    CONFIRMATION_FAILED = "CONFIRMATION_FAILED"
    RISK_REJECTED = "RISK_REJECTED"


@dataclass
class EntryError:
    """Single entry error with context."""
    error_type: EntryErrorType
    timestamp: datetime
    setup_type: str
    symbol: str
    timeframe: str
    context: dict  # regime, zone, trigger, etc.
    expected_outcome: str
    actual_outcome: str
    pnl_impact: float  # R-multiple impact


@dataclass
class ErrorAnalysisReport:
    """Aggregated error analysis for a period."""
    period_start: datetime
    period_end: datetime
    total_trades: int
    total_errors: int
    error_counts: dict[EntryErrorType, int]
    top_errors: list[tuple[EntryErrorType, int]]
    error_rate_by_setup: dict[str, float]
    error_rate_by_regime: dict[str, float]
    cost_impact_by_error: dict[EntryErrorType, float]
    recommendations: list[str]


class ErrorTaxonomyAnalyzer:
    """ENTRY-78: Analyze entry errors and generate feedback for Supervisor."""

    def analyze(self, errors: list[EntryError], trades: list[dict]) -> ErrorAnalysisReport:
        """Generate comprehensive error analysis."""
        error_counts = Counter(e.error_type for e in errors)
        total_errors = len(errors)
        total_trades = len(trades)

        # Top errors
        top_errors = error_counts.most_common(5)

        # Error rate by setup
        setup_errors = Counter(e.setup_type for e in errors)
        setup_trades = Counter(t.get("setup_type", "UNKNOWN") for t in trades)
        error_rate_by_setup = {
            s: setup_errors[s] / max(setup_trades[s], 1)
            for s in setup_trades
        }

        # Error rate by regime
        regime_errors = Counter(e.context.get("regime", "UNKNOWN") for e in errors)
        regime_trades = Counter(t.get("regime", "UNKNOWN") for t in trades)
        error_rate_by_regime = {
            r: regime_errors[r] / max(regime_trades[r], 1)
            for r in regime_trades
        }

        # Cost impact by error type
        cost_impact = {}
        for err_type in EntryErrorType:
            type_errors = [e for e in errors if e.error_type == err_type]
            if type_errors:
                cost_impact[err_type] = sum(e.pnl_impact for e in type_errors) / len(type_errors)

        # Generate recommendations
        recommendations = self._generate_recommendations(top_errors, error_rate_by_setup, error_rate_by_regime)

        return ErrorAnalysisReport(
            period_start=min((e.timestamp for e in errors), default=datetime.now(timezone.utc)),
            period_end=max((e.timestamp for e in errors), default=datetime.now(timezone.utc)),
            total_trades=total_trades,
            total_errors=total_errors,
            error_counts=dict(error_counts),
            top_errors=top_errors,
            error_rate_by_setup=error_rate_by_setup,
            error_rate_by_regime=error_rate_by_regime,
            cost_impact_by_error=cost_impact,
            recommendations=recommendations,
        )

    def _generate_recommendations(
        self,
        top_errors: list[tuple[EntryErrorType, int]],
        error_by_setup: dict,
        error_by_regime: dict,
    ) -> list[str]:
        recs = []

        for err_type, count in top_errors[:3]:
            if err_type == EntryErrorType.FALSE_BREAKOUT:
                recs.append("High FALSE_BREAKOUT rate: tighten breakout confirmation, require volume + orderflow")
            elif err_type == EntryErrorType.LATE_TRIGGER:
                recs.append("LATE_TRIGGER: reduce trigger confirmation bars, use more aggressive trigger")
            elif err_type == EntryErrorType.BAD_ZONE:
                recs.append("BAD_ZONE: improve zone freshness/strength filters, require confluence")
            elif err_type == EntryErrorType.ML_FALSE_POSITIVE:
                recs.append("ML_FALSE_POSITIVE: recalibrate ML, increase probability threshold, add regime filter")
            elif err_type == EntryErrorType.ORDERFLOW_FAILURE:
                recs.append("ORDERFLOW_FAILURE: make orderflow mandatory for this setup, or disable if unavailable")
            elif err_type == EntryErrorType.BAD_REGIME:
                recs.append("BAD_REGIME: add regime filter to skip this setup in unfavorable regimes")
            elif err_type == EntryErrorType.LOW_EV:
                recs.append("LOW_EV: raise EV threshold, improve target selection")
            elif err_type == EntryErrorType.SL_TOO_TIGHT:
                recs.append("SL_TOO_TIGHT: use structural SL + ATR buffer, avoid fixed multipliers")
            elif err_type == EntryErrorType.MISSED_ENTRY:
                recs.append("MISSED_ENTRY: increase max_chase, use limit orders at zone edge")
            elif err_type == EntryErrorType.EXECUTION_MISS:
                recs.append("EXECUTION_MISS: review fill model, adjust queue position assumptions")

        # Setup-specific
        for setup, rate in error_by_setup.items():
            if rate > 0.4:
                recs.append(f"Setup {setup} error rate {rate:.0%}: consider disabling or major revision")

        # Regime-specific
        for regime, rate in error_by_regime.items():
            if rate > 0.5:
                recs.append(f"Regime {regime} error rate {rate:.0%}: add regime gate")

        return recs


@dataclass
class ImprovementHypothesis:
    """ENTRY-79: Autonomous hypothesis generated from error analysis."""
    hypothesis_id: str
    source_error: EntryErrorType
    description: str
    proposed_change: dict  # e.g., {"component": "trigger", "param": "confirmation_bars", "from": 3, "to": 1}
    expected_impact: str  # e.g., "reduce LATE_TRIGGER by 50%"
    risk: str  # e.g., "may increase false triggers"
    priority: int  # 1=high, 5=low


class AutonomousHypothesisGenerator:
    """ENTRY-79: Supervisor generates hypotheses from error analysis."""

    def generate_hypotheses(self, error_report: ErrorAnalysisReport) -> list[ImprovementHypothesis]:
        """Generate improvement hypotheses from error patterns."""
        hypotheses = []

        for err_type, count in error_report.top_errors:
            if err_type == EntryErrorType.FALSE_BREAKOUT:
                hypotheses.append(ImprovementHypothesis(
                    hypothesis_id=f"HYP_{err_type.value}_{datetime.now(timezone.utc).strftime('%Y%m%d')}",
                    source_error=err_type,
                    description="Breakout confirmations failing - price breaks but reverses",
                    proposed_change={"component": "confirmation", "add_requirement": "volume_surge_1.5x + orderflow_absorption"},
                    expected_impact="Reduce FALSE_BREAKOUT by 40-60%",
                    risk="May miss valid breakouts with lower volume",
                    priority=1,
                ))

            elif err_type == EntryErrorType.LATE_TRIGGER:
                hypotheses.append(ImprovementHypothesis(
                    hypothesis_id=f"HYP_{err_type.value}_{datetime.now(timezone.utc).strftime('%Y%m%d')}",
                    source_error=err_type,
                    description="Triggers fire after move has extended",
                    proposed_change={"component": "trigger", "param": "max_wait_bars", "from": 20, "to": 10},
                    expected_impact="Catch entries earlier, reduce LATE_TRIGGER by 50%",
                    risk="May increase premature triggers",
                    priority=2,
                ))

            elif err_type == EntryErrorType.BAD_ZONE:
                hypotheses.append(ImprovementHypothesis(
                    hypothesis_id=f"HYP_{err_type.value}_{datetime.now(timezone.utc).strftime('%Y%m%d')}",
                    source_error=err_type,
                    description="Zones lack confluence or are stale",
                    proposed_change={"component": "zone", "param": "min_freshness", "from": 0.3, "to": 0.5, "require_confluence": True},
                    expected_impact="Filter weak zones, reduce BAD_ZONE by 30%",
                    risk="May reduce setup frequency",
                    priority=2,
                ))

            elif err_type == EntryErrorType.ML_FALSE_POSITIVE:
                hypotheses.append(ImprovementHypothesis(
                    hypothesis_id=f"HYP_{err_type.value}_{datetime.now(timezone.utc).strftime('%Y%m%d')}",
                    source_error=err_type,
                    description="ML overconfident in low-quality setups",
                    proposed_change={"component": "ml", "param": "prob_threshold", "from": 0.55, "to": 0.65, "add_regime_condition": True},
                    expected_impact="Reduce ML false positives by 40%",
                    risk="May reject valid ML signals",
                    priority=1,
                ))

            elif err_type == EntryErrorType.SL_TOO_TIGHT:
                hypotheses.append(ImprovementHypothesis(
                    hypothesis_id=f"HYP_{err_type.value}_{datetime.now(timezone.utc).strftime('%Y%m%d')}",
                    source_error=err_type,
                    description="Fixed ATR stops hit by noise before move continues",
                    proposed_change={"component": "sl_tp", "method": "structural_sl", "buffer_atr": 0.5},
                    expected_impact="Reduce premature stops by 50%",
                    risk="Larger stops reduce R:R, need better targets",
                    priority=2,
                ))

            elif err_type == EntryErrorType.EXECUTION_MISS:
                hypotheses.append(ImprovementHypothesis(
                    hypothesis_id=f"HYP_{err_type.value}_{datetime.now(timezone.utc).strftime('%Y%m%d')}",
                    source_error=err_type,
                    description="Limit orders not filled, price moves away",
                    proposed_change={"component": "execution", "policy": "LIMIT_MAKER", "max_chase_atr": 0.3},
                    expected_impact="Improve fill rate from 60% to 80%",
                    risk="Worse average entry price",
                    priority=3,
                ))

        # Sort by priority
        hypotheses.sort(key=lambda h: h.priority)
        return hypotheses


class FeedbackLoop:
    """ENTRY-77: Complete feedback loop Paper → Error Analysis → Hypothesis → Candidate → Backtest → WF → Paper."""

    def __init__(
        self,
        error_analyzer: ErrorTaxonomyAnalyzer,
        hypothesis_generator: AutonomousHypothesisGenerator,
        champion_selector: Any,  # ChampionSelector
        backtest_engine: Any,
        wf_validator: Any,  # WalkForwardValidator
    ):
        self.error_analyzer = error_analyzer
        self.hypothesis_generator = hypothesis_generator
        self.champion_selector = champion_selector
        self.backtest_engine = backtest_engine
        self.wf_validator = wf_validator

    def run_iteration(
        self,
        paper_trades: list[dict],
        paper_errors: list[EntryError],
    ) -> dict:
        """
        One iteration of self-improvement loop:
        1. Paper trading results + errors
        2. Error analysis
        3. Generate hypotheses
        4. Create new candidates from hypotheses
        5. Backtest candidates
        6. Walk-forward validation
        7. If passes → paper trading
        """
        # Step 1-2: Error Analysis
        error_report = self.error_analyzer.analyze(paper_errors, paper_trades)

        # Step 3: Generate Hypotheses
        hypotheses = self.hypothesis_generator.generate_hypotheses(error_report)

        # Step 4: Create Candidates (simplified - real impl creates strategy variants)
        candidates = []
        for hyp in hypotheses[:3]:  # Top 3 hypotheses
            candidate = self._create_candidate_from_hypothesis(hyp)
            candidates.append(candidate)

        # Step 5-6: Backtest + WF (framework - actual run is async)
        results = {
            "error_report": error_report,
            "hypotheses": hypotheses,
            "candidates": candidates,
            "status": "READY_FOR_BACKTEST",
        }

        return results

    def _create_candidate_from_hypothesis(self, hypothesis: ImprovementHypothesis) -> dict:
        """Create a new strategy candidate from hypothesis."""
        return {
            "hypothesis_id": hypothesis.hypothesis_id,
            "base_strategy": "current_champion",
            "modifications": hypothesis.proposed_change,
            "expected_improvement": hypothesis.expected_impact,
            "risk": hypothesis.risk,
        }