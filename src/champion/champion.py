"""
PHASE 17 — CHAMPION
ENTRY-74 — Candidate → Champion (7 gates)
ENTRY-75 — NO_VERIFIED_CHAMPION (honest result)
ENTRY-76 — Champion Feedback (expected vs actual)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class ChampionGate(str, Enum):
    OOS_PASS = "OOS_PASS"
    SAMPLE_PASS = "SAMPLE_PASS"
    COST_PASS = "COST_PASS"
    SLIPPAGE_PASS = "SLIPPAGE_PASS"
    REGIME_PASS = "REGIME_PASS"
    STATISTICAL_PASS = "STATISTICAL_PASS"
    PAPER_PASS = "PAPER_PASS"


@dataclass
class GateResult:
    gate: ChampionGate
    passed: bool
    metric_value: float
    threshold: float
    details: str


@dataclass
class ChampionCandidate:
    """A candidate that has passed all validation gates."""
    candidate_id: str
    setup_type: str
    params: dict
    gate_results: list[GateResult]
    oos_metrics: dict
    paper_metrics: dict | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def all_gates_passed(self) -> bool:
        return all(g.passed for g in self.gate_results)

    @property
    def failed_gates(self) -> list[ChampionGate]:
        return [g.gate for g in self.gate_results if not g.passed]


class ChampionSelector:
    """ENTRY-74: Candidate becomes Champion only after ALL 7 gates pass."""

    GATE_THRESHOLDS = {
        ChampionGate.OOS_PASS: {"min_profit_factor": 1.2, "min_expectancy": 0.1, "min_trades": 30},
        ChampionGate.SAMPLE_PASS: {"min_trades": 100, "min_windows": 5},
        ChampionGate.COST_PASS: {"max_cost_drag_pct": 0.30},  # costs don't eat >30% of gross
        ChampionGate.SLIPPAGE_PASS: {"max_slippage_degradation_pct": 0.25},  # 1.5x slippage
        ChampionGate.REGIME_PASS: {"min_regimes_profitable": 3, "max_regime_dd": 0.15},
        ChampionGate.STATISTICAL_PASS: {"min_pbo": 0.05, "min_dsr": 1.0, "min_wrc": 0.5},
        ChampionGate.PAPER_PASS: {"min_paper_trades": 50, "max_paper_dd": 0.10, "min_paper_pf": 1.1},
    }

    def evaluate_candidate(
        self,
        candidate_id: str,
        setup_type: str,
        params: dict,
        oos_metrics: dict,
        paper_metrics: dict | None = None,
    ) -> ChampionCandidate:
        """Run all 7 gates."""
        gate_results = []

        # Gate 1: OOS PASS
        gate_results.append(self._check_oos_pass(oos_metrics))

        # Gate 2: SAMPLE PASS
        gate_results.append(self._check_sample_pass(oos_metrics))

        # Gate 3: COST PASS
        gate_results.append(self._check_cost_pass(oos_metrics))

        # Gate 4: SLIPPAGE PASS
        gate_results.append(self._check_slippage_pass(oos_metrics))

        # Gate 5: REGIME PASS
        gate_results.append(self._check_regime_pass(oos_metrics))

        # Gate 6: STATISTICAL PASS (PBO/DSR/WRC)
        gate_results.append(self._check_statistical_pass(oos_metrics))

        # Gate 7: PAPER PASS (if paper data available)
        if paper_metrics:
            gate_results.append(self._check_paper_pass(paper_metrics))
        else:
            gate_results.append(GateResult(
                gate=ChampionGate.PAPER_PASS,
                passed=False,
                metric_value=0,
                threshold=1,
                details="No paper trading data available",
            ))

        return ChampionCandidate(
            candidate_id=candidate_id,
            setup_type=setup_type,
            params=params,
            gate_results=gate_results,
            oos_metrics=oos_metrics,
            paper_metrics=paper_metrics,
        )

    def _check_oos_pass(self, metrics: dict) -> GateResult:
        t = self.GATE_THRESHOLDS[ChampionGate.OOS_PASS]
        passed = (
            metrics.get("profit_factor", 0) >= t["min_profit_factor"]
            and metrics.get("expectancy", 0) >= t["min_expectancy"]
            and metrics.get("total_trades", 0) >= t["min_trades"]
        )
        return GateResult(
            gate=ChampionGate.OOS_PASS,
            passed=passed,
            metric_value=metrics.get("profit_factor", 0),
            threshold=t["min_profit_factor"],
            details=f"PF={metrics.get('profit_factor', 0):.2f}, Exp={metrics.get('expectancy', 0):.3f}, Trades={metrics.get('total_trades', 0)}",
        )

    def _check_sample_pass(self, metrics: dict) -> GateResult:
        t = self.GATE_THRESHOLDS[ChampionGate.SAMPLE_PASS]
        passed = (
            metrics.get("total_trades", 0) >= t["min_trades"]
            and metrics.get("windows", 0) >= t["min_windows"]
        )
        return GateResult(
            gate=ChampionGate.SAMPLE_PASS,
            passed=passed,
            metric_value=metrics.get("total_trades", 0),
            threshold=t["min_trades"],
            details=f"Trades={metrics.get('total_trades', 0)}, Windows={metrics.get('windows', 0)}",
        )

    def _check_cost_pass(self, metrics: dict) -> GateResult:
        t = self.GATE_THRESHOLDS[ChampionGate.COST_PASS]
        gross = metrics.get("gross_profit", 0)
        net = metrics.get("net_profit", 0)
        cost_drag = 1 - (net / gross) if gross > 0 else 1
        passed = cost_drag <= t["max_cost_drag_pct"]
        return GateResult(
            gate=ChampionGate.COST_PASS,
            passed=passed,
            metric_value=cost_drag,
            threshold=t["max_cost_drag_pct"],
            details=f"Cost drag={cost_drag:.1%}",
        )

    def _check_slippage_pass(self, metrics: dict) -> GateResult:
        t = self.GATE_THRESHOLDS[ChampionGate.SLIPPAGE_PASS]
        base_pf = metrics.get("profit_factor", 0)
        stress_pf = metrics.get("stress_1.5x_slippage_profit_factor", 0)
        degradation = (base_pf - stress_pf) / max(base_pf, 0.001) if base_pf > 0 else 1
        passed = degradation <= t["max_slippage_degradation_pct"]
        return GateResult(
            gate=ChampionGate.SLIPPAGE_PASS,
            passed=passed,
            metric_value=degradation,
            threshold=t["max_slippage_degradation_pct"],
            details=f"Base PF={base_pf:.2f}, 1.5x Slippage PF={stress_pf:.2f}, Deg={degradation:.1%}",
        )

    def _check_regime_pass(self, metrics: dict) -> GateResult:
        t = self.GATE_THRESHOLDS[ChampionGate.REGIME_PASS]
        regime_perf = metrics.get("regime_performance", {})
        profitable = sum(1 for v in regime_perf.values() if v > 0)
        max_dd = max((v for v in regime_perf.values() if v < 0), default=0)
        passed = profitable >= t["min_regimes_profitable"] and abs(max_dd) <= t["max_regime_dd"]
        return GateResult(
            gate=ChampionGate.REGIME_PASS,
            passed=passed,
            metric_value=profitable,
            threshold=t["min_regimes_profitable"],
            details=f"Profitable regimes={profitable}, Max regime DD={abs(max_dd):.1%}",
        )

    def _check_statistical_pass(self, metrics: dict) -> GateResult:
        t = self.GATE_THRESHOLDS[ChampionGate.STATISTICAL_PASS]
        pbo = metrics.get("pbo", 1.0)
        dsr = metrics.get("dsr", 0)
        wrc = metrics.get("wrc", 0)
        passed = pbo <= t["min_pbo"] and dsr >= t["min_dsr"] and wrc >= t["min_wrc"]
        return GateResult(
            gate=ChampionGate.STATISTICAL_PASS,
            passed=passed,
            metric_value=dsr,
            threshold=t["min_dsr"],
            details=f"PBO={pbo:.3f}, DSR={dsr:.2f}, WRC={wrc:.2f}",
        )

    def _check_paper_pass(self, metrics: dict) -> GateResult:
        t = self.GATE_THRESHOLDS[ChampionGate.PAPER_PASS]
        passed = (
            metrics.get("paper_trades", 0) >= t["min_paper_trades"]
            and metrics.get("paper_max_dd", 1) <= t["max_paper_dd"]
            and metrics.get("paper_profit_factor", 0) >= t["min_paper_pf"]
        )
        return GateResult(
            gate=ChampionGate.PAPER_PASS,
            passed=passed,
            metric_value=metrics.get("paper_profit_factor", 0),
            threshold=t["min_paper_pf"],
            details=f"Paper Trades={metrics.get('paper_trades', 0)}, DD={metrics.get('paper_max_dd', 0):.1%}, PF={metrics.get('paper_profit_factor', 0):.2f}",
        )


class NoVerifiedChampion(Exception):
    """ENTRY-75: Raised when NO candidate passes all gates."""
    def __init__(self, candidates: list[ChampionCandidate]):
        self.candidates = candidates
        failed_summary = []
        for c in candidates:
            failed = [g.gate.value for g in c.gate_results if not g.passed]
            failed_summary.append(f"{c.candidate_id}: {', '.join(failed)}")
        msg = "NO_VERIFIED_CHAMPION\n" + "\n".join(failed_summary)
        super().__init__(msg)


class ChampionRegistry:
    """Registry of verified champions."""

    def __init__(self):
        self.champions: list[ChampionCandidate] = []
        self.rejected: list[ChampionCandidate] = []

    def try_promote(self, candidate: ChampionCandidate) -> bool:
        if candidate.all_gates_passed:
            self.champions.append(candidate)
            return True
        else:
            self.rejected.append(candidate)
            return False

    def get_champion(self, setup_type: str | None = None) -> ChampionCandidate | None:
        """Get best champion (highest OOS expectancy)."""
        candidates = self.champions
        if setup_type:
            candidates = [c for c in candidates if c.setup_type == setup_type]
        if not candidates:
            return None
        return max(candidates, key=lambda c: c.oos_metrics.get("expectancy", 0))


class ChampionFeedback:
    """ENTRY-76: Expected vs Actual feedback after paper trading."""

    @dataclass
    class FeedbackReport:
        champion_id: str
        expected_entry_quality: float
        actual_execution_quality: float
        quality_gap: float
        expected_ev: float
        realized_ev: float
        ev_gap: float
        fill_rate_expected: float
        fill_rate_actual: float
        slippage_expected: float
        slippage_actual: float
        lessons: list[str]

    def generate_feedback(
        self,
        champion: ChampionCandidate,
        paper_results: dict,
    ) -> FeedbackReport:
        """Compare expected vs actual after paper trading."""
        expected_quality = champion.oos_metrics.get("avg_entry_quality", 0)
        actual_quality = paper_results.get("avg_execution_quality", 0)
        expected_ev = champion.oos_metrics.get("expectancy", 0)
        realized_ev = paper_results.get("realized_expectancy", 0)

        lessons = []
        if actual_quality < expected_quality - 0.1:
            lessons.append("Entry quality degraded in live conditions - review zone/trigger logic")
        if realized_ev < expected_ev * 0.7:
            lessons.append("EV realization < 70% of expected - check execution/slippage model")
        if paper_results.get("fill_rate", 1) < 0.8:
            lessons.append("Low fill rate - review limit order placement / chase logic")

        return self.FeedbackReport(
            champion_id=champion.candidate_id,
            expected_entry_quality=expected_quality,
            actual_execution_quality=actual_quality,
            quality_gap=expected_quality - actual_quality,
            expected_ev=expected_ev,
            realized_ev=realized_ev,
            ev_gap=expected_ev - realized_ev,
            fill_rate_expected=champion.oos_metrics.get("expected_fill_rate", 1),
            fill_rate_actual=paper_results.get("fill_rate", 1),
            slippage_expected=champion.oos_metrics.get("expected_slippage_bps", 0),
            slippage_actual=paper_results.get("avg_slippage_bps", 0),
            lessons=lessons,
        )