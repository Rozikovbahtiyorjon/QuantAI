"""
Edge Proven Gate — defer Champion optimization until edge is validated.

Problem: QuantAI currently spends heavy engineering on
  Champion / Tournament / Evolution / Promotion / Replacement / Rollback / Stability / Transition
before proving that ANY strategy has a robust, statistically significant edge.

Fix: HARD gate before any Champion logic.

    Research → [Edge Proven?] ──NO──► RESEARCH_ONLY (NO_CHAMPION, no Tournament)
                     │
                    YES
                     ▼
           Integrity → Statistical → Robustness → Selection → Tournament → Champion

Edge is proven IFF at least one candidate passes STRICT Research Integrity
(permissive=False) on real OOS data with:
  - Integrity Checks (PF, window share, DD, net, trades, std)
  - Statistical Validation (Sharpe>0, bootstrap p<0.05, deflated Sharpe>0, PBO<0.6)
  - Robustness (cost 1.5x PF>1, MC/stress if evaluated)
  - Selection Adjustment (deflated/ Bonferroni)

Until edge proven, ChampionPipeline stays in RESEARCH state and returns
NO_CHAMPION with reason NO_EDGE_PROVEN. All heavy Champion components
are bypassed (no Tournament, no Evolution, no Promotion).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.champion.research_integrity import IntegrityConfig, ResearchIntegrityEngine


@dataclass
class EdgeValidationResult:
    edge_proven: bool
    n_candidates: int
    n_passed_strict: int
    passed_ids: tuple[str, ...]
    failed_ids: tuple[str, ...]
    reasons: dict[str, tuple[str, ...]]  # per-candidate first failure
    gate_details: dict[str, Any] = field(default_factory=dict)


class EdgeProvenGate:
    """
    Strict gate: uses ResearchIntegrityEngine(permissive=False) on real evaluations.
    Permissive research tests (synthetic, no MC) will NOT prove edge — that is correct.
    Only real OOS-validated strategies with full metrics can prove edge.
    """

    def __init__(self, strict_config: IntegrityConfig | None = None):
        # Strict production thresholds
        self.strict_config = strict_config or IntegrityConfig(
            min_pf_median=1.05,
            min_profitable_window_share=0.45,
            max_drawdown_median_pct=-15.0,
            min_net_median_pct=0.0,
            min_trades_total=30,
            max_net_std_pct=10.0,
            min_sharpe_median=0.0,
            require_sharpe_significance=True,
            max_sharpe_p_value=0.05,
            max_pbo=0.6,
            min_deflated_sharpe=0.0,
            require_cost_robust=True,
            min_monte_carlo_score=0.3,
            min_stress_score=0.3,
            apply_deflated_correction=True,
            require_regime_stability=False,  # regime needs explicit labels; enable when pipeline provides them
            min_regimes_positive=3,
            min_trades_per_regime=5,
            permissive=False,  # STRICT
        )
        self.strict_engine = ResearchIntegrityEngine(self.strict_config)

    def check(self, evaluations: dict[str, dict]) -> EdgeValidationResult:
        """
        evaluations: same dict as ChampionPipeline.decide_promotion receives
                     (after evaluate_all). Must contain metrics, windows, etc.
        Returns edge_proven True iff ≥1 candidate passes strict integrity.
        """
        if not evaluations:
            return EdgeValidationResult(
                edge_proven=False,
                n_candidates=0,
                n_passed_strict=0,
                passed_ids=(),
                failed_ids=(),
                reasons={},
                gate_details={"error": "no evaluations"},
            )

        report = self.strict_engine.assess(evaluations)
        passed = tuple(report.eligible.keys())
        failed = tuple(report.rejected.keys())
        reasons = {sid: rep.reasons for sid, rep in report.candidates.items() if not rep.overall_passed}

        return EdgeValidationResult(
            edge_proven=len(passed) > 0,
            n_candidates=len(evaluations),
            n_passed_strict=len(passed),
            passed_ids=passed,
            failed_ids=failed,
            reasons=reasons,
            gate_details={
                "strict_report": report,
            },
        )

    def is_proven(self, evaluations: dict[str, dict]) -> bool:
        return self.check(evaluations).edge_proven
