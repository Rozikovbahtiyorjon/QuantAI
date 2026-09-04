"""
Complexity → Evidence Quality Gate

Enforces: Complexity ↑ must be accompanied by Evidence quality ↑↑

Otherwise autonomous system will accelerate overfitting by
working on data/results it overly trusts.

Complexity: n_params, n_indicators, n_trials, n_experiments, OOS touches
Evidence: PF, Sharpe, PBO, DSR, WRC, regime stability, cost robustness, sample size

Rule: Required evidence thresholds scale with complexity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
import math


@dataclass
class ComplexityScore:
    n_params: int
    n_indicators: int
    n_trials: int
    n_experiments: int
    oos_touches: int

    @property
    def score(self) -> float:
        # Complexity grows with params, indicators, trials, experiments
        # Use log scale to avoid explosion, but still monotonic
        # Calibrated so typical research (5 params, 10 indicators, 50 trials, 20 exps) = medium
        return (
            math.log1p(self.n_params) * 0.8
            + math.log1p(self.n_indicators) * 0.5
            + math.log1p(self.n_trials) * 0.4
            + math.log1p(self.n_experiments) * 0.3
            + math.log1p(self.oos_touches) * 0.6
        )

    def tier(self) -> str:
        s = self.score
        if s < 4.0:
            return "low"
        if s < 6.0:
            return "medium"
        if s < 8.0:
            return "high"
        return "very_high"


@dataclass
class EvidenceQuality:
    pf: float = 0.0
    sharpe: float = 0.0
    pbo: float = 1.0  # lower is better
    dsr: float = -1.0  # higher is better
    wrc_p: float = 1.0  # lower is better
    regime_stability: bool = False
    cost_robust: bool = False
    trades: int = 0
    oos_days: int = 0

    @property
    def score(self) -> float:
        # Evidence quality composite
        s = 0.0
        s += max(0, (self.pf - 1.0) * 2.0)  # PF 1.0 -> 0, 1.5 -> 1.0, 2.0 -> 2.0
        s += max(0, self.sharpe) * 0.5
        s += max(0, (0.6 - self.pbo) * 3.0)  # PBO 0.6 -> 0, 0.3 -> 0.9
        s += max(0, (self.dsr - 0.5) * 1.0)  # DSR 0.5 -> 0, 1.0 -> 0.5
        s += max(0, (0.05 - self.wrc_p) * 10.0)  # WRC p 0.05 -> 0, 0.01 -> 0.4
        s += 0.5 if self.regime_stability else 0
        s += 0.5 if self.cost_robust else 0
        # Sample size bonus
        s += min(1.0, self.trades / 100.0) * 0.5
        s += min(1.0, self.oos_days / 90.0) * 0.5
        return s


# Required evidence per complexity tier
REQUIRED_EVIDENCE = {
    "low": EvidenceQuality(pf=1.05, sharpe=0.0, pbo=0.6, dsr=0.0, wrc_p=0.1, trades=30, oos_days=30),
    "medium": EvidenceQuality(pf=1.2, sharpe=0.5, pbo=0.5, dsr=0.5, wrc_p=0.05, trades=50, oos_days=60),
    "high": EvidenceQuality(pf=1.4, sharpe=0.8, pbo=0.4, dsr=0.8, wrc_p=0.03, trades=100, oos_days=90),
    "very_high": EvidenceQuality(pf=1.6, sharpe=1.0, pbo=0.3, dsr=1.0, wrc_p=0.01, trades=200, oos_days=180),
}


def check_complexity_evidence(
    complexity: ComplexityScore,
    evidence: EvidenceQuality,
) -> tuple[bool, str, Dict[str, Any]]:
    """
    Check if evidence quality matches complexity tier.
    Returns (passed, reason, details)
    """
    tier = complexity.tier()
    required = REQUIRED_EVIDENCE[tier]

    failures = []

    if evidence.pf < required.pf:
        failures.append(f"PF {evidence.pf:.2f} < required {required.pf:.2f} for complexity {tier} (score {complexity.score:.2f})")

    if evidence.trades < required.trades:
        failures.append(f"trades {evidence.trades} < required {required.trades} for {tier}")

    if evidence.oos_days < required.oos_days:
        failures.append(f"OOS days {evidence.oos_days} < required {required.oos_days} for {tier}")

    if evidence.pbo > required.pbo:
        failures.append(f"PBO {evidence.pbo:.2f} > required {required.pbo:.2f} for {tier} (overfit risk)")

    if evidence.dsr < required.dsr:
        failures.append(f"DSR {evidence.dsr:.2f} < required {required.dsr:.2f} for {tier}")

    if evidence.wrc_p > required.wrc_p:
        failures.append(f"WRC p {evidence.wrc_p:.3f} > required {required.wrc_p:.3f} for {tier}")

    if tier in ("high", "very_high"):
        if not evidence.regime_stability:
            failures.append(f"regime stability required for {tier} complexity")
        if not evidence.cost_robust:
            failures.append(f"cost robustness required for {tier} complexity")

    details = {
        "complexity_score": complexity.score,
        "complexity_tier": tier,
        "evidence_score": evidence.score,
        "required": required.__dict__,
        "provided": evidence.__dict__,
        "failures": failures,
    }

    if failures:
        return False, f"Evidence quality insufficient for complexity {tier}: " + "; ".join(failures), details

    return True, f"Evidence quality matches complexity {tier}", details


# Integration with existing guards
def enforce_complexity_evidence_link(
    n_params: int,
    n_indicators: int,
    n_trials: int,
    n_experiments: int,
    oos_touches: int,
    evidence_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Main entry for autonomous loop.
    Call before allowing complexity increase.

    evidence_dict should contain: pf, sharpe, pbo, dsr, wrc_p, regime_stability, cost_robust, trades, oos_days
    """
    complexity = ComplexityScore(
        n_params=n_params,
        n_indicators=n_indicators,
        n_trials=n_trials,
        n_experiments=n_experiments,
        oos_touches=oos_touches,
    )

    evidence = EvidenceQuality(
        pf=float(evidence_dict.get("pf", 0) or 0),
        sharpe=float(evidence_dict.get("sharpe", 0) or 0),
        pbo=float(evidence_dict.get("pbo", 1.0) or 1.0),
        dsr=float(evidence_dict.get("dsr", -1) or -1),
        wrc_p=float(evidence_dict.get("wrc_p", 1.0) or 1.0),
        regime_stability=bool(evidence_dict.get("regime_stability", False)),
        cost_robust=bool(evidence_dict.get("cost_robust", False)),
        trades=int(evidence_dict.get("trades", 0) or 0),
        oos_days=int(evidence_dict.get("oos_days", 0) or 0),
    )

    passed, reason, details = check_complexity_evidence(complexity, evidence)

    if not passed:
        raise ValueError(f"COMPLEXITY-EVIDENCE LINK VIOLATION: {reason} | Complexity {complexity.score:.2f} requires Evidence {REQUIRED_EVIDENCE[complexity.tier()].score:.2f}, got {evidence.score:.2f}. Increase evidence quality before increasing complexity (P0.6).")

    return details
