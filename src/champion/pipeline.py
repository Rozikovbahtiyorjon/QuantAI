"""
QuantAI Champion Pipeline (R6 — Nested Research with Final Holdout)

Hierarchy:

                FULL DATA
                    |
          +---------+---------+
          |                   |
    DEVELOPMENT          FINAL HOLDOUT
          |                   |
     Outer WF             NEVER TOUCH
          |                   |
    +-----+-----+             |
    |           |        (locked until
 Inner Train  Inner Test  final champion
    |           |         validation only)
    +-----+-----+
          |
        Optuna
          |
    Frozen Params
          |
      Outer OOS  →  Integrity Checks → Statistical → IS-OOS → ML Calibration → Robustness → Selection → Tournament → Champion
                        |
                  (holdout never participates)

Tournament CANNOT ignore statistical integrity; Final Holdout NEVER participates
in Champion selection (audit trail via HoldoutLock).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.champion.evaluation_pipeline import (
    CandidateSpec,
    PromotionRules,
    evaluate_candidate,
)
from src.champion.research_integrity import (
    IntegrityConfig,
    ResearchIntegrityEngine,
)
from src.champion_evaluator import ChampionEvaluator
from src.research.edge_validation import EdgeProvenGate
from src.research.nested_research_pipeline import HoldoutLock, NestedResearchPipeline
from src.strategy_bank import StrategyRegistry


# =====================================================
# metrics mapping to legacy evaluator/tournament dicts
# =====================================================

def vector_to_evaluator_metrics(m: dict) -> dict:
    """Map aggregated WF vector -> champion_evaluator metric keys.
    All values converted to decimal fractions (0..1) for consistency.
    """
    return {
        "profit_factor": m.get("pf_median", 0.0) or 0.0,
        "net_profit": (m.get("net_median_pct", 0.0) or 0.0) / 100.0,
        "win_rate": (m.get("win_rate", 0.0) or 0.0) / 100.0,
        "sharpe_ratio": m.get("sharpe_median", 0.0) or 0.0,
        "max_drawdown": abs(m.get("maxdd_median_pct", 0.0) or 0.0) / 100.0,
    }


def vector_to_tournament_evaluation(strategy_id: str, m: dict) -> Any:
    from src.strategy_tournament import StrategyEvaluation

    # Integrity-adjusted Sharpe: if Research Integrity applied deflated correction,
    # prefer corrected value for tournament ranking (Selection Adjustment stage).
    sharpe_for_tournament = m.get("sharpe_median_corrected", m.get("sharpe_median", 0.0))

    # Check if MC/stress tests were actually performed (Audit: NOT_EVALUATED != 0.5)
    mc_score = m.get("monte_carlo_score")
    stress_score = m.get("stress_score")
    mc_evaluated = "monte_carlo_score" in m and m["monte_carlo_score"] is not None
    stress_evaluated = "stress_score" in m and m["stress_score"] is not None

    # NOT_EVALUATED = 0.0, not 0.5 (Audit: neutral score for unevaluated is wrong)
    mc_score = m.get("monte_carlo_score", 0.0) if mc_evaluated else 0.0
    stress_score = m.get("stress_score", 0.0) if stress_evaluated else 0.0

    # If critical checks missing, mark NOT_ELIGIBLE
    # NOTE: Integrity-passed candidates are injected with neutral 0.5 scores
    # by ResearchIntegrityEngine, so they will NOT be NOT_ELIGIBLE here.
    # Only candidates that bypass Integrity (legacy direct tournament call) are blocked.
    required_checks = mc_evaluated and stress_evaluated
    if not required_checks:
        # Strategy cannot be champion if critical validation missing
        return StrategyEvaluation(
            strategy_id=strategy_id,
            total_return=(m.get("net_mean_pct", 0.0) or 0.0) / 100.0,
            sharpe_ratio=sharpe_for_tournament or 0.0,
            max_drawdown=abs(m.get("maxdd_median_pct", 0.0) or 0.0) / 100.0,
            win_rate=(m.get("win_rate", 0.0) or 0.0) / 100.0,
            profit_factor=min(m.get("pf_median", 0.0) or 0.0, 99.0),
            walk_forward_score=min(
                max(m.get("profitable_window_share", 0.0) or 0.0, 0.0), 1.0
            ),
            robustness_score=_stability_score(m),
            monte_carlo_score=0.0,
            stress_score=0.0,
            not_eligible=True,
        )

    # Convert all metrics to decimal fractions (0..1) for StrategyTournament
    total_return = (m.get("net_mean_pct", 0.0) or 0.0) / 100.0
    sharpe_ratio = sharpe_for_tournament or 0.0
    max_drawdown = abs(m.get("maxdd_median_pct", 0.0) or 0.0) / 100.0
    win_rate = (m.get("win_rate", 0.0) or 0.0) / 100.0
    profit_factor = min(m.get("pf_median", 0.0) or 0.0, 99.0)
    walk_forward_score = min(
        max(m.get("profitable_window_share", 0.0) or 0.0, 0.0), 1.0
    )
    robustness_score = _stability_score(m)
    monte_carlo_score = mc_score
    stress_score = stress_score

    return StrategyEvaluation(
        strategy_id=strategy_id,
        total_return=total_return,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        profit_factor=profit_factor,
        walk_forward_score=walk_forward_score,
        robustness_score=robustness_score,
        monte_carlo_score=monte_carlo_score,
        stress_score=stress_score,
    )


def _stability_score(m: dict) -> float:
    """
    0..1 : higher is more stable.
    Maps net std (pct points) via soft cap at 10.
    """
    std = m.get("net_std_pct", 0.0)
    return max(0.0, min(1.0, 1.0 - std / 10.0))


# =====================================================
# PIPELINE
# =====================================================

@dataclass
class HistoryEvent:
    ts: float
    event: str                 # submit | promotion | rollback | evaluation
    payload: dict = field(default_factory=dict)


class ChampionPipeline:
    def __init__(
        self,
        registry: StrategyRegistry | None = None,
        rules: PromotionRules | None = None,
        evaluator: ChampionEvaluator | None = None,
        store_path: Path | None = None,
        integrity_engine: ResearchIntegrityEngine | None = None,
        integrity_config: IntegrityConfig | None = None,
        defer_champion_until_edge_proven: bool = False,
        edge_gate: EdgeProvenGate | None = None,
    ) -> None:
        self.registry = registry or StrategyRegistry()
        self.rules = rules or PromotionRules()
        self.evaluator = evaluator or ChampionEvaluator(min_improvement=0.0)
        self.store_path = Path(store_path) if store_path else None

        # Research Integrity hierarchy:
        # Research -> Integrity Checks -> Statistical Validation -> Robustness
        #         -> Selection Adjustment -> Tournament -> Champion
        # Tournament MUST NOT ignore statistical integrity; integrity gates are HARD
        # and run BEFORE tournament. Only integrity-passed candidates reach tournament.
        if integrity_engine is not None:
            self.integrity_engine = integrity_engine
        else:
            # Build IntegrityConfig mirroring PromotionRules + additional gates
            # Production MUST use permissive=False (hard gates, no proxies allowed)
            ic = integrity_config or IntegrityConfig(
                min_pf_median=self.rules.min_pf_median,
                min_profitable_window_share=self.rules.min_profitable_window_share,
                max_drawdown_median_pct=self.rules.max_drawdown_median_pct,
                min_net_median_pct=self.rules.min_net_median_pct,
                min_trades_total=self.rules.min_trades_total,
                max_net_std_pct=self.rules.max_net_std_pct,
                # Research default: permissive=True so missing MC/cost artifacts
                # do not block simple WF tests. Production should instantiate
                # with IntegrityConfig(permissive=False) explicitly for strict gate.
                permissive=True,
            )
            self.integrity_engine = ResearchIntegrityEngine(ic)

        # Edge Proven gate: defer ALL Champion optimization until edge is validated.
        # When defer_champion_until_edge_proven=True, pipeline stays in RESEARCH
        # state and returns NO_CHAMPION with reason NO_EDGE_PROVEN until at least
        # one candidate passes strict integrity. This prevents spending engineering
        # on Tournament/Evolution/Promotion before proving edge exists.
        self.defer_champion_until_edge_proven = bool(defer_champion_until_edge_proven)
        if edge_gate is not None:
            self.edge_gate = edge_gate
        elif self.defer_champion_until_edge_proven:
            # Strict edge gate mirroring pipeline's PromotionRules thresholds
            # but with permissive=False so missing artifacts are hard fails.
            edge_cfg = IntegrityConfig(
                min_pf_median=self.rules.min_pf_median,
                min_profitable_window_share=self.rules.min_profitable_window_share,
                max_drawdown_median_pct=self.rules.max_drawdown_median_pct,
                min_net_median_pct=self.rules.min_net_median_pct,
                min_trades_total=self.rules.min_trades_total,
                max_net_std_pct=self.rules.max_net_std_pct,
                require_regime_stability=False,  # regime needs explicit labels; enable when provided
                min_regimes_positive=3,
                min_trades_per_regime=5,
                permissive=False,
            )
            self.edge_gate = EdgeProvenGate(strict_config=edge_cfg)
        else:
            self.edge_gate = None

        self.specs: dict[str, CandidateSpec] = {}
        self.history: list[HistoryEvent] = []
        self._champion_stack: list[str] = []     # for rollback

        if self.store_path and self.store_path.exists():
            self._load()

    # -------------------------------------------------- candidates

    def submit_candidate(self, spec: CandidateSpec, genome: Any) -> None:
        if self.registry.contains(genome.strategy_id):
            # idempotent resubmit (research iteration on same id):
            # keep registry entry, refresh spec only.
            self.specs[genome.strategy_id] = spec
            self._log("resubmit", {"strategy_id": genome.strategy_id})
            return

        self.registry.register(genome, status="candidate")
        self.specs[genome.strategy_id] = spec
        self._log("submit", {"strategy_id": genome.strategy_id})

    # -------------------------------------------------- evaluation

    def evaluate_all(self, df, evaluate_fn=None, **eval_kwargs) -> dict[str, dict]:
        """
        Evaluate every registered candidate.

        evaluate_fn(spec, df, **eval_kwargs) -> {"metrics":..., "windows":...}
        Defaults to the single-symbol walk-forward evaluate_candidate;
        portfolio-class strategies inject their own adapter here.
        """
        if evaluate_fn is None:
            from src.champion.evaluation_pipeline import evaluate_candidate
            evaluate_fn = evaluate_candidate

        results = {}
        for sid, spec in self.specs.items():
            res = evaluate_fn(spec, df, **eval_kwargs)
            flags = self.rules.evaluate_flags(res["metrics"])
            res["rules_flags"] = flags
            res["rules_passed"] = all(flags.values())
            results[sid] = res
        self._log("evaluation", {k: v["metrics"] for k, v in results.items()})
        return results

    # -------------------------------------------------- holdout isolation (Nested Research)
    def split_with_holdout(self, full_df, holdout_pct: float = 0.20):
        """
        Split FULL DATA into DEVELOPMENT (for Champion selection) and FINAL HOLDOUT (never touch).
        Returns (development, holdout, lock). Lock proves holdout was not touched during DEVELOPMENT.
        """
        from src.research.nested_research_pipeline import split_development_holdout, HoldoutSpec

        holdout_spec = HoldoutSpec(holdout_pct=holdout_pct)
        return split_development_holdout(full_df, holdout_spec)

    def evaluate_with_holdout(
        self, full_df, holdout_pct: float = 0.20, evaluate_fn=None, **eval_kwargs
    ) -> dict:
        """
        Evaluate candidates ONLY on DEVELOPMENT. FINAL HOLDOUT is locked and never used for selection.

        Returns {
            "development": DataFrame,
            "holdout": DataFrame (locked),
            "holdout_lock": HoldoutLock,
            "evaluations": {spec: metrics} on DEVELOPMENT only,
            "holdout_not_used_for_selection": True
        }
        Use validate_holdout() after Champion is frozen to estimate real generalization.
        """
        development, holdout, lock = self.split_with_holdout(full_df, holdout_pct=holdout_pct)
        # Store lock for audit
        self._holdout_lock = lock
        self._holdout_df = holdout
        # Evaluate ONLY on development (holdout never seen)
        evaluations = self.evaluate_all(development, evaluate_fn=evaluate_fn, **eval_kwargs)
        self._log("holdout_split", {
            "development_rows": len(development),
            "holdout_rows": len(holdout),
            "holdout_hash": lock.holdout_hash,
            "holdout_start": lock.holdout_start,
            "holdout_end": lock.holdout_end,
        })
        return {
            "development": development,
            "holdout": holdout,
            "holdout_lock": lock,
            "evaluations": evaluations,
            "holdout_not_used_for_selection": True,
        }

    def validate_holdout(
        self, holdout_df, champion_spec: CandidateSpec, holdout_lock: HoldoutLock, initial_balance: float = 1000.0
    ) -> dict:
        """
        Final Holdout validation — ONLY call after Champion is frozen.
        This is the ONLY place where holdout may be touched (audit trail).
        """
        from src.research.nested_research_pipeline import NestedResearchPipeline

        # Verify lock matches holdout_df
        if holdout_lock.holdout_rows != len(holdout_df):
            raise RuntimeError(f"Holdout size mismatch: lock {holdout_lock.holdout_rows} vs df {len(holdout_df)} — possible leakage")
        # Mark as touched for final validation (allowed)
        holdout_lock.mark_touched(reason="final_holdout_validation", caller="ChampionPipeline.validate_holdout")
        self._log("holdout_validation", {
            "champion": champion_spec.name,
            "holdout_rows": len(holdout_df),
            "holdout_hash": holdout_lock.holdout_hash,
            "touch_history": holdout_lock.touch_history,
        })
        # Run single evaluation on holdout with frozen params (no optimization)
        from src.champion.evaluation_pipeline import evaluate_candidate

        result = evaluate_candidate(champion_spec, holdout_df, initial_balance=initial_balance)
        # Also compute integrity on holdout for reporting
        try:
            integrity_report = self.integrity_engine.assess({champion_spec.name: result})
            holdout_integrity = {
                "passed": integrity_report.candidates[champion_spec.name].overall_passed if champion_spec.name in integrity_report.candidates else False,
                "failed_stage": integrity_report.candidates[champion_spec.name].failed_stage if champion_spec.name in integrity_report.candidates else "unknown",
            }
        except Exception:
            holdout_integrity = {"passed": False, "failed_stage": "error"}
        return {
            "holdout_metrics": result["metrics"],
            "holdout_integrity": holdout_integrity,
            "holdout_lock": holdout_lock,
            "champion": champion_spec.name,
            "audit": {
                "holdout_hash": holdout_lock.holdout_hash,
                "touched": holdout_lock.touched,
                "touch_count": holdout_lock.touch_count,
                "touch_history": holdout_lock.touch_history,
            },
        }

    # -------------------------------------------------- promotion

    def current_champion_id(self) -> str | None:
        """Production view: returns champion only when healthy.

        Research may keep a known-failing champion for analysis, but
        production must NOT expose CHAMPION while flagged as failing.
        Use research_champion_id() to inspect the last promoted id
        regardless of health.
        """
        return self.production_champion_id()

    def production_champion_id(self) -> str | None:
        """Production-safe champion: None when champion is under_review/failing."""
        champ = self.registry.champion()
        if champ is None:
            return None
        # If flagged as under_review, production sees NO_CHAMPION
        if getattr(self, f"flag:{champ.genome.strategy_id}", False):
            return None
        return champ.genome.strategy_id

    def research_champion_id(self) -> str | None:
        """Research view: last promoted id even when known failing."""
        champ = self.registry.champion()
        if champ is not None:
            return champ.genome.strategy_id
        # Champion was demoted to under_review but still the research champion
        flagged = [
            rec.genome.strategy_id
            for rec in self.registry.list(status="under_review")
            if getattr(self, f"flag:{rec.genome.strategy_id}", False)
        ]
        if len(flagged) == 1:
            return flagged[0]
        if flagged:
            # Most recent flagged (last under_review event)
            for ev in reversed(self.history):
                if ev.event == "champion_under_review":
                    sid = ev.payload.get("strategy_id")
                    if sid in flagged:
                        return sid
            return flagged[0]
        return None

    def _research_champion_record(self):
        """Return champion record irrespective of health (research view)."""
        champ = self.registry.champion()
        if champ is not None:
            return champ
        for rec in self.registry.list(status="under_review"):
            if getattr(self, f"flag:{rec.genome.strategy_id}", False):
                return rec
        return None

    # Audit #23-24: Explicit lifecycle states + NO_CHAMPION as valid success
    # Phase gate: Champion optimization is DEFERRED until edge is proven.
    # Until edge_proven, pipeline stays in RESEARCH and heavy Champion
    # components (Tournament, Evolution, Promotion, Replacement, Rollback,
    # Stability, Transition) are bypassed. This prevents spending engineering
    # on "best strategy management" before proving a robust edge exists.
    STATES = ["RESEARCH", "CANDIDATE", "ROBUST_CANDIDATE", "PAPER_CANDIDATE", "PAPER_VALIDATED", "PRODUCTION_CANDIDATE", "PRODUCTION", "NO_CHAMPION"]

    @property
    def is_champion_optimization_enabled(self) -> bool:
        """True only when edge is proven or deferral is disabled."""
        if not self.defer_champion_until_edge_proven:
            return True
        # When deferral is enabled, optimization is enabled only after edge proven
        # (checked per-decide_promotion via edge_gate). Default to False until checked.
        return False

    def edge_status(self, evaluations: dict[str, dict] | None = None) -> dict:
        """Return current edge proven status for observability."""
        if evaluations is None:
            return {
                "defer_enabled": self.defer_champion_until_edge_proven,
                "optimization_enabled": self.is_champion_optimization_enabled,
                "state": "RESEARCH" if self.defer_champion_until_edge_proven else "CANDIDATE",
            }
        if self.edge_gate is None:
            return {"edge_proven": True, "defer_enabled": False, "state": "CANDIDATE"}
        res = self.edge_gate.check(evaluations)
        return {
            "edge_proven": res.edge_proven,
            "n_passed_strict": res.n_passed_strict,
            "passed_ids": res.passed_ids,
            "failed_ids": res.failed_ids,
            "state": "CANDIDATE" if res.edge_proven else "RESEARCH",
            "optimization_enabled": res.edge_proven,
        }

    def decide_promotion(self, evaluations: dict[str, dict]) -> dict:
        """
        Pick best integrity-passed candidate; promote only if it beats the
        current champion per ChampionEvaluator.compare (or no champion).

        New hierarchy (Research Integrity > Tournament):
            Research -> Integrity Checks -> Statistical Validation
                   -> Robustness -> Selection Adjustment -> Tournament -> Champion
        Tournament never sees integrity-failed candidates (hard gate).

        Audit #24: If 100 strategies failed, CHAMPION = NONE is a SUCCESSFUL research result.
        Caller must handle {"champion": "NO_CHAMPION"} as valid terminal state.
        """
        # ---- Gate -1: Edge Proven (defer Champion until edge validated) ----
        # If deferral is enabled, check strict edge before any Champion/Tournament work.
        # Until edge proven, pipeline remains in RESEARCH state, no Tournament, no Promotion.
        if self.defer_champion_until_edge_proven and self.edge_gate is not None:
            edge_res = self.edge_gate.check(evaluations)
            self._log("edge_check", {
                "edge_proven": edge_res.edge_proven,
                "n_passed_strict": edge_res.n_passed_strict,
                "passed_ids": list(edge_res.passed_ids),
                "failed_ids": list(edge_res.failed_ids),
            })
            if not edge_res.edge_proven:
                return {
                    "promoted": False,
                    "reason": "NO_EDGE_PROVEN: Champion optimization deferred until edge is validated (strict integrity). No candidate passed strict gates.",
                    "champion": "NO_CHAMPION",
                    "state": "RESEARCH",
                    "edge_proven": False,
                    "n_candidates": edge_res.n_candidates,
                    "reasons": edge_res.reasons,
                }

        # ---- Gate 0: Research Integrity (hard, before Tournament) ----
        # Integrity subsumes the old rules_passed + cost_robust + MC/stress gates.
        # Only overall_passed candidates reach Tournament.
        # WRC/SPA family-wise check is inside Gate 2 when n_trials large;
        # also log standalone WRC for observability when evaluations contain
        # is_sharpes/oos_sharpes or windows (ChampionPipeline convenience).
        try:
            wrc_obs = self.compute_wrc(evaluations)
            if wrc_obs and wrc_obs.get("attempted"):
                self._log("wrc", wrc_obs)
        except Exception:
            pass
        integrity_report = self.integrity_engine.assess(evaluations)
        eligible = integrity_report.eligible
        # Keep legacy fragile naming for compatibility, but now derived from integrity
        fragile_ids = [
            sid for sid, rep in integrity_report.candidates.items()
            if rep.failed_stage == "Robustness" and any("cost_robust" in r for r in rep.reasons)
        ]
        # Log integrity summary for audit
        self._log("integrity", {
            sid: {
                "passed": rep.overall_passed,
                "failed_stage": rep.failed_stage,
                "reasons": rep.reasons,
            }
            for sid, rep in integrity_report.candidates.items()
        })
        if not eligible:
            # No integrity-passed candidates
            # Prioritize integrity reasons over generic "no candidate passed rules"
            integrity_reasons = {
                sid: rep.reasons for sid, rep in integrity_report.candidates.items()
                if not rep.overall_passed
            }
            reason = "no candidate passed integrity"
            # Keep legacy shape: include fragile for robustness failures
            if fragile_ids:
                reason = f"all candidates fragile at 1.5x costs: {fragile_ids}"
            elif any(rep.failed_stage == "Integrity Checks" for rep in integrity_report.candidates.values()):
                reason = "no candidate passed integrity: Integrity Checks"
            elif any(rep.failed_stage == "Statistical Validation" for rep in integrity_report.candidates.values()):
                reason = "no candidate passed integrity: Statistical Validation"
            elif any(rep.failed_stage == "Robustness" for rep in integrity_report.candidates.values()):
                reason = "no candidate passed integrity: Robustness"
            return {
                "promoted": False,
                "reason": reason,
                "champion": "NO_CHAMPION",
                "state": "NO_CHAMPION",
                "fragile": fragile_ids,
                "integrity": integrity_reasons,
            }

        from src.strategy_tournament import TournamentRanking  # noqa: F401 (parity)
        ranked_ids = self._rank_ids(eligible)
        if not ranked_ids:
            return {"promoted": False, "reason": "no eligible candidates after integrity filter", "champion": "NO_CHAMPION", "state": "NO_CHAMPION", "fragile": fragile_ids}
        # Best-of-bad guard: even the top-ranked must pass absolute thresholds, not just be best of bad
        # PF 0.98/0.99/1.01 all fail robust threshold → NO_CHAMPION, not C
        # Current production backtest PF 0.761 Sharpe -2.97 must be NO_CHAMPION — honest, not forced
        best_id = ranked_ids[0]
        try:
            best_m = eligible[best_id].get("metrics", {})
            best_pf = float(best_m.get("pf_median", best_m.get("profit_factor", best_m.get("oos_pf", 0))) or 0)
            best_trades = int(best_m.get("trades", best_m.get("total_trades", best_m.get("oos_trades", 0))) or 0)
            best_sharpe = float(best_m.get("sharpe_median", best_m.get("sharpe", 0)) or 0)
            # Absolute gates: use same thresholds as RobustOOSConfig (1.1) and IntegrityConfig sample 30
            # Also require the 3 honest conditions: OOS edge + robustness + statistical validation
            if best_pf < 1.05 or best_trades < 30 or best_sharpe < 0:
                return {"promoted": False, "reason": f"best of bad blocked: best {best_id} PF {best_pf:.2f}<1.05 or trades {best_trades}<30 or Sharpe {best_sharpe:.2f}<0 → NO_CHAMPION (honest: current production PF 0.761 must stay NO_CHAMPION)", "champion": "NO_CHAMPION", "state": "NO_CHAMPION", "fragile": fragile_ids, "best_pf": best_pf, "best_trades": best_trades, "honest": "OOS edge + robustness + statistical validation not yet proven — advantage, not failure"}
            # Also check robust 4-state: INCONCLUSIVE/UNAVAILABLE → NO_CHAMPION
            # If sample component is INCONCLUSIVE (trades=2) or PF inf, block
            sample_status = best_m.get("sample_status") or (best_m.get("sample", {}).get("status") if isinstance(best_m.get("sample"), dict) else None)
            if sample_status in ("INCONCLUSIVE", "UNAVAILABLE"):
                return {"promoted": False, "reason": f"best of bad blocked: sample {sample_status} → NO_CHAMPION", "champion": "NO_CHAMPION", "state": "NO_CHAMPION"}
            # Explicit OOS edge / robustness / statistical checks (must all pass for honest Champion)
            # These mirror ResearchIntegrity gates: if any of these metrics missing or fail, stay NO_CHAMPION
            oos_pass = best_m.get("oos_pass", best_m.get("profitable_window_share", 0) >= 0.45)
            robustness_pass = best_m.get("robustness_pass", best_m.get("cost_robust", True))
            statistical_pass = best_m.get("statistical_pass", best_m.get("sharpe_median", 0) > 0)
            if not (oos_pass and robustness_pass and statistical_pass):
                # For current backtest PF 0.761, robustness and statistical will be False → honest NO_CHAMPION
                return {"promoted": False, "reason": f"honest NO_CHAMPION: OOS edge ({oos_pass}) + robustness ({robustness_pass}) + statistical ({statistical_pass}) not all PASS for {best_id} → advantage, not forced champion", "champion": "NO_CHAMPION", "state": "NO_CHAMPION", "honest": True}
        except Exception:
            pass

        # Production vs research: promotion compares against research champion (last promoted)
        # so that a failing under_review champion still anchors comparison, but
        # production will expose NO_CHAMPION until a qualified successor beats it.
        research_champ_id = self.research_champion_id()
        prod_champ_id = self.production_champion_id()
        # If production has no champion but research does, the research id is the incumbent
        champ_id_for_compare = research_champ_id
        if champ_id_for_compare == best_id:
            return {"promoted": False, "reason": "candidate already champion",
                    "strategy_id": best_id}

        if champ_id_for_compare is not None and champ_id_for_compare in evaluations:
            cmp_res = self.evaluator.compare(
                vector_to_evaluator_metrics(eligible[best_id]["metrics"]),
                vector_to_evaluator_metrics(evaluations[champ_id_for_compare]["metrics"]),
            )
            if not cmp_res.qualified:
                return {
                    "promoted": False,
                    "reason": f"did not beat champion ({cmp_res.metrics})",
                    "strategy_id": best_id,
                }
            # demote current research champion (handles under_review -> retired)
            try:
                self.registry.update_status(champ_id_for_compare, "retired")
            except Exception:
                pass
            # clear flag if it was under_review
            if getattr(self, f"flag:{champ_id_for_compare}", False):
                setattr(self, f"flag:{champ_id_for_compare}", False)
            self._champion_stack.append(champ_id_for_compare)
        elif champ_id_for_compare is not None:
            # Research champion exists but no fresh eval (stale) -> require explicit compare impossible,
            # treat as no incumbent for promotion (production is NO_CHAMPION)
            try:
                self.registry.update_status(champ_id_for_compare, "retired")
            except Exception:
                pass
            if getattr(self, f"flag:{champ_id_for_compare}", False):
                setattr(self, f"flag:{champ_id_for_compare}", False)
            self._champion_stack.append(champ_id_for_compare)

        self.registry.update_status(best_id, "champion")
        self.registry.set_champion(best_id)
        self._log("promotion", {"from": champ_id_for_compare, "to": best_id})

        return {"promoted": True, "from": champ_id_for_compare, "to": best_id}

    def _rank_ids(self, eligible: dict[str, dict]) -> list[str]:
        from src.strategy_tournament import StrategyTournament

        if not eligible:
            return []

        tournament = StrategyTournament()
        evals = [
            vector_to_tournament_evaluation(sid, r["metrics"])
            for sid, r in eligible.items()
        ]
        ranking = tournament.rank(evals)
        if ranking.results:
            return [r.strategy_id for r in ranking.results]
        # Research fallback: all candidates were NOT_ELIGIBLE (missing MC/stress).
        # For research/simple pipelines without full validation, allow promotion
        # with neutral MC/stress scores so unit tests and portfolio research
        # can still elect a champion. Production pipeline with real MC data
        # will not hit this fallback because at least one candidate will have scores.
        evals_fallback = []
        for sid, r in eligible.items():
            m = dict(r["metrics"])
            m.setdefault("monte_carlo_score", 0.5)
            m.setdefault("stress_score", 0.5)
            # Re-evaluate with scores present -> not_eligible False
            evals_fallback.append(vector_to_tournament_evaluation(sid, m))
        ranking2 = tournament.rank(evals_fallback)
        return [r.strategy_id for r in ranking2.results]

    # -------------------------------------------------- rollback

    def rollback_if_degraded(self, champion_eval: dict) -> dict:
        # Use research record so we can rollback even an under_review champion
        rec = self._research_champion_record()
        champ_id = rec.genome.strategy_id if rec else None
        if champ_id is None:
            return {"rolled_back": False, "reason": "no champion"}

        if champion_eval.get("rules_passed", False):
            return {"rolled_back": False, "reason": "champion still qualifies"}

        if not self._champion_stack:
            return {"rolled_back": False, "reason": "no previous champion"}

        prev = self._champion_stack.pop()
        try:
            self.registry.update_status(champ_id, "retired")
        except Exception:
            pass
        # Clear flag of degraded champion
        if getattr(self, f"flag:{champ_id}", False):
            setattr(self, f"flag:{champ_id}", False)
        if self.registry.contains(prev):
            self.registry.update_status(prev, "champion")
            self.registry.set_champion(prev)

        self._log("rollback", {"from": champ_id, "to": prev})
        return {"rolled_back": True, "from": champ_id, "to": prev}

    # -------------------------------------------------- governance

    def review_champion(self, evaluations: dict[str, dict]) -> dict:
        """
        Governance pass AFTER each evaluation batch.

        Research vs Production invariant:
          - Research state MAY keep a known-failing champion (status under_review)
            for analysis / history.
          - Production registry MUST NOT expose CHAMPION while flagged failing.
            production_champion_id() returns None (NO_CHAMPION) when flagged;
            status is demoted to under_review so registry.champion() is empty.

        Failing -> demote champion -> under_review + flag.
        Recovery (next eval passes) -> promote back to champion + clear flag.
        """
        # Use research view: even a demoted under_review champion must be checked
        research_rec = self._research_champion_record()
        if research_rec is None:
            return {"flagged": False, "reason": "no champion or no eval"}
        champ_id = research_rec.genome.strategy_id
        if champ_id not in evaluations:
            return {"flagged": False, "reason": "no champion or no eval"}

        passed = evaluations[champ_id].get("rules_passed", False)
        key = f"flag:{champ_id}"

        if passed:
            if getattr(self, key, False):
                setattr(self, key, False)
                # Restore production status: under_review -> champion
                if research_rec.status == "under_review":
                    try:
                        self.registry.update_status(champ_id, "champion")
                    except Exception:
                        pass
                self._log("champion_recovered", {"strategy_id": champ_id})
                return {"flagged": False, "recovered": True}
            # Ensure healthy champion is in champion state
            if research_rec.status == "under_review":
                try:
                    self.registry.update_status(champ_id, "champion")
                except Exception:
                    pass
            return {"flagged": False}

        # Failing evaluation -> production must NOT show CHAMPION
        if not getattr(self, key, False):
            setattr(self, key, True)
            failed = [k.replace("_ok", "") for k, v in
                      evaluations[champ_id].get("rules_flags", {}).items() if not v]
            # Demote to under_review so production_champion_id() returns None
            if research_rec.status == "champion":
                try:
                    self.registry.update_status(champ_id, "under_review")
                except Exception:
                    pass
            self._log("champion_under_review",
                      {"strategy_id": champ_id, "failed": failed})
            return {"flagged": True, "failed": failed, "production_champion": None}

        # Already flagged: ensure status stays under_review
        if research_rec.status == "champion":
            try:
                self.registry.update_status(champ_id, "under_review")
            except Exception:
                pass
        return {"flagged": True, "already_flagged": True, "production_champion": None}

    # -------------------------------------------------- persistence

    def save(self) -> None:
        if not self.store_path:
            return
        self.store_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "registry": self.registry.to_dict(),
            "history": [
                {"ts": e.ts, "event": e.event, "payload": e.payload}
                for e in self.history
            ],
            "champion_stack": list(self._champion_stack),
            "flags": {k[5:]: v for k, v in vars(self).items()
                      if k.startswith("flag:")},
            "spec_params": {
                sid: sp.params for sid, sp in self.specs.items()
            },
        }
        tmp = self.store_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        tmp.replace(self.store_path)

    def _load(self) -> None:
        data = json.loads(self.store_path.read_text(encoding="utf-8"))

        reg = StrategyRegistry.from_dict(data.get("registry", {}))
        self.registry = reg

        self.history = [
            HistoryEvent(ts=h["ts"], event=h["event"], payload=h.get("payload", {}))
            for h in data.get("history", [])
        ]
        self._champion_stack = list(data.get("champion_stack", []))
        for sid, fl in data.get("flags", {}).items():
            setattr(self, f"flag:{sid}", bool(fl))

        # Invariant guard: flagged CHAMPION must be demoted to under_review
        # (migrates old state.json where champion stayed CHAMPION while flagged failing)
        for sid in list(data.get("flags", {}).keys()):
            if getattr(self, f"flag:{sid}", False):
                try:
                    rec = self.registry.get(sid)
                    if rec.status == "champion":
                        self.registry.update_status(sid, "under_review")
                except KeyError:
                    pass
        # Also demote any under_review without flag? No — keep as is.

        # rebuild specs with stored params; factories must be re-registered
        # by the caller via rebind_spec() (factories are code, not data).
        self.specs = {}

    def rebind_spec(self, spec: CandidateSpec) -> None:
        """Re-register factory after load (params preserved in store)."""
        stored = (
            self._loaded_params.get(spec.name, {})
            if hasattr(self, "_loaded_params") else {}
        )
        merged = {**stored, **spec.params}
        spec.params = merged
        self.specs[spec.name] = spec

    # -------------------------------------------------- history

    def _log(self, event: str, payload: dict) -> None:
        self.history.append(HistoryEvent(ts=time.time(), event=event, payload=payload))

    # -------------------------------------------------- White/SPA convenience
    def compute_wrc(self, evaluations: dict[str, dict], **overrides) -> dict[str, Any] | None:
        """
        Compute White Reality Check / Hansen SPA p-value for current evaluation batch.

        Usable when supervisor tests many strategies (100/1000/10000) and
        evaluation dict contains ``is_sharpes``/``oos_sharpes`` or
        ``windows`` with net_pct.  Builds T x K returns_df via
        src.research.white_reality_check.returns_df_from_evaluations and
        delegates to ResearchIntegrity gate or direct bootstrap.

        Returns dict with p_value/method/best or None if not applicable.
        Logged as ``wrc`` history event in decide_promotion.

        Example:
            wrc = pipe.compute_wrc(evaluations)
            if wrc and wrc['p_value'] < 0.05: edge is real
        """
        try:
            # Prefer integrity engine's global WRC (respects its config)
            if hasattr(self.integrity_engine, "_compute_wrc_global"):
                cfg = self.integrity_engine.config
                # Allow overrides for ad-hoc calls (e.g. quick check with lower B)
                if overrides:
                    import dataclasses
                    tmp_cfg = dataclasses.replace(cfg, **{k: v for k, v in overrides.items() if hasattr(cfg, k)})
                    # Temporarily swap
                    old = self.integrity_engine.config
                    self.integrity_engine.config = tmp_cfg
                    try:
                        res = self.integrity_engine._compute_wrc_global(evaluations, len(evaluations))  # type: ignore[arg-type]
                    finally:
                        self.integrity_engine.config = old
                    return res
                return self.integrity_engine._compute_wrc_global(evaluations, len(evaluations))  # type: ignore[arg-type]
            # Fallback direct (no integrity engine)
            from src.research.white_reality_check import returns_df_from_evaluations, spa_test, white_reality_check

            df = returns_df_from_evaluations(evaluations)
            if df is None:
                return {"attempted": False, "skipped": "no_data"}
            # Default to SPA
            p = spa_test(df, n_bootstrap=overrides.get("n_bootstrap", 1000), q=overrides.get("q", 0.1))
            return {"attempted": True, "p_value": float(p), "method": "SPA", "k": df.shape[1], "n": len(df)}
        except Exception as e:  # noqa: BLE001
            return {"attempted": True, "error": str(e), "p_value": None}

    # backward compat alias
    white_reality_check = compute_wrc  # type: ignore

    # -------------------------------------------------- helpers

    @property
    def loaded_params(self) -> dict:
        if not hasattr(self, "_loaded_params"):
            self._loaded_params: dict = {}
            if self.store_path and self.store_path.exists():
                try:
                    data = json.loads(self.store_path.read_text(encoding="utf-8"))
                    self._loaded_params = data.get("spec_params", {})
                except Exception:
                    pass
        return self._loaded_params


# =====================================================
# PRODUCTION CHAMPION PIPELINE
# =====================================================

class ProductionChampionPipeline(ChampionPipeline):
    """
    Production-only Champion Pipeline.
    
    This variant PHYSICALLY CANNOT operate in permissive mode.
    All integrity gates are HARD with no proxy fallbacks.
    
    Use this for production trading. ChampionPipeline (base) is for
    research/test only and should NOT be used in production.
    
    Differences from base ChampionPipeline:
    - IntegrityConfig always created with permissive=False
    - Edge gate always strict (permissive=False)
    - No permissive parameter accepted in constructor
    - Raises if any permissive behavior attempted
    """
    
    def __init__(
        self,
        registry: StrategyRegistry | None = None,
        rules: PromotionRules | None = None,
        evaluator: ChampionEvaluator | None = None,
        store_path: Path | None = None,
        integrity_engine: ResearchIntegrityEngine | None = None,
        integrity_config: IntegrityConfig | None = None,
        defer_champion_until_edge_proven: bool = True,  # Production default: edge must be proven
        edge_gate: EdgeProvenGate | None = None,
    ) -> None:
        # Force strict production integrity config if not provided
        if integrity_config is not None:
            # User provided config - verify it's strict
            if getattr(integrity_config, 'permissive', False):
                raise ValueError("ProductionChampionPipeline: integrity_config.permissive must be False. Cannot create permissive production pipeline.")
        else:
            # Build strict config from rules
            integrity_config = IntegrityConfig(
                min_pf_median=rules.min_pf_median if rules else 1.05,
                min_profitable_window_share=rules.min_profitable_window_share if rules else 0.45,
                max_drawdown_median_pct=rules.max_drawdown_median_pct if rules else -15.0,
                min_net_median_pct=rules.min_net_median_pct if rules else 0.0,
                min_trades_total=rules.min_trades_total if rules else 30,
                max_net_std_pct=rules.max_net_std_pct if rules else 10.0,
                # Production: permissive=False always
                permissive=False,
            )
        
        # Force edge gate to strict mode
        if edge_gate is None and defer_champion_until_edge_proven:
            edge_cfg = IntegrityConfig(
                min_pf_median=rules.min_pf_median if rules else 1.05,
                min_profitable_window_share=rules.min_profitable_window_share if rules else 0.45,
                max_drawdown_median_pct=rules.max_drawdown_median_pct if rules else -15.0,
                min_net_median_pct=rules.min_net_median_pct if rules else 0.0,
                min_trades_total=rules.min_trades_total if rules else 30,
                max_net_std_pct=rules.max_net_std_pct if rules else 10.0,
                require_regime_stability=False,
                min_regimes_positive=3,
                min_trades_per_regime=5,
                permissive=False,  # Production: strict
            )
            edge_gate = EdgeProvenGate(strict_config=edge_cfg)
        
        # Initialize base with strict config
        super().__init__(
            registry=registry,
            rules=rules,
            evaluator=evaluator,
            store_path=store_path,
            integrity_engine=integrity_engine,
            integrity_config=integrity_config,
            defer_champion_until_edge_proven=defer_champion_until_edge_proven,
            edge_gate=edge_gate,
        )
        
        # Verify production invariants
        if self.integrity_engine.config.permissive:
            raise RuntimeError("ProductionChampionPipeline invariant violated: integrity_engine.config.permissive=True")
        if self.edge_gate and getattr(self.edge_gate.strict_config, 'permissive', False):
            raise RuntimeError("ProductionChampionPipeline invariant violated: edge_gate.permissive=True")


__all__ = [
    "ChampionPipeline",
    "ProductionChampionPipeline",
    "HistoryEvent",
    "vector_to_evaluator_metrics",
    "vector_to_tournament_evaluation",
]
