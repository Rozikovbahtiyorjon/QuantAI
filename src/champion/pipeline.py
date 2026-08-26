"""
QuantAI Champion Pipeline (R4)

Wires the previously isolated cluster into ONE controlled flow:

    CandidateSpec ──evaluate──▶ vector + rule flags
        │                              │
        ▼                              ▼
  StrategyRegistry  ◀── Tournament ranking + ChampionEvaluator compare
        │
  PROMOTE / RETAIN / ROLLBACK   (+ append-only history log)
        │
  Persistence: JSON store (registry state + history + params)

Feedback loop entry points live in src/champion/feedback.py.
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
from src.champion_evaluator import ChampionEvaluator
from src.strategy_bank import StrategyRegistry


# =====================================================
# metrics mapping to legacy evaluator/tournament dicts
# =====================================================

def vector_to_evaluator_metrics(m: dict) -> dict:
    """Map aggregated WF vector -> champion_evaluator metric keys."""
    return {
        "profit_factor": m["pf_median"],
        "net_profit": m["net_median_pct"],
        "win_rate": m["win_rate"],
        "sharpe_ratio": m["sharpe_median"],
        "max_drawdown": abs(m["maxdd_median_pct"]),   # positive magnitude
    }


def vector_to_tournament_evaluation(strategy_id: str, m: dict) -> Any:
    from src.strategy_tournament import StrategyEvaluation

    return StrategyEvaluation(
        strategy_id=strategy_id,
        total_return=m["net_mean_pct"],
        sharpe_ratio=m["sharpe_median"],
        max_drawdown=abs(m["maxdd_median_pct"]),
        win_rate=min(max(m["win_rate"], 0.0), 100.0) / 100.0,
        profit_factor=min(m["pf_median"], 99.0),
        walk_forward_score=min(
            max(m["profitable_window_share"], 0.0), 1.0
        ),
        robustness_score=_stability_score(m),
        monte_carlo_score=0.5,   # neutral until MC stage (R4+)
        stress_score=0.5,
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
    ) -> None:
        self.registry = registry or StrategyRegistry()
        self.rules = rules or PromotionRules()
        self.evaluator = evaluator or ChampionEvaluator(min_improvement=0.0)
        self.store_path = Path(store_path) if store_path else None

        self.specs: dict[str, CandidateSpec] = {}
        self.history: list[HistoryEvent] = []
        self._champion_stack: list[str] = []     # for rollback

        if self.store_path and self.store_path.exists():
            self._load()

    # -------------------------------------------------- candidates

    def submit_candidate(self, spec: CandidateSpec, genome: Any) -> None:
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

    # -------------------------------------------------- promotion

    def current_champion_id(self) -> str | None:
        champ = self.registry.champion()
        return champ.genome.strategy_id if champ else None

    def decide_promotion(self, evaluations: dict[str, dict]) -> dict:
        """
        Pick best rules-passing candidate; promote only if it beats the
        current champion per ChampionEvaluator.compare (or no champion).
        """
        eligible = {
            sid: r for sid, r in evaluations.items()
            if r["rules_passed"]
        }
        if not eligible:
            return {"promoted": False, "reason": "no candidate passed rules"}

        from src.strategy_tournament import TournamentRanking  # noqa: F401 (parity)
        ranked_ids = self._rank_ids(eligible)
        best_id = ranked_ids[0]

        champ_id = self.current_champion_id()
        if champ_id == best_id:
            return {"promoted": False, "reason": "candidate already champion",
                    "strategy_id": best_id}

        if champ_id is not None:
            cmp_res = self.evaluator.compare(
                vector_to_evaluator_metrics(eligible[best_id]["metrics"]),
                vector_to_evaluator_metrics(evaluations[champ_id]["metrics"]),
            )
            if not cmp_res.qualified:
                return {
                    "promoted": False,
                    "reason": f"did not beat champion ({cmp_res.metrics})",
                    "strategy_id": best_id,
                }
            # demote current champion
            self.registry.update_status(champ_id, "retired")
            self._champion_stack.append(champ_id)

        self.registry.update_status(best_id, "champion")
        self.registry.set_champion(best_id)
        self._log("promotion", {"from": champ_id, "to": best_id})

        return {"promoted": True, "from": champ_id, "to": best_id}

    def _rank_ids(self, eligible: dict[str, dict]) -> list[str]:
        from src.strategy_tournament import StrategyTournament

        tournament = StrategyTournament()
        evals = [
            vector_to_tournament_evaluation(sid, r["metrics"])
            for sid, r in eligible.items()
        ]
        ranking = tournament.rank(evals)
        return [r.strategy_id for r in ranking.results]

    # -------------------------------------------------- rollback

    def rollback_if_degraded(self, champion_eval: dict) -> dict:
        champ_id = self.current_champion_id()
        if champ_id is None:
            return {"rolled_back": False, "reason": "no champion"}

        if champion_eval.get("rules_passed", False):
            return {"rolled_back": False, "reason": "champion still qualifies"}

        if not self._champion_stack:
            return {"rolled_back": False, "reason": "no previous champion"}

        prev = self._champion_stack.pop()
        self.registry.update_status(champ_id, "retired")
        if self.registry.contains(prev):
            self.registry.update_status(prev, "champion")
            self.registry.set_champion(prev)

        self._log("rollback", {"from": champ_id, "to": prev})
        return {"rolled_back": True, "from": champ_id, "to": prev}

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


__all__ = [
    "ChampionPipeline",
    "HistoryEvent",
    "vector_to_evaluator_metrics",
    "vector_to_tournament_evaluation",
]
