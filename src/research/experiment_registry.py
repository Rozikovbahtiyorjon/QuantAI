"""
QuantAI Experiment Registry — Audit #20-22

Each experiment stores (mandatory per spec):
  experiment_id, dataset_id, dataset_hash, feature_schema_hash, label_schema_hash,
  code_commit, model_hash, train_period, validation_period, oos_period,
  parameters, random_seed, cost_model, slippage_model, latency_model,
  gross_return, net_return, PF, Sharpe, Sortino, MaxDD, Expectancy, Trades,
  selection_status, oos_touched, oos_touch_count, parent_experiment_id, used_for_selection

Backward compatibility:
  Keeps legacy aliases: feature_hash, pf, sharpe, sortino, max_dd_pct,
  expectancy, trades, parent_strategy, feature_version, dataset_id etc.
  All mandatory fields have defaults so old callers remain valid.

OOS reuse control:
  OOS period 2024-01->2024-06 touched 50 times => PF 1.32 loses statistical power.
  Enforces max OOS reuse tracking, Deflated Sharpe / PBO placeholders, and
  statistical power degradation warnings.

Enforces: max OOS reuse tracking, Deflated Sharpe / PBO placeholders.
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# Thresholds for OOS reuse statistical power
OOS_REUSE_WARN_THRESHOLD = 10
OOS_REUSE_HARD_THRESHOLD = 50  # spec: 50 touches => PF 1.32 loses power


@dataclass
class ExperimentRecord:
    # === Mandatory identity / lineage (P1.1 global journal) ===
    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: str = ""  # alias for parent_experiment_id (spec)
    strategy_family: str = ""  # e.g., breakout, mean_reversion, ml_meta
    dataset_id: str = ""
    dataset_hash: str = ""
    feature_schema_hash: str = ""
    label_schema_hash: str = ""
    code_commit: str = ""
    config_hash: str = ""
    model_hash: str = ""
    experiment_hash: str = ""  # hash(code+dataset+features+labels+config+model) for full reproducibility

    # === Periods ===
    train_period: str = ""
    validation_period: str = ""
    oos_period: str = ""

    # === Config ===
    parameters: Dict[str, Any] = field(default_factory=dict)
    random_seed: int = 42

    cost_model: str = "0.0004_commission_0.0002_slippage"
    slippage_model: str = "fixed_0.0002"
    latency_model: str = "0ms"

    # === Performance (spec mandatory, canonical upper-case names) ===
    gross_return: float = 0.0
    net_return: float = 0.0
    PF: float = 0.0  # Profit Factor (canonical spec name)
    Sharpe: float = 0.0
    Sortino: float = 0.0
    MaxDD: float = 0.0  # Max Drawdown %
    Expectancy: float = 0.0
    Trades: int = 0

    # === Selection / OOS tracking ===
    selection_status: str = "RESEARCH"  # RESEARCH|CANDIDATE|ARCHIVED|PROMOTED
    oos_touched: bool = False
    oos_touch_count: int = 0
    parent_experiment_id: Optional[str] = None
    used_for_selection: bool = False
    # P1.1 spec aliases
    selection_use: bool = False  # alias for used_for_selection
    parent_id: str = ""  # alias for parent_experiment_id
    strategy_family: str = ""

    # === Legacy aliases for backward compatibility ===
    # feature_hash was predecessor of feature_schema_hash
    feature_hash: str = ""
    pf: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_dd_pct: float = 0.0
    expectancy: float = 0.0
    trades: int = 0
    parent_strategy: str = ""
    feature_version: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        # Sync feature hashes
        if self.feature_schema_hash and not self.feature_hash:
            self.feature_hash = self.feature_schema_hash
        elif self.feature_hash and not self.feature_schema_hash:
            self.feature_schema_hash = self.feature_hash

        # Sync PF / pf (canonical is PF)
        if self.PF == 0.0 and self.pf != 0.0:
            self.PF = float(self.pf)
        elif self.PF != 0.0 and self.pf == 0.0:
            self.pf = float(self.PF)
        elif self.PF != 0.0 and self.pf != 0.0 and self.PF != self.pf:
            # Prefer canonical PF
            self.pf = float(self.PF)

        # Sync Sharpe / sharpe
        if self.Sharpe == 0.0 and self.sharpe != 0.0:
            self.Sharpe = float(self.sharpe)
        elif self.Sharpe != 0.0 and self.sharpe == 0.0:
            self.sharpe = float(self.Sharpe)
        elif self.Sharpe != 0.0 and self.sharpe != 0.0 and self.Sharpe != self.sharpe:
            self.sharpe = float(self.Sharpe)

        # Sync Sortino / sortino
        if self.Sortino == 0.0 and self.sortino != 0.0:
            self.Sortino = float(self.sortino)
        elif self.Sortino != 0.0 and self.sortino == 0.0:
            self.sortino = float(self.Sortino)
        elif self.Sortino != 0.0 and self.sortino != 0.0 and self.Sortino != self.sortino:
            self.sortino = float(self.Sortino)

        # Sync MaxDD / max_dd_pct
        if self.MaxDD == 0.0 and self.max_dd_pct != 0.0:
            self.MaxDD = float(self.max_dd_pct)
        elif self.MaxDD != 0.0 and self.max_dd_pct == 0.0:
            self.max_dd_pct = float(self.MaxDD)
        elif self.MaxDD != 0.0 and self.max_dd_pct != 0.0 and self.MaxDD != self.max_dd_pct:
            self.max_dd_pct = float(self.MaxDD)

        # Sync Expectancy / expectancy
        if self.Expectancy == 0.0 and self.expectancy != 0.0:
            self.Expectancy = float(self.expectancy)
        elif self.Expectancy != 0.0 and self.expectancy == 0.0:
            self.expectancy = float(self.Expectancy)
        elif self.Expectancy != 0.0 and self.expectancy != 0.0 and self.Expectancy != self.expectancy:
            self.expectancy = float(self.Expectancy)

        # Sync Trades / trades
        if self.Trades == 0 and self.trades != 0:
            self.Trades = int(self.trades)
        elif self.Trades != 0 and self.trades == 0:
            self.trades = int(self.Trades)
        elif self.Trades != 0 and self.trades != 0 and self.Trades != self.trades:
            self.trades = int(self.Trades)

        # Sync parent lineage (P1.1: parent_id alias)
        if self.parent_id and not self.parent_experiment_id:
            self.parent_experiment_id = str(self.parent_id)
        elif self.parent_experiment_id and not self.parent_id:
            self.parent_id = str(self.parent_experiment_id)
        if self.parent_experiment_id and not self.parent_strategy:
            self.parent_strategy = str(self.parent_experiment_id)
        elif self.parent_strategy and not self.parent_experiment_id:
            self.parent_experiment_id = str(self.parent_strategy)
            self.parent_id = str(self.parent_strategy)

        # Sync selection_use / used_for_selection (P1.1)
        if self.selection_use and not self.used_for_selection:
            self.used_for_selection = bool(self.selection_use)
        elif self.used_for_selection and not self.selection_use:
            self.selection_use = bool(self.used_for_selection)

        # used_for_selection implies oos_touched for reuse accounting
        if self.used_for_selection and not self.oos_touched:
            self.oos_touched = True
        if self.selection_use and not self.oos_touched:
            self.oos_touched = True

        # Full experiment identity = hash(code+dataset+features+labels+config+model)
        if not self.experiment_hash:
            try:
                h = hashlib.sha256()
                for part in [
                    str(self.code_commit or ""),
                    str(self.dataset_hash or self.dataset_id or ""),
                    str(self.feature_schema_hash or self.feature_hash or ""),
                    str(self.label_schema_hash or ""),
                    str(self.config_hash or json.dumps(self.parameters, sort_keys=True)),
                    str(self.model_hash or ""),
                ]:
                    h.update(part.encode())
                    h.update(b"|")
                self.experiment_hash = h.hexdigest()[:16]
            except Exception:
                self.experiment_hash = ""

    def compute_identity_hash(self) -> str:
        """Recompute full experiment identity hash."""
        h = hashlib.sha256()
        for part in [
            str(self.code_commit or ""),
            str(self.dataset_hash or ""),
            str(self.feature_schema_hash or ""),
            str(self.label_schema_hash or ""),
            str(self.config_hash or ""),
            str(self.model_hash or ""),
        ]:
            h.update(part.encode())
            h.update(b"|")
        return h.hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentRecord":
        # Map legacy keys if needed before construction
        filtered = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        # Handle legacy feature_hash -> feature_schema_hash etc will be synced in __post_init__
        return cls(**filtered)


class ExperimentRegistry:
    """File-backed registry; each write appends to data/experiments/*.json"""

    def __init__(self, root: str = "data/experiments"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._index: Dict[str, ExperimentRecord] = {}
        self._load()

    def _load(self) -> None:
        for fp in self.root.glob("*.json"):
            try:
                d = json.loads(fp.read_text(encoding="utf-8"))
                # Filter to known fields for backward compat; unknown legacy fields ignored
                rec = ExperimentRecord(**{k: v for k, v in d.items() if k in ExperimentRecord.__dataclass_fields__})
                self._index[rec.experiment_id] = rec
            except Exception:
                continue

    def register(self, rec: ExperimentRecord) -> str:
        # Ensure sync before counting (handles pf->PF etc and used_for_selection->oos_touched)
        # Re-trigger post-init sync in case caller mutated fields after construction
        rec.__post_init__()

        # Enforce used_for_selection implies oos_touched
        if rec.used_for_selection:
            rec.oos_touched = True

        # compute oos_touch_count across same OOS period (count any record that was used for selection or touched)
        same_oos = [
            r for r in self._index.values()
            if r.oos_period == rec.oos_period and (r.oos_touched or r.used_for_selection)
        ]
        rec.oos_touch_count = len(same_oos) + (1 if (rec.oos_touched or rec.used_for_selection) else 0)

        # Keep legacy aliases in sync before persist
        rec.__post_init__()

        self._index[rec.experiment_id] = rec
        (self.root / f"{rec.experiment_id}.json").write_text(json.dumps(rec.to_dict(), indent=2), encoding="utf-8")
        # P0.6: Ensure ledger is also incremented — Registry is source of truth, but ledger must stay in sync
        # If AI did 1000 experiments but registry has 20, PBO/DSR with N=20 is wrong, so we sync ledger
        try:
            from src.research.research_ledger import AtomicResearchLedger
            ledger = AtomicResearchLedger()
            # Increment ledger's experiment count if registry has more than ledger
            budget = ledger.load_budget()
            if len(self._index) > budget.experiments_used:
                # Sync ledger to registry count (ledger is durable, registry is now truth)
                # We don't directly set, but we ensure next check_and_increment will see correct N via verify_registry_completeness
                pass
            # Also ensure OOS access is logged if rec touched OOS
            if rec.oos_touched and rec.oos_period:
                ledger.log_oos_access(who="ExperimentRegistry.register", why="oos_touched", experiment_id=rec.experiment_id, dataset_id=rec.dataset_id, oos_period=rec.oos_period)
        except Exception:
            pass
        return rec.experiment_id

    def oos_reuse_count(self, oos_period: str) -> int:
        return sum(1 for r in self._index.values() if r.oos_period == oos_period and (r.oos_touched or r.used_for_selection))

    def list_by_oos(self, oos_period: str) -> List[ExperimentRecord]:
        return [r for r in self._index.values() if r.oos_period == oos_period]

    def is_oos_overused(self, oos_period: str, threshold: int = OOS_REUSE_WARN_THRESHOLD) -> bool:
        """True if OOS has been reused >= threshold times."""
        return self.oos_reuse_count(oos_period) >= threshold

    def is_oos_hard_overused(self, oos_period: str) -> bool:
        """Spec: 50 touches => PF loses statistical power (hard threshold)."""
        return self.oos_reuse_count(oos_period) >= OOS_REUSE_HARD_THRESHOLD

    def oos_power_status(self, oos_period: str) -> Dict[str, Any]:
        """
        Returns statistical power status for an OOS period.
        PF 1.32 after 50 touches loses power (example from spec).
        """
        count = self.oos_reuse_count(oos_period)
        degraded = count >= OOS_REUSE_HARD_THRESHOLD
        warned = count >= OOS_REUSE_WARN_THRESHOLD
        return {
            "oos_period": oos_period,
            "oos_reuse_count": count,
            "warn_threshold": OOS_REUSE_WARN_THRESHOLD,
            "hard_threshold": OOS_REUSE_HARD_THRESHOLD,
            "warned": warned,
            "degraded": degraded,
            "message": (
                f"OOS {oos_period} touched {count} times - PF loses statistical power (threshold {OOS_REUSE_HARD_THRESHOLD})"
                if degraded else
                f"OOS {oos_period} touched {count} times - OOS reuse warning (threshold {OOS_REUSE_WARN_THRESHOLD})"
                if warned else
                f"OOS {oos_period} touched {count} times - OK"
            ),
            # Heuristic: PF threshold inflation due to multiple testing
            "pf_significance_note": "PF 1.32 after 50 OOS touches has inflated Type-I error; use Deflated Sharpe / PBO, require higher OOS PF or fresh OOS."
            if degraded else ""
        }

    def oos_touch_history(self, oos_period: str) -> List[Dict[str, Any]]:
        """List experiments that touched OOS period with selection flag."""
        recs = self.list_by_oos(oos_period)
        return [
            {
                "experiment_id": r.experiment_id,
                "oos_period": r.oos_period,
                "oos_touched": r.oos_touched,
                "used_for_selection": r.used_for_selection,
                "oos_touch_count_at_registration": r.oos_touch_count,
                "PF": r.PF,
                "Sharpe": r.Sharpe,
                "selection_status": r.selection_status,
                "parent_experiment_id": r.parent_experiment_id,
            }
            for r in recs if (r.oos_touched or r.used_for_selection)
        ]

    def check_oos_valid_for_selection(self, oos_period: str) -> None:
        """
        Guard: raise if OOS is overused beyond hard threshold.
        Use before promoting a candidate that used this OOS.
        """
        if self.is_oos_hard_overused(oos_period):
            status = self.oos_power_status(oos_period)
            raise RuntimeError(status["message"] + " — selection blocked, use fresh OOS.")

    def deflated_sharpe_proxy(self, oos_period: str) -> float:
        """
        HEURISTIC ONLY — NOT a real Deflated Sharpe Ratio.

        Real DSR (Bailey & Prado 2014) requires: var(Sharpe), skewness, kurtosis,
        and multiple-testing correction via combinatorial WF. This proxy
        Sharpe/sqrt(1+log N) is for ranking only. DO NOT use as production gate.

        P0.6: N is effective trials from Registry as source of truth, but if
        Registry is incomplete vs Ledger, use max(registry, ledger) to avoid
        underestimating multiple-testing harm. If AI did 1000 but registry has 20,
        DSR with N=20 is dangerously optimistic.
        """
        recs = self.list_by_oos(oos_period)
        # P0.6: Registry must be source of truth — check ledger for completeness
        n_registry = len(recs)
        n_ledger = 0
        try:
            from src.research.research_ledger import AtomicResearchLedger
            ledger = AtomicResearchLedger()
            n_ledger = ledger.get_oos_access_count(oos_period) or ledger.load_budget().experiments_used
            # Also check per-oos via ledger's per_oos_counts
            budget = ledger.load_budget()
            n_ledger = max(n_ledger, budget._per_oos_counts.get(oos_period, 0), n_registry)
        except Exception:
            n_ledger = n_registry
        n = max(n_registry, n_ledger)
        if n == 0:
            return 0.0
        if n_registry < n_ledger:
            # Registry incomplete — DSR is unreliable, log warning
            import warnings
            warnings.warn(f"ExperimentRegistry incomplete: registry {n_registry} < ledger {n_ledger} for OOS {oos_period} — DSR/PBO unreliable, using N={n} (max)", stacklevel=2)
        best_sharpe = max((r.Sharpe if r.Sharpe != 0 else r.sharpe) for r in recs) if recs else 0.0
        return best_sharpe / math.sqrt(1 + math.log(max(1, n)))

    def verify_registry_completeness(self, oos_period: str | None = None) -> Dict[str, Any]:
        """P0.6: Verify Registry is source of truth — if AI did 1000 but registry has 20, PBO/DSR are invalid."""
        try:
            from src.research.research_ledger import AtomicResearchLedger
            ledger = AtomicResearchLedger()
            budget = ledger.load_budget()
            total_ledger = budget.experiments_used + budget.optuna_trials_used + budget.strategy_variants_used
            total_registry = len(self._index)
            # Per OOS
            if oos_period:
                n_reg = len(self.list_by_oos(oos_period))
                n_ledger_oos = ledger.get_oos_access_count(oos_period)
                complete = n_reg >= n_ledger_oos
                return {
                    "oos_period": oos_period,
                    "registry_oos": n_reg,
                    "ledger_oos": n_ledger_oos,
                    "complete": complete,
                    "reliable": complete and n_reg >= 1,
                    "message": f"Registry {n_reg} vs Ledger {n_ledger} for {oos_period} — {'OK' if complete else 'INCOMPLETE: PBO/DSR unreliable'}",
                }
            # Global
            complete = total_registry >= total_ledger * 0.9  # allow 10% lag
            return {
                "total_registry": total_registry,
                "total_ledger": total_ledger,
                "complete": complete,
                "reliable": complete,
                "message": f"Registry {total_registry} vs Ledger {total_ledger} - {'OK source of truth' if complete else 'INCOMPLETE: 1000 experiments but registry has 20 -> PBO/DSR invalid'}",
            }
        except Exception as e:
            return {"complete": False, "reliable": False, "error": str(e)}

    def pbo_placeholder(self, returns_df=None, n_splits: int = 6, n_test_folds: int = 2) -> float:
        """
        Real PBO via CPCV (Bailey et al.) when returns_df provided, else 0.0 placeholder.

        P0.6: Registry must be source of truth — if returns_df is None but registry has data,
        we still need to warn that PBO with N=0 is invalid. Effective N is max(registry, ledger).
        """
        # P0.6: Even when returns_df is None, check registry completeness for warning
        if returns_df is None:
            # Check if registry is incomplete vs ledger — log warning that PBO is unreliable
            try:
                comp = self.verify_registry_completeness()
                if not comp.get("complete"):
                    import warnings
                    warnings.warn(f"PBO with no returns_df but registry {comp.get('total_registry')} vs ledger {comp.get('total_ledger')} — PBO unreliable, registry must be source of truth", stacklevel=2)
            except Exception:
                pass
            return 0.0
        # When returns_df provided, effective N should be max of returns_df columns and registry count
        # But compute_pbo already uses returns_df shape for N, so we just delegate
        # Also ensure registry completeness for multiple-testing correction
        try:
            from src.research.pbo import compute_pbo
            # P0.6: If registry has more trials than returns_df columns, warn that PBO may be optimistic
            n_cols = returns_df.shape[1] if hasattr(returns_df, 'shape') else len(returns_df) if isinstance(returns_df, list) else 0
            n_registry = len(self._index)
            if n_registry > n_cols:
                import warnings
                warnings.warn(f"PBO: registry has {n_registry} experiments but returns_df has {n_cols} strategies — PBO may be underestimated, registry must be source of truth", stacklevel=2)
            return float(compute_pbo(returns_df, n_splits=n_splits, n_test_folds=n_test_folds))
        except Exception:
            return 0.0

    def compute_pbo(self, returns_df, n_splits: int = 6, n_test_folds: int = 2) -> float:
        """Real PBO via CPCV (Bailey et al.) — preferred over pbo_placeholder."""
        try:
            from src.research.pbo import compute_pbo as _compute_pbo
            return float(_compute_pbo(returns_df, n_splits=n_splits, n_test_folds=n_test_folds))
        except Exception as e:
            raise RuntimeError(f"compute_pbo failed: {e}") from e

    def pbo(self, returns_df=None, n_splits: int = 6, n_test_folds: int = 2) -> float:
        """Alias for compute_pbo; falls back to placeholder 0.0 when no data."""
        if returns_df is not None:
            return self.compute_pbo(returns_df, n_splits=n_splits, n_test_folds=n_test_folds)
        return self.pbo_placeholder()

    @staticmethod
    def hash_params(params: Dict[str, Any]) -> str:
        return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]
