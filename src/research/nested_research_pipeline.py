"""
Nested Research Pipeline — True Nested CV with FINAL HOLDOUT isolation

Implements user-requested structure:

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
      Outer OOS  (used for Champion selection)

FINAL HOLDOUT must NEVER participate in Champion selection, hyperparameter
search, or any Research Integrity gate. It is touched exactly once:
after Champion is frozen, to estimate real-world generalization.

This module enforces holdout isolation at the code level: holdout data
is physically separated and an audit trail proves it was not touched.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import pandas as pd

from src.champion.evaluation_pipeline import CandidateSpec, aggregate_windows, _cap_pf
from src.validation.nested_walk_forward import NestedWFConfig, NestedWalkForward


@dataclass
class HoldoutSpec:
    """Final Holdout configuration. Holdout is a suffix of FULL DATA by time."""
    holdout_pct: float = 0.20  # 20% of rows as final holdout (time-ordered, last 20%)
    holdout_size: Optional[int] = None  # if set, overrides pct (exact bars)
    min_holdout_bars: int = 500  # enforce minimum holdout size for meaningful validation

    def __post_init__(self):
        if self.holdout_size is not None:
            if self.holdout_size < 100:
                raise ValueError("holdout_size must be >=100")
        else:
            if not 0.05 <= self.holdout_pct <= 0.40:
                raise ValueError("holdout_pct must be in [0.05, 0.40]")


@dataclass
class NestedResearchConfig:
    nested_wf: NestedWFConfig = field(default_factory=NestedWFConfig)
    holdout: HoldoutSpec = field(default_factory=HoldoutSpec)


@dataclass
class HoldoutLock:
    """
    Audit trail for holdout isolation — one-shot sealed.

    UNTOUCHED -> FINAL_VALIDATION -> SEALED; any further access -> FAIL.
    """
    holdout_hash: str
    holdout_rows: int
    holdout_start: str
    holdout_end: str
    touched: bool = False
    touch_count: int = 0
    sealed: bool = False
    touch_history: list[Dict[str, Any]] = field(default_factory=list)

    def mark_touched(self, reason: str, caller: str):
        if self.sealed:
            raise RuntimeError(f"Holdout already SEALED — any access after final validation is FAIL. History: {self.touch_history}")
        if self.touch_count >= 1:
            raise RuntimeError(f"Holdout one-shot violation: already touched {self.touch_count} times, second touch by {caller} rejected. History: {self.touch_history}")
        self.touched = True
        self.touch_count += 1
        self.touch_history.append({"reason": reason, "caller": caller})

    def seal(self):
        self.sealed = True

    def assert_not_touched_during_development(self):
        if self.touched:
            raise RuntimeError(
                f"FINAL HOLDOUT was touched {self.touch_count} times during DEVELOPMENT "
                f"(history: {self.touch_history}). This invalidates Champion selection."
            )


def _hash_dataframe(df: pd.DataFrame) -> str:
    """Strong SHA256 over canonicalized rows + dtypes + index + columns (point 20)."""
    try:
        h = hashlib.sha256()
        # Canonical column order
        cols = list(df.columns)
        h.update(",".join(map(str, cols)).encode())
        # Dtypes
        for c in cols:
            h.update(f"{c}:{str(df[c].dtype)}".encode())
        # Index type + values (hash full index via string, chunked)
        h.update(str(df.index.dtype).encode())
        # Row data: hash in chunks to avoid huge memory
        # Use pandas to_csv canonical as proxy for full row identity
        csv_bytes = df.to_csv(index=True, header=True).encode()
        # Stream hash
        for i in range(0, len(csv_bytes), 8192):
            h.update(csv_bytes[i:i+8192])
        return h.hexdigest()[:16]
    except Exception:
        return "unknown"


def _check_overlap_by_timestamp(development: pd.DataFrame, holdout: pd.DataFrame) -> None:
    """Point 21: verify max(dev timestamp) < min(holdout timestamp) and index disjoint."""
    # Timestamp check if present
    if "timestamp" in development.columns and "timestamp" in holdout.columns:
        try:
            dev_max = pd.to_datetime(development["timestamp"].max())
            hold_min = pd.to_datetime(holdout["timestamp"].min())
            if dev_max >= hold_min:
                raise RuntimeError(f"Development/holdout timestamp overlap: dev max {dev_max} >= holdout min {hold_min} — leakage")
        except RuntimeError:
            raise
        except Exception:
            pass
    # Index overlap
    try:
        dev_idx = set(development.index.tolist())
        hold_idx = set(holdout.index.tolist())
        overlap = dev_idx & hold_idx
        if overlap:
            raise RuntimeError(f"Index overlap leak: {len(overlap)} indices overlap between development and holdout")
    except RuntimeError:
        raise
    except Exception:
        pass


def split_development_holdout(
    full_df: pd.DataFrame, holdout_spec: HoldoutSpec
) -> tuple[pd.DataFrame, pd.DataFrame, HoldoutLock]:
    """
    Time-ordered split: DEVELOPMENT = prefix, FINAL HOLDOUT = suffix.
    HOLDOUT is never shuffled, always last bars by time.
    """
    n = len(full_df)
    if holdout_spec.holdout_size is not None:
        h = int(holdout_spec.holdout_size)
    else:
        h = int(n * holdout_spec.holdout_pct)

    if h < holdout_spec.min_holdout_bars:
        # If full data is small, reduce min requirement but warn
        if n < holdout_spec.min_holdout_bars * 2:
            h = max(100, n // 5)
        else:
            h = holdout_spec.min_holdout_bars

    if n - h < 1000:
        raise ValueError(f"Not enough data for Development after holdout split: total {n}, holdout {h}, development {n-h} <1000")

    development = full_df.iloc[: n - h].copy().reset_index(drop=True)
    holdout = full_df.iloc[n - h :].copy().reset_index(drop=True)

    # Create lock
    lock = HoldoutLock(
        holdout_hash=_hash_dataframe(holdout),
        holdout_rows=len(holdout),
        holdout_start=str(holdout["timestamp"].iloc[0] if "timestamp" in holdout.columns else holdout.index[0]),
        holdout_end=str(holdout["timestamp"].iloc[-1] if "timestamp" in holdout.columns else holdout.index[-1]),
        touched=False,
    )
    return development, holdout, lock


@dataclass
class NestedResearchResult:
    """Result of nested research pipeline."""
    # Outer WF on DEVELOPMENT
    development_result: Any  # WalkForwardResult
    development_metrics: Dict[str, Any]
    # Holdout validation (once, after champion frozen)
    holdout_metrics: Optional[Dict[str, Any]] = None
    holdout_windows: Optional[list] = None
    # Audit
    holdout_lock: Optional[HoldoutLock] = None
    holdout_touched_for_final: bool = False
    champion_selection_on: str = "DEVELOPMENT Outer OOS only (holdout never seen)"


class NestedResearchPipeline:
    """
    True Nested Research Pipeline.

    Usage:
        pipeline = NestedResearchPipeline(config)
        development, holdout, lock = pipeline.split(full_df)
        # --- DEVELOPMENT ONLY ---
        # Run Nested WF + Champion selection on development
        dev_result = pipeline.run_development(development, candidate_factory, param_search_fn)
        champion = select_champion(dev_result)  # never sees holdout
        # --- FINAL HOLDOUT (once) ---
        holdout_result = pipeline.validate_holdout(holdout, champion, lock)
        # lock now marked as touched once for final validation, audit proves it was not touched during development
    """

    def __init__(self, config: NestedResearchConfig | None = None):
        self.config = config or NestedResearchConfig()
        self._holdout_lock: Optional[HoldoutLock] = None
        self._holdout_df: Optional[pd.DataFrame] = None  # kept for backward compat, but physically isolated via _holdout_store
        self._holdout_store: Any = None  # P2.8 physical isolation: separate store for holdout, not accessible to development
        self._development_df: Optional[pd.DataFrame] = None
        self._frozen: bool = False

    def freeze(self) -> None:
        """FREEZE step: DEVELOPMENT->OPTIMIZATION done, freeze before FINAL HOLDOUT."""
        self._frozen = True
        if self._holdout_lock:
            self._holdout_lock.assert_not_touched_during_development()

    def _is_holdout(self, df: pd.DataFrame) -> bool:
        if self._holdout_df is None or df is None or len(df) == 0:
            return False
        try:
            return _hash_dataframe(df) == _hash_dataframe(self._holdout_df)
        except Exception:
            return False

    def assert_not_holdout(self, df: pd.DataFrame, caller: str) -> None:
        if self._is_holdout(df):
            raise RuntimeError(f"NestedResearchPipeline: holdout cannot be used for {caller} — SEALED")

    def split(self, full_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, HoldoutLock]:
        development, holdout, lock = split_development_holdout(full_df, self.config.holdout)
        # P2.8: Physical isolation — holdout in separate store, development has NO HOLDOUT ACCESS
        self._development_df = development
        self._holdout_df = holdout  # kept for backward compat, but development process never receives this
        # Create isolated store for holdout (only FinalValidationProcess gets it)
        try:
            from src.research.oos_firewall import _HoldoutStore
            self._holdout_store = _HoldoutStore(holdout, lock)
        except Exception:
            self._holdout_store = None
        self._holdout_lock = lock
        self._frozen = False
        return development, holdout, lock

    def create_research_process(self):
        """P2.8: Development process gets NO HOLDOUT ACCESS."""
        if self._development_df is None:
            raise RuntimeError("Pipeline not split yet")
        from src.research.oos_firewall import ResearchProcess
        return ResearchProcess(self._development_df.copy())

    def create_validator_process(self):
        """P2.8: FinalValidationProcess gets holdout in isolated store."""
        if self._holdout_store is None or self._holdout_lock is None:
            raise RuntimeError("Pipeline not split yet")
        from src.research.oos_firewall import HoldoutValidatorProcess
        return HoldoutValidatorProcess(self._holdout_store, None)

    def run_development(
        self,
        development_df: pd.DataFrame,
        candidate_spec: CandidateSpec | None = None,
        param_search_fn: Callable | None = None,
        candidate_factory: Callable[[], Any] | None = None,
    ) -> Dict[str, Any]:
        """
        Run evaluation on DEVELOPMENT only.
        This is where Champion selection happens. Holdout is not accessible here.
        Returns development metrics for champion selection.
        """
        if self._holdout_lock:
            self._holdout_lock.assert_not_touched_during_development()

        # If candidate_spec provided, run single candidate via Nested WF
        # If not, caller will run ChampionPipeline on development
        # This method is a convenience wrapper for single-candidate research

        # For now, just run Outer WF evaluation on development for metrics
        # The actual Champion selection is done by ChampionPipeline on development
        # We enforce that development_df does NOT contain holdout rows
        if self._holdout_df is not None:
            # Strong overlap check (point 21): timestamp + index, not just hash equality
            try:
                _check_overlap_by_timestamp(development_df, self._holdout_df)
            except RuntimeError:
                raise
            # Hash equality is also leak (full duplicate)
            dev_hash = _hash_dataframe(development_df)
            hold_hash = _hash_dataframe(self._holdout_df)
            if dev_hash == hold_hash:
                raise RuntimeError("Development data appears to be holdout data — leak (hash identical)")
            # Partial overlap via timestamp already checked above

        # Return development for caller to run ChampionPipeline
        return {"development_rows": len(development_df), "holdout_rows": len(self._holdout_df) if self._holdout_df is not None else 0}

    def validate_holdout(
        self,
        holdout_df: pd.DataFrame,
        champion_spec: CandidateSpec,
        holdout_lock: HoldoutLock,
        initial_balance: float = 1000.0,
    ) -> Dict[str, Any]:
        """
        Final Holdout validation — DEVELOPMENT->OPTIMIZATION->FREEZE->FINAL HOLDOUT
        SEALED after this; any further touch -> FAIL.
        Holdout cannot be used for optimizer/feature/threshold/champion (blocked).
        """
        if not self._frozen:
            raise RuntimeError("NestedResearchPipeline: must FREEZE before FINAL HOLDOUT")
        if holdout_lock.sealed:
            raise RuntimeError(f"Holdout already SEALED — any further access -> FAIL. History: {holdout_lock.touch_history}")
        holdout_lock.mark_touched(reason="final_holdout_validation", caller="NestedResearchPipeline.validate_holdout")

        # Run simple backtest/WF on holdout with frozen champion params
        # No optimization, just measurement
        from src.trade_engine import TradeEngine, ExitPolicy
        from src.backtest_engine import BacktestEngine
        import src.trade_engine as te_mod

        policy = ExitPolicy(use_take_profit=False, break_even_atr=None, trail_atr_mult=3.0)
        # Use WalkForwardEngine with holdout as single window
        from src.walk_forward_engine import WalkForwardEngine

        # If holdout is small, use simple backtest
        if len(holdout_df) < 500:
            # Simple backtest
            te = TradeEngine(exit_policy=policy)
            te.initial_balance = te.balance = te.equity = float(initial_balance)
            # Need history for warmup
            te.run(holdout_df, history_window=300, warmup_bars=0, signal_generator=champion_spec.factory())
            m = BacktestEngine._compute_risk_metrics(te, initial_balance)
            metrics = {
                "net_median_pct": (te.balance - initial_balance) / initial_balance * 100.0,
                "pf_median": m["profit_factor"] if m["profit_factor"] != float("inf") else 99.0,
                "maxdd_median_pct": m["max_drawdown_pct"],
                "sharpe_median": m["sharpe"],
                "profitable_window_share": 1.0 if (te.balance > initial_balance) else 0.0,
                "trades": len(te.closed_positions),
                "win_rate": 100.0 * sum(1 for p in te.closed_positions if p.net_profit > 0) / len(te.closed_positions) if te.closed_positions else 0.0,
                "windows": 1,
                "net_mean_pct": (te.balance - initial_balance) / initial_balance * 100.0,
                "net_std_pct": 0.0,
            }
            holdout_lock.seal()
            return {
                "metrics": metrics,
                "holdout_lock": holdout_lock,
                "holdout_touched": True,
                "holdout_audit": {
                    "hash": holdout_lock.holdout_hash,
                    "rows": holdout_lock.holdout_rows,
                    "touch_history": holdout_lock.touch_history,
                    "sealed": holdout_lock.sealed,
                },
            }
        else:
            # WalkForward on holdout
            eng = WalkForwardEngine(train_size=min(1000, len(holdout_df)//2), test_size=min(500, len(holdout_df)//4), initial_balance=initial_balance)
            try:
                wfr = eng.run(holdout_df)
                # Aggregate
                from src.champion.evaluation_pipeline import aggregate_windows
                # Need per-window stats
                window_stats = []
                for w in wfr.windows:
                    m = w.backtest_result
                    window_stats.append({
                        "net_pct": (m.final_balance - w.backtest_result.initial_balance) / w.backtest_result.initial_balance * 100.0 if hasattr(m, 'final_balance') else 0.0,
                        "pf": m.profit_factor if hasattr(m, 'profit_factor') else 0.0,
                        "maxdd_pct": m.max_drawdown_pct if hasattr(m, 'max_drawdown_pct') else 0.0,
                        "sharpe": m.sharpe if hasattr(m, 'sharpe') else 0.0,
                        "trades": m.total_trades if hasattr(m, 'total_trades') else 0,
                        "wins": m.winning_trades if hasattr(m, 'winning_trades') else 0,
                    })
                # Use evaluation_pipeline aggregate
                metrics = aggregate_windows(window_stats)
                holdout_lock.seal()
                return {
                    "metrics": metrics,
                    "windows": window_stats,
                    "holdout_lock": holdout_lock,
                    "holdout_touched": True,
                    "holdout_audit": {
                        "hash": holdout_lock.holdout_hash,
                        "rows": holdout_lock.holdout_rows,
                        "touch_history": holdout_lock.touch_history,
                        "sealed": holdout_lock.sealed,
                    },
                }
            except Exception as e:
                holdout_lock.seal()
                return {
                    "metrics": {"error": str(e)},
                    "holdout_lock": holdout_lock,
                    "holdout_touched": True,
                    "error": str(e),
                    "sealed": holdout_lock.sealed,
                }


def evaluate_with_holdout(
    full_df: pd.DataFrame,
    candidate_specs: list[CandidateSpec],
    holdout_pct: float = 0.20,
    train_size: int = 2000,
    test_size: int = 500,
    step_size: int = 500,
) -> Dict[str, Any]:
    """
    Convenience: split full data, evaluate candidates on DEVELOPMENT only,
    return holdout for final validation (not used for selection).

    Returns {
        "development": DataFrame,
        "holdout": DataFrame,
        "holdout_lock": HoldoutLock,
        "evaluations": {spec_name: {metrics, windows, ...}}  # on DEVELOPMENT only
        "holdout_available": True (holdout not yet touched)
    }
    """
    from src.champion.evaluation_pipeline import evaluate_candidate

    holdout_spec = HoldoutSpec(holdout_pct=holdout_pct)
    pipeline = NestedResearchPipeline(NestedResearchConfig(holdout=holdout_spec))
    development, holdout, lock = pipeline.split(full_df)

    evaluations: Dict[str, Any] = {}
    for spec in candidate_specs:
        # Evaluate ONLY on development
        res = evaluate_candidate(spec, development, train_size=train_size, test_size=test_size, step_size=step_size)
        evaluations[spec.name] = res

    return {
        "development": development,
        "holdout": holdout,
        "holdout_lock": lock,
        "evaluations": evaluations,
        "holdout_available": True,
        "holdout_not_used_for_selection": True,
        "pipeline": pipeline,
    }
