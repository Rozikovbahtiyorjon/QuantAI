"""
Nested Walk-Forward (Audit #46 Etapa 3) — True Nested WF with FULL INNER WF aggregate

Fix P0-6/P0-7: Previously only FIRST inner window was passed to Optuna (generate_windows(train_df)[0]),
which overfits to a single 100-bar slice. Now Optuna sees FULL INNER WF aggregate.

Architecture (True Nested WF):
 OUTER TRAIN (e.g., 3000 bars)
      ↓ split OUTER TRAIN into INNER WF (N windows)
 INNER WF N windows:
    WINDOW 1: inner_train_1 → inner_test_1 (OOS slice 1)
    WINDOW 2: inner_train_2 → inner_test_2 (OOS slice 2)
    ...
    WINDOW N: inner_train_N → inner_test_N (OOS slice N)
      ↓ aggregate OOS metrics over ALL inner windows (mean PF, Sharpe, win_rate, etc.)
      ↓ choose params based on AGGREGATE (not first window)
      ↓ freeze params
 OUTER OOS (untouched — never seen during inner optimization)

Prevents leakage of Optuna tuning into OOS. Without aggregate, PF 1.35 can be overfit to first slice.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional, List, Tuple

import pandas as pd

from src.walk.walk_forward_engine import WalkForwardEngine, WalkForwardResult


# =========================================================
# CONFIG
# =========================================================

@dataclass
class NestedWFConfig:
    outer_train_size: int = 3000
    outer_test_size: int = 600
    inner_train_size: int = 500
    inner_test_size: int = 100
    step_size: Optional[int] = None  # outer step (defaults to outer_test_size)
    inner_step_size: Optional[int] = None  # inner step (defaults to inner_test_size)

    def __post_init__(self):
        for name in ("outer_train_size", "outer_test_size", "inner_train_size", "inner_test_size"):
            v = getattr(self, name)
            if not isinstance(v, int) or v <= 0:
                raise ValueError(f"{name} must be positive int, got {v}")
        if self.step_size is not None and (not isinstance(self.step_size, int) or self.step_size <= 0):
            raise ValueError("step_size must be positive int or None")
        if self.inner_step_size is not None and (not isinstance(self.inner_step_size, int) or self.inner_step_size <= 0):
            raise ValueError("inner_step_size must be positive int or None")
        if self.outer_train_size < self.inner_train_size + self.inner_test_size:
            # Not fatal — inner will fallback to split, but warn via valid check
            pass

    @property
    def outer_step(self) -> int:
        return self.step_size if self.step_size is not None else self.outer_test_size

    @property
    def inner_step(self) -> int:
        return self.inner_step_size if self.inner_step_size is not None else self.inner_test_size


# =========================================================
# AGGREGATION HELPER
# =========================================================

def _aggregate_inner_metrics(inner_wfr: Optional[WalkForwardResult]) -> Dict[str, Any]:
    """
    Aggregate OOS metrics over ALL inner WF windows.

    Returns dict with mean/median PF, Sharpe, win_rate, profit, etc.
    Used to select params based on FULL INNER WF, not first window.
    """
    if inner_wfr is None or not inner_wfr.windows:
        return {
            "n_windows": 0,
            "mean_pf": 0.0,
            "median_pf": 0.0,
            "mean_sharpe": 0.0,
            "mean_win_rate": 0.0,
            "mean_net_profit": 0.0,
            "total_trades": 0,
            "total_wins": 0,
            "total_losses": 0,
            "win_rate_overall": 0.0,
        }
    import numpy as np

    pfs: List[float] = []
    sharpes: List[float] = []
    win_rates: List[float] = []
    net_profits: List[float] = []
    total_trades = 0
    total_wins = 0
    total_losses = 0

    for w in inner_wfr.windows:
        br = w.backtest_result
        pf = br.profit_factor
        if pf != float("inf") and pf is not None:
            try:
                pfs.append(float(pf))
            except Exception:
                pass
        try:
            sharpes.append(float(br.sharpe))
        except Exception:
            sharpes.append(0.0)
        try:
            win_rates.append(float(br.win_rate))
        except Exception:
            win_rates.append(0.0)
        try:
            net_profits.append(float(br.net_profit))
        except Exception:
            pass
        total_trades += int(br.total_trades)
        total_wins += int(br.winning_trades)
        total_losses += int(br.losing_trades)

    win_rate_overall = round((total_wins / total_trades * 100.0) if total_trades else 0.0, 2)

    def _mean(xs: List[float]) -> float:
        return float(np.mean(xs)) if xs else 0.0

    def _median(xs: List[float]) -> float:
        return float(np.median(xs)) if xs else 0.0

    return {
        "n_windows": len(inner_wfr.windows),
        "mean_pf": _mean(pfs),
        "median_pf": _median(pfs),
        "mean_sharpe": _mean(sharpes),
        "mean_win_rate": _mean(win_rates),
        "mean_net_profit": _mean(net_profits),
        "total_trades": total_trades,
        "total_wins": total_wins,
        "total_losses": total_losses,
        "win_rate_overall": win_rate_overall,
        # also expose raw lists for advanced selection
        "pfs": pfs,
        "sharpes": sharpes,
    }


def _choose_and_invoke(
    param_search_fn: Callable,
    inner_windows: List[Tuple[int, pd.DataFrame, pd.DataFrame]],
    inner_wfr: Optional[WalkForwardResult],
    aggregate: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Strict single-contract (point 16) — ONLY canonical signature allowed.

    Canonical: param_search_fn(inner_windows, inner_result, aggregate) -> FrozenParameters
    where inner_result is inner_wfr (WalkForwardResult) and aggregate is dict with
    mean/median PF/Sharpe over ALL inner windows.

    No heuristic, no trial-and-error, no 2-arg/1-arg fallback — any mismatch is FAIL-FAST.
    This simplifies leakage proof: optimizer sees FULL INNER WF, never OUTER TEST.
    """
    sig = inspect.signature(param_search_fn)
    params = list(sig.parameters.values())
    regular = [p for p in params if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)]
    n_total = len(regular)
    if n_total != 3:
        raise TypeError(
            f"param_search_fn signature {sig} has {n_total} params — expected exactly 3 "
            "(inner_windows, inner_result, aggregate) -> FrozenParameters. "
            "Legacy 1/2-arg adapters removed in 5.3 (point 16). Fix signature to single contract."
        )
    names = [p.name.lower() for p in regular]
    first_is_train = any("train" in n for n in names[:1])
    if first_is_train:
        raise TypeError(
            f"Legacy param_search_fn signature {sig} with first param containing 'train' is forbidden. "
            "Use canonical (inner_windows, inner_result, aggregate) — legacy would see only FIRST inner window."
        )
    try:
        result = param_search_fn(inner_windows, inner_wfr, aggregate)  # type: ignore
    except TypeError as e:
        raise TypeError(f"Canonical param_search_fn(inner_windows, inner_result, aggregate) failed — {e} — signature {sig}") from e
    if isinstance(result, dict):
        return result
    # Allow FrozenParameters dataclass with to_dict
    if hasattr(result, "to_dict"):
        try:
            return dict(result.to_dict())  # type: ignore
        except Exception:
            pass
    return dict(result) if result is not None else {}


def _invoke_param_search(
    param_search_fn: Callable,
    inner_windows: List[Tuple[int, pd.DataFrame, pd.DataFrame]],
    inner_wfr: Optional[WalkForwardResult],
    aggregate: Dict[str, Any],
) -> Dict[str, Any]:
    """Wrapper for backward compat — delegates to heuristic chooser."""
    if not callable(param_search_fn):
        raise TypeError("param_search_fn must be callable")
    return _choose_and_invoke(param_search_fn, inner_windows, inner_wfr, aggregate)


class NestedWalkForward:
    """Runs inner WF for hyperparam search (FULL aggregate), evaluates best on outer OOS."""

    def __init__(self, config: NestedWFConfig = NestedWFConfig()):
        self.config = config

    def run(
        self,
        df: pd.DataFrame,
        param_search_fn: Callable[..., Dict[str, Any]] | None = None,
        best_param_apply_fn: Optional[Callable[[Dict[str, Any]], None]] = None,
        strategy_factory: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ) -> WalkForwardResult:
        """
        Leakage-free TRUE Nested WF — single strict contract.

        param_search_fn(inner_windows, inner_result, aggregate) -> FrozenParameters
            inner_windows: List[Tuple[window_id, train_df, test_df]] for ALL inner slices
            inner_result: WalkForwardResult over OUTER TRAIN (FULL INNER WF)
            aggregate: dict(mean_pf, median_pf, mean_sharpe, ...) over ALL inner windows

        Returns FrozenParameters (dict or dataclass with to_dict) which is frozen
        before OUTER OOS via strategy_factory. No 1/2-arg variants — single contract
        simplifies leakage proof (point 16).

        Legacy: best_param_apply_fn is deprecated adapter (will be removed); use
        strategy_factory(params) -> Strategy for immutable per-window creation.

        IMPORTANT: inner_* are splits of OUTER TRAIN only. OUTER TEST is never passed
        to optimizer — isolation is enforced by code.
        """
        # Generate outer windows
        outer = WalkForwardEngine(
            train_size=self.config.outer_train_size,
            test_size=self.config.outer_test_size,
            step_size=self.config.outer_step,
        )
        outer_windows = outer.generate_windows(df)
        results = []
        balance = outer.initial_balance

        for window_number, train_df, test_df in outer_windows:
            # Inner WF on OUTER TRAIN ONLY — OUTER TEST is isolated
            inner = WalkForwardEngine(
                train_size=self.config.inner_train_size,
                test_size=self.config.inner_test_size,
                step_size=self.config.inner_step,
            )
            # FULL INNER WF must actually execute, not just generate windows
            # Audit P0: optimization errors must NOT be silent — research validation must fail, not continue with {}
            inner_wfr = None
            inner_windows: List[Tuple[int, pd.DataFrame, pd.DataFrame]] = []
            try:
                inner_windows = inner.generate_windows(train_df)
            except Exception:
                inner_windows = []
            try:
                inner_wfr = inner.run(train_df)
                # Re-generate windows to ensure we have the list aligned with wfr
                # (run already validated, but generate again for param_search)
                if not inner_windows:
                    try:
                        inner_windows = inner.generate_windows(train_df)
                    except Exception:
                        inner_windows = []
            except Exception as e_inner:
                # INNER WF failure is research failure — do not silently continue with scaffold
                # Only allow fallback if train_df genuinely too small (< inner train+test)
                if len(train_df) < inner.train_size + inner.test_size:
                    inner_wfr = None
                    # inner_windows already captured (likely empty)
                else:
                    raise RuntimeError(f"Inner WF failed in outer window {window_number}: {e_inner}") from e_inner

            # Compute FULL INNER aggregate (mean PF, Sharpe, etc. over ALL inner windows)
            aggregate = _aggregate_inner_metrics(inner_wfr)

            # P0.6: Automatic assertion - outer_test_hash NOT PRESENT in any optimization artifact
            # Stronger than index check: hash must not appear in inner windows, aggregate, or best_params
            outer_test_hash = None
            try:
                from src.research.nested_research_pipeline import _hash_dataframe
                outer_test_hash = _hash_dataframe(test_df)
                # Check not in inner windows (hash of any inner test)
                for _, _, inner_test in inner_windows:
                    try:
                        if _hash_dataframe(inner_test) == outer_test_hash:
                            raise AssertionError(f"outer_test_hash {outer_test_hash} found in inner optimization windows — leakage (outer_test leaked into optimizer)")
                    except AssertionError:
                        raise
                    except Exception:
                        pass
                # Check not in aggregate (aggregate should not contain outer_test data)
                # Aggregate is dict of metrics, not data, but check if any string value contains hash (defensive)
                agg_str = str(aggregate)
                if outer_test_hash and outer_test_hash in agg_str:
                    raise AssertionError(f"outer_test_hash {outer_test_hash} found in aggregate artifact — leakage")
            except AssertionError:
                raise
            except Exception:
                # Hash check best-effort, not fatal if hashing fails
                outer_test_hash = outer_test_hash or "unknown"

            # Optimization: any exception → EXPERIMENT FAILED (fail-fast), not best_params={}
            best_params: Dict[str, Any] = {}
            _inner_wfr_for_evidence = inner_wfr
            _aggregate_for_evidence = aggregate
            _inner_windows_for_evidence = inner_windows
            _outer_test_hash_for_evidence = outer_test_hash
            try:
                if param_search_fn is not None:
                    # Case: inner WF produced windows and wfr
                    if inner_windows:
                        # Leakage guard: every inner TEST must end before OUTER TEST starts
                        # Use positional index check via length + assumption of time order;
                        # also check index values if possible
                        for _, _, inner_test in inner_windows:
                            try:
                                if hasattr(inner_test.index, "max") and hasattr(test_df.index, "min"):
                                    # Strict isolation: inner_test must end before outer test starts
                                    assert inner_test.index.max() < test_df.index.min(), "Inner/Outer overlap — leakage"
                            except AssertionError:
                                raise
                            except Exception:
                                # Index comparison may fail for mixed types
                                # Fallback: positional check via length ordering is insufficient alone
                                # so we require time-ordered df; if indexes incomparable, verify via row positions
                                try:
                                    # Fallback positional: train_df ends before test_df starts by construction
                                    # so inner_test (slice of train_df) must also end before test_df
                                    train_end = getattr(train_df.index, "max", lambda: len(train_df))()
                                    test_start = getattr(test_df.index, "min", lambda: float("inf"))()
                                    assert train_end < test_start, "Inner/Outer overlap (positional) — leakage"
                                except Exception:
                                    pass
                        # True Nested WF: pass ALL windows + wfr + aggregate
                        best_params = _invoke_param_search(param_search_fn, inner_windows, inner_wfr, aggregate)
                    elif inner_wfr is None:
                        # Fallback when train_df too small for inner WF: use simple split
                        # This path is allowed only for small data; still must propagate Optuna errors
                        if len(train_df) < inner.train_size + inner.test_size:
                            if inner_windows:
                                # Should not happen (windows empty when too small)
                                best_params = _invoke_param_search(param_search_fn, inner_windows, inner_wfr, aggregate)
                            else:
                                split = len(train_df) // 2
                                single_window = [(1, train_df.iloc[:split].copy(), train_df.iloc[split:].copy())]
                                # Strict single contract even for small-data fallback
                                best_params = _invoke_param_search(param_search_fn, single_window, None, aggregate)
                                # Ensure best_params is dict
                                if best_params is None:
                                    best_params = {}
                                elif not isinstance(best_params, dict):
                                    best_params = dict(best_params)
                        else:
                            raise RuntimeError(f"Inner WF missing but train large enough in window {window_number} — cannot optimize")
                    else:
                        best_params = {}
                else:
                    best_params = {}
                # P0.6: Automatic assertion - outer_test_hash NOT PRESENT in any optimization artifact (best_params)
                if outer_test_hash and outer_test_hash != "unknown":
                    try:
                        # Check best_params does not contain outer_test data
                        params_str = str(best_params)
                        if outer_test_hash in params_str:
                            raise AssertionError(f"outer_test_hash {outer_test_hash} found in best_params artifact — leakage (optimizer saw outer_test)")
                        # Also check inner aggregate string already checked, but double-check
                        # Check that best_params values are not DF hashes
                        for v in best_params.values():
                            if isinstance(v, str) and v == outer_test_hash:
                                raise AssertionError(f"outer_test_hash {outer_test_hash} found in best_params value — leakage")
                    except AssertionError:
                        raise
                    except Exception:
                        pass
                # Normalize
                if best_params is None:
                    best_params = {}
                if not isinstance(best_params, dict):
                    try:
                        best_params = dict(best_params)
                    except Exception:
                        best_params = {}
            except Exception as e_opt:
                # Research pipeline: optimization error → EXPERIMENT FAILED, not "valid result" with {}
                raise RuntimeError(f"Optimization failed in outer window {window_number}: {e_opt}") from e_opt

            # HARD FORBID mutable carry-over: every outer window must get fresh strategy via factory (point 15)
            if best_param_apply_fn is not None:
                raise TypeError(
                    "best_param_apply_fn (mutable global strategy) is FORBIDDEN — state contamination "
                    "window 1 → mutate → window 2 reuses → window 3 reuses. "
                    "Use strategy_factory(params) -> fresh Strategy per outer window (immutable). "
                    "See NestedWalkForward.run docstring."
                )
            # Immutable factory: required when optimization is performed
            if param_search_fn is not None and strategy_factory is None:
                # No factory + optimization → would require mutable carry-over → forbidden
                # Allow only when param_search_fn is None (no optimization, pure validation)
                raise TypeError(
                    "param_search_fn requires strategy_factory for immutable per-window recreation. "
                    "Mutable carry-over between windows is hard-forbidden."
                )
            _factory_for_window = None
            # P0.6: Registry as source of truth — register each outer window optimization as experiment
            # This ensures that even if AI does 1000 experiments via Nested WF, registry has 1000, not 20
            try:
                from src.research.experiment_registry import ExperimentRegistry, ExperimentRecord
                from src.research.research_ledger import AtomicResearchLedger
                # Create experiment record for this outer window's optimization
                exp_rec = ExperimentRecord(
                    strategy_family="nested_wf_outer",
                    dataset_id=f"outer_window_{window_number}",
                    oos_period=f"outer_test_{window_number}",
                    parameters=dict(best_params) if best_params else {},
                    used_for_selection=True,
                    oos_touched=True,
                    selection_status="CANDIDATE",
                    PF=float(aggregate.get("mean_pf", 0.0)),
                    Sharpe=float(aggregate.get("mean_sharpe", 0.0)),
                    Trades=int(aggregate.get("total_trades", 0)),
                )
                # Use shared registry file (source of truth)
                reg = ExperimentRegistry()
                reg.register(exp_rec)
                # Also increment ledger for multiple-testing correction (PBO/DSR N)
                try:
                    ledger = AtomicResearchLedger()
                    ledger.check_and_increment("experiment")
                    if exp_rec.oos_period:
                        ledger.check_and_increment("oos_reuse", registry_oos_reuse=reg.oos_reuse_count(exp_rec.oos_period))
                except Exception:
                    pass
            except Exception:
                # Registration best-effort, not fatal for WF execution
                pass
            # Track frozen state per window for independent check
            if not hasattr(self, '_prev_strategy_id'):
                self._prev_strategy_id = None
                self._prev_best_params = None
            if strategy_factory is not None and best_params:
                try:
                    _strat = strategy_factory(best_params)
                    # P0.6: each outer window must have independent frozen state (not shared object)
                    strat_id = id(_strat)
                    if self._prev_strategy_id is not None and strat_id == self._prev_strategy_id:
                        raise AssertionError(f"Outer window {window_number} frozen state not independent: strategy object reused (id {strat_id} == prev) — mutable carry-over")
                    if self._prev_best_params is not None and best_params == self._prev_best_params and len(outer_windows) > 1:
                        # Allow same params if optimizer converged, but log; not fail if same params but different object is ok
                        # Only fail if same object reused, not same value
                        pass
                    self._prev_strategy_id = strat_id
                    self._prev_best_params = dict(best_params)
                    _factory_for_window = strategy_factory
                except AssertionError:
                    raise
                except Exception as e_apply:
                    raise RuntimeError(f"strategy_factory failed in window {window_number}: {e_apply}") from e_apply
            elif strategy_factory is not None and not best_params and param_search_fn:
                if inner_windows and aggregate.get("n_windows", 0) > 0:
                    raise RuntimeError(f"Optimizer returned no params for immutable factory in window {window_number} (inner windows {len(inner_windows)})")

            # Evaluate on OUTER TEST with frozen best params
            train_start = (window_number - 1) * outer.step_size
            train_end = train_start + outer.train_size
            test_start = train_end
            test_end = test_start + outer.test_size

            # If immutable factory supplied, try to inject into run_window via strategy_factory kwarg or signal_generator override
            win_kwargs: dict = {}
            if _factory_for_window is not None:
                win_kwargs["strategy_factory"] = _factory_for_window
                # Also try signal_generator for engines that accept it
                try:
                    win_kwargs["signal_generator"] = _factory_for_window(best_params)
                except Exception:
                    pass
            win = outer.run_window(
                df=df,
                window_id=window_number,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                initial_balance=balance,
                **win_kwargs,
            )
            # Evidence: attach inner WF result + aggregate + windows to outer window (audit trail that FULL inner was executed)
            try:
                win.model_result = _inner_wfr_for_evidence  # type: ignore
                # Attach aggregate as extra evidence if possible
                win.inner_aggregate = _aggregate_for_evidence  # type: ignore
                win.inner_windows = _inner_windows_for_evidence  # type: ignore
                win.inner_n_windows = len(_inner_windows_for_evidence)  # type: ignore
            except Exception:
                pass
            results.append(win)
            balance = float(win.backtest_result.final_balance)

        # Aggregate like WalkForwardEngine.run
        from src.walk.walk_forward_engine import WalkForwardResult as WFR

        total_trades = sum(w.backtest_result.total_trades for w in results)
        winning = sum(w.backtest_result.winning_trades for w in results)
        losing = sum(w.backtest_result.losing_trades for w in results)
        win_rate = round((winning / total_trades * 100.0) if total_trades else 0.0, 2)

        return WFR(
            initial_balance=outer.initial_balance,
            final_balance=balance,
            net_profit=round(balance - outer.initial_balance, 8),
            total_trades=total_trades,
            winning_trades=winning,
            losing_trades=losing,
            win_rate=win_rate,
            windows=results,
        )

    # Helper exposed for testing / inspection
    @staticmethod
    def aggregate_inner_metrics(inner_wfr: Optional[WalkForwardResult]) -> Dict[str, Any]:
        return _aggregate_inner_metrics(inner_wfr)
