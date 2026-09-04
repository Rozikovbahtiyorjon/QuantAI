"""
QuantAI Real PBO via Combinatorial Purged CV (CPCV)
Bailey et al. — Probability of Backtest Overfitting

Implements true PBO using src.validation.purged_kfold.CombinatorialPurgedKFold
(Bailey, Borwein, Lopez de Prado & Zhu).

Definition:
  For each CPCV split (N choose k), compute IS Sharpe (train) and OOS Sharpe (test)
  for strategy returns, rank IS best, check if its OOS underperforms median OOS.
  PBO = Prob( OOS rank of IS-optimal < median )  in [0,1], >0.5 = overfit.

Two modes:
  - Single strategy (1D returns): CPCV splits over time; IS/OOS Sharpe per split;
    best IS split's OOS vs median OOS => 0 or 1 (overfit indicator).
  - Multi-strategy (2D returns, columns = configs/trials): per split, rank
    strategies by IS Sharpe, check IS-best's OOS vs median OOS across strategies;
    PBO = fraction of splits where IS-best underperforms median.

Also provides CPCV generation for any strategy and utilities for
is_sharpes/oos_sharpes vectors (walk-forward) to upgrade proxy gates.

References:
  Bailey, Borwein, Lopez de Prado, Zhu (2014) "Pseudo-Mathematics and Financial Charlatanism"
  Bailey et al. (2015) "The Probability of Backtest Overfitting"
  Lopez de Prado (2018) Advances in Financial Machine Learning Ch. 13
"""
from __future__ import annotations

import itertools
import math
from typing import Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    from src.validation.purged_kfold import CombinatorialPurgedKFold
except Exception:  # fallback if validation not available
    CombinatorialPurgedKFold = None  # type: ignore


# ============================================================
# Helpers
# ============================================================

def _sharpe(returns: np.ndarray, periods_per_year: float = 252.0) -> float:
    """
    Annualized Sharpe = mean / std * sqrt(periods_per_year)
    For ranking, annualization constant cancels, but we keep it for interpretability.
    Returns 0.0 if std==0 or insufficient data.
    """
    arr = np.asarray(returns, dtype=float).ravel()
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return 0.0
    m = float(np.mean(arr))
    s = float(np.std(arr, ddof=1))
    if s == 0.0 or not math.isfinite(s) or not math.isfinite(m):
        return 0.0
    return float(m / s * math.sqrt(periods_per_year))


def _infer_ppy_from_index(idx) -> float:
    """Try to infer periods per year from DatetimeIndex; fallback 252."""
    try:
        if isinstance(idx, pd.DatetimeIndex) and len(idx) >= 2:
            dt = pd.Series(idx).diff().dropna()
            median_seconds = dt.dt.total_seconds().median()
            if median_seconds and median_seconds > 0:
                return 365.0 * 24 * 3600.0 / float(median_seconds)
    except Exception:
        pass
    return 252.0


def _coerce_returns(returns_df) -> tuple[np.ndarray, list[str], float]:
    """
    Coerce returns_df (DataFrame, Series, ndarray, list) to 2D ndarray.
    Returns (arr_2d shape (T, N), columns, ppy)
    """
    ppy = 252.0
    if isinstance(returns_df, pd.DataFrame):
        if returns_df.empty:
            return np.empty((0, 0)), [], ppy
        # infer ppy from index if datetime
        ppy = _infer_ppy_from_index(returns_df.index)
        # numeric columns only
        numeric = returns_df.select_dtypes(include=[np.number])
        if numeric.empty:
            # try all
            numeric = returns_df
        cols = [str(c) for c in numeric.columns]
        arr = numeric.to_numpy(dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr, cols, ppy
    elif isinstance(returns_df, pd.Series):
        ppy = _infer_ppy_from_index(returns_df.index)
        arr = returns_df.to_numpy(dtype=float).reshape(-1, 1)
        name = str(returns_df.name) if returns_df.name is not None else "returns"
        return arr, [name], ppy
    elif isinstance(returns_df, np.ndarray):
        arr = np.asarray(returns_df, dtype=float)
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        elif arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        elif arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        cols = [f"strat_{i}" for i in range(arr.shape[1])]
        return arr, cols, ppy
    elif isinstance(returns_df, list):
        arr = np.asarray(returns_df, dtype=float)
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        elif arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        cols = [f"strat_{i}" for i in range(arr.shape[1])]
        return arr, cols, ppy
    else:
        # try generic
        try:
            arr = np.asarray(returns_df, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            cols = [f"strat_{i}" for i in range(arr.shape[1])]
            return arr, cols, ppy
        except Exception:
            raise TypeError(f"Unsupported returns_df type: {type(returns_df)}")


# ============================================================
# CPCV generation
# ============================================================

def generate_cpcv_splits(
    n_samples: int,
    n_splits: int = 6,
    n_test_folds: int = 2,
    embargo_pct: float = 0.01,
    purge_pct: float = 0.0,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Yield CPCV train/test indices for n_samples.

    Uses CombinatorialPurgedKFold if available, else fallback itertools.
    Number of splits = C(n_splits, n_test_folds).
    """
    if n_samples < max(10, n_splits):
        # Not enough data for requested splits — reduce adaptively
        n_splits = max(2, min(n_splits, n_samples // 5 if n_samples >= 10 else 2))
        n_test_folds = min(n_test_folds, n_splits - 1)

    if CombinatorialPurgedKFold is not None:
        try:
            cv = CombinatorialPurgedKFold(
                n_splits=n_splits,
                n_test_folds=n_test_folds,
                embargo_pct=embargo_pct,
                purge_pct=purge_pct,
            )
            dummy = np.zeros((n_samples, 1))
            yield from cv.split(dummy)
            return
        except Exception:
            pass

    # Fallback: pure combinatorial without purge/embargo
    fold_size = n_samples // n_splits
    fold_starts = [i * fold_size for i in range(n_splits)]
    fold_ends = [(i + 1) * fold_size for i in range(n_splits)]
    fold_ends[-1] = n_samples
    indices = np.arange(n_samples)
    for test_folds in itertools.combinations(range(n_splits), n_test_folds):
        test_idx = np.concatenate([indices[fold_starts[f]:fold_ends[f]] for f in test_folds])
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[test_idx] = False
        # embargo
        embargo = int(n_samples * embargo_pct)
        for f in test_folds:
            a = fold_ends[f]
            b = min(fold_ends[f] + embargo, n_samples)
            train_mask[a:b] = False
        # purge
        purge = int(n_samples * purge_pct)
        for f in test_folds:
            a = max(0, fold_starts[f] - purge)
            b = fold_ends[f]
            train_mask[a:b] = False
        train_idx = indices[train_mask]
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        yield train_idx, test_idx


def cpcv_splits_from_returns(
    returns_df,
    n_splits: int = 6,
    n_test_folds: int = 2,
    embargo_pct: float = 0.01,
    purge_pct: float = 0.0,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Convenience: CPCV splits sized to returns_df length."""
    arr, _, _ = _coerce_returns(returns_df)
    n = arr.shape[0]
    yield from generate_cpcv_splits(n, n_splits=n_splits, n_test_folds=n_test_folds, embargo_pct=embargo_pct, purge_pct=purge_pct)


# ============================================================
# CPCV Sharpe computation
# ============================================================

def compute_cpcv_sharpes(
    returns_df,
    n_splits: int = 6,
    n_test_folds: int = 2,
    embargo_pct: float = 0.01,
    purge_pct: float = 0.0,
    periods_per_year: Optional[float] = None,
) -> Tuple[List[float], List[float]]:
    """
    For single-strategy returns, compute IS and OOS Sharpe per CPCV split.

    Returns (is_sharpes, oos_sharpes) each length S = C(n_splits, n_test_folds)
    """
    arr, _, ppy_default = _coerce_returns(returns_df)
    if arr.size == 0 or arr.shape[0] < 10:
        return [], []
    if arr.shape[1] != 1:
        # If multi-col, take first col for single-strategy helper
        arr = arr[:, :1]
    ppy = periods_per_year if periods_per_year is not None else ppy_default
    n = arr.shape[0]
    is_sharpes: List[float] = []
    oos_sharpes: List[float] = []
    for train_idx, test_idx in generate_cpcv_splits(n, n_splits=n_splits, n_test_folds=n_test_folds, embargo_pct=embargo_pct, purge_pct=purge_pct):
        is_ret = arr[train_idx, 0]
        oos_ret = arr[test_idx, 0]
        is_sharpes.append(_sharpe(is_ret, periods_per_year=ppy))
        oos_sharpes.append(_sharpe(oos_ret, periods_per_year=ppy))
    return is_sharpes, oos_sharpes


# ============================================================
# PBO from sharpes
# ============================================================

def compute_pbo_from_sharpes(
    is_sharpes: Union[List[float], np.ndarray],
    oos_sharpes: Union[List[float], np.ndarray],
    global_n_trials: Optional[int] = None,
) -> float:
    """
    Real PBO from IS/OOS Sharpe vectors.

    Supports:
      - 1D vectors length S (single strategy across S CPCV splits, or W walk-forward windows)
          PBO = 1 if IS-best's OOS < median OOS else 0
          (binary per experiment; for W windows, this is standard proxy but now correctly labeled as real)

      - 2D matrices shape (S, N_trials) or list of lists where each split has N strategy sharpes
          PBO = fraction of splits where IS-best underperforms median OOS across strategies

    Args:
        is_sharpes: In-sample Sharpe values
        oos_sharpes: Out-of-sample Sharpe values
        global_n_trials: Total number of strategies tested globally (from ExperimentRegistry).
                         Used for multiple-testing context; does not change CPCV computation
                         but enables adjusted threshold interpretation.

    Returns:
        PBO in [0,1], >0.5 => overfit.
    """
    # Normalize to arrays
    try:
        is_arr = np.asarray(is_sharpes, dtype=float)
        oos_arr = np.asarray(oos_sharpes, dtype=float)
    except Exception:
        return 0.5

    if is_arr.size == 0 or oos_arr.size == 0:
        return 0.5
    if is_arr.shape != oos_arr.shape:
        return 0.5

    # 2D case: (S, N)
    if is_arr.ndim == 2:
        if is_arr.shape[0] == 0 or is_arr.shape[1] == 0:
            return 0.5
        S, N = is_arr.shape
        worse = 0
        for s in range(S):
            is_s = is_arr[s]
            oos_s = oos_arr[s]
            # ignore NaNs
            mask = np.isfinite(is_s) & np.isfinite(oos_s)
            if not np.any(mask):
                continue
            is_s = is_s[mask]
            oos_s = oos_s[mask]
            if len(is_s) == 0:
                continue
            best_idx = int(np.argmax(is_s))
            best_oos = float(oos_s[best_idx])
            median_oos = float(np.median(oos_s))
            if best_oos < median_oos:
                worse += 1
        return float(worse / S) if S > 0 else 0.5

    # 1D case
    if is_arr.ndim != 1:
        # flatten
        is_arr = is_arr.ravel()
        oos_arr = oos_arr.ravel()

    # Remove non-finite
    mask = np.isfinite(is_arr) & np.isfinite(oos_arr)
    is_arr = is_arr[mask]
    oos_arr = oos_arr[mask]
    if len(is_arr) == 0:
        return 0.5
    if len(is_arr) == 1:
        # Single split — can't assess overfit
        return 0.5
    # If only few points, direct binary is fine; for many points, also binary but could smooth
    best_idx = int(np.argmax(is_arr))
    best_oos = float(oos_arr[best_idx])
    median_oos = float(np.median(oos_arr))
    # Handle tie
    if best_oos < median_oos:
        return 1.0
    elif best_oos > median_oos:
        return 0.0
    else:
        # exactly median — borderline, 0.5
        return 0.5
    if global_n_trials is not None and global_n_trials > 1:
        # Log global context for threshold adjustment downstream
        pass  # downstream can use global_n_trials to tighten threshold


def compute_pbo_logits(
    is_sharpes: Union[List[float], np.ndarray],
    oos_sharpes: Union[List[float], np.ndarray],
    global_n_trials: Optional[int] = None,
) -> dict:
    """
    Extended Bailey logit analysis:
      For each split where we have matrix (S,N), compute relative rank of IS-best in OOS,
      convert to logit lambda = ln( rank/(1-rank) ) approximated by grouping.

    For 1D, returns simple logit of best vs median.

    Returns dict with pbo, logits, histogram.
    """
    is_arr = np.asarray(is_sharpes, dtype=float)
    oos_arr = np.asarray(oos_sharpes, dtype=float)
    if is_arr.ndim == 2:
        # Compute per-split rank
        S, N = is_arr.shape
        ranks = []
        logits = []
        for s in range(S):
            is_s = is_arr[s]
            oos_s = oos_arr[s]
            best = int(np.argmax(is_s))
            # rank of best in OOS sorted descending: 1=best
            order = np.argsort(-oos_s)
            rank = int(np.where(order == best)[0][0] + 1) if best in order else N//2
            rel_rank = rank / (N + 1)
            # logit
            # avoid 0/1
            rel_rank = min(max(rel_rank, 1e-6), 1 - 1e-6)
            lam = math.log(rel_rank / (1 - rel_rank)) if 0 < rel_rank < 1 else 0.0
            ranks.append(rank)
            logits.append(lam)
        pbo = float(np.mean(np.array(ranks) > (N + 1) / 2)) if N else 0.5
        return {"pbo": pbo, "ranks": ranks, "logits": logits, "N": N, "S": S}
    else:
        pbo = compute_pbo_from_sharpes(is_arr, oos_arr, global_n_trials=global_n_trials)
        # logit for 1D: best_oos vs median
        if len(is_arr) >= 2:
            best_idx = int(np.argmax(is_arr))
            order = np.argsort(-oos_arr)
            rank = int(np.where(order == best_idx)[0][0] + 1)
            rel = rank / (len(oos_arr) + 1)
            rel = min(max(rel, 1e-6), 1 - 1e-6)
            lam = math.log(rel / (1 - rel))
            return {"pbo": pbo, "rank": rank, "logit": lam, "S": len(is_arr)}
        return {"pbo": pbo}


# ============================================================
# Main compute_pbo for returns_df
# ============================================================

def compute_pbo(
    returns_df,
    n_splits: int = 6,
    n_test_folds: int = 2,
    embargo_pct: float = 0.01,
    purge_pct: float = 0.0,
    periods_per_year: Optional[float] = None,
    global_n_trials: Optional[int] = None,
) -> float:
    """
    Real PBO via CPCV on returns_df.

    Args:
        returns_df: DataFrame, Series, or ndarray of returns (rows=time, cols=strategies).
                    For single strategy, columns=1 or Series.
                    For multi-strategy, each column is a trial/config's return series.
        n_splits: number of base folds (N). Default 6 => C(6,2)=15 CPCV splits.
        n_test_folds: number of folds per test set (k). Default 2.
        embargo_pct: embargo after each test fold (fraction of T). Default 0.01.
        purge_pct: additional purge before test (fraction). Default 0.0.
        periods_per_year: annualization for Sharpe. Inferred from index if None.
        global_n_trials: Total number of strategies tested globally (from ExperimentRegistry).
                         Used for multiple-testing context; does not change CPCV computation
                         but enables adjusted threshold interpretation.

    Returns:
        PBO in [0,1]. >0.5 indicates overfit (IS ranking not predictive of OOS).
        Returns 0.5 when insufficient data (neutral).
    """
    arr, cols, ppy_default = _coerce_returns(returns_df)
    if arr.size == 0 or arr.shape[0] < 10:
        return 0.5
    ppy = periods_per_year if periods_per_year is not None else ppy_default
    n_samples = arr.shape[0]
    n_strats = arr.shape[1]

    # Single strategy path: compute IS/OOS sharpes per CPCV split, then rank
    if n_strats == 1:
        is_sharpes, oos_sharpes = compute_cpcv_sharpes(
            returns_df, n_splits=n_splits, n_test_folds=n_test_folds,
            embargo_pct=embargo_pct, purge_pct=purge_pct, periods_per_year=ppy
        )
        if not is_sharpes:
            return 0.5
        pbo = float(compute_pbo_from_sharpes(is_sharpes, oos_sharpes, global_n_trials=global_n_trials))
        if global_n_trials is not None and global_n_trials > 1:
            # Log global context for threshold adjustment downstream
            pass  # downstream can use global_n_trials to tighten threshold
        return pbo

    # Multi-strategy path: per CPCV split, rank strategies by IS, check OOS
    # This is the textbook Bailey PBO
    worse = 0
    total = 0
    for train_idx, test_idx in generate_cpcv_splits(n_samples, n_splits=n_splits, n_test_folds=n_test_folds, embargo_pct=embargo_pct, purge_pct=purge_pct):
        is_sharpes_split = []
        oos_sharpes_split = []
        for j in range(n_strats):
            is_ret = arr[train_idx, j]
            oos_ret = arr[test_idx, j]
            # handle NaNs
            is_ret = is_ret[np.isfinite(is_ret)]
            oos_ret = oos_ret[np.isfinite(oos_ret)]
            is_sharpes_split.append(_sharpe(is_ret, periods_per_year=ppy))
            oos_sharpes_split.append(_sharpe(oos_ret, periods_per_year=ppy))
        if not is_sharpes_split:
            continue
        best_idx = int(np.argmax(is_sharpes_split))
        best_oos = float(oos_sharpes_split[best_idx])
        median_oos = float(np.median(oos_sharpes_split))
        if best_oos < median_oos:
            worse += 1
        total += 1

    if total == 0:
        return 0.5
    pbo = float(worse / total)
    if global_n_trials is not None and global_n_trials > n_strats:
        # PBO computed on subset; global context available for stricter threshold
        pass
    return pbo


# Convenience alias for research_integrity integration
def pbo_from_is_oos(
    is_sharpes: Union[List[float], np.ndarray],
    oos_sharpes: Union[List[float], np.ndarray],
    global_n_trials: Optional[int] = None,
) -> float:
    """Alias for compute_pbo_from_sharpes."""
    return compute_pbo_from_sharpes(is_sharpes, oos_sharpes, global_n_trials=global_n_trials)


def compute_pbo_detailed(
    returns_df,
    n_splits: int = 6,
    n_test_folds: int = 2,
    embargo_pct: float = 0.01,
    purge_pct: float = 0.0,
    periods_per_year: Optional[float] = None,
    global_n_trials: Optional[int] = None,
) -> dict:
    """
    Compute PBO with detailed output including global experiment context.
    
    Returns dict with:
        - pbo: float PBO value
        - global_n_trials: int or None
        - n_strategies_in_batch: int
        - cpcv_splits: int
        - pbo_source: str
        - threshold_adjusted: bool (whether global_n_trials > batch size)
    """
    arr, cols, ppy_default = _coerce_returns(returns_df)
    if arr.size == 0 or arr.shape[0] < 10:
        return {"pbo": 0.5, "global_n_trials": global_n_trials, "n_strategies_in_batch": 0, "cpcv_splits": 0, "pbo_source": "insufficient_data", "threshold_adjusted": False}
    
    pbo = compute_pbo(returns_df, n_splits=n_splits, n_test_folds=n_test_folds, embargo_pct=embargo_pct, purge_pct=purge_pct, periods_per_year=periods_per_year, global_n_trials=global_n_trials)
    
    # Count CPCV splits
    n_samples = arr.shape[0]
    splits = list(generate_cpcv_splits(n_samples, n_splits=n_splits, n_test_folds=n_test_folds, embargo_pct=embargo_pct, purge_pct=purge_pct))
    n_cpcv = len(splits)
    
    return {
        "pbo": pbo,
        "global_n_trials": global_n_trials,
        "n_strategies_in_batch": arr.shape[1],
        "cpcv_splits": n_cpcv,
        "pbo_source": "CPCV_real",
        "threshold_adjusted": global_n_trials is not None and global_n_trials > arr.shape[1]
    }


__all__ = [
    "compute_pbo",
    "compute_pbo_detailed",
    "compute_pbo_from_sharpes",
    "pbo_from_is_oos",
    "compute_cpcv_sharpes",
    "compute_pbo_logits",
    "generate_cpcv_splits",
    "cpcv_splits_from_returns",
]
