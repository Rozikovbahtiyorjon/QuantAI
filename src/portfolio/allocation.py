"""
P2.19 Portfolio Allocation — Risk Parity, HRP, Correlation-aware sizing

Implements:
- Risk Parity (equal risk contribution)
- HRP (Hierarchical Risk Parity, Lopez de Prado)
- Correlation-aware sizing (inverse vol * correlation penalty)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List


def risk_parity_weights(cov: pd.DataFrame, max_iter: int = 1000, tol: float = 1e-8) -> pd.Series:
    """
    Risk Parity: w_i * (cov @ w)_i = constant for all i.
    Uses CCD (cyclical coordinate descent) — equal risk contribution.
    """
    n = len(cov)
    if n == 0:
        return pd.Series(dtype=float)
    if n == 1:
        return pd.Series([1.0], index=cov.index)
    # init equal weight
    w = np.ones(n) / n
    cov_m = cov.values
    for _ in range(max_iter):
        w_prev = w.copy()
        for i in range(n):
            # Risk contribution of i: w_i * (cov @ w)_i
            # Update w_i to equalize
            # Use simple iteration: w_i = sqrt( (target_risk) / cov_ii ) approximated
            # Target risk = total variance / n
            # More stable: use Spinu 2013 CCD formula
            a = np.sqrt(cov_m[i, i])
            # Compute marginal risk
            sigma = np.sqrt(w @ cov_m @ w)
            if sigma == 0:
                continue
            # Risk contribution
            rc = w[i] * (cov_m[i] @ w) / sigma
            # Target
            target = sigma / n
            # Update
            if rc > 0:
                w[i] *= np.sqrt(target / rc)
        # Normalize
        w = np.maximum(w, 1e-6)
        w /= w.sum()
        if np.max(np.abs(w - w_prev)) < tol:
            break
    return pd.Series(w, index=cov.index)


def hrp_weights(cov: pd.DataFrame) -> pd.Series:
    """
    Hierarchical Risk Parity (Lopez de Prado 2016).
    1. Cluster via single-linkage on correlation distance
    2. Quasi-diagonalize
    3. Recursive bisection with inverse variance
    """
    if len(cov) <= 1:
        return pd.Series([1.0], index=cov.index) if len(cov)==1 else pd.Series(dtype=float)
    # Correlation from cov
    std = np.sqrt(np.diag(cov.values))
    std = np.maximum(std, 1e-6)
    corr = cov.values / np.outer(std, std)
    corr = np.clip(corr, -1, 1)
    # Ensure diagonal exactly 1
    np.fill_diagonal(corr, 1.0)
    dist = np.sqrt(0.5 * (1 - corr))
    np.fill_diagonal(dist, 0.0)
    # Hierarchical clustering (scipy if available, else simple)
    try:
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import squareform
        # squareform needs condensed, but dist is square; use squareform to convert
        condensed = squareform(dist, checks=False)
        link = linkage(condensed, method='single')
        order = leaves_list(link)
        ordered_idx = cov.index[order].tolist()
        cov_ordered = cov.loc[ordered_idx, ordered_idx]
    except Exception:
        # Fallback: sort by volatility
        order = np.argsort(std)
        ordered_idx = cov.index[order].tolist()
        cov_ordered = cov.loc[ordered_idx, ordered_idx]
    # Recursive bisection
    def _get_ivp(cov_sub: pd.DataFrame) -> float:
        iv = 1.0 / np.diag(cov_sub.values)
        iv /= iv.sum()
        return iv
    def _cluster_var(cov_sub: pd.DataFrame) -> float:
        iv = _get_ivp(cov_sub)
        return float(iv @ cov_sub.values @ iv)
    def _recurse(cov_sub: pd.DataFrame) -> pd.Series:
        if len(cov_sub) == 1:
            return pd.Series([1.0], index=cov_sub.index)
        if len(cov_sub) == 2:
            v0 = _cluster_var(cov_sub.iloc[:1, :1])
            v1 = _cluster_var(cov_sub.iloc[1:, 1:])
            w0 = 1 - v0 / (v0 + v1) if (v0+v1)>0 else 0.5
            return pd.Series([w0, 1-w0], index=cov_sub.index)
        mid = len(cov_sub)//2
        left = cov_sub.iloc[:mid, :mid]
        right = cov_sub.iloc[mid:, mid:]
        var_left = _cluster_var(left)
        var_right = _cluster_var(right)
        alloc_left = 1 - var_left / (var_left + var_right) if (var_left+var_right)>0 else 0.5
        # Recurse
        w_left = _recurse(left) * alloc_left
        w_right = _recurse(right) * (1-alloc_left)
        return pd.concat([w_left, w_right])
    w_ordered = _recurse(cov_ordered)
    # Restore original order
    w = w_ordered.reindex(cov.index).fillna(0)
    w /= w.sum()
    return w


def correlation_aware_weights(cov: pd.DataFrame, signal_strength: Dict[str, float] | None = None) -> pd.Series:
    """
    Correlation-aware sizing: inverse vol * (1 - avg_corr) * signal_strength
    """
    if len(cov)==0:
        return pd.Series(dtype=float)
    vols = np.sqrt(np.diag(cov.values))
    # Avg correlation per asset
    corr = cov.values / np.outer(vols, vols)
    avg_corr = np.nanmean(np.abs(corr - np.eye(len(corr))), axis=1)
    # Weight = (1/vol) * (1 - avg_corr) * signal
    inv_vol = 1.0 / np.maximum(vols, 1e-6)
    penalty = 1.0 - avg_corr
    w = inv_vol * penalty
    if signal_strength:
        for i, sym in enumerate(cov.index):
            w[i] *= abs(signal_strength.get(sym, 1.0))
    w = np.maximum(w, 0)
    if w.sum() > 0:
        w /= w.sum()
    return pd.Series(w, index=cov.index)


def get_allocation(
    returns: pd.DataFrame,
    method: str = "hrp",
    signal_strength: Dict[str, float] | None = None,
    max_position_pct: float = 0.4,
) -> Dict[str, float]:
    """
    Unified entry: returns DataFrame (columns symbols, rows time), method in {risk_parity, hrp, correlation_aware}
    Returns dict symbol -> weight (sum 1.0, capped by max_position_pct)
    """
    if returns.empty or len(returns.columns) < 1:
        return {}
    cov = returns.cov()
    # Annualization not needed for weights (scale invariant)
    if method == "risk_parity":
        w = risk_parity_weights(cov)
    elif method == "hrp":
        w = hrp_weights(cov)
    elif method == "correlation_aware":
        w = correlation_aware_weights(cov, signal_strength)
    else:
        raise ValueError(f"unknown allocation method {method}")
    # Cap max position
    w = w.clip(upper=max_position_pct)
    w /= w.sum()  # renormalize after cap
    return w.to_dict()
