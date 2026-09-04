"""
Bootstrap validation — Audit §21: White's Reality Check / SPA / multiple-testing

Provides block bootstrap for PBO and Sharpe CI.
"""

from __future__ import annotations

import numpy as np
from typing import List


def block_bootstrap_sharpe(returns: List[float], block: int = 20, n_iter: int = 500) -> dict:
    """Block bootstrap to preserve autocorrelation (4h bars)."""
    if len(returns) < block:
        return {"sharpe": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "p_value": 1.0}
    arr = np.array(returns, dtype=float)
    n = len(arr)
    sharpes = []
    rng = np.random.default_rng(42)
    for _ in range(n_iter):
        # block resampling
        blocks = []
        while len(blocks) * block < n:
            start = rng.integers(0, n - block + 1)
            blocks.extend(arr[start : start + block].tolist())
        sample = np.array(blocks[:n])
        m = float(np.mean(sample))
        s = float(np.std(sample, ddof=1))
        sharpes.append(m / s * np.sqrt(365 * 6) if s > 0 else 0.0)  # 4h
    lower, upper = np.percentile(sharpes, [2.5, 97.5])
    p_val = float(np.mean(np.array(sharpes) <= 0))
    return {"sharpe": float(np.mean(sharpes)), "ci_lower": float(lower), "ci_upper": float(upper), "p_value": p_val, "n_iter": n_iter}


def pbo_combinatorial(is_sharpes: List[float], oos_sharpes: List[float]) -> float:
    """
    Simplified PBO: prob that best IS Sharpe underperforms median OOS.
    Real Bailey et al. needs combinatorial splits; this is proxy.
    """
    if not is_sharpes or not oos_sharpes:
        return 0.5
    best_is_idx = int(np.argmax(is_sharpes))
    best_oos = oos_sharpes[best_is_idx]
    median_oos = float(np.median(oos_sharpes))
    return float(best_oos < median_oos)
