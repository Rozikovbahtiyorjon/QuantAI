"""
Drift detection: PSI + KS-test
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index.
    Bins by quantile of expected (deciles by default).
    Smoothing epsilon to avoid log(0).
    Thresholds: <0.1 no shift, 0.1-0.25 small, >0.25 large
    """
    # Quantile breakpoints from expected
    quantiles = np.linspace(0, 100, bins + 1)
    breakpoints = np.percentile(expected, quantiles)
    # Ensure strictly increasing (deduplicate via epsilon)
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) <= 2:
        return 0.0

    # Bin counts
    expected_percents = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    # Smoothing to avoid division by zero
    epsilon = 1e-4
    expected_percents = np.where(expected_percents == 0, epsilon, expected_percents)
    actual_percents = np.where(actual_percents == 0, epsilon, actual_percents)

    psi_values = (actual_percents - expected_percents) * np.log(
        actual_percents / expected_percents
    )
    return float(np.sum(psi_values))


def ks_test(expected: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    """
    Kolmogorov-Smirnov test.
    Returns (statistic, pvalue). p < 0.05 => significant drift.
    """
    result = ks_2samp(expected, actual)
    return float(result.statistic), float(result.pvalue)


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str] | None = None,
    psi_threshold: float = 0.2,
    ks_p_threshold: float = 0.05,
) -> pd.DataFrame:
    """
    Per-feature drift report.
    """
    if features is None:
        features = list(set(reference.columns) & set(current.columns))

    rows = []
    for feat in features:
        exp = reference[feat].dropna().values
        act = current[feat].dropna().values
        if len(exp) == 0 or len(act) == 0:
            continue
        psi_val = psi(exp, act)
        ks_stat, ks_p = ks_test(exp, act)
        drifted = (psi_val > psi_threshold) or (ks_p < ks_p_threshold)
        rows.append(
            {
                "feature": feat,
                "psi": round(psi_val, 4),
                "ks_stat": round(ks_stat, 4),
                "ks_pvalue": round(ks_p, 4),
                "drifted": drifted,
                "psi_threshold": psi_threshold,
                "ks_p_threshold": ks_p_threshold,
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("psi", ascending=False).reset_index(drop=True)
    return df
