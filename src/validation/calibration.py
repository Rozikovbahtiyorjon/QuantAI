"""
Calibration validation — P(confidence bucket) -> actual net return

Balanced Accuracy 0.39 with 3 classes > random (0.33), but trading needs:
  confidence 0.35 -> -0.02% net
  confidence 0.50 -> +0.01%
  confidence 0.65 -> +0.07%
  confidence 0.80 -> +0.19%
Monotonic: higher confidence => higher actual net return.

This module validates that ML confidence is calibrated to future net return,
not just direction accuracy. Useful for both classifier (P(win)) and
regressor (E[net]).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np


@dataclass
class BucketCalibration:
    bucket: str  # e.g. "0.30-0.40"
    n: int
    mean_pred: float  # mean confidence / predicted E[net] in bucket
    mean_actual: float  # mean actual net return in bucket
    win_rate: float  # for classifier
    calibration_error: float  # |mean_pred - win_rate| or |pred - actual|


@dataclass
class CalibrationReport:
    buckets: List[BucketCalibration]
    spearman_corr: float  # correlation between mean_pred and mean_actual across buckets
    pearson_corr: float
    monotonic: bool  # mean_actual strictly increasing with bucket
    calibration_error: float  # mean absolute error across buckets
    n_buckets: int
    n_samples: int
    passed: bool
    reason: str


def _spearman(x: List[float], y: List[float]) -> float:
    if len(x) < 3:
        return 0.0
    try:
        from scipy.stats import spearmanr
        corr, _ = spearmanr(x, y)
        return float(corr) if not np.isnan(corr) else 0.0
    except ImportError:
        # Fallback rank correlation
        rx = np.argsort(np.argsort(x))
        ry = np.argsort(np.argsort(y))
        if np.std(rx) == 0 or np.std(ry) == 0:
            return 0.0
        return float(np.corrcoef(rx, ry)[0, 1])


def _pearson(x: List[float], y: List[float]) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def evaluate_calibration(
    y_true_net: List[float],  # actual net return per trade/sample
    y_pred_conf: List[float],  # predicted confidence (P(win) or E[net])
    n_buckets: int = 5,
    bucket_edges: List[float] | None = None,
) -> CalibrationReport:
    """
    Bin by predicted confidence, compute actual net return per bucket.

    Args:
        y_true_net: actual net return (e.g. -0.02% .. +0.19%)
        y_pred_conf: predicted confidence (0.35 .. 0.80) or predicted E[net]
        n_buckets: number of quantile buckets (if bucket_edges None)
        bucket_edges: explicit edges, e.g. [0.3,0.4,0.5,0.6,0.7,0.8,0.9]

    Returns:
        CalibrationReport with monotonic and correlation checks.
    """
    y_true = np.array(y_true_net, dtype=float)
    y_pred = np.array(y_pred_conf, dtype=float)
    n = len(y_true)
    if n == 0 or len(y_pred) != n:
        return CalibrationReport([], 0.0, 0.0, False, 1.0, 0, n, False, "no samples")

    # Bucket by predicted confidence
    if bucket_edges is not None:
        edges = np.array(bucket_edges, dtype=float)
    else:
        # Quantile buckets to ensure balanced n per bucket
        qs = np.linspace(0, 1, n_buckets + 1)
        edges = np.quantile(y_pred, qs)
        # Deduplicate edges
        edges = np.unique(edges)
        if len(edges) < 3:
            # Fallback uniform
            edges = np.linspace(float(np.min(y_pred)), float(np.max(y_pred)), n_buckets + 1)

    buckets: List[BucketCalibration] = []
    for i in range(len(edges) - 1):
        lo, hi = float(edges[i]), float(edges[i + 1])
        # Last bucket inclusive
        if i == len(edges) - 2:
            mask = (y_pred >= lo) & (y_pred <= hi)
        else:
            mask = (y_pred >= lo) & (y_pred < hi)
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        m_pred = float(np.mean(y_pred[idx]))
        m_actual = float(np.mean(y_true[idx]))
        # For win_rate, count actual >0
        win = float(np.mean(y_true[idx] > 0)) if len(idx) else 0.0
        # Calibration error: for regressor, |pred - actual|; for classifier, |prob - win_rate|
        # Heuristic: if pred in [0,1] (prob), compare to win_rate, else compare to mean_actual
        if 0 <= m_pred <= 1 and np.all((y_pred >= 0) & (y_pred <= 1)):
            err = abs(m_pred - win)
        else:
            err = abs(m_pred - m_actual)
        buckets.append(BucketCalibration(
            bucket=f"{lo:.2f}-{hi:.2f}",
            n=int(len(idx)),
            mean_pred=m_pred,
            mean_actual=m_actual,
            win_rate=win,
            calibration_error=err,
        ))

    if len(buckets) < 3:
        return CalibrationReport(buckets, 0.0, 0.0, False, 1.0, len(buckets), n, False, "too few buckets")

    preds = [b.mean_pred for b in buckets]
    actuals = [b.mean_actual for b in buckets]

    spear = _spearman(preds, actuals)
    pear = _pearson(preds, actuals)
    # Monotonic: actual strictly increasing with pred (allow small noise)
    monotonic = all(actuals[i] <= actuals[i + 1] + 1e-9 for i in range(len(actuals) - 1))
    # For calibration, also check monotonic with tolerance
    # More robust: spearman >0.5 indicates monotonic
    monotonic_strict = spear > 0.5

    cal_err = float(np.mean([b.calibration_error for b in buckets]))

    # Pass criteria: monotonic and positive correlation and low calibration error
    # Trading usefulness: higher confidence must predict higher actual net return
    passed = monotonic_strict and pear > 0.3 and cal_err < 0.5
    # For regression where pred is E[net] in same units as actual, also check pear >0.3
    # For classifier where pred is P(win), check win_rate monotonic

    reason = ""
    if not monotonic_strict:
        reason = f"non-monotonic spearman {spear:.2f} (need >0.5)"
    elif pear <= 0.3:
        reason = f"weak pearson {pear:.2f} (need >0.3)"
    elif cal_err >= 0.5:
        reason = f"high calibration error {cal_err:.3f}"

    return CalibrationReport(
        buckets=buckets,
        spearman_corr=float(spear),
        pearson_corr=float(pear),
        monotonic=bool(monotonic_strict),
        calibration_error=float(cal_err),
        n_buckets=len(buckets),
        n_samples=n,
        passed=bool(passed),
        reason=reason,
    )


def evaluate_expected_return_calibration(
    y_true_net: List[float],
    y_pred_expected: List[float],
    hurdle: float = 0.0,
) -> CalibrationReport:
    """
    Specialized for ExpectedReturnModel: y_pred is E[net], hurdle is TAKE threshold.
    Checks that predicted E[net] correlates with actual net.
    """
    # Use quantile buckets on predicted expected return
    return evaluate_calibration(y_true_net, y_pred_expected, n_buckets=5)


def is_trading_useful(report: CalibrationReport) -> bool:
    """
    Trading usefulness: confidence bucket monotonic and actual net return
    increases with confidence. Example:

      0.35 -> -0.02%
      0.50 -> +0.01%
      0.65 -> +0.07%
      0.80 -> +0.19%  monotonic => useful

    vs random:

      0.35 -> +0.02%
      0.50 -> -0.01%  non-monotonic => not useful

    Balanced Accuracy 0.39 > random 0.33 but if calibration is flat,
    model is not trading-useful.
    """
    return report.passed and report.monotonic and report.pearson_corr > 0.3


# =========================================================
# Brier Score, ECE, Reliability / Calibration Curve
# For probabilistic classifier: 0.80 predicted prob must mean ~80% empirical
# Not just high XGBoost score.
# =========================================================

@dataclass
class BrierECEReport:
    brier_score: float  # mean((p - y)^2), 0=perfect, 0.25=random
    brier_skill: float  # 1 - brier / brier_baseline (baseline = always predict base rate)
    ece: float  # Expected Calibration Error, weighted |acc - conf|
    mce: float  # Maximum Calibration Error
    reliability_curve: List[Dict[str, float]]  # per bucket: conf, acc, n, gap
    passed_brier: bool
    passed_ece: bool
    passed: bool
    reason: str


def compute_brier_score(y_true_binary: List[int], y_pred_prob: List[float]) -> float:
    """Brier Score for binary classifier: mean((p - y)^2)."""
    y_true = np.array(y_true_binary, dtype=float)
    y_pred = np.array(y_pred_prob, dtype=float)
    if len(y_true) == 0:
        return 1.0
    return float(np.mean((y_pred - y_true) ** 2))


def compute_ece(
    y_true_binary: List[int],
    y_pred_prob: List[float],
    n_buckets: int = 10,
) -> Tuple[float, float, List[Dict[str, float]]]:
    """
    Expected Calibration Error.
    Bins by predicted prob, computes |acc - conf| weighted.
    Returns (ece, mce, reliability_curve).
    """
    y_true = np.array(y_true_binary, dtype=int)
    y_pred = np.array(y_pred_prob, dtype=float)
    n = len(y_true)
    if n == 0:
        return 1.0, 1.0, []

    # Use uniform buckets [0,0.1)...[0.9,1.0]
    edges = np.linspace(0.0, 1.0, n_buckets + 1)
    ece = 0.0
    mce = 0.0
    curve: List[Dict[str, float]] = []
    for i in range(n_buckets):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if i == n_buckets - 1:
            mask = (y_pred >= lo) & (y_pred <= hi)
        else:
            mask = (y_pred >= lo) & (y_pred < hi)
        idx = np.where(mask)[0]
        n_b = len(idx)
        if n_b == 0:
            curve.append({"bucket": f"{lo:.1f}-{hi:.1f}", "conf": (lo + hi) / 2, "acc": 0.0, "n": 0, "gap": 0.0})
            continue
        acc = float(np.mean(y_true[idx]))  # empirical frequency
        conf = float(np.mean(y_pred[idx]))  # mean predicted prob
        gap = abs(acc - conf)
        ece += (n_b / n) * gap
        mce = max(mce, gap)
        curve.append({"bucket": f"{lo:.1f}-{hi:.1f}", "conf": conf, "acc": acc, "n": int(n_b), "gap": float(gap)})
    return float(ece), float(mce), curve


def calibration_curve_data(
    y_true_binary: List[int],
    y_pred_prob: List[float],
    n_buckets: int = 10,
) -> List[Dict[str, float]]:
    """Alias for reliability curve data (sklearn-compatible)."""
    _, _, curve = compute_ece(y_true_binary, y_pred_prob, n_buckets=n_buckets)
    return curve


def evaluate_brier_ece(
    y_true_binary: List[int],
    y_pred_prob: List[float],
    max_brier: float = 0.25,
    max_ece: float = 0.10,
    n_buckets: int = 10,
) -> BrierECEReport:
    """
    Evaluate Brier Score and ECE for probabilistic predictions.

    0.80 predicted must mean ~80% empirical frequency, not just high score.

    Thresholds:
      Brier <0.25 (random 0.25 for balanced binary, <0.20 good)
      ECE <0.10 (10% avg miscalibration, <0.05 well-calibrated)

    Returns BrierECEReport with reliability_curve.
    """
    brier = compute_brier_score(y_true_binary, y_pred_prob)
    # Baseline Brier: always predict base rate
    base_rate = float(np.mean(y_true_binary)) if len(y_true_binary) else 0.5
    brier_baseline = float(np.mean((np.full(len(y_true_binary), base_rate) - np.array(y_true_binary)) ** 2)) if len(y_true_binary) else 0.25
    brier_skill = 1.0 - (brier / brier_baseline) if brier_baseline > 0 else 0.0

    ece, mce, curve = compute_ece(y_true_binary, y_pred_prob, n_buckets=n_buckets)

    passed_brier = brier < max_brier
    passed_ece = ece < max_ece
    passed = passed_brier and passed_ece

    reason = ""
    if not passed_brier:
        reason = f"Brier {brier:.3f} >= {max_brier} (need <{max_brier})"
    elif not passed_ece:
        reason = f"ECE {ece:.3f} >= {max_ece} (need <{max_ece}, MCE {mce:.3f})"

    return BrierECEReport(
        brier_score=float(brier),
        brier_skill=float(brier_skill),
        ece=float(ece),
        mce=float(mce),
        reliability_curve=curve,
        passed_brier=bool(passed_brier),
        passed_ece=bool(passed_ece),
        passed=bool(passed),
        reason=reason,
    )


def evaluate_reliability_curve(
    y_true_binary: List[int],
    y_pred_prob: List[float],
    n_buckets: int = 10,
) -> Dict[str, Any]:
    """
    Returns data for plotting Reliability Diagram / Calibration Curve.

    For well-calibrated model, curve should be close to diagonal (conf == acc).
    Use with matplotlib: x=conf, y=acc, size=n, diagonal reference.
    """
    ece, mce, curve = compute_ece(y_true_binary, y_pred_prob, n_buckets=n_buckets)
    brier = compute_brier_score(y_true_binary, y_pred_prob)
    return {
        "curve": curve,
        "ece": ece,
        "mce": mce,
        "brier": brier,
        "diagonal": [{"conf": float(i / 10), "acc": float(i / 10)} for i in range(11)],
    }
