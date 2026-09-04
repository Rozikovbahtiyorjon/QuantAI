"""
QuantAI Deflated Sharpe Ratio — Bailey & Prado 2014

Real DSR per:
  Bailey, D.H. & López de Prado, M. (2014) "The Deflated Sharpe Ratio:
  Correcting for Selection Bias, Backtest Overfitting and Non-Normality"
  Journal of Portfolio Management 40(5) + Advances in Financial ML Ch14.

Formula (Keel/InteractiveML summary):
  DSR = Φ( (SR_obs - SR_0) / sqrt(Var[SR]) )
  Var[SR] = (1 - γ3·SR + (γ4 -1)/4 · SR²) / (T-1)   # Mertens/Lo with skew/kurt
  SR_0 = sqrt(V) * ((1-γ)·Φ⁻¹(1-1/N) + γ·Φ⁻¹(1-1/(N·e)))  # Expected max
  Φ, Φ⁻¹ = Normal CDF / quantile, γ≈0.57721566 Euler-Mascheroni

Inputs:
  observed_sharpe : estimated Sharpe (annualized or same units as V)
  n_trials        : number of independent trials/backtests (N)
  sample_len      : number of return observations (T)
  returns_skew    : skewness γ3 of returns
  returns_kurtosis: kurtosis γ4 — Pearson (3=Normal) is expected; Fisher excess (0=Normal)
                    is auto-detected if value in [-2.5, 2.5) and converted (+3).

This module provides:
  - deflated_sharpe_ratio(...)
  - expected_max_sharpe(...)
  - is_dsr_significant(...)
  - probabilistic_sharpe_ratio(...) helper
"""

from __future__ import annotations

import math
from typing import Optional

# Euler-Mascheroni constant
EMC = 0.57721566490153286060651209
EULER_E = math.e

# Try scipy, fallback to pure-python approximations
try:
    from scipy.stats import norm as _scipy_norm  # type: ignore

    def _norm_cdf(x: float) -> float:
        return float(_scipy_norm.cdf(x))

    def _norm_ppf(p: float) -> float:
        # clip to avoid inf
        p = min(max(float(p), 1e-12), 1 - 1e-12)
        return float(_scipy_norm.ppf(p))

    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False

    def _norm_cdf(x: float) -> float:
        # Φ(x) = 0.5 * [1+ erf(x/sqrt2)]
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    # Acklam approximation for Φ⁻¹
    def _norm_ppf(p: float) -> float:
        p = min(max(float(p), 1e-12), 1 - 1e-12)
        # Coefficients
        a1 = -3.969683028665376e01
        a2 = 2.209460984245205e02
        a3 = -2.759285104469687e02
        a4 = 1.383577518672690e02
        a5 = -3.066479806614716e01
        a6 = 2.506628277459239e00
        b1 = -5.447609879822406e01
        b2 = 1.615858368580409e02
        b3 = -1.556989798598866e02
        b4 = 6.680131188771972e01
        b5 = -1.328068155288572e01
        c1 = -7.784894002430293e-03
        c2 = -3.223964580411365e-01
        c3 = -2.400758277161838e00
        c4 = -2.549732539343734e00
        c5 = 4.374664141464968e00
        c6 = 2.938163982698783e00
        d1 = 7.784695709041462e-03
        d2 = 3.224671290700398e-01
        d3 = 2.445134137142996e00
        d4 = 3.754408661907416e00
        lo = 0.02425
        hi = 1 - lo
        if p < lo:
            q = math.sqrt(-2 * math.log(p))
            return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / (
                (((d1 * q + d2) * q + d3) * q + d4) * q + 1
            )
        if p <= hi:
            q = p - 0.5
            r = q * q
            return (
                (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q
            ) / (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1)
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / (
            (((d1 * q + d2) * q + d3) * q + d4) * q + 1
        )


def _to_pearson_kurtosis(kurt: float) -> float:
    """
    Normalize kurtosis to Pearson (Fisher excess 0 → Pearson 3).
    Heuristic: if kurt in [-2.5, 2.5) treat as Fisher excess and add 3.
    Pearson kurtosis is always >=1 (excess >= -2). Values below 1 are impossible
    as Pearson, so they must be Fisher.
    """
    k = float(kurt)
    # Fisher excess range for normal 0, heavy tails maybe 5-20 excess
    # Pearson range for normal 3, heavy tails 8-23
    # Overlap in 3-5 ambiguous; we treat <2.5 as Fisher
    if -3.0 <= k < 2.5:
        # Likely Fisher excess — convert to Pearson
        return k + 3.0
    return k


def _sharpe_variance(
    observed_sharpe: float, sample_len: int, skew: float, kurtosis: float
) -> float:
    """
    Variance of Sharpe estimator under non-normality (Mertens/Lo).
      Var = (1 - γ3·SR + (γ4-1)/4·SR²) / (T-1)
    γ4 is Pearson kurtosis (3=Normal).
    """
    T = int(sample_len)
    if T <= 1:
        return 1e-12
    sr = float(observed_sharpe)
    g3 = float(skew)
    g4 = _to_pearson_kurtosis(float(kurtosis))
    # Clamp kurt to reasonable range to avoid negative var from extreme skew/kurt*SR²
    # g4 >=1 by definition; enforce
    if g4 < 1.0:
        g4 = 1.0
    var = (1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr * sr) / (T - 1)
    # Numerical guard: variance must be positive
    if var <= 1e-12 or not math.isfinite(var):
        # fallback to variance under null (SR=0) or tiny positive
        var = max(1e-12, 1.0 / max(1, T - 1))
    return float(var)


def probabilistic_sharpe_ratio(
    observed_sharpe: float,
    benchmark_sharpe: float,
    sample_len: int,
    returns_skew: float = 0.0,
    returns_kurtosis: float = 3.0,
) -> float:
    """
    Probabilistic Sharpe Ratio (Bailey & López de Prado 2012):
      PSR(SR*) = Φ( (SR - SR*) / sqrt(Var) )
    where Var is _sharpe_variance with non-normality correction.
    Returns probability that true Sharpe > benchmark.
    """
    var = _sharpe_variance(observed_sharpe, sample_len, returns_skew, returns_kurtosis)
    se = math.sqrt(var)
    if se <= 1e-12:
        return 0.5 if observed_sharpe <= benchmark_sharpe else 1.0
    z = (float(observed_sharpe) - float(benchmark_sharpe)) / se
    # Clip z to avoid overflow in cdf
    # norm cdf approaches 0/1 quickly beyond +-8
    if z > 8:
        return 1.0
    if z < -8:
        return 0.0
    return float(_norm_cdf(z))


def expected_max_sharpe(
    n_trials: int,
    sample_len: Optional[int] = None,
    variance: Optional[float] = None,
    returns_skew: float = 0.0,
    returns_kurtosis: float = 3.0,
    var_sharpe: Optional[float] = None,
    **kwargs,
) -> float:
    """
    Expected maximum Sharpe under null of N independent zero-skill trials.

    Bailey & Prado 2014 Eq.6:
      E[max SR] = sqrt(V) * ((1-γ)·Φ⁻¹(1-1/N) + γ·Φ⁻¹(1-1/(N·e)))

    Parameters
    ----------
    n_trials : int
        Number of independent trials (N). N<=1 → 0.
    sample_len : int | None
        Number of observations T. If variance not given, V = 1/(T-1) (null).
        Can also be passed positionally as second arg.
    variance : float | None
        Variance of Sharpe across trials V[{SR}]. If given, overrides sample_len.
        Alias var_sharpe.
    returns_skew, returns_kurtosis : kept for API compatibility; not used for
        expected max under null (Var at SR=0 is 1/(T-1) independent of skew/kurt).
        They are accepted so calls like expected_max_sharpe(N, T, skew, kurt) work.

    Returns
    -------
    float
        Expected maximum Sharpe (SR_0) threshold to beat.
    """
    # Handle aliases and flexible calling
    # Allow sample_len to be passed as variance if it looks like variance (small float)
    # and vice versa. Detect by kwarg names.
    if variance is None and var_sharpe is not None:
        variance = float(var_sharpe)
    # Check kwargs for alternative names
    if variance is None:
        for k in ("var", "var_trials", "variance_trials", "v"):
            if k in kwargs and kwargs[k] is not None:
                variance = float(kwargs[k])
                break
    if sample_len is None:
        for k in ("n_obs", "T", "sample_size", "len", "n"):
            if k in kwargs and kwargs[k] is not None:
                sample_len = int(kwargs[k])
                break
    # Support second positional being variance when sample_len is small float
    # This is handled by caller explicitly passing keyword; otherwise we interpret
    # sample_len as T if integer-like >=2
    N = int(n_trials)
    if N <= 1:
        return 0.0

    # Resolve variance V
    V: float
    if variance is not None:
        V = float(variance)
        if V <= 0 or not math.isfinite(V):
            V = 1e-12
    elif sample_len is not None:
        T = int(sample_len)
        if T <= 1:
            V = 1.0
        else:
            # Under null SR=0, Var = 1/(T-1) irrespective of skew/kurt
            # (skew/kurt terms vanish at SR=0)
            V = 1.0 / (T - 1)
    else:
        # No information — assume standardized variance 1.0 (conservative upper bound)
        # This makes expected max larger (harder to pass), which is safe.
        V = 1.0

    # Compute expected maximum quantile
    # Clip probabilities to avoid ppf inf
    try:
        p1 = 1.0 - 1.0 / N
        p2 = 1.0 - 1.0 / (N * EULER_E)
        z1 = _norm_ppf(p1)
        z2 = _norm_ppf(p2)
    except Exception:
        return 0.0

    expected_z = (1.0 - EMC) * z1 + EMC * z2
    return float(math.sqrt(max(V, 1e-12)) * expected_z)


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    sample_len: int,
    returns_skew: float,
    returns_kurtosis: float,
    variance_trials: Optional[float] = None,
) -> float:
    """
    Deflated Sharpe Ratio (Bailey & Prado 2014).

    DSR = PSR( SR_0 ) = Φ( (SR_obs - SR_0) / sqrt(Var) )
    where
      Var = (1 - skew·SR + (kurt-1)/4·SR²)/(T-1)
      SR_0 = expected_max_sharpe(N, T)  # selection bias threshold

    Parameters
    ----------
    observed_sharpe : float
        Estimated Sharpe ratio of best strategy.
    n_trials : int
        Number of independent trials (N).
    sample_len : int
        Number of return observations (T). Must be >1.
    returns_skew : float
        Skewness of returns (γ3).
    returns_kurtosis : float
        Kurtosis γ4 — Pearson (3=Normal). Fisher excess (0=Normal) auto-converted
        if value in [-3, 2.5).

    Returns
    -------
    float
        DSR in [0,1] — probability skill is real after correcting for
        multiple testing and non-normality. >0.95 is conventional significance.

    Notes
    -----
    - N=1 → DSR reduces to PSR with benchmark 0 (i.e., prob SR>0).
    - High N, negative skew, or fat tails (high kurt) deflate DSR.
    - See src/research/experiment_registry.deflated_sharpe_proxy for heuristic
      baseline that this replaces.
    """
    sr = float(observed_sharpe)
    N = int(n_trials)
    T = int(sample_len)
    skew = float(returns_skew)
    kurt = float(returns_kurtosis)

    if T <= 1:
        # Not enough data — return 0.5 (uninformative)
        return 0.5
    if not math.isfinite(sr) or not math.isfinite(skew) or not math.isfinite(kurt):
        return 0.5
    if N <= 1:
        # No selection bias — DSR = PSR vs 0
        return probabilistic_sharpe_ratio(sr, 0.0, T, skew, _to_pearson_kurtosis(kurt))

    # Variance with non-normality correction
    var = _sharpe_variance(sr, T, skew, kurt)

    # Expected max under null; if variance_trials given use it, else use Var under null (1/(T-1))
    # For consistency with DSR definition, SR_0 uses variance across trials V.
    # If caller provides variance_trials, use it; else derive from sample_len.
    # Note: Bailey defines V as variance across trials’ SRs. If not measured, V ≈ var under null.
    # We use expected_max_sharpe which will compute V=1/(T-1) if no variance supplied.
    if variance_trials is not None:
        sr0 = expected_max_sharpe(N, variance=float(variance_trials))
    else:
        # Use sample_len to infer V; skew/kurt not needed for SR0 (null) but pass for API
        sr0 = expected_max_sharpe(N, sample_len=T)

    # DSR as PSR against SR_0
    return probabilistic_sharpe_ratio(sr, sr0, T, skew, kurt)


def is_dsr_significant(dsr: float, threshold: float = 0.95) -> bool:
    """
    Check DSR significance.

    Parameters
    ----------
    dsr : float
        Deflated Sharpe Ratio probability in [0,1].
    threshold : float, default 0.95
        Significance level (1-α). 0.95 corresponds to p<0.05.

    Returns
    -------
    bool
        True if dsr >= threshold.
    """
    try:
        return float(dsr) >= float(threshold)
    except Exception:
        return False


def deflated_sharpe_ratio_detailed(
    observed_sharpe: float,
    n_trials: int,
    sample_len: int,
    returns_skew: float,
    returns_kurtosis: float,
    variance_trials: Optional[float] = None,
    global_n_trials: Optional[int] = None,
) -> dict:
    """
    Compute DSR with detailed output including global experiment context.
    
    Returns dict with:
        - dsr: float Deflated Sharpe Ratio
        - expected_max_sharpe: float
        - probabilistic_sharpe: float
        - sharpe_variance: float
        - n_trials: int (local)
        - global_n_trials: int or None
        - sample_len: int
        - skew: float
        - kurtosis: float
        - threshold_adjusted: bool
        - significant_095: bool
        - significant_099: bool
    """
    sr = float(observed_sharpe)
    N = int(n_trials)
    T = int(sample_len)
    skew = float(returns_skew)
    kurt = float(returns_kurtosis)
    
    dsr = deflated_sharpe_ratio(sr, N, T, skew, kurt, variance_trials)
    exp_max = expected_max_sharpe(N, sample_len=T, returns_skew=skew, returns_kurtosis=kurt, variance=variance_trials)
    psr = probabilistic_sharpe_ratio(sr, 0.0, T, skew, kurt)
    var = _sharpe_variance(sr, T, skew, kurt)
    
    return {
        "dsr": float(dsr),
        "expected_max_sharpe": float(exp_max),
        "probabilistic_sharpe": float(psr),
        "sharpe_variance": float(var),
        "n_trials": N,
        "global_n_trials": global_n_trials,
        "sample_len": T,
        "skew": skew,
        "kurtosis": kurt,
        "threshold_adjusted": global_n_trials is not None and global_n_trials > N,
        "significant_095": float(dsr) >= 0.95,
        "significant_099": float(dsr) >= 0.99,
    }


# Convenience alias for research_integrity fallback detection
__all__ = [
    "deflated_sharpe_ratio",
    "deflated_sharpe_ratio_detailed",
    "expected_max_sharpe",
    "is_dsr_significant",
    "probabilistic_sharpe_ratio",
    "EMC",
]
