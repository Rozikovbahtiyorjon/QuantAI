"""
QuantAI Research Integrity Engine

Hierarchy (per user requirement):

    Research
      ↓
    Integrity Checks
      ↓
    Statistical Validation
      ↓
    Robustness
      ↓
    Selection Adjustment
      ↓
    Tournament
      ↓
    Champion

Tournament MUST NOT be able to ignore statistical integrity.
All gates before Tournament are HARD — failing candidates are
removed before ranking. Tournament only sees integrity-passed pool.

This module implements the four gates and their composition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

try:
    import pandas as pd  # noqa: F401  for regime gate isinstance check
except Exception:  # pragma: no cover
    pd = None  # type: ignore
from typing import Any, Optional


# =====================================================
# CONFIG
# =====================================================

@dataclass
class IntegrityConfig:
    """Thresholds for the four gates. Conservative defaults."""

    # ---- Integrity Checks (PromotionRules) ----
    min_pf_median: float = 1.05
    min_profitable_window_share: float = 0.45
    max_drawdown_median_pct: float = -15.0
    min_net_median_pct: float = 0.0
    min_trades_total: int = 30
    max_net_std_pct: float = 10.0

    # ---- Statistical Validation ----
    min_sharpe_median: float = 0.0
    require_sharpe_significance: bool = True
    max_sharpe_p_value: float = 0.05
    max_pbo: float = 0.6
    min_deflated_sharpe: float = 0.0

    # ---- White Reality Check / SPA (data-snooping) ----
    # White (2000), Hansen (2005), Politis & Romano stationary bootstrap.
    # When n_trials large (e.g. >100), best-of-many luck inflates.
    # WRC/SPA bootstrap p must be < alpha to claim edge.
    wrc_enabled: bool = True
    wrc_min_trials: int = 100
    wrc_max_p_value: float = 0.05
    wrc_n_bootstrap: int = 1000
    wrc_q: float = 0.1  # stationary bootstrap restart prob, mean block 1/q
    wrc_use_spa: bool = True  # True => Hansen SPA (more powerful), False => White RC
    wrc_seed: int = 42

    # ---- IS-OOS Consistency (alpha reality) ----
    # Breakout-type IS good PF -> OOS deterioration must block real alpha.
    # IS != OOS => research hypothesis only.
    require_is_oos_consistency: bool = True
    max_pf_deterioration: float = 0.50  # (IS-OOS)/IS <=50%  e.g. IS 1.6 -> OOS >=0.8
    min_pf_ratio: float = 0.60  # OOS/IS >=0.60
    max_sharpe_deterioration: float = 0.70  # Sharpe drop <=70%
    max_is_oos_pbo: float = 0.60  # PBO <0.6

    # ---- Robustness ----
    require_cost_robust: bool = True
    min_monte_carlo_score: float = 0.3
    min_stress_score: float = 0.3

    # ---- Selection Adjustment ----
    n_trials_for_correction: int | None = None  # None => len(evaluations)
    apply_deflated_correction: bool = True
    min_corrected_sharpe: float = -1.0  # -1 disables

    # ---- ML Calibration (trading usefulness) ----
    # Balanced Accuracy 0.39 > random 0.33 but not enough.
    # Need P(conf bucket) -> actual net return monotonic:
    # 0.35->-0.02% 0.50->+0.01% 0.65->+0.07% 0.80->+0.19% => useful
    # Also need Brier/ECE: 0.80 predicted must mean ~80% empirical frequency.
    require_ml_calibration: bool = True
    min_calibration_spearman: float = 0.5  # monotonic rank correlation
    min_calibration_pearson: float = 0.3
    max_calibration_error: float = 0.5
    max_brier_score: float = 0.25  # Brier <0.25, 0.25=random for balanced binary
    max_ece: float = 0.10  # Expected Calibration Error <10%

    # ---- Regime Stability (Gate 6) ----
    # Split crypto into 7 regimes: Bull, Bear, Sideways, High Vol, Low Vol, Crash, Recovery
    # Require positive expectancy in at least min_regimes_positive regimes (3-4),
    # but MUST report WHEN strategy WORKS / WHEN FAILS (not necessarily positive in all).
    require_regime_stability: bool = True
    min_regimes_positive: int = 3
    min_trades_per_regime: int = 5

    # ---- Robust OOS Edge — MAX ROBUST OOS EDGE KPI (Task 14) ----
    # Replaces MAX PF with stable, cost/latency/slippage-robust OOS edge+
    # selection-bias adjustment (DSR/PBO/WRC). Weighted 8-component score.
    require_robust_oos_edge: bool = True
    min_robust_oos_edge_score: float = 0.70  # score >0.70 and critical pass
    robust_oos_edge_min_pf: float = 1.10  # stable PF threshold for edge gate
    robust_oos_edge_max_dd_pct: float = -15.0
    robust_oos_edge_min_trades: int = 30
    robust_oos_edge_min_expectancy: float = 0.0
    robust_oos_edge_min_regimes_positive: int = 3
    robust_oos_edge_min_dsr: float = 0.95
    robust_oos_edge_max_pbo: float = 0.60
    robust_oos_edge_max_wrc_p: float = 0.05
    # Weight overrides (must sum 1.0; None => use RobustOOSConfig defaults)
    robust_oos_edge_weights: dict[str, float] | None = None

    # Mode: strict production — unified strict policy (P0.6)
    # UNKNOWN ≠ PASS everywhere: required evidence absent → BLOCKED without local exceptions
    # Permissive research mode is DEPRECATED — even in research, missing Monte Carlo/stress/DSR/PBO/WRC/regime/selection → BLOCKED
    permissive: bool = False  # must remain False — do not set True to bypass missing evidence


@dataclass
class GateResult:
    passed: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateIntegrityReport:
    strategy_id: str
    integrity_passed: bool  # Integrity Checks gate
    statistical_passed: bool
    is_oos_passed: bool
    ml_calibration_passed: bool
    robustness_passed: bool
    selection_passed: bool
    regime_stability_passed: bool = True
    robust_oos_edge_passed: bool = True
    overall_passed: bool = False
    failed_stage: str | None = None  # first failed stage name
    reasons: tuple[str, ...] = field(default_factory=tuple)
    gate_details: dict[str, GateResult] = field(default_factory=dict)
    # Adjusted metrics for Tournament (selection-corrected)
    adjusted_metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class IntegrityReport:
    candidates: dict[str, CandidateIntegrityReport]
    eligible: dict[str, dict]  # strategy_id -> evaluation dict (only overall_passed)
    rejected: dict[str, tuple[str, ...]]  # strategy_id -> reasons
    n_trials: int
    deflated_correction_applied: bool


# =====================================================
# ENGINE
# =====================================================

class ResearchIntegrityEngine:
    """
    Enforces Research Integrity hierarchy above Tournament.

    Each gate is hard — Tournament never sees failing candidates.
    """

    def __init__(self, config: IntegrityConfig | None = None):
        cfg = config or IntegrityConfig()
        # P0.6 unified strict: UNKNOWN != PASS everywhere — force permissive False, no local exceptions
        cfg.permissive = False
        self.config = cfg

    # ---------- Gate 1: Integrity Checks ----------
    def _gate_integrity(self, sid: str, evaluation: dict) -> GateResult:
        m = evaluation.get("metrics", {})
        flags = evaluation.get("rules_flags")
        # If flags already computed via PromotionRules, use them
        if flags is not None:
            failed = [k for k, v in flags.items() if not v]
            if failed:
                return GateResult(False, tuple(f"integrity:{f}" for f in failed), {"flags": flags})
            return GateResult(True, (), {"flags": flags})

        # Otherwise compute directly against thresholds
        c = self.config
        checks = {
            "pf_ok": m.get("pf_median", 0) >= c.min_pf_median,
            "window_share_ok": m.get("profitable_window_share", 0) >= c.min_profitable_window_share,
            "drawdown_ok": m.get("maxdd_median_pct", -100) >= c.max_drawdown_median_pct,
            "net_ok": m.get("net_median_pct", -100) >= c.min_net_median_pct,
            "trades_ok": m.get("trades", 0) >= c.min_trades_total,
            "stability_ok": m.get("net_std_pct", 1000) <= c.max_net_std_pct,
        }
        failed = [k for k, v in checks.items() if not v]
        if failed:
            return GateResult(False, tuple(f"integrity:{f}" for f in failed), {"checks": checks})
        return GateResult(True, (), {"checks": checks})

    # ---------- White/SPA global computation (Gate 2 helper) ----------
    def _compute_wrc_global(self, evaluations: dict[str, dict], n_trials: int) -> dict[str, Any]:
        """
        Family-wise White/SPA test for data-snooping.

        Builds T x K returns DataFrame from evaluations (per-window net_pct
        or sharpes) via returns_df_from_evaluations, then calls
        white_reality_check / spa_test with stationary bootstrap.

        Returns dict with p_value etc. or skipped.
        """
        c = self.config
        if not c.wrc_enabled:
            return {"attempted": False, "skipped": "wrc_disabled"}
        if n_trials < c.wrc_min_trials:
            return {"attempted": False, "skipped": f"n_trials {n_trials} < wrc_min_trials {c.wrc_min_trials}"}
        try:
            from src.research.white_reality_check import returns_df_from_evaluations, white_reality_check, spa_test

            returns_df = returns_df_from_evaluations(evaluations)
            if returns_df is None or returns_df.empty:
                return {
                    "attempted": True,
                    "p_value": None,
                    "error": "insufficient_returns_data",
                    "details": "returns_df_from_evaluations returned None (need >=2 strategies, >=10 windows each; check evaluation['windows'] with net_pct)",
                    "skipped": "no_data",
                }
            if len(returns_df) < 10 or returns_df.shape[1] < 2:
                return {
                    "attempted": True,
                    "p_value": None,
                    "error": "too_small",
                    "details": f"returns_df shape {returns_df.shape} too small for WRC (need >=10 rows, >=2 cols)",
                    "skipped": "too_small",
                }
            method = "SPA" if c.wrc_use_spa else "WRC"
            if c.wrc_use_spa:
                res = spa_test(
                    returns_df,
                    benchmark=0,
                    n_bootstrap=c.wrc_n_bootstrap,
                    q=c.wrc_q,
                    seed=c.wrc_seed,
                    studentized=True,
                    return_details=True,
                    global_n_trials=n_trials,
                )
                p = float(res.p_value) if hasattr(res, "p_value") else float(res)  # type: ignore
                best = getattr(res, "best_strategy", None)
                # Extract n/k for details
                n_rows = getattr(res, "n", len(returns_df))
                k_cols = getattr(res, "k", returns_df.shape[1])
            else:
                res = white_reality_check(
                    returns_df,
                    benchmark=0,
                    n_bootstrap=c.wrc_n_bootstrap,
                    q=c.wrc_q,
                    seed=c.wrc_seed,
                    studentized=False,
                    return_details=True,
                    global_n_trials=n_trials,
                )
                p = float(res.p_value) if hasattr(res, "p_value") else float(res)  # type: ignore
                best = getattr(res, "best_strategy", None)
                n_rows = getattr(res, "n", len(returns_df))
                k_cols = getattr(res, "k", returns_df.shape[1])

            return {
                "attempted": True,
                "p_value": float(p),
                "method": method,
                "best_strategy": best,
                "n": int(n_rows),
                "k": int(k_cols),
                "n_bootstrap": int(c.wrc_n_bootstrap),
                "q": float(c.wrc_q),
                "passed": float(p) < c.wrc_max_p_value,
                "threshold": float(c.wrc_max_p_value),
            }
        except Exception as e:  # noqa: BLE001
            return {
                "attempted": True,
                "p_value": None,
                "error": f"{type(e).__name__}: {e}",
                "skipped": "exception",
            }

    # ---------- Gate 2: Statistical Validation ----------
    def _gate_statistical(self, sid: str, evaluation: dict, n_trials: int, wrc_global: dict[str, Any] | None = None) -> GateResult:
        c = self.config
        m = evaluation.get("metrics", {})
        reasons: list[str] = []
        details: dict[str, Any] = {}

        # Sharpe median threshold
        sharpe = float(m.get("sharpe_median", 0.0))
        details["sharpe_median"] = sharpe
        if sharpe < c.min_sharpe_median:
            reasons.append(f"stat:sharpe_median {sharpe:.3f} < {c.min_sharpe_median}")

        # Bootstrap Sharpe significance (if window data available)
        # evaluation["windows"] contains per-window net_pct/sharpe; we can use net_pct as proxy returns
        # For real validation we would use per-trade returns, but median sharpe + CI is sufficient as gate.
        windows = evaluation.get("windows", [])
        # Also handle legacy evaluation shape where windows are not present but metrics has net_std etc.
        if c.require_sharpe_significance and windows:
            try:
                from src.validation.bootstrap import block_bootstrap_sharpe
                # Use window net_pct as returns proxy (better than nothing)
                returns = [float(w.get("net_pct", 0.0)) for w in windows] or [sharpe]
                # Need at least block size; if too few windows, skip significance
                if len(returns) >= 6:
                    bs = block_bootstrap_sharpe(returns, block=min(6, len(returns)//2 or 1), n_iter=200)
                    details["bootstrap"] = bs
                    if bs["p_value"] > c.max_sharpe_p_value:
                        reasons.append(f"stat:sharpe_p_value {bs['p_value']:.3f} > {c.max_sharpe_p_value}")
                    if bs["ci_lower"] <= 0 and sharpe <= 0:
                        reasons.append(f"stat:sharpe_ci_lower {bs['ci_lower']:.3f} <=0")
                else:
                    details["bootstrap_skipped"] = "too_few_windows"
            except Exception as e:
                details["bootstrap_error"] = str(e)
                if not c.permissive:
                    reasons.append(f"stat:bootstrap_error {e}")

        # PBO check — real CPCV via src.research.pbo when possible (Bailey et al.)
        pbo = m.get("pbo")
        pbo_source = "metrics"
        pbo_unavailable = False
        if pbo is None:
            # Prefer real PBO via CPCV: try returns_df first, then is/oos sharpes
            returns_df = evaluation.get("returns_df")
            if returns_df is None:
                returns_df = evaluation.get("returns")
            if returns_df is not None:
                try:
                    from src.research.pbo import compute_pbo
                    pbo = compute_pbo(returns_df, global_n_trials=n_trials)
                    details["pbo_computed"] = pbo
                    details["pbo_global_n_trials"] = n_trials
                    pbo_source = "CPCV_real_returns"
                except Exception as e:
                    details["pbo_returns_error"] = str(e)
                    pbo = None
            # Fallback: real PBO from is/oos sharpes via CPCV-rank logic
            if pbo is None:
                is_sharpes = evaluation.get("is_sharpes")
                oos_sharpes = evaluation.get("oos_sharpes")
                if is_sharpes is not None and oos_sharpes is not None:
                    try:
                        from src.research.pbo import compute_pbo_from_sharpes
                        pbo = compute_pbo_from_sharpes(is_sharpes, oos_sharpes, global_n_trials=n_trials)
                        details["pbo_computed"] = pbo
                        details["pbo_global_n_trials"] = n_trials
                        pbo_source = "CPCV_real_sharpes"
                    except Exception as e:
                        details["pbo_real_error"] = str(e)
                        pbo_unavailable = True
                else:
                    pbo_unavailable = True
        # If real PBO unavailable, in strict mode FAIL; in permissive allow proxy for diagnostics only
        if pbo is None and pbo_unavailable:
            if not c.permissive:
                details["pbo"] = None
                details["pbo_source"] = "FAILED — real PBO unavailable, proxy forbidden in production"
                reasons.append("stat:pbo_unavailable — real PBO calculation failed; proxy not permitted in production")
            else:
                # Permissive: allow legacy proxy for diagnostic purposes only
                try:
                    from src.validation.bootstrap import pbo_combinatorial
                    is_sharpes = evaluation.get("is_sharpes")
                    oos_sharpes = evaluation.get("oos_sharpes")
                    if is_sharpes is not None and oos_sharpes is not None:
                        pbo = pbo_combinatorial(is_sharpes, oos_sharpes)
                        details["pbo_computed"] = pbo
                        details["pbo_global_n_trials"] = n_trials
                        pbo_source = "proxy_fallback (permissive)"
                    else:
                        pbo_unavailable = True
                except Exception:
                    pbo_unavailable = True
        if pbo is not None:
            details["pbo"] = pbo
            details["pbo_source"] = pbo_source
            if float(pbo) > c.max_pbo:
                reasons.append(f"stat:pbo {pbo:.3f} > {c.max_pbo} [{pbo_source}]")
        elif pbo_unavailable and not c.permissive:
            # Already added failure reason above, but ensure details reflect unavailability
            details["pbo_unavailable"] = True

        # Deflated Sharpe (multiple-testing adjustment)
        # Real DSR Bailey & Prado 2014 with skew/kurtosis when available, fallback to proxy heuristic
        if n_trials > 1:
            _real_dsr_computed = False
            _real_dsr = None
            _exp_max = None
            _skew_for_dsr = None
            _kurt_for_dsr = None
            _sample_len_for_dsr = None
            try:
                # Resolve skew/kurt/sample_len from evaluation
                # Priority: metrics fields -> evaluation top-level -> compute from windows
                _m_for_dsr = m
                _windows_for_dsr = windows
                # Check metrics for skew/kurt
                for _k in ("returns_skew", "skew", "skewness", "returns_skewness"):
                    if _k in _m_for_dsr and _m_for_dsr[_k] is not None:
                        _skew_for_dsr = float(_m_for_dsr[_k])
                        break
                if _skew_for_dsr is None:
                    for _k in ("returns_skew", "skew", "skewness"):
                        if _k in evaluation and evaluation[_k] is not None:
                            _skew_for_dsr = float(evaluation[_k])
                            break
                for _k in ("returns_kurtosis", "kurtosis", "kurt", "k", "returns_kurt"):
                    if _k in _m_for_dsr and _m_for_dsr[_k] is not None:
                        _kurt_for_dsr = float(_m_for_dsr[_k])
                        break
                if _kurt_for_dsr is None:
                    for _k in ("returns_kurtosis", "kurtosis", "kurt"):
                        if _k in evaluation and evaluation[_k] is not None:
                            _kurt_for_dsr = float(evaluation[_k])
                            break
                for _k in ("sample_len", "n_obs", "T", "sample_size", "n_returns"):
                    if _k in _m_for_dsr and _m_for_dsr[_k] is not None:
                        _sample_len_for_dsr = int(_m_for_dsr[_k])
                        break
                if _sample_len_for_dsr is None:
                    for _k in ("sample_len", "n_obs", "T"):
                        if _k in evaluation and evaluation[_k] is not None:
                            _sample_len_for_dsr = int(evaluation[_k])
                            break
                # Compute from windows if still missing
                if (_skew_for_dsr is None or _kurt_for_dsr is None or _sample_len_for_dsr is None) and _windows_for_dsr:
                    _rets = [float(_w.get("net_pct", 0.0)) for _w in _windows_for_dsr if "net_pct" in _w]
                    # Fallback to net_return or return
                    if not _rets:
                        _rets = [float(_w.get("net_return", 0.0)) for _w in _windows_for_dsr if "net_return" in _w]
                    if len(_rets) >= 4:
                        if _skew_for_dsr is None or _kurt_for_dsr is None:
                            try:
                                import numpy as _np  # noqa
                                from scipy.stats import skew as _s_skew, kurtosis as _s_kurt
                                if _skew_for_dsr is None and len(_rets) >= 3:
                                    _skew_for_dsr = float(_s_skew(_rets, bias=False))
                                if _kurt_for_dsr is None and len(_rets) >= 4:
                                    # Pearson kurtosis (3 = Normal)
                                    _kurt_for_dsr = float(_s_kurt(_rets, fisher=False, bias=False))
                            except Exception:
                                # Lightweight fallback: use numpy manual if scipy unavailable
                                try:
                                    import numpy as _np2
                                    _arr = _np2.array(_rets, dtype=float)
                                    _mean = float(_np2.mean(_arr))
                                    _std = float(_np2.std(_arr, ddof=1))
                                    if _std > 1e-12:
                                        _z = (_arr - _mean) / _std
                                        if _skew_for_dsr is None:
                                            _skew_for_dsr = float(_np2.mean(_z**3))
                                        if _kurt_for_dsr is None:
                                            _kurt_for_dsr = float(_np2.mean(_z**4))
                                    else:
                                        if _skew_for_dsr is None:
                                            _skew_for_dsr = 0.0
                                        if _kurt_for_dsr is None:
                                            _kurt_for_dsr = 3.0
                                except Exception:
                                    pass
                        if _sample_len_for_dsr is None:
                            _sample_len_for_dsr = len(_rets)
                # If still sample_len missing but windows exist, use windows length
                if _sample_len_for_dsr is None and _windows_for_dsr:
                    _sample_len_for_dsr = len(_windows_for_dsr)
                # Need all three to compute real DSR
                if _skew_for_dsr is not None and _kurt_for_dsr is not None and _sample_len_for_dsr is not None and int(_sample_len_for_dsr) > 1:
                    from src.research.dsr import deflated_sharpe_ratio as _dsr_func, expected_max_sharpe as _ems_func, is_dsr_significant as _is_sig, deflated_sharpe_ratio_detailed as _dsr_detailed
                    _real_dsr = _dsr_func(sharpe, n_trials, int(_sample_len_for_dsr), float(_skew_for_dsr), float(_kurt_for_dsr))
                    try:
                        _exp_max = _ems_func(n_trials, sample_len=int(_sample_len_for_dsr), returns_skew=float(_skew_for_dsr), returns_kurtosis=float(_kurt_for_dsr))
                    except Exception:
                        _exp_max = _ems_func(n_trials, int(_sample_len_for_dsr))
                    # Also compute detailed version with global context
                    _dsr_detail = _dsr_detailed(sharpe, n_trials, int(_sample_len_for_dsr), float(_skew_for_dsr), float(_kurt_for_dsr), global_n_trials=n_trials)
                    details.update(_dsr_detail)
                    details["deflated_sharpe"] = float(_real_dsr)
                    details["deflated_sharpe_real"] = float(_real_dsr)
                    details["deflated_sharpe_method"] = "real_dsr_BaileyPrado2014"
                    details["dsr_strict"] = True
                    details["expected_max_sharpe"] = float(_exp_max) if _exp_max is not None else None
                    details["dsr_skew"] = float(_skew_for_dsr)
                    details["dsr_kurtosis"] = float(_kurt_for_dsr)
                    details["sample_len"] = int(_sample_len_for_dsr)
                    details["n_trials"] = n_trials
                    details["dsr_significant"] = bool(_is_sig(float(_real_dsr), threshold=0.95))
                    _real_dsr_computed = True
                    # Gate: real DSR significance
                    # Interpret min_deflated_sharpe as probability threshold when real DSR is used.
                    # Default 0.0 -> use sensible DSR thresholds: 0.95 for strict, 0.5 for permissive
                    # If user set explicit threshold >0, respect max of thresholds.
                    _dsr_threshold = 0.95 if not c.permissive else 0.5
                    if c.min_deflated_sharpe > _dsr_threshold:
                        _dsr_threshold = float(c.min_deflated_sharpe)
                    elif c.min_deflated_sharpe > 0 and c.min_deflated_sharpe < 0.5:
                        # User set very low threshold like 0.0 -> keep our sensible threshold
                        pass
                    details["dsr_threshold"] = float(_dsr_threshold)
                    if float(_real_dsr) < _dsr_threshold:
                        reasons.append(f"stat:deflated_sharpe DSR {float(_real_dsr):.3f} < {_dsr_threshold} (expected_max {float(_exp_max):.3f}, skew {float(_skew_for_dsr):.2f}, kurt {float(_kurt_for_dsr):.2f}, T={int(_sample_len_for_dsr)}, N={n_trials})")
            except Exception as _e:
                details["dsr_error"] = str(_e)
                _real_dsr_computed = False
            if not _real_dsr_computed:
                if c.permissive:
                    # Permissive (research): allow proxy for diagnostic purposes only
                    deflated = sharpe / math.sqrt(1 + math.log(n_trials))
                    details["deflated_sharpe"] = float(deflated)
                    details["deflated_sharpe_method"] = "proxy Sharpe/sqrt(1+log N) — NOT strict DSR (permissive)"
                    details["dsr_strict"] = False
                    details["n_trials"] = n_trials
                    if _skew_for_dsr is not None:
                        details["dsr_skew_available_but_failed"] = float(_skew_for_dsr)
                    if _kurt_for_dsr is not None:
                        details["dsr_kurt_available_but_failed"] = float(_kurt_for_dsr)
                    if float(deflated) < c.min_deflated_sharpe:
                        reasons.append(f"stat:deflated_sharpe {float(deflated):.3f} < {c.min_deflated_sharpe} (proxy, not strict)")
                else:
                    # Strict (production): FAIL — DSR unavailable, proxy not allowed as gate
                    details["deflated_sharpe"] = None
                    details["deflated_sharpe_method"] = "FAILED — real DSR unavailable, proxy forbidden in strict mode"
                    details["dsr_strict"] = True
                    details["n_trials"] = n_trials
                    details["dsr_unavailable"] = True
                    reasons.append("stat:deflated_sharpe_unavailable — real DSR calculation failed; proxy not permitted in production")

        # ---- White Reality Check / SPA (family-wise data-snooping) ----
        # When n_trials large, best-of-many is biased.  WRC/SPA bootstrap
        # under stationary bootstrap (Politis & Romano) tests H0: max_k E[f_k]<=0
        # vs bootstrap null.  Large p => best is fluke, block edge claim.
        # wrc_global is computed once per assess() from all evaluations.
        if wrc_global is not None:
            details["wrc"] = wrc_global
            if wrc_global.get("attempted"):
                p = wrc_global.get("p_value")
                method = wrc_global.get("method", "WRC/SPA")
                if p is not None:
                    details["wrc_p_value"] = float(p)
                    details["wrc_method"] = method
                    details["wrc_best"] = wrc_global.get("best_strategy")
                    details["wrc_k"] = wrc_global.get("k")
                    details["wrc_n"] = wrc_global.get("n")
                    details["wrc_threshold"] = float(c.wrc_max_p_value)
                    # FAIL if p > alpha: cannot claim edge, data-snooping risk
                    if float(p) > c.wrc_max_p_value:
                        reasons.append(
                            f"stat:wrc_p_value {float(p):.3f} > {c.wrc_max_p_value} "
                            f"[{method} n={wrc_global.get('n')} k={wrc_global.get('k')} B={wrc_global.get('n_bootstrap')} q={wrc_global.get('q')} best={wrc_global.get('best_strategy')}] "
                            f"(data-snooping: best of {n_trials} not significant)"
                        )
                else:
                    # Computation attempted but no p (insufficient data / error)
                    err = wrc_global.get("error") or wrc_global.get("skipped") or "unknown"
                    details["wrc_error"] = err
                    details["wrc_details"] = wrc_global.get("details")
                    # In strict mode, missing WRC when required is a FAIL (cannot validate edge)
                    if not c.permissive:
                        reasons.append(f"stat:wrc_missing {err} (need WRC/SPA when n_trials={n_trials}>={c.wrc_min_trials})")
                    else:
                        details["wrc_skipped_permissive"] = True
            else:
                details["wrc_skipped"] = wrc_global.get("skipped")

        if reasons:
            return GateResult(False, tuple(reasons), details)
        return GateResult(True, (), details)

    # ---------- Gate 3: Robustness ----------
    def _gate_robustness(self, sid: str, evaluation: dict) -> GateResult:
        c = self.config
        m = evaluation.get("metrics", {})
        reasons: list[str] = []
        details: dict[str, Any] = {}

        # Cost robustness
        has_cost_data = ("cost_robust" in m) or ("cost_stress" in evaluation)
        cost_robust = m.get("cost_robust", None)
        if "cost_stress" in evaluation:
            try:
                from src.validation.cost_stress import is_cost_robust
                cost_robust = is_cost_robust(evaluation["cost_stress"])
                has_cost_data = True
                details["cost_robust_computed"] = cost_robust
            except Exception as e:
                details["cost_robust_error"] = str(e)
                cost_robust = False
                has_cost_data = True
        details["cost_robust"] = cost_robust
        details["has_cost_data"] = has_cost_data
        if c.require_cost_robust and has_cost_data and not cost_robust:
            reasons.append("robust:cost_robust False (PF<1 at 1.5x costs)")
        elif c.require_cost_robust and not has_cost_data and not c.permissive:
            # In strict mode, missing cost data is a fail
            reasons.append("robust:cost_robust missing")

        # Monte Carlo / Stress scores (if evaluated)
        mc = m.get("monte_carlo_score")
        stress = m.get("stress_score")
        # Also check tournament-level scores that may be in metrics
        if mc is not None:
            details["monte_carlo_score"] = mc
            if float(mc) < c.min_monte_carlo_score:
                reasons.append(f"robust:monte_carlo {mc:.3f} < {c.min_monte_carlo_score}")
        elif not c.permissive:
            # Strict mode: missing Monte Carlo evidence is a FAIL (not pass)
            reasons.append("robust:monte_carlo missing (required in strict mode)")

        if stress is not None:
            details["stress_score"] = stress
            if float(stress) < c.min_stress_score:
                reasons.append(f"robust:stress {stress:.3f} < {c.min_stress_score}")
        elif not c.permissive:
            # Strict mode: missing Stress evidence is a FAIL (not pass)
            reasons.append("robust:stress missing (required in strict mode)")

        # Drawdown / stability already checked in integrity, but robustness double-checks
        # Could add additional robustness checks like profitable_window_share stability

        if reasons:
            return GateResult(False, tuple(reasons), details)
        return GateResult(True, (), details)

    # ---------- Gate 3.5: IS-OOS Consistency (alpha reality) ----------
    def _gate_is_oos(self, sid: str, evaluation: dict, n_trials: int = 0) -> GateResult:
        """
        Core alpha check: IS != OOS => research hypothesis only, not real alpha.

        Breakout often shows IS PF 1.6 -> OOS PF 0.9 deterioration.
        This gate blocks such cases from becoming real alpha.
        """
        c = self.config
        if not c.require_is_oos_consistency:
            return GateResult(True, (), {})

        is_oos = evaluation.get("is_oos")
        # Fallback: if evaluation has is_metrics/oos metrics separately
        if is_oos is None:
            # No IS-OOS data available — in permissive research mode allow,
            # in strict production require data (fail)
            if c.permissive:
                return GateResult(True, (), {"is_oos_missing": "permissive pass"})
            return GateResult(False, ("is_oos:missing IS-OOS data (no is_oos)",), {"is_oos": None})

        reasons: list[str] = []
        details: dict[str, Any] = dict(is_oos)

        pf_ratio = float(is_oos.get("pf_ratio", 0.0))
        pf_det = float(is_oos.get("pf_deterioration", 0.0))
        sharpe_det = float(is_oos.get("sharpe_deterioration", 0.0))
        # Upgrade PBO to real CPCV when is_sharpes/oos_sharpes or returns_df available
        pbo_proxy = float(is_oos.get("pbo", 0.5))
        pbo = None
        pbo_source = "unavailable"
        pbo_unavailable = False
        # Try real CPCV: first via returns_df, then via sharpes
        returns_df = evaluation.get("returns_df")
        if returns_df is None:
            returns_df = evaluation.get("returns")
        if returns_df is not None:
            try:
                from src.research.pbo import compute_pbo as _cpcv_pbo
                pbo = float(_cpcv_pbo(returns_df, global_n_trials=n_trials))
                pbo_source = "CPCV_real_returns"
                details["pbo_real"] = pbo
                details["pbo_proxy_original"] = pbo_proxy
                details["pbo_global_n_trials"] = n_trials
            except Exception as e:
                details["pbo_returns_error"] = str(e)
                pbo_unavailable = True
        else:
            is_sharpes = evaluation.get("is_sharpes")
            oos_sharpes = evaluation.get("oos_sharpes")
            if is_sharpes is not None and oos_sharpes is not None:
                try:
                    from src.research.pbo import compute_pbo_from_sharpes as _real_pbo
                    pbo = float(_real_pbo(is_sharpes, oos_sharpes, global_n_trials=n_trials))
                    pbo_source = "CPCV_real_sharpes"
                    details["pbo_real"] = pbo
                    details["pbo_proxy_original"] = pbo_proxy
                    details["pbo_global_n_trials"] = n_trials
                except Exception as e:
                    details["pbo_real_error"] = str(e)
                    pbo_unavailable = True
            else:
                pbo_unavailable = True
        # If real PBO unavailable, in strict mode FAIL; in permissive use proxy for diagnostics
        if pbo is None and pbo_unavailable:
            if not c.permissive:
                details["pbo"] = None
                details["pbo_source"] = "FAILED — real PBO unavailable, proxy forbidden in production"
                reasons.append("is_oos:pbo_unavailable — real PBO calculation failed; proxy not permitted in production")
            else:
                # Permissive: allow proxy for diagnostic purposes only
                pbo = pbo_proxy
                pbo_source = "proxy_fallback (permissive)"
        oos_pf = float(is_oos.get("oos_pf", 0.0))
        is_pf = float(is_oos.get("is_pf", 0.0))

        details.update({"pf_ratio": pf_ratio, "pf_deterioration": pf_det, "pbo": pbo, "pbo_source": pbo_source})

        # PF ratio OOS/IS must stay high — sharp drop indicates overfit
        if pf_ratio < c.min_pf_ratio:
            reasons.append(f"is_oos:pf_ratio {pf_ratio:.2f} < {c.min_pf_ratio} (IS {is_pf:.2f} -> OOS {oos_pf:.2f})")
        if pf_det > c.max_pf_deterioration:
            reasons.append(f"is_oos:pf_deterioration {pf_det:.0%} > {c.max_pf_deterioration:.0%} (IS PF {is_pf:.2f} -> OOS {oos_pf:.2f})")
        # Sharpe deterioration
        if abs(sharpe_det) > c.max_sharpe_deterioration and is_oos.get("is_sharpe", 0.0) > 0.3:
            # Only enforce when IS Sharpe was meaningful (>0.3)
            reasons.append(f"is_oos:sharpe_deterioration {sharpe_det:.0%} > {c.max_sharpe_deterioration:.0%}")
        # PBO high means IS ranking does not predict OOS (overfit) — real CPCV when available
        if pbo is not None and pbo > c.max_is_oos_pbo:
            reasons.append(f"is_oos:pbo {pbo:.2f} > {c.max_is_oos_pbo} [{pbo_source}]")

        # Absolute OOS must still be viable if IS was good
        # If IS PF >=1.3 but OOS PF <1.0 => clear deterioration
        if is_pf >= 1.3 and oos_pf < 1.0:
            reasons.append(f"is_oos:is_good_but_oos_bad IS PF {is_pf:.2f} -> OOS PF {oos_pf:.2f}")

        if reasons:
            return GateResult(False, tuple(reasons), details)
        return GateResult(True, (), details)

    # ---------- Gate 3.7: ML Calibration (trading usefulness) ----------
    def _gate_ml_calibration(self, sid: str, evaluation: dict) -> GateResult:
        """
        Trading usefulness: confidence must correlate with actual net return.

        Example useful:
          0.35 -> -0.02%  0.50 -> +0.01%  0.65 -> +0.07%  0.80 -> +0.19%  monotonic

        vs not useful (BA 0.39 but flat):
          0.35 -> +0.02%  0.50 -> -0.01%  non-monotonic

        Balanced Accuracy 0.39 > random 0.33 (3 classes) is not sufficient.
        """
        c = self.config
        if not c.require_ml_calibration:
            return GateResult(True, (), {})

        # Try to get calibration data from evaluation
        # For meta-labeler: y_true_net and y_pred_conf / y_pred_expected
        y_true = evaluation.get("y_true_net") or evaluation.get("calibration_y_true")
        y_pred = evaluation.get("y_pred_conf") or evaluation.get("calibration_y_pred") or evaluation.get("y_pred_expected")

        # Fallback: try to derive from windows (less accurate, but for walk-forward)
        # If no per-sample data, try to use bucket example from metrics if provided
        # For now, if no data, permissive mode passes, strict fails
        if y_true is None or y_pred is None:
            # Try alternative keys: ml_calibration data
            calib = evaluation.get("ml_calibration")
            if calib and isinstance(calib, dict):
                y_true = calib.get("y_true_net")
                y_pred = calib.get("y_pred_conf")
            if y_true is None or y_pred is None:
                if c.permissive:
                    return GateResult(True, (), {"calibration_missing": "permissive pass"})
                return GateResult(False, ("ml_calibration:missing y_true/y_pred (no calibration data)",), {"y_true": y_true, "y_pred": y_pred})

        try:
            from src.validation.calibration import evaluate_calibration, evaluate_brier_ece

            report = evaluate_calibration(list(y_true), list(y_pred), n_buckets=4)
            details = {
                "buckets": [(b.bucket, b.mean_pred, b.mean_actual, b.n) for b in report.buckets],
                "spearman": report.spearman_corr,
                "pearson": report.pearson_corr,
                "monotonic": report.monotonic,
                "cal_error": report.calibration_error,
                "passed": report.passed,
            }
            # Check bucket monotonic first (trading usefulness)
            if not report.passed:
                reasons = []
                if not report.monotonic:
                    reasons.append(f"ml_calibration:non-monotonic spearman {report.spearman_corr:.2f} need >{c.min_calibration_spearman}")
                if report.pearson_corr <= c.min_calibration_pearson:
                    reasons.append(f"ml_calibration:pearson {report.pearson_corr:.2f} <= {c.min_calibration_pearson}")
                if report.calibration_error >= c.max_calibration_error:
                    reasons.append(f"ml_calibration:error {report.calibration_error:.3f} >= {c.max_calibration_error}")
                if not reasons:
                    reasons.append(f"ml_calibration:failed {report.reason}")
                return GateResult(False, tuple(reasons), details)

            # For probabilistic classifier (y_pred in [0,1] AND y_true is binary win 0/1),
            # also check Brier/ECE/Reliability: 0.80 must mean ~80% empirical.
            # For regression E[net] (y_pred is expected net return, y_true is actual net),
            # Brier is not applicable — bucket monotonic already validates E[net] calibration.
            is_prob = all(0 <= float(p) <= 1 for p in y_pred) and len(y_pred) > 0
            # Check if y_true is binary win (0/1) — then Brier is meaningful
            is_binary_y_true = all(float(v) in (0.0, 1.0) for v in y_true) and len(y_true) > 0
            # Also consider net>0 as win for Brier if y_true is net but y_pred is prob
            # Only do Brier when y_pred is prob AND (y_true is binary OR we can derive win)
            # For E[net] regression, y_pred is net return (e.g. -0.02..0.19) not prob, so skip
            # Heuristic: if y_pred in [0,1] and y_true is binary win, do Brier; if y_true is net, skip Brier and rely on bucket monotonic
            if is_prob and is_binary_y_true:
                try:
                    brier_rep = evaluate_brier_ece([int(float(v)) for v in y_true], list(y_pred), max_brier=c.max_brier_score, max_ece=c.max_ece, n_buckets=10)
                    details["brier"] = brier_rep.brier_score
                    details["brier_skill"] = brier_rep.brier_skill
                    details["ece"] = brier_rep.ece
                    details["mce"] = brier_rep.mce
                    details["reliability_curve"] = brier_rep.reliability_curve
                    if not brier_rep.passed:
                        reasons = []
                        if not brier_rep.passed_brier:
                            reasons.append(f"ml_calibration:Brier {brier_rep.brier_score:.3f} >= {c.max_brier_score} (need <{c.max_brier_score})")
                        if not brier_rep.passed_ece:
                            reasons.append(f"ml_calibration:ECE {brier_rep.ece:.3f} >= {c.max_ece} (need <{c.max_ece}, MCE {brier_rep.mce:.3f})")
                        if brier_rep.reason and not reasons:
                            reasons.append(f"ml_calibration:{brier_rep.reason}")
                        return GateResult(False, tuple(reasons), details)
                    details["reliability_passed"] = True
                except Exception as e:
                    details["brier_ece_error"] = str(e)
                    if not c.permissive:
                        return GateResult(False, (f"ml_calibration:brier_ece_error {e}",), details)
            elif is_prob and not is_binary_y_true:
                # y_pred is prob but y_true is net return (continuous) — bucket monotonic already covers E[net] calibration,
                # Brier on win derived from net>0 would be noisy. Skip Brier, rely on bucket monotonic which already passed.
                details["brier_skipped"] = "y_true is net return, not binary — bucket monotonic is primary (E[net] calibration)"
                pass

            return GateResult(True, (), details)
        except Exception as e:
            if c.permissive:
                return GateResult(True, (), {"calibration_error": str(e), "permissive": True})
            return GateResult(False, (f"ml_calibration:error {e}",), {})

    # ---------- Gate 4: Selection Adjustment ----------
    def _gate_selection(self, sid: str, evaluation: dict, n_trials: int) -> GateResult:
        c = self.config
        m = evaluation.get("metrics", {})
        details: dict[str, Any] = {}
        # Deflated correction
        sharpe = float(m.get("sharpe_median", 0.0))
        n = c.n_trials_for_correction or n_trials
        corrected = sharpe
        if c.apply_deflated_correction and n > 1:
            corrected = sharpe / math.sqrt(1 + math.log(n))
            details["sharpe_corrected"] = corrected
            details["sharpe_original"] = sharpe
            details["n_trials"] = n
            if corrected < c.min_corrected_sharpe:
                return GateResult(False, (f"selection:corrected_sharpe {corrected:.3f} < {c.min_corrected_sharpe}",), details)

        # Holm/Bonferroni style p-value adjustment placeholder
        # If evaluation has p_value, adjust: p_adj = p * n (Bonferroni)
        p_val = evaluation.get("p_value")
        if p_val is None:
            # Try from bootstrap details if computed
            p_val = details.get("p_value")
        if p_val is not None and n > 1:
            p_adj = min(1.0, float(p_val) * n)
            details["p_value_adjusted"] = p_adj
            if p_adj > 0.05:
                # In strict mode this would be a fail, but we keep permissive
                if not c.permissive:
                    return GateResult(False, (f"selection:p_adjusted {p_adj:.3f} >0.05",), details)

        return GateResult(True, (), details)

    # ---------- Gate 6: Regime Stability ----------
    def _gate_regime_stability(self, sid: str, evaluation: dict) -> GateResult:
        """
        Gate 6 — Regime Stability (Task 5).

        Split crypto into 7 regimes: Bull, Bear, Sideways, High Vol, Low Vol, Crash, Recovery.
        Require positive expectancy (mean pnl >0) in at least min_regimes_positive regimes.
        System MUST KNOW when strategy works / when fails — works/fails lists are reported
        even when verdict passes. Negative expectancy in some regimes is ALLOWED.

        Thresholds from IntegrityConfig:
            require_regime_stability: bool (default True)
            min_regimes_positive: int (default 3)
            min_trades_per_regime: int (default 5)

        Input contract (evaluation dict):
            - regime_labels: Series/list of regime strings same length as trades/windows
            - trades_df / trades / trade_pnls / windows / window_stats: pnl source
            - OR regime_stability: precomputed dict with verdict/works/fails

        If regime data missing:
            - permissive=True  => PASS with warning (research mode, no regime data yet)
            - permissive=False => FAIL (strict production needs regime proof)

        When data present, delegates to src.research.regime_stability.evaluate_regime_stability
        and checks n_positive >= min_regimes_positive.
        """
        c = self.config
        if not getattr(c, "require_regime_stability", False):
            return GateResult(True, (), {"regime_stability": "disabled"})

        # Fast path: precomputed regime_stability in evaluation
        pre = evaluation.get("regime_stability") or evaluation.get("regime_stability_result")
        if isinstance(pre, dict) and "verdict" in pre:
            details = {"regime_stability": pre, "source": "precomputed"}
            if bool(pre.get("verdict")):
                return GateResult(True, (), details)
            # verdict False
            works = pre.get("works", [])
            fails = pre.get("fails", [])
            n_pos = pre.get("n_positive", 0)
            return GateResult(
                False,
                (f"regime_stability:n_positive {n_pos} < {c.min_regimes_positive} (works={works}, fails={fails})",),
                details,
            )

        # Try to locate regime_labels + pnl source
        regime_labels = (
            evaluation.get("regime_labels")
            or evaluation.get("window_regimes")
            or evaluation.get("regimes")
            or evaluation.get("regime_series")
        )
        # Also handle case where df with regime column produced labels via classify_regimes
        # If evaluation contains df and no labels, try auto-classify (best effort)
        if regime_labels is None and "df" in evaluation and pd is not None and isinstance(evaluation["df"], pd.DataFrame):
            try:
                from src.research.regime_stability import classify_regimes  # lazy

                df_for_regime = evaluation["df"]
                # if windows present, map window center to regime, else bar-wise
                regime_series = classify_regimes(df_for_regime)
                windows = evaluation.get("windows") or evaluation.get("window_stats")
                if windows and isinstance(windows, list) and len(windows) > 0:
                    # Map each window to regime of its end bar (or majority)
                    # For simplicity, use regime at window-aligned indices
                    # We need to slice regime_series to match windows count
                    # Assume windows correspond to OOS test windows covering df tail sequentially
                    # Use last len(windows) regimes as proxy (conservative)
                    if len(regime_series) >= len(windows):
                        # take tail regimes per window
                        # step = len(df) // len(windows) approx, just take evenly sampled
                        step = max(1, len(regime_series) // len(windows))
                        sampled = [str(regime_series.iloc[min((i + 1) * step - 1, len(regime_series) - 1)]) for i in range(len(windows))]
                        regime_labels = sampled
                    else:
                        regime_labels = regime_series.tolist()[: len(windows)]
                else:
                    # bar-wise, need trades length? fallback to bar regimes
                    regime_labels = regime_series.tolist()
            except Exception as e:  # noqa: BLE001
                if c.permissive:
                    return GateResult(True, (), {"regime_stability_skipped": f"auto-classify failed: {e}", "permissive": True})
                return GateResult(False, (f"regime_stability:auto_classify_error {e}",), {})

        if regime_labels is None:
            # No regime data at all
            if c.permissive:
                return GateResult(True, (), {"regime_stability_skipped": "no regime data (permissive pass)", "require_regime_stability": True})
            return GateResult(False, ("regime_stability:missing regime_labels (no trades/windows regime mapping)",), {})

        # Locate pnl source
        pnl_source = None
        pnl_obj = None
        # Priority: trades_df, trades, trade_pnls, windows, window_stats, pnls
        for key in ("trades_df", "trades", "trade_pnls", "pnls", "pnl_series", "returns", "trade_returns"):
            if key in evaluation and evaluation[key] is not None:
                pnl_obj = evaluation[key]
                pnl_source = key
                break
        if pnl_obj is None:
            for key in ("windows", "window_stats"):
                if key in evaluation and evaluation[key] is not None:
                    pnl_obj = evaluation[key]
                    pnl_source = key
                    break
        if pnl_obj is None:
            if c.permissive:
                return GateResult(True, (), {"regime_stability_skipped": "no pnl data for regime check (permissive)", "regime_labels_len": len(regime_labels) if hasattr(regime_labels, "__len__") else "unknown"})
            return GateResult(False, ("regime_stability:missing pnl data (windows/trades) for regime_labels",), {"regime_labels_len": len(regime_labels) if hasattr(regime_labels, "__len__") else None})

        # Delegate to evaluator
        try:
            from src.research.regime_stability import evaluate_regime_stability

            result = evaluate_regime_stability(
                pnl_obj,
                regime_labels,
                min_regimes_positive=int(getattr(c, "min_regimes_positive", 3)),
                min_trades_per_regime=int(getattr(c, "min_trades_per_regime", 5)),
            )
        except ValueError as ve:
            # Mismatch or insufficient data => permissive pass, strict fail
            if c.permissive:
                return GateResult(True, (), {"regime_stability_error": str(ve), "permissive": True, "source": pnl_source})
            return GateResult(False, (f"regime_stability:error {ve}",), {"source": pnl_source})
        except Exception as e:  # noqa: BLE001
            if c.permissive:
                return GateResult(True, (), {"regime_stability_error": str(e), "permissive": True})
            return GateResult(False, (f"regime_stability:error {e}",), {})

        details = {"regime_stability": result, "source": pnl_source}
        # Also surface per-regime for observability
        n_pos = int(result.get("n_positive", 0))
        works = result.get("works", [])
        fails = result.get("fails", [])
        if bool(result.get("verdict")):
            return GateResult(True, (), details)
        return GateResult(
            False,
            (f"regime_stability:n_positive {n_pos} < {getattr(c, 'min_regimes_positive', 3)} (works={works}, fails={fails}, insufficient={result.get('insufficient',[])})",),
            details,
        )

    # ---------- Gate 7: Robust OOS Edge — MAX ROBUST OOS EDGE KPI (Task 14) ----------
    def _gate_robust_oos_edge(self, sid: str, evaluation: dict) -> GateResult:
        """
        Task 14 — MAX ROBUST OOS EDGE gate.

        Replaces MAX PF.  Requires:
          - weighted 8-component score > 0.70  (expectancy, PF, DD, sample,
            regime, cost, slippage/latency, DSR/WRC)
          - ALL 8 critical components pass (expectancy, PF, DD, sample,
            regime, cost_robust, slippage_latency, selection_bias)
        Score computed via src.research.robust_oos_edge.compute_robust_oos_edge.
        Blocks promotion before Champion (hard gate, after all other gates).
        """
        c = self.config
        if not getattr(c, "require_robust_oos_edge", True):
            return GateResult(True, (), {"robust_oos_edge": "disabled"})
        try:
            from src.research.robust_oos_edge import RobustOOSConfig, compute_robust_oos_edge

            # In permissive (research) mode, inherit thresholds from IntegrityConfig
            # so legacy loose tests (min_trades 0, min_pf 0) still pass. In strict
            # production, enforce robust defaults (PF 1.1, trades 30, DD -15).
            is_permissive = bool(getattr(c, "permissive", False))
            if is_permissive:
                # Inherit from IntegrityConfig base thresholds (loose tests use 0)
                rc_min_pf = float(getattr(c, "min_pf_median", 1.05))
                rc_min_trades = int(getattr(c, "min_trades_total", 30))
                rc_max_dd = float(getattr(c, "max_drawdown_median_pct", -15.0))
                # min_expectancy: Integrity uses min_net_median_pct; if very negative (loose -100) keep robust 0.0
                base_exp = float(getattr(c, "min_net_median_pct", 0.0))
                rc_min_exp = float(getattr(c, "robust_oos_edge_min_expectancy", 0.0)) if base_exp < -5 else base_exp
                rc_min_reg = int(getattr(c, "min_regimes_positive", 3))
            else:
                # PRODUCTION: Use strict thresholds and require ALL 8 components to pass
                rc_min_pf = float(getattr(c, "robust_oos_edge_min_pf", 1.1))
                rc_min_trades = int(getattr(c, "robust_oos_edge_min_trades", 30))
                rc_max_dd = float(getattr(c, "robust_oos_edge_max_dd_pct", -15.0))
                rc_min_exp = float(getattr(c, "robust_oos_edge_min_expectancy", 0.0))
                rc_min_reg = int(getattr(c, "robust_oos_edge_min_regimes_positive", 3))

            rc = RobustOOSConfig(
                min_expectancy=rc_min_exp,
                min_pf=rc_min_pf,
                max_dd_pct=rc_max_dd,
                min_trades=rc_min_trades,
                min_regimes_positive=rc_min_reg,
                total_regimes=7,
                min_dsr=float(getattr(c, "robust_oos_edge_min_dsr", 0.95)),
                max_pbo=float(getattr(c, "robust_oos_edge_max_pbo", 0.60)),
                max_wrc_p=float(getattr(c, "robust_oos_edge_max_wrc_p", 0.05)),
                min_score=float(getattr(c, "min_robust_oos_edge_score", 0.70)),
                permissive=is_permissive,
            )
            # weight overrides
            w = getattr(c, "robust_oos_edge_weights", None)
            if isinstance(w, dict):
                for k, v in w.items():
                    # support both w_expectancy and expectancy
                    target = k if k.startswith("w_") else f"w_{k}"
                    if hasattr(rc, target):
                        try:
                            setattr(rc, target, float(v))
                        except Exception:
                            pass
                    elif hasattr(rc, k):
                        try:
                            setattr(rc, k, float(v))
                        except Exception:
                            pass

            result = compute_robust_oos_edge(evaluation, config=rc)
            # result is RobustOOSResult (tuple-unpackable + .score/.components)
            score = float(result.score) if hasattr(result, "score") else float(result[0])  # type: ignore
            comps = result.components if hasattr(result, "components") else result[1]  # type: ignore
            # details for audit
            try:
                comp_detail = comps.to_detail_dict() if hasattr(comps, "to_detail_dict") else {}
            except Exception:
                comp_detail = {}
            try:
                res_dict = result.to_dict() if hasattr(result, "to_dict") else {}
            except Exception:
                res_dict = {}
            details: dict[str, Any] = {
                "robust_oos_edge_score": float(score),
                "threshold": float(rc.min_score),
                "passed": bool(getattr(result, "passed", score > rc.min_score)),
                "components": comp_detail,
                "result": res_dict,
                "config": {
                    "min_expectancy": rc.min_expectancy,
                    "min_pf": rc.min_pf,
                    "max_dd_pct": rc.max_dd_pct,
                    "min_trades": rc.min_trades,
                    "min_regimes": rc.min_regimes_positive,
                    "min_dsr": rc.min_dsr,
                    "max_pbo": rc.max_pbo,
                    "max_wrc_p": rc.max_wrc_p,
                    "min_score": rc.min_score,
                },
            }
            # Determine if passed: for production, require ALL critical components (all 8)
            # For permissive (research), use is_robust_edge which checks score > threshold + critical
            if is_permissive:
                from src.research.robust_oos_edge import is_robust_edge

                passed = bool(is_robust_edge(score, comps, threshold=rc.min_score, critical=rc.critical_components, config=rc))
            else:
                # PRODUCTION: Must pass score threshold AND ALL 8 components
                passed = float(score) > float(rc.min_score)
                if passed:
                    # Check ALL 8 components pass
                    all_critical = (
                        "expectancy",
                        "pf_stable",
                        "dd",
                        "sample",
                        "regime",
                        "cost_robust",
                        "slippage_latency",
                        "selection_bias",
                    )
                    for crit in all_critical:
                        try:
                            comp = comps[crit] if hasattr(comps, "__getitem__") else getattr(comps, crit, None)  # type: ignore
                            if comp is None:
                                passed = False
                                details.setdefault("production_check", {})[crit] = "missing => fail"
                                break
                            # comp may be ComponentResult or dict
                            is_pass = bool(getattr(comp, "passed", False)) if not isinstance(comp, dict) else bool(comp.get("passed", False))
                            if not is_pass:
                                passed = False
                                details.setdefault("production_check", {})[crit] = "failed"
                                break
                            details.setdefault("production_check", {})[crit] = "passed"
                        except Exception:
                            passed = False
                            details.setdefault("production_check", {})[crit] = "error"
                            break

            if passed:
                return GateResult(True, (), details)

            # Build reasons
            reasons: list[str] = []
            if score <= float(rc.min_score):
                reasons.append(f"robust_oos_edge:score {score:.3f} <= {rc.min_score:.2f} (need >{rc.min_score:.2f})")
            # critical failures (all 8 are critical in production)
            critical_check = rc.critical_components if is_permissive else (
                "expectancy", "pf_stable", "dd", "sample",
                "regime", "cost_robust", "slippage_latency", "selection_bias"
            )
            for crit in critical_check:
                try:
                    comp = comps[crit] if hasattr(comps, "__getitem__") else getattr(comps, crit, None)  # type: ignore
                    if comp is None:
                        reasons.append(f"robust_oos_edge:critical {crit} missing => fail")
                        continue
                    # comp may be ComponentResult
                    if isinstance(comp, dict):
                        is_pass = bool(comp.get("passed", False))
                        reason = comp.get("reason", "")
                    else:
                        is_pass = bool(getattr(comp, "passed", False))
                        reason = getattr(comp, "reason", "")
                    if not is_pass:
                        reasons.append(f"robust_oos_edge:critical {crit} failed ({reason})")
                except Exception as e:
                    reasons.append(f"robust_oos_edge:critical {crit} error {e}")
            if not reasons:
                # list all failing components for visibility
                try:
                    d = comps.as_dict() if hasattr(comps, "as_dict") else {}
                    for k, v in d.items():
                        is_pass = bool(getattr(v, "passed", False)) if not isinstance(v, dict) else bool(v.get("passed", False))
                        if not is_pass:
                            reason = getattr(v, "reason", "") if not isinstance(v, dict) else v.get("reason", "")
                            reasons.append(f"robust_oos_edge:{k} failed ({reason})")
                except Exception:
                    pass
            if not reasons:
                reasons.append(f"robust_oos_edge:score {score:.3f} fails gate (unknown)")
            return GateResult(False, tuple(reasons), details)
        except Exception as e:  # noqa: BLE001
            if bool(getattr(c, "permissive", False)):
                return GateResult(True, (), {"robust_oos_edge_error": str(e), "permissive": True})
            return GateResult(False, (f"robust_oos_edge:error {e}",), {"error": str(e)})

    # ---------- Helper: global n_trials from ExperimentRegistry ----------
    def _get_global_n_trials(self, evaluations: dict[str, dict], registry: Optional["ExperimentRegistry"] = None) -> int:
        """
        Get global experiment count for selection bias correction.

        Uses ExperimentRegistry to count all experiments in the same
        hypothesis family / OOS period / dataset. Falls back to len(evaluations)
        if registry unavailable or no matching experiments found.

        This prevents selection bias where Supervisor tests 5000 strategies
        but only 5 reach this batch — n_trials must be 5000, not 5.
        """
        if registry is None:
            # No registry available — fall back to batch size (conservative, under-corrects)
            return len(evaluations)

        # Determine OOS period / dataset from evaluations
        # Strategy: find most common oos_period / dataset_id across evaluations
        oos_periods: dict[str, int] = {}
        dataset_ids: dict[str, int] = {}

        for ev in evaluations.values():
            # Check evaluation top-level
            oos = ev.get("oos_period") or ev.get("OOS_period")
            if oos:
                oos_periods[oos] = oos_periods.get(oos, 0) + 1
            ds = ev.get("dataset_id") or ev.get("dataset")
            if ds:
                dataset_ids[ds] = dataset_ids.get(ds, 0) + 1

            # Check is_oos sub-dict
            is_oos = ev.get("is_oos")
            if isinstance(is_oos, dict):
                oos = is_oos.get("oos_period") or is_oos.get("OOS_period")
                if oos:
                    oos_periods[oos] = oos_periods.get(oos, 0) + 1
                ds = is_oos.get("dataset_id") or is_oos.get("dataset")
                if ds:
                    dataset_ids[ds] = dataset_ids.get(ds, 0) + 1

        # Query registry using most common OOS period first, then dataset
        if oos_periods:
            primary_oos = max(oos_periods, key=oos_periods.get)
            count = registry.oos_reuse_count(primary_oos)
            if count > 0:
                return count

        if dataset_ids:
            primary_ds = max(dataset_ids, key=dataset_ids.get)
            # list_by_oos uses oos_period; try dataset as OOS period fallback
            # (some workflows may store dataset_id in oos_period field)
            count = registry.oos_reuse_count(primary_ds)
            if count > 0:
                return count

        # Fallback: total experiments in registry (broad hypothesis family)
        total = len(registry._index)
        if total > 0:
            return total

        # Ultimate fallback: batch size
        return len(evaluations)

    # ---------- Full assessment ----------
    def assess(self, evaluations: dict[str, dict], registry: Optional["ExperimentRegistry"] = None) -> IntegrityReport:
        n_trials = self._get_global_n_trials(evaluations, registry)
        # Family-wise WRC/SPA: compute ONCE for the whole batch (Gate 2)
        # Needed when supervisor tests 100/1000/10000 strategies.
        wrc_global: dict[str, Any] | None = None
        try:
            wrc_global = self._compute_wrc_global(evaluations, n_trials)
        except Exception as e:  # noqa: BLE001
            wrc_global = {"attempted": True, "p_value": None, "error": str(e), "skipped": "exception"}

        candidates: dict[str, CandidateIntegrityReport] = {}
        eligible: dict[str, dict] = {}
        rejected: dict[str, tuple[str, ...]] = {}

        for sid, ev in evaluations.items():
            g1 = self._gate_integrity(sid, ev)
            g2 = self._gate_statistical(sid, ev, n_trials, wrc_global) if g1.passed else GateResult(False, ("skipped: integrity failed",), {})
            g_is = self._gate_is_oos(sid, ev, n_trials) if g1.passed and g2.passed else GateResult(False, ("skipped: prior gate failed",), {})
            g_ml = self._gate_ml_calibration(sid, ev) if g1.passed and g2.passed and g_is.passed else GateResult(False, ("skipped: prior gate failed",), {})
            g3 = self._gate_robustness(sid, ev) if g1.passed and g2.passed and g_is.passed and g_ml.passed else GateResult(False, ("skipped: prior gate failed",), {})
            g4 = self._gate_selection(sid, ev, n_trials) if g1.passed and g2.passed and g_is.passed and g_ml.passed and g3.passed else GateResult(False, ("skipped",), {})
            g_regime = self._gate_regime_stability(sid, ev) if g1.passed and g2.passed and g_is.passed and g_ml.passed and g3.passed and g4.passed else GateResult(False, ("skipped",), {})
            g_edge = self._gate_robust_oos_edge(sid, ev) if g1.passed and g2.passed and g_is.passed and g_ml.passed and g3.passed and g4.passed and g_regime.passed else GateResult(False, ("skipped",), {})

            overall = g1.passed and g2.passed and g_is.passed and g_ml.passed and g3.passed and g4.passed and g_regime.passed and g_edge.passed
            failed_stage = None
            if not g1.passed:
                failed_stage = "Integrity Checks"
            elif not g2.passed:
                failed_stage = "Statistical Validation"
            elif not g_is.passed:
                failed_stage = "IS-OOS Consistency"
            elif not g_ml.passed:
                failed_stage = "ML Calibration"
            elif not g3.passed:
                failed_stage = "Robustness"
            elif not g4.passed:
                failed_stage = "Selection Adjustment"
            elif not g_regime.passed:
                failed_stage = "Regime Stability"
            elif not g_edge.passed:
                failed_stage = "Robust OOS Edge"

            reasons: tuple[str, ...] = ()
            if not overall:
                # Collect reasons from first failed gate
                if not g1.passed:
                    reasons = g1.reasons
                elif not g2.passed:
                    reasons = g2.reasons
                elif not g_is.passed:
                    reasons = g_is.reasons
                elif not g_ml.passed:
                    reasons = g_ml.reasons
                elif not g3.passed:
                    reasons = g3.reasons
                elif not g4.passed:
                    reasons = g4.reasons
                elif not g_regime.passed:
                    reasons = g_regime.reasons
                else:
                    reasons = g_edge.reasons

            # Selection-adjusted metrics for Tournament (use corrected Sharpe)
            adjusted = dict(ev.get("metrics", {}))
            # Inject corrected Sharpe for tournament if available
            if g4.details.get("sharpe_corrected") is not None:
                adjusted["sharpe_median_corrected"] = g4.details["sharpe_corrected"]
                # Tournament will use sharpe_median; we provide corrected value via extra key
                # The pipeline's vector_to_tournament will be updated to prefer corrected if present
            # Ensure Tournament's NOT_ELIGIBLE gate is satisfied for integrity-passed pool.
            # Integrity already enforced robustness; inject neutral MC/stress so Tournament
            # does not discard integrity-approved candidates due to missing validation artifacts.
            if overall:
                if "monte_carlo_score" not in adjusted or adjusted["monte_carlo_score"] is None:
                    adjusted["monte_carlo_score"] = 0.5 if self.config.permissive else 0.0
                if "stress_score" not in adjusted or adjusted["stress_score"] is None:
                    adjusted["stress_score"] = 0.5 if self.config.permissive else 0.0
            gate_details = {
                "integrity": g1,
                "statistical": g2,
                "is_oos": g_is,
                "ml_calibration": g_ml,
                "robustness": g3,
                "selection": g4,
                "regime_stability": g_regime,
                "robust_oos_edge": g_edge,
            }

            report = CandidateIntegrityReport(
                strategy_id=sid,
                integrity_passed=g1.passed,
                statistical_passed=g2.passed,
                is_oos_passed=g_is.passed,
                ml_calibration_passed=g_ml.passed,
                robustness_passed=g3.passed,
                selection_passed=g4.passed,
                regime_stability_passed=g_regime.passed,
                robust_oos_edge_passed=g_edge.passed,
                overall_passed=overall,
                failed_stage=failed_stage,
                reasons=reasons,
                gate_details=gate_details,
                adjusted_metrics=adjusted,
            )
            candidates[sid] = report
            if overall:
                # Tournament sees only adjusted metrics
                ev_copy = dict(ev)
                ev_copy["metrics"] = adjusted
                # Preserve windows etc.
                eligible[sid] = ev_copy
            else:
                rejected[sid] = reasons

        return IntegrityReport(
            candidates=candidates,
            eligible=eligible,
            rejected=rejected,
            n_trials=n_trials,
            deflated_correction_applied=self.config.apply_deflated_correction,
        )

    def filter_eligible(self, evaluations: dict[str, dict], registry: Optional["ExperimentRegistry"] = None) -> tuple[dict[str, dict], IntegrityReport]:
        report = self.assess(evaluations, registry)
        return report.eligible, report
