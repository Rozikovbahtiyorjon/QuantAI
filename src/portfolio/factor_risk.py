"""
P2.18 Factor-adjusted Risk — gross/net exposure, crypto beta, correlation, factor concentration

Computes:
- gross exposure = sum |w_i|
- net exposure = sum w_i (signed, long - short)
- crypto beta = w' * beta_vector (beta vs BTC or market)
- correlation = avg pairwise |corr|
- factor concentration = Herfindahl index of factor exposures
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict
from dataclasses import dataclass


@dataclass
class FactorRiskReport:
    gross_exposure: float
    net_exposure: float
    beta: float
    avg_correlation: float
    factor_concentration: float  # 0 diversified, 1 concentrated
    factor_exposures: Dict[str, float]
    passed: bool
    reason: str


def compute_factor_risk(
    weights: Dict[str, float],
    returns: pd.DataFrame,
    betas: Dict[str, float] | None = None,
    factor_map: Dict[str, str] | None = None,
    max_gross: float = 1.5,
    max_beta: float = 0.6,
    max_concentration: float = 0.5,
) -> FactorRiskReport:
    """
    Args:
        weights: symbol -> weight (signed, e.g. BTC 0.3, ETH -0.1). Gross = sum |w|
        returns: DataFrame of returns for correlation/beta estimation
        betas: symbol -> beta vs market (BTC). If None, estimated vs BTC or first column.
        factor_map: symbol -> factor (e.g. BTC-> Layer1, ETH->Layer1). For concentration.
        max_gross, max_beta, max_concentration: thresholds for passed
    """
    if not weights:
        return FactorRiskReport(0,0,0,0,0,{}, True, "empty portfolio")
    gross = float(sum(abs(v) for v in weights.values()))
    net = float(sum(weights.values()))
    # Beta
    if betas is None and not returns.empty:
        # Estimate beta vs BTC (or market = equal-weight)
        # Use BTC as market proxy if present, else first column
        market_col = "BTCUSDT" if "BTCUSDT" in returns.columns else returns.columns[0]
        market_ret = returns[market_col].dropna()
        betas_est = {}
        for sym in weights:
            if sym in returns.columns:
                s_ret = returns[sym].dropna()
                # Align
                aligned = pd.concat([s_ret, market_ret], axis=1, join='inner').dropna()
                if len(aligned) > 30:
                    cov = aligned.cov().iloc[0,1]
                    var_m = aligned[market_col].var()
                    beta = cov / var_m if var_m>1e-9 else 1.0
                else:
                    beta = 1.0
                betas_est[sym] = beta
            else:
                betas_est[sym] = 1.0
        betas = betas_est
    if betas:
        beta = float(sum(weights[sym] * betas.get(sym, 1.0) for sym in weights))
    else:
        beta = float(net)  # fallback
    # Correlation
    avg_corr = 0.0
    if not returns.empty and len(weights) > 1:
        cols = [c for c in weights if c in returns.columns]
        if len(cols) > 1:
            corr = returns[cols].corr().values
            # avg off-diagonal
            mask = ~np.eye(len(cols), dtype=bool)
            avg_corr = float(np.mean(np.abs(corr[mask]))) if mask.any() else 0.0
    # Factor concentration (Herfindahl)
    factor_exposures: Dict[str, float] = {}
    if factor_map:
        for sym, w in weights.items():
            f = factor_map.get(sym, "Other")
            factor_exposures[f] = factor_exposures.get(f, 0) + abs(w)
        # Normalize to gross
        if gross > 0:
            for k in factor_exposures:
                factor_exposures[k] /= gross
        # Herfindahl = sum p_i^2
        hhi = float(sum(v**2 for v in factor_exposures.values())) if factor_exposures else 0.0
    else:
        # Without factor map, concentration = Herfindahl of weights
        if gross>0:
            p = [abs(v)/gross for v in weights.values()]
            hhi = float(sum(x*x for x in p))
        else:
            hhi = 0.0
        factor_exposures = {k: abs(v)/gross if gross else 0 for k,v in weights.items()}
    # Pass conditions
    passed = (gross <= max_gross) and (abs(beta) <= max_beta) and (hhi <= max_concentration)
    reasons = []
    if gross > max_gross:
        reasons.append(f"gross {gross:.2f}>{max_gross}")
    if abs(beta) > max_beta:
        reasons.append(f"beta {beta:.2f}>{max_beta}")
    if hhi > max_concentration:
        reasons.append(f"concentration {hhi:.2f}>{max_concentration}")
    reason = "; ".join(reasons) if reasons else "OK"
    return FactorRiskReport(
        gross_exposure=gross,
        net_exposure=net,
        beta=beta,
        avg_correlation=avg_corr,
        factor_concentration=hhi,
        factor_exposures=factor_exposures,
        passed=passed,
        reason=reason,
    )
