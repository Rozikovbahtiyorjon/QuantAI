"""
Risk correlation — Audit #26-27: Correlation-adjusted exposure & factor exposure.

Prevents BTC+ETH+SOL long being treated as 3 independent positions when they are 1 CRYPTO BETA factor.
Computes gross/net exposure, margin usage, liquidation distance for leveraged portfolios.

Enhanced Task 7: also provides gross/net, beta to market, factor concentration (Herfindahl/max weight)
via src.risk.factor_risk canonical implementation. This module re-exports for backward compat.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class CorrelationExposure:
    gross_exposure: float
    net_exposure: float
    correlation_adjusted_exposure: float
    factor_exposure: Dict[str, float]
    max_correlation: float
    warning: str = ""
    # Task 7 extensions (optional, for enriched report)
    factor_weights: Dict[str, float] | None = None
    max_factor_weight: float = 0.0
    herfindahl: float = 0.0
    portfolio_beta: float = 0.0


# --- Gross / Net exposure helpers (Task 7) ---------------------------------

def compute_gross_exposure(positions: Dict[str, float]) -> float:
    """Gross exposure = sum |notional|."""
    return float(sum(abs(float(v)) for v in positions.values()))


def gross_exposure(positions: Dict[str, float]) -> float:
    return compute_gross_exposure(positions)


def compute_net_exposure(positions: Dict[str, float]) -> float:
    """Net exposure = sum signed notional."""
    return float(sum(float(v) for v in positions.values()))


def net_exposure(positions: Dict[str, float]) -> float:
    return compute_net_exposure(positions)


# --- Beta to market (Task 7) ------------------------------------------------

def compute_beta(asset_returns: pd.Series, market_returns: pd.Series) -> float:
    """Beta = Cov(asset, market)/Var(market). Delegates to factor_risk."""
    try:
        from src.risk.factor_risk import compute_beta as _cb
        return _cb(asset_returns, market_returns)
    except Exception:
        if len(asset_returns) < 2 or len(market_returns) < 2:
            return 0.0
        try:
            a, m = asset_returns.align(market_returns, join="inner")
            a = a.dropna()
            m = m.dropna()
            if len(a) < 2 or len(m) < 2:
                return 0.0
            cov = float(a.cov(m))
            var = float(m.var())
            if var == 0 or not math.isfinite(cov) or not math.isfinite(var):
                return 0.0
            beta = cov / var
            return float(beta) if math.isfinite(beta) else 0.0
        except Exception:
            return 0.0


def beta_to_market(asset_returns: pd.Series, market_returns: pd.Series) -> float:
    return compute_beta(asset_returns, market_returns)


def compute_portfolio_beta(
    positions: Dict[str, float],
    betas: Dict[str, float],
    market_symbol: str = "BTC",
) -> float:
    try:
        from src.risk.factor_risk import compute_portfolio_beta as _cpb
        return _cpb(positions, betas, market_symbol=market_symbol)
    except Exception:
        gross = compute_gross_exposure(positions)
        if gross == 0:
            return 0.0
        total = sum(abs(float(w)) * float(betas.get(sym, 1.0)) for sym, w in positions.items())
        return float(total / gross)


def crypto_beta(positions: Dict[str, float], betas: Optional[Dict[str, float]] = None) -> float:
    return compute_portfolio_beta(positions, betas or {})


# --- Factor exposure & concentration (Task 7) --------------------------------

def compute_factor_exposure(
    positions: Dict[str, float],
    factor_map: Optional[Dict[str, str]] = None,
) -> Dict[str, float]:
    try:
        from src.risk.factor_risk import compute_factor_exposure as _cfe
        return _cfe(positions, factor_map)
    except Exception:
        factor_map = factor_map or {}
        exp: Dict[str, float] = {}
        for s, n in positions.items():
            f = factor_map.get(s, s)
            exp[f] = exp.get(f, 0.0) + float(n)
        return exp


def factor_exposure(positions: Dict[str, float], factor_map: Optional[Dict[str, str]] = None) -> Dict[str, float]:
    return compute_factor_exposure(positions, factor_map)


def compute_factor_concentration(
    positions: Dict[str, float],
    factor_map: Optional[Dict[str, str]] = None,
    gross: Optional[float] = None,
) -> Dict[str, float]:
    try:
        from src.risk.factor_risk import compute_factor_concentration as _cfc
        return _cfc(positions, factor_map, gross=gross)
    except Exception:
        f_exp = compute_factor_exposure(positions, factor_map)
        g = gross if gross is not None else compute_gross_exposure(positions)
        if g == 0:
            g = sum(abs(v) for v in f_exp.values()) or 1.0
        weights = {k: abs(v) / g for k, v in f_exp.items()}
        max_w = max(weights.values()) if weights else 0.0
        hhi = sum(w * w for w in weights.values())
        return {"max_factor_weight": float(max_w), "herfindahl": float(hhi), "n_factors": len(weights), "factor_exposure": f_exp, "factor_weights": weights}


def factor_concentration(positions: Dict[str, float], factor_map: Optional[Dict[str, str]] = None) -> Dict[str, float]:
    return compute_factor_concentration(positions, factor_map)


def herfindahl_index(factor_weights: Dict[str, float]) -> float:
    try:
        from src.risk.factor_risk import herfindahl_index as _hhi
        return _hhi(factor_weights)
    except Exception:
        return float(sum(float(w) ** 2 for w in factor_weights.values()))


def max_factor_weight(factor_weights: Dict[str, float]) -> float:
    if not factor_weights:
        return 0.0
    return float(max(factor_weights.values()))


def correlation_adjusted_exposure(
    positions: Dict[str, float],  # symbol -> notional pct (e.g., BTC 0.05 = 5%)
    corr_matrix: pd.DataFrame,  # symbols x symbols correlation matrix
    factor_map: Dict[str, str] | None = None,  # symbol -> factor (e.g., BTC -> CRYPTO_BETA)
) -> CorrelationExposure:
    """Adjust exposure for correlation; returns factor exposures.
    
    Enhanced with factor concentration (Herfindahl/max weight) and portfolio beta.
    """
    symbols = list(positions.keys())
    if len(symbols) <= 1:
        ge = compute_gross_exposure(positions)
        ne = compute_net_exposure(positions)
        # factor exposure for single
        f_exp = compute_factor_exposure(positions, factor_map)
        conc = compute_factor_concentration(positions, factor_map, gross=ge)
        return CorrelationExposure(
            gross_exposure=ge,
            net_exposure=ne,
            correlation_adjusted_exposure=ge,
            factor_exposure=f_exp,
            max_correlation=0.0,
            warning="",
            factor_weights=conc["factor_weights"],
            max_factor_weight=conc["max_factor_weight"],
            herfindahl=conc["herfindahl"],
            portfolio_beta=0.0,
        )

    # Build vector and sub-matrix
    vec = np.array([positions[s] for s in symbols], dtype=float)
    try:
        sub = corr_matrix.loc[symbols, symbols].values.astype(float)
        sub = np.nan_to_num(sub, nan=0.0)
        sub = np.clip(sub, -1.0, 1.0)
        for i in range(len(symbols)):
            sub[i, i] = 1.0
    except Exception as e:
        # Fail-closed: missing correlation data for multi-asset must not assume diversified (eye → 0)
        # Assume worst-case high correlation 0.9 for unknown pair (conservative, not eye)
        sub = np.full((len(symbols), len(symbols)), 0.9, dtype=float)
        for i in range(len(symbols)):
            sub[i, i] = 1.0

    # Factor exposure
    factor_exp = compute_factor_exposure(positions, factor_map)
    conc = compute_factor_concentration(positions, factor_map, gross=float(np.sum(np.abs(vec))))

    max_corr = float(np.max(sub[np.triu_indices(len(symbols), k=1)])) if len(symbols) > 1 else 0.0
    warning = ""
    if max_corr > 0.85:
        warning = f"High correlation {max_corr:.2f} — positions are single factor, reduce size"
    if any(abs(v) > 0.15 for v in factor_exp.values()):
        warning += " | Factor exposure >15%"
    if conc["max_factor_weight"] > 0.70:
        warning += f" | Factor concentration {conc['max_factor_weight']:.0%} >70%"
    if conc["herfindahl"] > 0.60:
        warning += f" | Herfindahl {conc['herfindahl']:.2f} >0.60"

    return CorrelationExposure(
        gross_exposure=float(np.sum(np.abs(vec))),
        net_exposure=float(np.sum(vec)),
        correlation_adjusted_exposure=corr_adj,
        factor_exposure=factor_exp,
        max_correlation=max_corr,
        warning=warning,
        factor_weights=conc["factor_weights"],
        max_factor_weight=conc["max_factor_weight"],
        herfindahl=conc["herfindahl"],
        portfolio_beta=0.0,
    )


def liquidation_distance(entry: float, position_size: float, leverage: float, maintenance_margin: float = 0.005) -> float:
    """Distance to liquidation for cross-margin long/short."""
    if leverage <= 0 or entry <= 0:
        return float("inf")
    # Simplified: liquidation = entry * (1 - 1/leverage + maintenance_margin) for long
    # For short: entry * (1 + 1/leverage - maintenance_margin)
    # Here signed position_size determines direction
    if position_size > 0:  # long
        liq = entry * (1 - 1 / leverage + maintenance_margin)
        return (entry - liq) / entry
    else:  # short
        liq = entry * (1 + 1 / leverage - maintenance_margin)
        return (liq - entry) / entry
