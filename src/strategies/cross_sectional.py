"""
QuantAI Cross-Sectional Momentum (P-C1)

Portfolio-class strategy: periodically rank assets by trailing total
return and hold the top-K equally weighted until next rebalance.

Causality by construction:
    ranks use prices up to and INCLUDING the rebalance bar;
    period PnL uses strictly LATER bars.

This module has its own compact vectorized backtester because the
single-symbol TradeEngine cannot express cross-sectional portfolios.
Results are shaped to plug into the R4 champion contract
(same window-aggregation schema as evaluate_candidate).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CrossSectionParams:
    lookback_days: int = 14        # trailing return used for ranking
    top_k: int = 2                 # number of assets held
    rebalance_days: int = 7        # holding period length
    fee_per_side: float = 0.0005   # conservative per-side cost

    # ---- Risk layer (P-C1.A) ----
    # Volatility targeting: scale exposure so that the held basket's
    # trailing annualized vol ~= target. None disables.
    target_ann_vol: float | None = None
    vol_lookback_days: int = 30

    # Drawdown gate: when equity drawdown breaches soft stop, stay FLAT
    # at subsequent rebalances until drawdown recovers above re-entry.
    # None disables.
    dd_soft_stop_pct: float | None = None      # e.g. -25.0
    dd_reentry_pct: float | None = None        # e.g. -12.5

    # Intra-basket weighting: 'equal' or 'inv_vol' (trailing per-name
    # volatility weighting — smooths basket WITHOUT market timing).
    weighting: str = "equal"
    weight_vol_days: int = 30

    def __post_init__(self) -> None:
        if self.lookback_days < 1:
            raise ValueError("lookback_days must be >= 1")
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")
        if self.rebalance_days < 1:
            raise ValueError("rebalance_days must be >= 1")
        if self.fee_per_side < 0:
            raise ValueError("fee_per_side cannot be negative")
        if self.target_ann_vol is not None and self.target_ann_vol <= 0:
            raise ValueError("target_ann_vol must be positive")
        if self.dd_soft_stop_pct is not None:
            if self.dd_reentry_pct is None:
                raise ValueError("dd_reentry_pct required with dd_soft_stop_pct")
            if not (self.dd_reentry_pct > self.dd_soft_stop_pct):
                raise ValueError(
                    "dd_reentry_pct must be closer to zero than dd_soft_stop_pct"
                )
        if self.weighting not in ("equal", "inv_vol"):
            raise ValueError("weighting must be 'equal' or 'inv_vol'")
        if self.weight_vol_days < 2:
            raise ValueError("weight_vol_days must be >= 2")


def backtest(prices: pd.DataFrame, params: CrossSectionParams) -> dict:
    """
    Run cross-sectional momentum over a wide price frame
    (index=dates ascending, columns=symbols).

    Returns dict with:
        equity      Series (net, starts at 1.0)
        periods     list[dict] per holding period
        stats       {total_ret_pct, maxdd_pct, sharpe, pf, trades, wins}
    """
    p = params
    prices = prices.sort_index()
    rets = prices.pct_change(fill_method=None)

    n = len(prices)
    if n <= p.lookback_days + p.rebalance_days:
        raise ValueError("not enough history for given lookback/rebalance")

    equity = [1.0]
    periods: list[dict] = []
    prev_weights: dict[str, float] = {}
    in_dd_pause = False

    t = p.lookback_days
    while t < n - 1:
        # ---- drawdown gate (state machine, causal) ----
        if p.dd_soft_stop_pct is not None:
            peak = max(equity)
            dd_now = (equity[-1] / peak - 1.0) * 100.0

            if in_dd_pause:
                if dd_now >= p.dd_reentry_pct:
                    in_dd_pause = False
            elif dd_now <= p.dd_soft_stop_pct:
                in_dd_pause = True

        # ---- ranking: causal (uses rows <= t) ----
        row_now = prices.iloc[t]
        row_past = prices.iloc[t - p.lookback_days]

        mom = (row_now / row_past - 1.0).dropna()

        if len(mom) == 0 or in_dd_pause:
            equity.append(equity[-1])          # flat bar (cash)
            periods.append({
                "start": prices.index[t],
                "end": prices.index[min(t + p.rebalance_days, n - 1)],
                "picked": [],
                "gross": 0.0,
                "cost": 0.0,
                "net": 0.0,
                "flat": True,
            })
            prev_weights = {}                   # full exit on next entry
            t += p.rebalance_days
            continue

        picked = list(mom.nlargest(min(p.top_k, len(mom))).index)

        # ---- intra-basket weights (causal) ----
        if p.weighting == "inv_vol":
            hist = rets.iloc[max(0, t - p.weight_vol_days): t][picked]
            vols = hist.std(ddof=1).replace(0, np.nan).dropna()
            if len(vols) == len(picked):
                inv = 1.0 / vols
                weights = (inv / inv.sum()).to_dict()
            else:
                weights = {s: 1.0 / len(picked) for s in picked}
        else:
            weights = {s: 1.0 / len(picked) for s in picked}

        # ---- vol targeting scale (causal: trailing returns only) ----
        scale = 1.0
        if p.target_ann_vol is not None:
            lb_vol = min(p.vol_lookback_days, t)
            hist = rets.iloc[t - lb_vol : t][picked].mean(axis=1, skipna=True).dropna()
            if len(hist) > 2:
                realized_daily = float(hist.std(ddof=1))
                realized_ann = realized_daily * math.sqrt(365.0)
                if realized_ann > 0:
                    scale = min(1.0, float(p.target_ann_vol) / realized_ann)

        # ---- holding period returns (BUY & HOLD semantics) ----
        # Static weights fixed at entry; no intra-week micro-rebalancing.
        # Matches live broker: enter at close(t), exit at close(t+rb).
        end_i = min(t + p.rebalance_days, n - 1)
        p_start = prices.iloc[t][picked].astype(float)
        p_end = prices.iloc[end_i][picked].astype(float)

        gross_period = float(
            (
                (p_end / p_start - 1.0)
                * pd.Series(weights)
            ).dropna().sum()
        )

        # ---- costs on weight turnover ----
        turned = sum(abs(weights.get(s, 0.0) - prev_weights.get(s, 0.0))
                     for s in set(weights) | set(prev_weights))
        cost = p.fee_per_side * 2.0 * turned
        net_period = (gross_period - cost) * scale
        prev_weights = weights

        eq_prev = equity[-1]
        equity.append(eq_prev * (1.0 + net_period))

        periods.append({
            "start": prices.index[t],
            "end": prices.index[min(t + p.rebalance_days, n - 1)],
            "picked": picked,
            "gross": gross_period,
            "cost": cost,
            "net": net_period,
            "scale": round(scale, 4),
            "flat": False,
        })

        t += p.rebalance_days

    eq = pd.Series(equity[1:] if len(equity) > 1 else equity,
                   index=[per["end"] for per in periods]) if periods else \
          pd.Series([1.0], index=[prices.index[-1]])

    total_ret = (eq.iloc[-1] / 1.0 - 1.0) * 100.0 if len(eq) else 0.0

    # drawdown
    peaks = eq.cummax()
    dd = (eq / peaks - 1.0) * 100.0
    maxdd = float(dd.min()) if len(dd) else 0.0

    # sharpe from period returns (weekly-scale annualization)
    nets = np.array([per["net"] for per in periods])
    if len(nets) > 1 and np.std(nets, ddof=1) > 0:
        per_year = 365.0 / p.rebalance_days
        sharpe = float(
            nets.mean() / np.std(nets, ddof=1) * math.sqrt(per_year)
        )
    else:
        sharpe = 0.0

    pos_sum = nets[nets > 0].sum() if len(nets) else 0.0
    neg_sum = abs(nets[nets < 0].sum()) if len(nets) else 0.0
    pf = pos_sum / neg_sum if neg_sum > 0 else (99.0 if pos_sum > 0 else 0.0)

    trades = len(periods) * p.top_k              # asset-slot activity metric
    slot_rets = []
    for per in periods:
        slot_rets.extend([per["net"]] * p.top_k)  # conservative slot proxy
    wins = int(sum(1 for r in slot_rets if r > 0))

    stats_out = {
        "total_ret_pct": round(total_ret, 4),
        "maxdd_pct": round(maxdd, 4),
        "sharpe": round(sharpe, 3),
        "pf": round(pf, 3),
        "trades": trades,
        "wins": wins,
        "periods_n": len(periods),
    }

    return {"equity": eq, "periods": periods, "stats": stats_out}


__all__ = ["CrossSectionParams", "backtest"]
