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

    def __post_init__(self) -> None:
        if self.lookback_days < 1:
            raise ValueError("lookback_days must be >= 1")
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")
        if self.rebalance_days < 1:
            raise ValueError("rebalance_days must be >= 1")
        if self.fee_per_side < 0:
            raise ValueError("fee_per_side cannot be negative")


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
    prev_set: set[str] = set()

    t = p.lookback_days
    while t < n - 1:
        # ---- ranking: causal (uses rows <= t) ----
        row_now = prices.iloc[t]
        row_past = prices.iloc[t - p.lookback_days]

        mom = (row_now / row_past - 1.0).dropna()

        if len(mom) == 0:
            t += p.rebalance_days
            continue

        picked = list(mom.nlargest(min(p.top_k, len(mom))).index)

        # ---- holding period returns ----
        seg = rets.iloc[t + 1 : t + 1 + p.rebalance_days][picked]
        daily_port = seg.mean(axis=1, skipna=True).dropna()

        gross_period = float((1.0 + daily_port).prod() - 1.0) if len(daily_port) else 0.0

        # ---- costs on turnover ----
        new_set = set(picked)
        turned_over = len(new_set.symmetric_difference(prev_set))
        changed_frac = turned_over / max(len(new_set | prev_set), 1)
        cost = p.fee_per_side * 2.0 * changed_frac * len(picked) / max(p.top_k, 1)
        net_period = gross_period - cost
        prev_set = new_set

        eq_prev = equity[-1]
        equity.append(eq_prev * (1.0 + net_period))

        periods.append({
            "start": prices.index[t],
            "end": prices.index[min(t + p.rebalance_days, n - 1)],
            "picked": picked,
            "gross": gross_period,
            "cost": cost,
            "net": net_period,
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
