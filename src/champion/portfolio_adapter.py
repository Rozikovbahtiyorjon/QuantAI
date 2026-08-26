"""
Portfolio-candidate adapter (P-C1 -> R4 contract).

Bridges cross-sectional (multi-symbol) strategies into the champion
evaluation schema: sequential NON-overlapping OOS blocks replace the
single-symbol WF windows; aggregation reuses the same function, so
PromotionRules / Tournament / ChampionEvaluator work unchanged.
"""

from __future__ import annotations

import pandas as pd

from src.champion.evaluation_pipeline import aggregate_windows
from src.strategies.cross_sectional import CrossSectionParams, backtest


def evaluate_portfolio_candidate(
    spec,
    wide_prices: pd.DataFrame,
    *,
    test_days: int = 180,
    initial_balance: float = 10_000.0,
) -> dict:
    """
    spec.factory() must return a CrossSectionParams instance
    (params-as-strategy convention for portfolio candidates).

    Sequential OOS blocks of `test_days`; no fitting anywhere, so
    every block is honest out-of-sample by construction.
    """

    params = spec.factory()
    if not isinstance(params, CrossSectionParams):
        raise TypeError(
            "portfolio candidate factory must return CrossSectionParams"
        )

    prices = wide_prices.sort_index()
    n = len(prices)
    if n <= test_days:
        raise ValueError("not enough history for the requested test_days")

    window_stats = []
    start = 0
    while start + test_days <= n:
        block = prices.iloc[max(0, start - params.lookback_days - 5): start + test_days]
        # small pre-window so lookback has data on the first rebalance;
        # PnL accounting still only spans in-block period via backtest()
        full = wide_prices_slice_stats(block, params)
        window_stats.append(full)
        start += test_days

    return {"metrics": aggregate_windows(window_stats), "windows": window_stats}


def wide_prices_slice_stats(block: pd.DataFrame, params: CrossSectionParams) -> dict:
    res = backtest(block, params)
    s = res["stats"]

    net_pct = s["total_ret_pct"]

    return {
        "net_pct": net_pct,
        "pf": s["pf"],
        "maxdd_pct": s["maxdd_pct"],
        "sharpe": s["sharpe"],
        "trades": s["trades"],
        "wins": s["wins"],
        "periods": s["periods_n"],
    }


__all__ = ["evaluate_portfolio_candidate"]
