"""
Portfolio-candidate adapter (P-C1 -> R4 contract).

Bridges cross-sectional (multi-symbol) strategies into the champion
evaluation schema: sequential NON-overlapping OOS blocks replace the
single-symbol WF windows; aggregation reuses the same function, so
PromotionRules / Tournament / ChampionEvaluator work unchanged.

FIX task-6: separate portfolio observations vs actual fills / round trips.
  - window_stats['trades'] / aggregated metrics['trades'] MUST be
    actual completed round trips (actual_trades), not slot observations.
  - n_observations / n_fills / fill_rate exposed for transparency but
    NOT used by integrity gates.
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

    # aggregate_windows sums 'trades'/'wins' and computes win_rate;
    # since wide_prices_slice_stats now maps trades -> actual_trades
    # (round trips), the aggregated metrics are comparable to
    # single-asset backtest (TradeEngine.closed_positions).
    metrics = aggregate_windows(window_stats)

    # surface transparent slot vs fill diagnostics aggregated across windows
    # (not used by gates, but useful for reporting)
    try:
        total_obs = sum(w.get("n_observations", 0) for w in window_stats)
        total_fills = sum(w.get("n_fills", 0) for w in window_stats)
        total_actual = sum(w.get("actual_trades", w.get("trades", 0)) for w in window_stats)
        metrics["n_observations"] = total_obs
        metrics["n_fills"] = total_fills
        metrics["actual_trades"] = total_actual
        metrics["fill_rate"] = round(total_fills / total_obs, 4) if total_obs else 0.0
        # also expose pf_on_fills median for convenience
        pfs = [w.get("pf_on_fills", w.get("pf", 0.0)) for w in window_stats]
        if pfs:
            import statistics
            metrics["pf_on_fills_median"] = statistics.median(pfs)
        wrs = [w.get("win_rate_on_fills", 0.0) for w in window_stats]
        if wrs:
            import statistics
            metrics["win_rate_on_fills_median"] = statistics.median(wrs)
    except Exception:
        pass

    return {"metrics": metrics, "windows": window_stats}


def wide_prices_slice_stats(block: pd.DataFrame, params: CrossSectionParams) -> dict:
    res = backtest(block, params)
    s = res["stats"]

    net_pct = s["total_ret_pct"]

    # Integrity gates must use actual fills (round trips), not observations.
    # s["trades"] / s["wins"] are already mapped to actual_trades / wins_on_fills
    # inside cross_sectional.backtest for comparability.
    return {
        "net_pct": net_pct,
        "pf": s.get("pf_on_fills", s["pf"]),
        "pf_on_fills": s.get("pf_on_fills", s["pf"]),
        "pf_period": s.get("pf_period", s["pf"]),
        "maxdd_pct": s["maxdd_pct"],
        "sharpe": s["sharpe"],
        # --- gate-facing (round trips) ---
        "trades": s.get("actual_trades", s["trades"]),
        "actual_trades": s.get("actual_trades", s["trades"]),
        "wins": s.get("wins_on_fills", s["wins"]),
        "wins_on_fills": s.get("wins_on_fills", s["wins"]),
        "win_rate_on_fills": s.get("win_rate_on_fills", s.get("win_rate", 0.0)),
        # --- transparency (slot observations, not used by gates) ---
        "n_observations": s.get("n_observations", 0),
        "n_fills": s.get("n_fills", 0),
        "fill_rate": s.get("fill_rate", 0.0),
        "portfolio_rebalance_observations": s.get("portfolio_rebalance_observations", s.get("periods_n", 0)),
        "n_periods": s.get("n_periods", s.get("periods_n", 0)),
        "periods": s["periods_n"],
    }


__all__ = ["evaluate_portfolio_candidate", "wide_prices_slice_stats"]
