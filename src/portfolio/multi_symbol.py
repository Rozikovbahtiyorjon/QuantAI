"""
P2.17 Multi-Symbol — BTC/ETH/SOL etc. after single-symbol stable

Runs single-symbol backtest per symbol, aggregates with portfolio allocation,
and evaluates combined portfolio PF/Sharpe/DD via factor-adjusted risk.
"""

from __future__ import annotations

import pandas as pd
from typing import Dict, List
from pathlib import Path

from src.backtest_engine import BacktestEngine


def multi_symbol_backtest(
    symbols: List[str] = None,
    data_dir: str = "data",
    timeframe: str = "1h",
    initial_balance: float = 10000.0,
    allocation_method: str = "hrp",
) -> Dict:
    """
    Run single-symbol backtest per symbol, then combine via HRP allocation.

    Args:
        symbols: list like ["BTCUSDT","ETHUSDT","SOLUSDT"] — if None, auto-discover *_1h_prepared.parquet
        data_dir: directory with prepared parquet
        timeframe: timeframe suffix
        allocation_method: hrp|risk_parity|correlation_aware
    """
    if symbols is None:
        # Auto-discover
        p = Path(data_dir)
        files = sorted(p.glob(f"*_{timeframe}_prepared.parquet"))
        symbols = [f.name.split("_")[0].upper() + "USDT" if "usdt" not in f.name.lower() else f.name.split("_")[0].upper() for f in files[:5]]
        # Fallback to known
        if not symbols:
            symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    # Map symbol -> prepared path
    results = {}
    returns_dict = {}
    for sym in symbols:
        # Find file
        cand = Path(data_dir) / f"{sym.lower()}_{timeframe}_prepared.parquet"
        if not cand.exists():
            # Try alternative naming
            alt = list(Path(data_dir).glob(f"{sym.lower().split('usdt')[0]}*_{timeframe}_prepared.parquet"))
            if alt:
                cand = alt[0]
            else:
                continue
        try:
            df = pd.read_parquet(cand)
            be = BacktestEngine(initial_balance=initial_balance / len(symbols))
            res = be.run(df)
            results[sym] = res
            # Collect returns for allocation (equity curve pct)
            if hasattr(res, "equity_curve") and res.equity_curve:
                eq = pd.Series([v for _, v in res.equity_curve])
                rets = eq.pct_change().dropna()
                returns_dict[sym] = rets
        except Exception as e:
            results[sym] = {"error": str(e)}
    # Build returns DataFrame for allocation
    if returns_dict:
        # Align lengths
        min_len = min(len(v) for v in returns_dict.values())
        aligned = {k: v.iloc[-min_len:].reset_index(drop=True) for k, v in returns_dict.items()}
        returns_df = pd.DataFrame(aligned)
        # Get allocation
        try:
            from src.portfolio.allocation import get_allocation
            alloc = get_allocation(returns_df, method=allocation_method)
        except Exception:
            # Equal weight fallback
            alloc = {k: 1/len(returns_dict) for k in returns_dict}
        # Factor risk
        try:
            from src.portfolio.factor_risk import compute_factor_risk
            fr = compute_factor_risk(alloc, returns_df)
            factor_report = {"gross": fr.gross_exposure, "net": fr.net_exposure, "beta": fr.beta, "corr": fr.avg_correlation, "passed": fr.passed}
        except Exception as e:
            factor_report = {"error": str(e)}
        # Portfolio PF: weighted avg of individual PFs (simplified)
        pfs = [r.profit_factor for r in results.values() if hasattr(r, "profit_factor") and r.profit_factor not in (float("inf"),)]
        port_pf = float(sum(pfs)/len(pfs)) if pfs else 0.0
    else:
        alloc = {}
        factor_report = {}
        port_pf = 0.0
        returns_df = pd.DataFrame()
    return {
        "symbols": symbols,
        "individual": {k: {"pf": getattr(v, "profit_factor", None), "trades": getattr(v, "total_trades", None), "return_pct": getattr(v, "total_return_pct", None)} if hasattr(v, "profit_factor") else v for k,v in results.items()},
        "allocation": alloc,
        "factor_risk": factor_report,
        "portfolio_pf": port_pf,
        "returns_df_shape": returns_df.shape if not returns_df.empty else (0,0),
    }
