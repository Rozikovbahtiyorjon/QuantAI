#!/usr/bin/env python
"""
Measure PF after P0 — must be run AFTER fixing python runtime.

Usage:
  python scripts/measure_pf_after_p0.py
  python scripts/measure_pf_after_p0.py --file data/btcusdt_15m_prepared.parquet

What it does:
  - Loads prepared parquet (indicators already fixed to ffill-only)
  - Runs BacktestEngine (TradeEngine now uses RiskOrchestrator)
  - Runs trading_readiness gate (PF<1, bankrupt, Sharpe<0 -> FAIL)
  - Prints baseline to compare with pre-P0: PF 0.412 Sharpe -6.875 -100%

Requires:
  - pip install -r requirements.txt
  - python 3.12 (see pyvenv.cfg broken pythoncore-3.14)
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.backtest_engine import BacktestEngine
from src.validation.gate import check_trading_readiness, check_backtest_smoke

DATA_CANDIDATES = [
    Path("data/btcusdt_15m_prepared.parquet"),
    Path("data/ethusdt_15m_prepared.parquet"),
    Path("data/solusdt_15m_prepared.parquet"),
]

def pick_file(arg: str | None) -> Path:
    if arg and Path(arg).exists():
        return Path(arg)
    for p in DATA_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError("No *_prepared.parquet found in data/")

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=None, help="prepared parquet path")
    args = ap.parse_args()

    fp = pick_file(args.file)
    print(f"[measure] file: {fp} ({fp.stat().st_size/1e6:.1f} MB)")
    df = pd.read_parquet(fp)
    print(f"[measure] rows: {len(df)} cols: {list(df.columns)[:10]}...")

    # Warmup NaNs from ffill-only indicators should be dropped by engine's minimum_rows
    # BacktestEngine.validate_data will fail if NaNs remain in required cols.
    # If you see NaN error here, re-prepare data with fixed indicators.py
    be = BacktestEngine(initial_balance=1000.0)
    res = be.run(df)

    print("\n" + "="*64)
    print("P0 BASELINE (compare to pre-P0: PF 0.412 DD -100% -100%)")
    print("="*64)
    print(f"Trades: {res.total_trades}  WinRate: {res.win_rate:.1f}%  PF: {res.profit_factor}")
    print(f"Return: {res.total_return_pct:.2f}%  DD: {res.max_drawdown_pct:.2f}%  Sharpe: {res.sharpe:.3f}")
    print(f"Expectancy: {(res.net_profit/res.total_trades) if res.total_trades else 0:.4f}  Bankrupt: {res.final_balance<=0}")
    print(f"Final: {res.final_balance:.2f} / 1000.00")
    be.print_report(res)

    print("\n--- Gate checks ---")
    hb = check_backtest_smoke(Path("data"))
    tr = check_trading_readiness(Path("data"))
    print(f"[{hb.status}] {hb.name}: {hb.details}  {hb.metrics}")
    print(f"[{tr.status}] {tr.name}: {tr.details}  {tr.metrics}")
    if tr.status == "FAIL":
        print("\n[EXPECTED] trading_readiness FAIL until triple-barrier + retrain improves PF>1")
    else:
        print("\n[UNEXPECTED PASS] — verify with walk-forward, not single backtest")

if __name__ == "__main__":
    main()
