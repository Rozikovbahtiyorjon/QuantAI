#!/usr/bin/env python
"""Compare simple vs triple-barrier labeling on same data."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.dataset_builder import DatasetBuilder, DatasetConfig

FP = Path("data/btcusdt_15m.parquet")
if not FP.exists():
    FP = Path("data/btcusdt_15m_prepared.parquet")
    _prepared = True
else:
    _prepared = False

df = pd.read_parquet(FP) if FP.suffix==".parquet" else pd.read_csv(FP)
print(f"rows {len(df)} file {FP}")

for method in ("simple","triple_barrier"):
    cfg = DatasetConfig(label_method=method, future_bars=5, target_profit=0.002, warmup_bars=200,
                        calculate_indicators=not _prepared, drop_nan=True,
                        tb_pt=0.012, tb_sl=0.008, tb_use_atr=True)
    builder = DatasetBuilder(cfg)
    ds = builder.build(df)
    stats = DatasetBuilder.statistics(ds)
    print(f"\n[{method}] rows={stats['rows']} BUY {stats['buy']} ({stats['buy_percent']:.1f}%) SELL {stats['sell']} ({stats['sell_percent']:.1f}%) HOLD {stats['hold']} ({stats['hold_percent']:.1f}%)")
    if method=="triple_barrier" and "tb_barrier" in ds.columns:
        print(ds["tb_barrier"].value_counts().to_dict())
