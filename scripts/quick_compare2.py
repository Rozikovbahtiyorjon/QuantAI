import sys
sys.path.insert(0, '.')
import pandas as pd
from src.dataset_builder import DatasetBuilder, DatasetConfig
from pathlib import Path
fp = Path('data/btcusdt_15m.parquet')
df = pd.read_parquet(fp)
df_small = df.head(5000)
for method in ('simple','triple_barrier'):
    cfg = DatasetConfig(label_method=method, future_bars=5, target_profit=0.002, warmup_bars=200, calculate_indicators=True, tb_pt=0.012, tb_sl=0.008, tb_use_atr=True)
    builder = DatasetBuilder(cfg)
    ds = builder.build(df_small)
    stats = DatasetBuilder.statistics(ds)
    print(f"[{method}] rows={stats['rows']} BUY {stats['buy']} SELL {stats['sell']} HOLD {stats['hold']}")
    print(f"  buy_pct {stats['buy_percent']:.1f} sell_pct {stats['sell_percent']:.1f} hold_pct {stats['hold_percent']:.1f}")
    if 'tb_barrier' in ds.columns:
        print(ds['tb_barrier'].value_counts().to_dict())
    print('cols', len(ds.columns), list(ds.columns)[:12])
