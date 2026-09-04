import pandas as pd
from src.dataset_builder import DatasetBuilder, DatasetConfig
from pathlib import Path
fp = Path('data/btcusdt_15m.parquet')
if not fp.exists():
    fp = Path('data/btcusdt_15m_prepared.parquet')
    prepared=True
else:
    prepared=False
print('fp', fp, 'prepared', prepared)
df = pd.read_parquet(fp) if fp.suffix=='.parquet' else pd.read_csv(fp)
print('rows', len(df), 'cols', len(df.columns))
df_small = df.head(5000)
for method in ('simple','triple_barrier'):
    cfg = DatasetConfig(label_method=method, future_bars=5, target_profit=0.002, warmup_bars=200, calculate_indicators=not prepared, drop_nan=True, tb_pt=0.012, tb_sl=0.008, tb_use_atr=True)
    builder = DatasetBuilder(cfg)
    ds = builder.build(df_small)
    stats = DatasetBuilder.statistics(ds)
    print(f"[{method}] rows={stats['rows']} BUY {stats['buy']} ({stats['buy_percent']:.1f}%) SELL {stats['sell']} ({stats['sell_percent']:.1f}%) HOLD {stats['hold']} ({stats['hold_percent']:.1f}%)")
    if method=='triple_barrier' and 'tb_barrier' in ds.columns:
        print(ds['tb_barrier'].value_counts().to_dict())
