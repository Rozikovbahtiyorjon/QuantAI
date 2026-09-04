import sys
sys.path.insert(0, '.')
import pandas as pd
from src.indicators import add_indicators
from pathlib import Path

# Regenerate prepared files with fixed ffill-only indicators
for name in ['btcusdt_4h_prepared.parquet','btcusdt_1h_prepared.parquet','btcusdt_15m_prepared.parquet']:
    fp = Path(f'data/{name}')
    if not fp.exists():
        print(f"skip {name} not found")
        continue
    print(f"Regenerating {name} ...")
    df = pd.read_parquet(fp)
    # Extract OHLCV
    ohlcv = df[['timestamp','open','high','low','close','volume']].copy()
    # Re-add indicators with fixed logic
    df_new = add_indicators(ohlcv)
    # Keep same path, backup
    fp.with_suffix('.bak.parquet').write_bytes(fp.read_bytes()) if not (fp.parent / (fp.stem + '.bak.parquet')).exists() else None
    df_new.to_parquet(fp, index=False)
    print(f"  done {len(df_new)} rows NaN {df_new.isna().sum().sum()} cols {len(df_new.columns)}")

print("Regen done")
# Verify 4h
df4 = pd.read_parquet('data/btcusdt_4h_prepared.parquet')
print(f"4h verify PF test quick")
from src.backtest_engine import BacktestEngine
be = BacktestEngine(initial_balance=1000.0)
res = be.run(df4)
print(f"4h after regen: PF {res.profit_factor:.3f} ret {res.total_return_pct:.1f}% DD {res.max_drawdown_pct:.1f}% trades {res.total_trades}")
