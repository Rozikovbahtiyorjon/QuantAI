import sys
sys.path.insert(0, '.')
import pandas as pd
from src.indicators import add_indicators
from src.strategy.mean_reversion_signal import MeanReversionSignalGenerator, MeanReversionConfig

df = pd.read_parquet('data/btcusdt_4h_prepared_fixed.parquet').head(2000)
df = add_indicators(df)

cfg = MeanReversionConfig(max_adx=60.0)
gen = MeanReversionSignalGenerator(config=cfg)

trades = 0
for i in range(100, len(df)):
    result = gen.generate(df.iloc[:i+1])
    if result.signal != 'HOLD':
        row = df.iloc[i-1]
        print(f'Trade at {i}: {result.signal} bb_pos={row.get("bb_position",0):.3f} rsi={row.get("rsi",50):.1f} adx={row.get("adx",0):.1f}')
        trades += 1

print(f'Total trades: {trades}')