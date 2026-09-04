import sys
sys.path.insert(0, '.')
import pandas as pd
from src.indicators import add_indicators

df = pd.read_parquet('data/btcusdt_4h_prepared_fixed.parquet')
df = add_indicators(df)
print('Columns:', [c for c in df.columns if 'bb' in c or 'rsi' in c or 'adx' in c])

df['buy_cond'] = (df['bb_position'] <= 0.1) & (df['rsi'] <= 30) & (df['adx'] <= 60)
df['sell_cond'] = (df['bb_position'] >= 0.9) & (df['rsi'] >= 70) & (df['adx'] <= 60)

print(f'Buy signals: {df["buy_cond"].sum()}')
print(df[df['buy_cond']].head(5)[['close','rsi','adx','bb_position']].to_string())
print()
print(df[df['sell_cond']].head(5)[['close','rsi','adx','bb_position']].to_string())
print()
print(f'Sell signals: {df["sell_cond"].sum()}')
print()
print('adx stats:', df['adx'].describe())
print('bb_position stats:', df['bb_position'].describe())
print('rsi stats:', df['rsi'].describe())