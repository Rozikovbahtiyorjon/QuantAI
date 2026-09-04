import sys
sys.path.insert(0, '.')
import pandas as pd
from src.indicators import add_indicators

df = pd.read_parquet('data/btcusdt_4h_prepared_fixed.parquet')
df = add_indicators(df)

# Check individual conditions
bb_low = df['bb_position'] <= 0.1
rsi_low = df['rsi'] <= 30
adx_low = df['adx'] <= 60

print(f'bb_position <= 0.1: {bb_low.sum()} ({bb_low.mean()*100:.1f}%)')
print(f'rsi <= 30: {rsi_low.sum()} ({rsi_low.mean()*100:.1f}%)')
print(f'adx <= 60: {adx_low.sum()} ({adx_low.mean()*100:.1f}%)')
print(f'All three: {(bb_low & rsi_low & adx_low).sum()}')

# Check a few rows where bb_position <= 0.1
bb_low_df = df[bb_low]
print(f'\nRows with bb_pos <= 0.1: {len(bb_low_df)}')
print(bb_low_df[['close','rsi','adx','bb_position']].head(10).to_string())

# Check where rsi <= 30
rsi_low_df = df[rsi_low]
print(f'\nRows with rsi <= 30: {len(rsi_low_df)}')
print(rsi_low_df[['close','rsi','adx','bb_position']].head(10).to_string())

# Check intersection
both = bb_low & rsi_low
print(f'\nbb <= 0.1 AND rsi <= 30: {both.sum()}')
both_df = df[both]
if len(both_df) > 0:
    print(both_df[['close','rsi','adx','bb_position']].head(10).to_string())

# Check all three
all_three = bb_low & rsi_low & (df['adx'] <= 60)
print(f'\nAll three: {all_three.sum()}')
if all_three.sum() > 0:
    print(df[all_three][['close','rsi','adx','bb_position']].head(10).to_string())