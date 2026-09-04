import sys
sys.path.insert(0, '.')
import pandas as pd
import glob
from src.strategies.cross_sectional import CrossSectionParams, backtest

# Load 4h data for cross-sectional
files = glob.glob('data/*_4h_prepared.parquet')
print('Available files:', files)

prices = {}
for fp in files:
    symbol = fp.split('\\')[-1].replace('_4h_prepared.parquet', '')
    df = pd.read_parquet(fp)
    prices[symbol] = df['close']

prices_df = pd.DataFrame(prices)
prices_df = prices_df.sort_index()
print(f'Price matrix: {prices_df.shape}')
print('Columns:', prices_df.columns.tolist())

params = CrossSectionParams(lookback_days=14, top_k=2, rebalance_days=7)
result = backtest(prices_df, params)
print(f'Cross-sectional 4h: ret={result["stats"]["total_ret_pct"]:.1f}% DD={result["stats"]["maxdd_pct"]:.1f}% PF={result["stats"]["pf"]:.2f} Sharpe={result["stats"]["sharpe"]:.2f} trades={result["stats"]["trades"]}')