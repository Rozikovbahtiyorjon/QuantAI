import sys
sys.path.insert(0, '.')
import pandas as pd
from src.indicators import add_indicators
from src.backtest_engine import BacktestEngine

files = [
    ('BTC 15m', 'data/btcusdt_15m.parquet'),
    ('BTC 1h', 'data/btcusdt_1h.parquet'),
    ('BTC 4h', 'data/btcusdt_4h.parquet'),
    ('ETH 15m', 'data/ethusdt_15m.parquet'),
    ('ETH 1h', 'data/ethusdt_1h.parquet'),
    ('SOL 15m', 'data/solusdt_15m.parquet'),
]

print(f"{'Symbol':<10} {'Rows':>6} {'Trades':>6} {'PF':>6} {'Sharpe':>7} {'DD%':>7} {'Ret%':>7} {'Win%':>6} {'Final':>8}")
print("-"*80)
for name, fp in files:
    try:
        raw = pd.read_parquet(fp)
        df = add_indicators(raw)
        be = BacktestEngine(initial_balance=1000.0)
        res = be.run(df)
        print(f"{name:<10} {len(df):>6} {res.total_trades:>6} {res.profit_factor:>6.2f} {res.sharpe:>7.2f} {res.max_drawdown_pct:>7.1f} {res.total_return_pct:>7.1f} {res.win_rate:>6.1f} {res.final_balance:>8.2f}")
    except Exception as e:
        print(f"{name:<10} ERROR {e}")
