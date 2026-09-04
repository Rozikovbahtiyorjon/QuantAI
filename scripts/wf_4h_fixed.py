import sys
sys.path.insert(0, '.')
import pandas as pd
from src.walk.walk_forward_engine import WalkForwardEngine
from src.backtest_engine import BacktestEngine

fp = 'data/btcusdt_4h_prepared_fixed.parquet'
df = pd.read_parquet(fp)
print(f"4h fixed {len(df)} rows")
# Also test fresh from raw 1h resampled? Use fixed directly
be = BacktestEngine(initial_balance=1000.0)
res = be.run(df)
print(f"BACKTEST 4h PF {res.profit_factor:.3f} Sharpe {res.sharpe:.2f} DD {res.max_drawdown_pct:.1f}% ret {res.total_return_pct:.1f}% trades {res.total_trades} win {res.win_rate:.1f}%")

# Walk-Forward 4h: train 3000 (~ 1.4 years), test 600 (~ 3 months), step 600
wf = WalkForwardEngine(train_size=3000, test_size=600, step_size=600, initial_balance=1000.0)
wf_res = wf.run(df)
print(f"WF 4h: windows {wf_res.total_windows} trades {wf_res.total_trades} profit {wf_res.net_profit:.2f} win {wf_res.win_rate:.1f}% final {wf_res.final_balance:.2f}")
for w in wf_res.windows:
    bt = w.backtest_result
    print(f"  W{w.window_id} train {w.train_start}-{w.train_end} test {w.test_start}-{w.test_end} trades {bt.total_trades} PF {bt.profit_factor:.2f} ret {bt.total_return_pct:.1f}% Sharpe {bt.sharpe:.2f} DD {bt.max_drawdown_pct:.1f}%")

# Overwrite main file with fixed
df.to_parquet('data/btcusdt_4h_prepared.parquet', index=False)
print("Overwrote btcusdt_4h_prepared.parquet with fixed")

from pathlib import Path
from src.validation.gate import check_trading_readiness
cr = check_trading_readiness(Path("data"))
print(f"Gate after fix: {cr.status} {cr.details} {cr.metrics}")
