import sys
sys.path.insert(0, '.')
import pandas as pd
from src.walk.walk_forward_engine import WalkForwardEngine
from src.backtest_engine import BacktestEngine

fp = 'data/btcusdt_4h_prepared_fixed.parquet'
# Ensure main file is fixed version
import pathlib
if pathlib.Path('data/btcusdt_4h_prepared_fixed.parquet').exists():
    import shutil
    shutil.copy('data/btcusdt_4h_prepared_fixed.parquet', 'data/btcusdt_4h_prepared.parquet')

df = pd.read_parquet('data/btcusdt_4h_prepared.parquet')
print(f"4h tuned {len(df)} thr 0.60 tw 1.2")

be = BacktestEngine(initial_balance=1000.0)
res = be.run(df)
print(f"BACKTEST 4h TUNED PF {res.profit_factor:.3f} Sharpe {res.sharpe:.2f} DD {res.max_drawdown_pct:.1f}% ret {res.total_return_pct:.1f}% trades {res.total_trades} win {res.win_rate:.1f}% PF")

wf = WalkForwardEngine(train_size=3000, test_size=600, step_size=600, initial_balance=1000.0)
wf_res = wf.run(df)
print(f"WF TUNED 4h: windows {wf_res.total_windows} trades {wf_res.total_trades} profit {wf_res.net_profit:.2f} win {wf_res.win_rate:.1f}% final {wf_res.final_balance:.2f}")
for w in wf_res.windows:
    bt = w.backtest_result
    print(f"  W{w.window_id} PF {bt.profit_factor:.2f} ret {bt.total_return_pct:.1f}% Sharpe {bt.sharpe:.2f} DD {bt.max_drawdown_pct:.1f}% trades {bt.total_trades}")

from pathlib import Path
from src.validation.gate import check_trading_readiness, check_backtest_smoke
print("Gates:")
print(check_backtest_smoke(Path("data")).__dict__)
print(check_trading_readiness(Path("data")).__dict__)
