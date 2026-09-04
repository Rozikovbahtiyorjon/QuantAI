import sys
sys.path.insert(0, '.')
import pandas as pd
from src.indicators import add_indicators
from src.walk.walk_forward_engine import WalkForwardEngine
from src.backtest_engine import BacktestEngine

raw = pd.read_parquet('data/btcusdt_1h.parquet')
print(f"raw 1h {len(raw)}")
df = add_indicators(raw)
print(f"prepared {len(df)} NaN {df.isna().sum().sum()}")

# Simple backtest full
be = BacktestEngine(initial_balance=1000.0)
res = be.run(df)
print(f"SIMPLE 1h FULL: PF {res.profit_factor:.3f} Sharpe {res.sharpe:.3f} DD {res.max_drawdown_pct:.1f}% ret {res.total_return_pct:.1f}% trades {res.total_trades} win {res.win_rate:.1f}%")

# Walk-Forward simple
wf = WalkForwardEngine(train_size=5000, test_size=1000, step_size=1000, initial_balance=1000.0)
wf_res = wf.run(df)
print(f"WF SIMPLE 1h: windows {wf_res.total_windows} trades {wf_res.total_trades} PF? profit {wf_res.net_profit:.2f} win {wf_res.win_rate:.1f}% ret {(wf_res.final_balance-1000)/10:.1f}%")
for w in wf_res.windows:
    bt = w.backtest_result
    print(f"  W{w.window_id} {w.train_start}-{w.test_end} trades {bt.total_trades} PF {bt.profit_factor:.2f} ret {bt.total_return_pct:.1f}% Sharpe {bt.sharpe:.2f}")

from pathlib import Path
from src.validation.gate import check_trading_readiness
tmp = Path("data/tmp_1h.parquet")
df.to_parquet(tmp)
cr = check_trading_readiness(Path("data"))
print(f"Gate: {cr.status} {cr.details}")

# Also quick head 8000 comparison for noise check
df8 = df.head(8000)
be8 = BacktestEngine(initial_balance=1000.0).run(df8)
print(f"SIMPLE 1h HEAD8000: PF {be8.profit_factor:.3f} ret {be8.total_return_pct:.1f}% trades {be8.total_trades}")
