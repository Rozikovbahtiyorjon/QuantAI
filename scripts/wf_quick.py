import sys
sys.path.insert(0, '.')
import pandas as pd
from src.walk.walk_forward_engine import WalkForwardEngine

for fp in ['data/btcusdt_15m_prepared.parquet', 'data/btcusdt_15m.parquet']:
    try:
        df = pd.read_parquet(fp)
        print(f"Loaded {fp} rows {len(df)}")
        break
    except: pass

# Use prepared (indicators already fixed to ffill) — but need to ensure ffill fix applied?
# Our fix requires re-preparing. Prepared files still have old bfill data.
# So load raw and prepare fresh for correct walk-forward
raw = pd.read_parquet('data/btcusdt_15m.parquet')
print(f"raw {len(raw)}")
from src.indicators import add_indicators
df_prepared = add_indicators(raw.head(8000))
print(f"prepared {len(df_prepared)} has NaN? {df_prepared.isna().sum().sum()}")

from src.backtest_engine import BacktestEngine
be = BacktestEngine(initial_balance=1000.0)
res = be.run(df_prepared)
print(f"Simple backtest head8000: trades {res.total_trades} PF {res.profit_factor:.3f} ret {res.total_return_pct:.2f}% DD {res.max_drawdown_pct:.2f}% Sharpe {res.sharpe:.2f}")

# Walk-forward without ML
wf = WalkForwardEngine(train_size=3000, test_size=800, step_size=800, initial_balance=1000.0)
wf_res = wf.run(df_prepared)
print(f"WF: windows {wf_res.total_windows} trades {wf_res.total_trades} profit {wf_res.net_profit:.2f} win {wf_res.win_rate:.1f}%")
for w in wf_res.windows[:3]:
    print(f"  W{w.window_id} profit {w.backtest_result.net_profit:.2f} PF {w.backtest_result.profit_factor:.2f} trades {w.backtest_result.total_trades}")

from src.validation.gate import check_trading_readiness
# need to save temp prepared for gate
import tempfile
tmp = "data/tmp_prepared.parquet"
df_prepared.to_parquet(tmp)
from pathlib import Path
cr = check_trading_readiness(Path("data"))
print(f"Gate trading_readiness: {cr.status} {cr.details} {cr.metrics}")
