import sys
sys.path.insert(0, '.')
import pandas as pd
from src.backtest_engine import BacktestEngine
from src.walk.walk_forward_engine import WalkForwardEngine

from src.strategy.signal_generator import SignalGenerator, SignalConfig
from src.strategy.breakout_signal import BreakoutSignalGenerator, BreakoutConfig
from src.strategy.mean_reversion_signal import MeanReversionSignalGenerator, MeanReversionConfig

import src.trade_engine as te_mod

fp = 'data/btcusdt_4h_prepared.parquet'
df = pd.read_parquet(fp)
print(f"Loaded {len(df)} rows from {fp}")

# Override signal generator
import src.trade_engine as te_mod
from src.strategy.signal_generator import SignalGenerator, SignalConfig

def make_factory(gen_fn):
    def factory(df_hist):
        gen = gen_fn()
        return gen.generate(df_hist)
    return factory

results = {}

# --- Family A: Baseline (Regime-Adaptive) ---
print("\n=== FAMILY A: Baseline (Regime-Adaptive) ===")
cfg_a = SignalConfig(use_regime_adaptive=True, use_ml=False)
te_mod.generate_signal_result = make_factory(lambda: SignalGenerator(config=cfg_a))
be = BacktestEngine(initial_balance=1000.0)
res = be.run(df)
print(f"BACKTEST A: PF {res.profit_factor:.3f} ret {res.total_return_pct:.1f}% DD {res.max_drawdown_pct:.1f}% trades {res.total_trades} win {res.win_rate:.1f}%")

from src.walk.walk_forward_engine import WalkForwardEngine
wf = WalkForwardEngine(train_size=3000, test_size=600, step_size=600, initial_balance=1000.0)
wf_res = wf.run(df)
print(f"WF A: windows {wf_res.total_windows} trades {wf_res.total_trades} profit {wf_res.net_profit:.2f} win {wf_res.win_rate:.1f}% final {wf_res.final_balance:.2f}")
for w in wf_res.windows:
    bt = w.backtest_result
    print(f"  W{w.window_id} PF {bt.profit_factor:.2f} ret {bt.total_return_pct:.1f}% Sharpe {bt.sharpe:.2f} DD {bt.max_drawdown_pct:.1f}% trades {bt.total_trades}")

# --- Family B: Breakout ---
print("\n=== FAMILY B: Breakout ===")
from src.strategy.breakout_signal import BreakoutSignalGenerator, BreakoutConfig
cfg_b = BreakoutConfig(channel_bars=96, min_adx=20.0, sl_atr_mult=3.0, cooldown_bars=12)
te_mod.generate_signal_result = make_factory(lambda: BreakoutSignalGenerator(config=cfg_b))
be = BacktestEngine(initial_balance=1000.0)
res = be.run(df)
print(f"BACKTEST B: PF {res.profit_factor:.3f} ret {res.total_return_pct:.1f}% DD {res.max_drawdown_pct:.1f}% trades {res.total_trades} win {res.win_rate:.1f}%")

wf = WalkForwardEngine(train_size=3000, test_size=600, step_size=600, initial_balance=1000.0)
wf_res = wf.run(df)
print(f"WF B: windows {wf_res.total_windows} trades {wf_res.total_trades} profit {wf_res.net_profit:.2f} win {wf_res.win_rate:.1f}% final {wf_res.final_balance:.2f}")
for w in wf_res.windows:
    bt = w.backtest_result
    print(f"  W{w.window_id} PF {bt.profit_factor:.2f} ret {bt.total_return_pct:.1f}% Sharpe {bt.sharpe:.2f} DD {bt.max_drawdown_pct:.1f}% trades {bt.total_trades}")

# --- Family D: Mean Reversion ---
print("\n=== FAMILY D: Mean Reversion ===")
from src.strategy.mean_reversion_signal import MeanReversionSignalGenerator, MeanReversionConfig
te_mod.generate_signal_result = make_factory(lambda: MeanReversionSignalGenerator(MeanReversionConfig(max_adx=60.0)))

be = BacktestEngine(initial_balance=1000.0)
res = be.run(df)
print(f"BACKTEST D: PF {res.profit_factor:.3f} ret {res.total_return_pct:.1f}% DD {res.max_drawdown_pct:.1f}% trades {res.total_trades} win {res.win_rate:.1f}%")

wf = WalkForwardEngine(train_size=3000, test_size=600, step_size=600, initial_balance=1000.0)
wf_res = wf.run(df)
print(f"WF D: windows {wf_res.total_windows} trades {wf_res.total_trades} profit {wf_res.net_profit:.2f} win {wf_res.win_rate:.1f}% final {wf_res.final_balance:.2f}")
for w in wf_res.windows:
    bt = w.backtest_result
    print(f"  W{w.window_id} PF {bt.profit_factor:.2f} ret {bt.total_return_pct:.1f}% Sharpe {bt.sharpe:.2f} DD {bt.max_drawdown_pct:.1f}% trades {bt.total_trades}")

print("\n=== SUMMARY ===")
print(f"{'Family':<12} {'Backtest PF':>10} {'WF PF range':>12} {'WF Profit':>10} {'WF Win%':>8} {'WF Sharpe':>10} {'Max DD':>8}")
print("-" * 80)
# We'll compute WF PF ranges from the window results
print("Done!")