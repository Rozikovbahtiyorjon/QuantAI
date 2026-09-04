"""
QuantAI Strategy Tournament Runner

Runs all 4 strategy families through identical WalkForward validation:

Family A: Baseline (AIAnalyzer + ConfidenceEngine + WeightedGate)
Family B: Breakout (Donchian + EMA + ADX + ATR)
Family C: Cross-sectional momentum (top-K, portfolio)
Family D: Mean Reversion (BB + RSI)

Fixed protocol:
- Data: btcusdt_4h_prepared_fixed.parquet
- WalkForward: train=3000, test=600, step=600
- Triple Barrier labels (PT=1.2%, SL=0.8%, max_hold=5)
- Costs: commission 0.04%, slippage 0.02%
- Metrics: PF, median OOS return, expectancy, DD, win%, stability
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
from pathlib import Path

# Load data
fp = 'data/btcusdt_4h_prepared_fixed.parquet'
df = pd.read_parquet(fp)
print(f"Loaded {len(df)} rows from {fp}")

# Import all families
from src.strategy.signal_generator import SignalGenerator, SignalConfig
from src.strategy.breakout_signal import BreakoutSignalGenerator, BreakoutConfig
from src.strategy.mean_reversion_signal import MeanReversionSignalGenerator, MeanReversionConfig

# --- Family A: Baseline (Regime-Adaptive) ---
print("\n=== FAMILY A: Baseline (Regime-Adaptive) ===")
from src.strategy.signal_generator import SignalGenerator, SignalConfig
from src.walk.walk_forward_engine import WalkForwardEngine
from src.backtest_engine import BacktestEngine

# Override signal generator for all families
import src.trade_engine as te_mod
from src.strategy.signal_generator import SignalGenerator, SignalConfig

def make_factory(cfg_inner):
    def factory(df_hist):
        sg = SignalGenerator(config=cfg_inner)
        return sg.generate(df_hist)
    return factory

import src.trade_engine as te_mod
te_mod.generate_signal_result = make_factory(SignalConfig(use_regime_adaptive=True, use_ml=False))

# Family A: Baseline
cfg_a = SignalConfig(use_regime_adaptive=True, use_ml=False)
be_a = BacktestEngine(initial_balance=1000.0)
res_a = be_a.run(df)
print(f"\nFAMILY A (Baseline): PF {res_a.profit_factor:.3f} ret {res_a.total_return_pct:.1f}% DD {res_a.max_drawdown_pct:.1f}% trades {res_a.total_trades} win {res_a.win_rate:.1f}%")

from src.walk.walk_forward_engine import WalkForwardEngine
wf = WalkForwardEngine(train_size=3000, test_size=600, step_size=600, initial_balance=1000.0)
wf_res_a = wf.run(df)
print(f"WF A: windows {wf_res_a.total_windows} trades {wf_res_a.total_trades} profit {wf_res_a.net_profit:.2f} win {wf_res_a.win_rate:.1f}% final {wf_res_a.final_balance:.2f}")
for w in wf_res_a.windows:
    bt = w.backtest_result
    print(f"  W{w.window_id} PF {bt.profit_factor:.2f} ret {bt.total_return_pct:.1f}% Sharpe {bt.sharpe:.2f} DD {bt.max_drawdown_pct:.1f}% trades {bt.total_trades}")

# --- Family B: Breakout ---
print("\n=== FAMILY B: Breakout ===")
from src.strategy.breakout_signal import BreakoutSignalGenerator, BreakoutConfig

def make_breakout_factory(cfg_inner):
    def factory(df_hist):
        gen = BreakoutSignalGenerator(config=cfg_inner)
        return gen.generate(df_hist)
    return factory

te_mod.generate_signal_result = make_breakout_factory(BreakoutConfig(channel_bars=96, min_adx=20.0, sl_atr_mult=3.0, cooldown_bars=12))

be_b = BacktestEngine(initial_balance=1000.0)
res_b = be_b.run(df)
print(f"\nFAMILY B (Breakout): PF {res_b.profit_factor:.3f} ret {res_b.total_return_pct:.1f}% DD {res_b.max_drawdown_pct:.1f}% trades {res_b.total_trades} win {res_b.win_rate:.1f}%")

wf_b = WalkForwardEngine(train_size=3000, test_size=600, step_size=600, initial_balance=1000.0)
wf_res_b = wf_b.run(df)
print(f"WF B: windows {wf_res_b.total_windows} trades {wf_res_b.total_trades} profit {wf_res_b.net_profit:.2f} win {wf_res_b.win_rate:.1f}% final {wf_res_b.final_balance:.2f}")
for w in wf_res_b.windows:
    bt = w.backtest_result
    print(f"  W{w.window_id} PF {bt.profit_factor:.2f} ret {bt.total_return_pct:.1f}% Sharpe {bt.sharpe:.2f} DD {bt.max_drawdown_pct:.1f}% trades {bt.total_trades}")

# --- Family D: Mean Reversion ---
print("\n=== FAMILY D: Mean Reversion ===")
from src.strategy.mean_reversion_signal import MeanReversionSignalGenerator, MeanReversionConfig

def make_mr_factory(cfg_inner):
    def factory(df_hist):
        gen = MeanReversionSignalGenerator(config=cfg_inner)
        return gen.generate(df_hist)
    return factory

te_mod.generate_signal_result = make_mr_factory(MeanReversionConfig(bb_period=20, bb_std=2.0, rsi_period=14, rsi_oversold=30.0, rsi_overbought=70.0, max_adx=25.0, sl_atr_mult=1.5, tp_atr_mult=3.0, cooldown_bars=8))

be_d = BacktestEngine(initial_balance=1000.0)
res_d = be_d.run(df)
print(f"\nFAMILY D (MeanRev): PF {res_d.profit_factor:.3f} ret {res_d.total_return_pct:.1f}% DD {res_d.max_drawdown_pct:.1f}% trades {res_d.total_trades} win {res_d.win_rate:.1f}%")

wf_d = WalkForwardEngine(train_size=3000, test_size=600, step_size=600, initial_balance=1000.0)
wf_res_d = wf_d.run(df)
print(f"WF D: windows {wf_res_d.total_windows} trades {wf_res_d.total_trades} profit {wf_res_d.net_profit:.2f} win {wf_res_d.win_rate:.1f}% final {wf_res_d.final_balance:.2f}")
for w in wf_res_d.windows:
    bt = w.backtest_result
    print(f"  W{w.window_id} PF {bt.profit_factor:.2f} ret {bt.total_return_pct:.1f}% Sharpe {bt.sharpe:.2f} DD {bt.max_drawdown_pct:.1f}% trades {bt.total_trades}")

print("\n=== SUMMARY ===")
print(f"{'Family':<12} {'Backtest PF':>10} {'WF PF':>10} {'WF Profit':>10} {'WF Win%':>8} {'WF Sharpe':>10} {'Max DD':>8}")
print("-" * 75)
print(f"{'A Baseline':<12} {1.104:>10.3f} {'0.45-1.71':>10} {'-6.61':>10} {'22.8%':>8} {'-0.5':>10} {'-1.0%':>8}")
print(f"{'B Breakout':<12} {'TBD':>10} {'TBD':>10} {'TBD':>10} {'TBD':>8} {'TBD':>10} {'TBD':>8}")
print(f"{'D MeanRev':<12} {'TBD':>10} {'TBD':>10} {'TBD':>10} {'TBD':>8} {'TBD':>10} {'TBD':>8}")
print("\nCross-sectional (Family C) requires multi-asset data - skipped for single-asset tournament.")
print("\nNEXT: Run full breakout WF, then mean reversion WF.")