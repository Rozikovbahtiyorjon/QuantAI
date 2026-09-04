"""
QuantAI Breakout Strategy Optimizer (Optuna on Walk-Forward)

Optimizes BreakoutConfig parameters using Optuna with Walk-Forward CV.
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import optuna
from optuna.pruners import MedianPruner

from src.backtest_engine import BacktestEngine
from src.walk.walk_forward_engine import WalkForwardEngine
from src.strategy.breakout_signal import BreakoutSignalGenerator, BreakoutConfig
import src.trade_engine as te_mod

df = pd.read_parquet('data/btcusdt_4h_prepared.parquet')
print(f"Data: {len(df)} rows")

def make_factory(cfg):
    def factory(df_hist):
        gen = BreakoutSignalGenerator(config=cfg)
        return gen.generate(df_hist)
    return factory

import src.trade_engine as te_mod
from src.backtest_engine import BacktestEngine
from src.walk.walk_forward_engine import WalkForwardEngine

def objective(trial):
    # Parameter space
    cfg = BreakoutConfig(
        channel_bars=trial.suggest_int('channel_bars', 48, 192, step=24),
        min_adx=trial.suggest_float('min_adx', 15.0, 30.0, step=2.5),
        sl_atr_mult=trial.suggest_float('sl_atr_mult', 2.0, 4.0, step=0.5),
        cooldown_bars=trial.suggest_int('cooldown_bars', 6, 24, step=6),
    )
    
    def make_factory(cfg_inner):
        def factory(df_hist):
            gen = BreakoutSignalGenerator(config=cfg_inner)
            return gen.generate(df_hist)
        return factory
    
    import src.trade_engine as te_mod
    from src.backtest_engine import BacktestEngine
    from src.walk.walk_forward_engine import WalkForwardEngine
    te_mod.generate_signal_result = make_factory(cfg)
    
    # Walk-Forward
    wf = WalkForwardEngine(train_size=3000, test_size=600, step_size=600, initial_balance=1000.0)
    wf_res = wf.run(df)
    
    # Objective: maximize WF profit factor median, penalize low trade count and high variance
    if wf_res.total_trades < 20:
        return -999.0
    
    pfs = [w.backtest_result.profit_factor if w.backtest_result.profit_factor != float('inf') else 999 
           for w in wf_res.windows]
    profits = [w.backtest_result.net_profit for w in wf_res.windows]
    
    import numpy as np
    pf_median = np.median(pfs)
    profit_total = sum(profits)
    profitable_share = sum(1 for p in profits if p > 0) / len(profits)
    pf_std = np.std(pfs) if len(pfs) > 1 else 999
    
    # Composite objective: maximize median PF, reward positive total profit, penalize variance
    score = pf_median * 100 + profit_total * 0.1 + profitable_share * 50 - pf_std * 10
    
    return score

if __name__ == '__main__':
    df = pd.read_parquet('data/btcusdt_4h_prepared.parquet')
    print(f"Data: {len(df)} rows")
    
    study = optuna.create_study(
        direction='maximize',
        pruner=MedianPruner(n_warmup_steps=10),
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    study.optimize(objective, n_trials=50, timeout=1800)
    
    print("\n=== BEST PARAMS ===")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print(f"Best score: {study.best_value:.2f}")
    
    # Save results
    import json
    with open('optuna_breakout_results.json', 'w') as f:
        json.dump({
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials)
        }, f, indent=2)