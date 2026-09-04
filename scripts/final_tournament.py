import sys
sys.path.insert(0, '.')
import pandas as pd
from src.backtest_engine import BacktestEngine
from src.indicators import add_indicators
from src.strategy.signal_generator import SignalGenerator, SignalConfig
from src.strategy.breakout_signal import BreakoutSignalGenerator, BreakoutConfig
from src.strategy.mean_reversion_signal import MeanReversionSignalGenerator, MeanReversionConfig
import src.trade_engine as te_mod

df = pd.read_parquet('data/btcusdt_4h_prepared.parquet')
df = add_indicators(df)
print(f'Data: {len(df)} rows')

print('=== FINAL STRATEGY TOURNAMENT ===')

def run_strategy(name, gen_factory):
    te_mod.generate_signal_result = gen_factory
    be = __import__('src.backtest_engine', fromlist=['BacktestEngine']).BacktestEngine(initial_balance=1000.0)
    res = be.run(df)
    pf = res.profit_factor if res.profit_factor != float('inf') else 999
    return {
        'name': name,
        'pf': pf,
        'ret': res.total_return_pct,
        'dd': res.max_drawdown_pct,
        'trades': res.total_trades,
        'win': res.win_rate
    }

def make_factory(gen_fn):
    def factory(df_hist):
        gen = gen_fn()
        return gen.generate(df_hist)
    return factory

# A: Baseline
print("=== FAMILY A: Baseline ===")
te_mod.generate_signal_result = make_factory(lambda: SignalGenerator(SignalConfig(use_regime_adaptive=True, use_ml=False)).generate)
res_a = run_strategy('A Baseline', lambda df_hist: SignalGenerator(SignalConfig(use_regime_adaptive=True, use_ml=False)).generate(df_hist))

# B: Breakout
print("=== FAMILY B: Breakout ===")
from src.strategy.breakout_signal import BreakoutSignalGenerator, BreakoutConfig
b_gen = lambda df_hist: BreakoutSignalGenerator(BreakoutConfig(channel_bars=96, min_adx=20.0, sl_atr_mult=3.0, cooldown_bars=12)).generate(df_hist)
res_b = run_strategy('B Breakout', b_gen)

# D: Mean Reversion
print("=== FAMILY D: Mean Reversion ===")
from src.strategy.mean_reversion_signal import MeanReversionSignalGenerator, MeanReversionConfig
d_gen = lambda df_hist: MeanReversionSignalGenerator(MeanReversionConfig(max_adx=60.0)).generate(df_hist)
res_d = run_strategy('D MeanRev', d_gen)

# C: Cross-Sectional Momentum (Family C)
print("\n=== FAMILY C: Cross-Sectional Momentum ===")
from src.strategies.cross_sectional import CrossSectionParams, backtest
import glob

# Load multi-asset 1h data for cross-sectional
files = glob.glob('data/*_1h_prepared.parquet')
prices = {}
for fp in glob.glob('data/*_1h_prepared.parquet'):
    symbol = fp.split('\\')[-1].replace('_1h_prepared.parquet', '')
    df = pd.read_parquet(fp)
    prices[symbol] = df['close']

prices_df = pd.DataFrame(prices).sort_index()

params = CrossSectionParams(lookback_days=14, top_k=2, rebalance_days=7)
result = backtest(prices_df, params)

res_c = {
    'name': 'C Cross-Sectional',
    'pf': result['stats']['pf'],
    'ret': result['stats']['total_ret_pct'],
    'dd': result['stats']['maxdd_pct'],
    'trades': result['stats']['trades'],
    'win': result['stats'].get('wins', 0) / max(1, result['stats']['trades']) * 100
}
print(f"Cross-sectional 1h: PF={res_c['pf']:.3f} ret={res_c['ret']:.1f}% DD={res_c['dd']:.1f}% trades={res_c['trades']} win={res_c['win']:.1f}%")

print('\n=== FINAL RESULTS ===')
print(f'{"Family":<12} {"PF":>6} {"Ret%":>7} {"DD%":>7} {"Trades":>7} {"Win%":>6}')
print('-' * 50)
for r in [res_a, res_b, res_c, res_d]:
    print(f'{r["name"]:<12} {r["pf"]:>6.3f} {r["ret"]:>7.1f} {r["dd"]:>7.1f} {r["trades"]:>7} {r["win"]:>6.1f}')

print("\nDone!")