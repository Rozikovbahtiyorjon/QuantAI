import sys
sys.path.insert(0, '.')
import pandas as pd
from src.backtest_engine import BacktestEngine
import src.trade_engine as te_mod
from src.strategy.mean_reversion_signal import MeanReversionSignalGenerator, MeanReversionConfig

df = pd.read_parquet('data/btcusdt_4h_prepared.parquet')
print('Data:', len(df), 'rows, has bb_position:', 'bb_position' in df.columns)

def make_factory(cfg_inner):
    def factory(df_hist):
        gen = MeanReversionSignalGenerator(config=cfg_inner)
        return gen.generate(df_hist)
    return factory

te_mod.generate_signal_result = make_factory(MeanReversionConfig(max_adx=60.0))

be = __import__('src.backtest_engine', fromlist=['BacktestEngine']).BacktestEngine(initial_balance=1000.0)
res = be.run(df)
pf = res.profit_factor if res.profit_factor != float('inf') else 999
print(f'MeanRev: PF={pf:.3f} ret={res.total_return_pct:.1f}% DD={res.max_drawdown_pct:.1f}% trades={res.total_trades} win%={res.win_rate:.1f}% final={res.final_balance:.1f}')