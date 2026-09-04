import sys
sys.path.insert(0, '.')
import pandas as pd
from src.indicators import add_indicators
from src.backtest_engine import BacktestEngine
from src.strategy.signal_generator import SignalGenerator, SignalConfig
from src.trade_engine import TradeEngine

raw = pd.read_parquet('data/btcusdt_4h_prepared_fixed.parquet')[['timestamp','open','high','low','close','volume']]
df = add_indicators(raw)
print(f"4h {len(df)}")

# Grid search WeightedGate + trend weight
thresholds = [0.60, 0.65, 0.70, 0.75]
long_thrs = [0.50, 0.55, 0.60]
trend_weights = [1.2, 1.5, 1.8]

best = None
results = []
for thr in thresholds:
    for lthr in long_thrs:
        for tw in trend_weights:
            cfg = SignalConfig(
                min_confidence=60.0,
                use_ml=False,
                use_order_flow=False,
                use_weighted_gate=True,
                weighted_gate_threshold=thr,
                weighted_gate_min_confidence=60.0,
                weighted_gate_long_threshold=lthr,
                weighted_gate_short_threshold=lthr,
                trend_weight=tw,
                momentum_weight=1.2,
                volume_weight=1.1,
                volatility_weight=1.0,
            )
            # Inject into TradeEngine via signal_generator param
            from src.trade_engine import TradeEngine
            te = TradeEngine()
            # Monkey patch generate_signal_result to use our cfg
            import src.trade_engine as te_mod
            orig_factory = te_mod.generate_signal_result
            def make_factory(cfg_inner):
                def factory(df_hist):
                    sg = SignalGenerator(config=cfg_inner)
                    return sg.generate(df_hist)
                return factory
            te_mod.generate_signal_result = make_factory(cfg)
            try:
                be = BacktestEngine(initial_balance=1000.0)
                # Need to pass signal_generator? TradeEngine.run will use factory
                res = be.run(df)
                pf = res.profit_factor if res.profit_factor != float('inf') else 0
                results.append((pf, res.total_return_pct, res.max_drawdown_pct, res.total_trades, thr, lthr, tw, res.sharpe))
                if best is None or pf > best[0]:
                    best = (pf, res.total_return_pct, thr, lthr, tw, res)
                print(f"thr {thr:.2f} lthr {lthr:.2f} tw {tw:.1f} => PF {pf:.3f} ret {res.total_return_pct:.1f}% DD {res.max_drawdown_pct:.1f}% trades {res.total_trades} Sharpe {res.sharpe:.2f}")
            except Exception as e:
                print(f"thr {thr} lthr {lthr} tw {tw} ERROR {e}")
            finally:
                te_mod.generate_signal_result = orig_factory

print("\n=== BEST ===")
if best:
    pf, ret, thr, lthr, tw, res = best
    print(f"Best PF {pf:.3f} ret {ret:.1f}% thr {thr} lthr {lthr} tw {tw} trades {res.total_trades} Sharpe {res.sharpe:.2f} DD {res.max_drawdown_pct:.1f}%")
    # Print top 5
    results_sorted = sorted(results, key=lambda x: x[0], reverse=True)[:5]
    for r in results_sorted:
        print(f" PF {r[0]:.3f} ret {r[1]:.1f}% DD {r[2]:.1f}% trades {r[3]} thr {r[4]} lthr {r[5]} tw {r[6]} Sharpe {r[7]:.2f}")
