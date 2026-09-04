import sys
sys.path.insert(0, '.')
import pandas as pd
from src.ml_walk_forward import MLWalkForwardEngine
from src.ml_config import MLConfig
from src.dataset_builder import DatasetConfig

df = pd.read_parquet('data/btcusdt_15m.parquet').head(8000)
print(f"df {len(df)}")

for label in ['simple','triple_barrier']:
    print(f"\n=== {label} ===")
    ml_config = MLConfig(n_estimators=40, max_depth=5, cv_type='purged', n_splits=3, embargo_pct=0.01, purge_pct=0.01)
    dataset_config = DatasetConfig(label_method=label, future_bars=5, warmup_bars=200, calculate_indicators=True, tb_pt=0.012, tb_sl=0.008, tb_use_atr=True)
    engine = MLWalkForwardEngine(train_size=3000, test_size=800, step_size=800, initial_balance=1000.0, ml_config=ml_config, dataset_config=dataset_config)
    res = engine.run(df)
    print(f"ML WF {label}: windows {res.total_windows} trades {res.total_trades} profit {res.net_profit:.2f} win {res.win_rate:.1f}% BalAcc {res.avg_balanced_accuracy:.3f} F1 {res.avg_f1:.3f}")
    engine.print_report(res)
