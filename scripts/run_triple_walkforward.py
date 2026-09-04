"""ML Walk-Forward with Triple-Barrier for btcusdt_15m."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.ml_walk_forward import MLWalkForwardEngine
from src.ml_config import MLConfig
from src.dataset_builder import DatasetConfig

fp = Path("data/btcusdt_15m.parquet")
print(f"Loading {fp} ...")
df = pd.read_parquet(fp)
print(f"Rows {len(df)}")

ml_config = MLConfig(
    n_estimators=150,  # reduced for speed (was 300)
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    use_class_weights=True,
    cv_type="purged",  # not combinatorial for speed (was 10 folds)
    n_splits=5,
    embargo_pct=0.01,
    purge_pct=0.01,
    use_ensemble=False,
)

dataset_config = DatasetConfig(
    label_method="triple_barrier",
    future_bars=5,
    warmup_bars=200,
    tb_pt=0.012,
    tb_sl=0.008,
    tb_use_atr=True,
    tb_atr_pt_mult=3.0,
    tb_atr_sl_mult=1.5,
    tb_min_net_return=None,
    calculate_indicators=True,
)

print(f"ML Config: {ml_config}")
print(f"Dataset: {dataset_config.label_method} pt={dataset_config.tb_pt} sl={dataset_config.tb_sl} atr={dataset_config.tb_use_atr}")

# Smaller windows for 15m: train 3000 (~1 month), test 500 (~5 days), step 500
engine = MLWalkForwardEngine(
    train_size=3000,
    test_size=500,
    step_size=500,
    initial_balance=1000.0,
    ml_config=ml_config,
    dataset_config=dataset_config,
)

print("Running ML Walk-Forward ... (expect ~8-12 min for 107k rows)")
result = engine.run(df)
engine.print_report(result)

# Also print trading readiness expectation
print("\n=== TRADING READINESS PREVIEW ===")
if result.net_profit > 0 and result.total_trades > 100:
    print(f"PASS candidate: profit {result.net_profit:.2f} trades {result.total_trades} win {result.win_rate:.1f}%")
else:
    print(f"FAIL: profit {result.net_profit:.2f} trades {result.total_trades} — need PF>1 walk-forward")
