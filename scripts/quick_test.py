import sys
sys.path.insert(0, '.')
import pandas as pd
from src.indicators import add_indicators
from src.feature_engine import build_features
from src.dataset_builder import DatasetBuilder, DatasetConfig
from src.labeling import TripleBarrierConfig, triple_barrier_label

# Quick integration test
df = pd.read_parquet('data/btcusdt_4h_prepared_fixed.parquet').head(100)
print('Testing pipeline...')

# Test indicators
df2 = add_indicators(df)
print('Indicators:', df2.shape)

# Test features
feat = build_features(df2)
print('Features:', len(feat))

# Test dataset builder with triple barrier
cfg = DatasetConfig(label_method='triple_barrier', future_bars=5, warmup_bars=50, calculate_indicators=True)
builder = DatasetBuilder(cfg)
ds = builder.build(df.head(2000))
print(f'Dataset: {ds.shape}, target dist: ' + str(ds['target'].value_counts().to_dict()))

# Test triple barrier labeling
from src.labeling import TripleBarrierConfig, triple_barrier_label
cfg2 = TripleBarrierConfig(pt=0.012, sl=0.008, max_holding_bars=5)
r = triple_barrier_label(df2, 50, cfg2)  # Use index 50 instead of 100
print(f'Triple barrier label: {r}')

print('All tests passed!')