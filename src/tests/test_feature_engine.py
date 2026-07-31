"""
==========================================
Feature Engine Test
==========================================
"""

from src.data_loader import load_binance_data
from src.indicators import add_indicators
from src.feature_engine import build_features


df = load_binance_data()

df = add_indicators(df)

features = build_features(df)

print()

print("=" * 60)
print("FEATURE ENGINE")
print("=" * 60)

for key, value in features.items():

    print(f"{key:30s} {value:.6f}")

print("=" * 60)