"""
==========================================
Feature Engine Test
==========================================
"""

from src.data_loader import load_binance_data
from src.indicators import add_indicators
from src.feature_engine import build_features

print("Loading data...")

df = load_binance_data()

print("Calculating indicators...")

df = add_indicators(df)

print("Building features...")

features = build_features(df)

print()

print("=" * 60)
print("FEATURE ENGINE")
print("=" * 60)

for key, value in features.items():

    print(f"{key:30s} {value:.6f}")

print("=" * 60)