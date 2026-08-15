"""
==========================================
Dataset Builder Test
==========================================
"""

from src.data_loader import load_binance_data
from src.indicators import add_indicators
from src.dataset_builder import build_dataset

print("Loading data...")

df = load_binance_data()

print("Calculating indicators...")

df = add_indicators(df)

print("Building dataset...")

dataset = build_dataset(df)

print()

print("=" * 60)
print("DATASET")
print("=" * 60)

print(f"Rows    : {len(dataset)}")
print(f"Columns : {len(dataset.columns)}")

print()

print(dataset.head())

print()

print("=" * 60)
print("Target Distribution")
print("=" * 60)

print(dataset["target"].value_counts())

print("=" * 60)