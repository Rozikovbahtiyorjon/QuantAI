"""
====================================================
Machine Learning Engine Test
====================================================
"""

from src.data_loader import load_binance_data
from src.indicators import add_indicators
from src.dataset_builder import build_dataset
from src.ml_engine import train_model

print("Loading data...")

df = load_binance_data()

print("Calculating indicators...")

df = add_indicators(df)

print("Building dataset...")

dataset = build_dataset(df)

print()

print("=" * 60)
print("DATASET READY")
print("=" * 60)

print(f"Rows    : {len(dataset)}")
print(f"Columns : {len(dataset.columns)}")

print()

engine, result = train_model(dataset)

print()

print("=" * 60)
print("TRAINING FINISHED")
print("=" * 60)