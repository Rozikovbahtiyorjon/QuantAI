"""
====================================================
QuantAI ML Engine Diagnostic Test
====================================================
"""

from collections import Counter

from src.data_loader import load_binance_data
from src.indicators import add_indicators
from src.dataset_builder import build_dataset
from src.ml_engine import train_model


print("=" * 60)
print("QUANTAI ML ENGINE DIAGNOSTIC")
print("=" * 60)

print()
print("Loading data...")

df = load_binance_data()

print()
print("Calculating indicators...")

df = add_indicators(df)

print()
print("Building dataset...")

dataset = build_dataset(df)

print()
print("=" * 60)
print("DATASET READY")
print("=" * 60)

print(f"Rows    : {len(dataset)}")
print(f"Columns : {len(dataset.columns)}")

# ====================================================
# TARGET COLUMN
# ====================================================

print()
print("=" * 60)
print("TARGET COLUMN DIAGNOSTIC")
print("=" * 60)

target_candidates = [
    "target",
    "label",
    "signal",
    "y",
]

target_column = None

for column in target_candidates:

    if column in dataset.columns:
        target_column = column
        break

if target_column is None:

    print()
    print("ERROR: Target column not found.")

    print()
    print("Available columns:")
    print(list(dataset.columns))

    raise SystemExit(1)


print(f"Target column : {target_column}")

# ====================================================
# CLASS DISTRIBUTION
# ====================================================

target = dataset[target_column]

counts = target.value_counts().sort_index()

total = len(target)

print()
print("-" * 60)
print("CLASS DISTRIBUTION")
print("-" * 60)

for cls, count in counts.items():

    percentage = (
        count / total * 100
        if total > 0
        else 0
    )

    print(
        f"Class {cls!s:>5} : "
        f"{count:>6} "
        f"({percentage:>6.2f}%)"
    )

print("-" * 60)

# ====================================================
# EXPECTED TRADING CLASSES
# ====================================================

print()
print("TRADING CLASS INTERPRETATION")
print("-" * 60)

class_map = {
    1: "BUY",
    -1: "SELL",
    0: "HOLD",
}

for cls in sorted(counts.index):

    name = class_map.get(
        cls,
        "UNKNOWN",
    )

    print(
        f"{cls:>5} = {name}"
    )

# ====================================================
# TRAIN MODEL
# ====================================================

print()
print("=" * 60)
print("TRAINING XGBOOST")
print("=" * 60)

engine, result = train_model(dataset)

# ====================================================
# MODEL CLASSES
# ====================================================

print()
print("=" * 60)
print("MODEL CLASSES")
print("=" * 60)

print(
    "Classes:",
    engine.model.classes_
)

# ====================================================
# FEATURE IMPORTANCE
# ====================================================

print()
print("=" * 60)
print("TOP FEATURES")
print("=" * 60)

try:

    importance = engine.feature_importance(
        top=20,
    )

    print(
        importance.to_string(
            index=False,
        )
    )

except Exception as exc:

    print(
        f"Feature importance error: {exc}"
    )

# ====================================================
# PERFORMANCE
# ====================================================

print()
print("=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(
    f"Accuracy  : {result.accuracy:.4f}"
)

print(
    f"Precision : {result.precision:.4f}"
)

print(
    f"Recall    : {result.recall:.4f}"
)

print(
    f"F1 Score  : {result.f1:.4f}"
)

print("=" * 60)

print()
print("DIAGNOSTIC FINISHED")
print("=" * 60)