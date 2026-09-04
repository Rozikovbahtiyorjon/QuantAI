"""
Bulk register 436 parquet datasets — Audit §33
"""

import sys
sys.path.insert(0, ".")
import pathlib
from src.research.dataset_registry import DatasetRegistry

reg = DatasetRegistry()
count = 0
for p in pathlib.Path("data").glob("*.parquet"):
    if "archive" in str(p) or ".bak" in str(p) or "tmp" in str(p):
        continue
    # infer symbol/timeframe from filename like btcusdt_4h_prepared.parquet
    name = p.stem  # btcusdt_4h_prepared
    parts = name.split("_")
    if len(parts) >= 2:
        symbol = parts[0].upper()
        tf = parts[1]
        dataset_id = f"{symbol}_{tf.upper()}_v7"
        try:
            reg.register(str(p), dataset_id, symbol, tf)
            count += 1
        except Exception as e:
            print(f"skip {p}: {e}")
print(f"Registered {count} datasets")
print(f"Registry has {len(reg.list_all())} entries")
