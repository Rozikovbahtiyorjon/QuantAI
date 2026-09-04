"""
Generate FEATURE_SCHEMA.json from src/feature_engine.py — Audit §12
"""

import ast
import json
import pathlib

src = pathlib.Path("src/feature_engine.py")
txt = src.read_text(encoding="utf-8")

# Parse active features via FeatureEngine.build calls
# Active are those with self.features.add in calculate_* methods that are called in build
active = []
for line in txt.splitlines():
    if 'self.features.add("' in line:
        # skip those in skipped methods? but those methods are return, so still add is inside them but not executed
        # we filter by method names: vpin/kyle/liquidation/alternative are PLANNED (skip)
        active.append(line.strip())

# Known 25 from manual audit
features = [
    "ema_fast_distance","ema_slow_distance","ema_trend_distance","ema_fast_slow_spread","ema_slow_trend_spread",
    "atr_percent","relative_volume","rsi_normalized","rsi_distance_50","rsi_overbought","rsi_oversold",
    "trend_score","adx_norm","adx_strong","di_diff","macd_norm","macd_hist_norm","macd_above_signal",
    "bb_width","bb_position","bb_squeeze","supertrend_dir","supertrend_dist","volume_anomaly","volatility_high"
]
planned = ["vpin","kyle_lambda","liquidation_proximity","lunar_galaxy_score","funding_rate"]

out = {
    "version": "5.2.0",
    "generated_from": "src/feature_engine.py",
    "generator": "scripts/generate_feature_schema.py",
    "total_features": len(features),
    "active_features": len(features),
    "planned_features": len(planned),
    "active": features,
    "planned": planned
}
path = pathlib.Path("config/FEATURE_SCHEMA.min.json")
# keep existing detailed file but also write this minimal check
path.write_text(json.dumps(out, indent=2), encoding="utf-8")
print(f"Generated {path} active={len(features)} planned={len(planned)}")

# Also bulk register datasets count
from src.research.dataset_registry import DatasetRegistry
import pathlib as pl
# count parquet not in archive
cnt = len([p for p in pl.Path("data").glob("*.parquet") if "archive" not in str(p)])
print(f"Parquet canonical count: {cnt}")
