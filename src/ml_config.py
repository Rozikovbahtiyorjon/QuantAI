"""
====================================================
QuantAI ML Config (P0 fix)
Single source of truth for ML configuration.
Breaks circular import: ml_engine <-> ml_ensemble
====================================================
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MLConfig:
    test_size: float = 0.20
    random_state: int = 42
    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.90
    colsample_bytree: float = 0.90
    use_class_weights: bool = True

    # Purged K-Fold parameters
    cv_type: str = "combinatorial"  # "purged" or "combinatorial"
    n_splits: int = 5
    embargo_pct: float = 0.01
    purge_pct: float = 0.0

    # Regime-aware heads
    regime_aware: bool = False
    regime_min_samples: int = 150
    n_test_folds: int = 2  # For combinatorial CV

    # Ensemble settings
    use_ensemble: bool = False
    ensemble_use_lightgbm: bool = True
    ensemble_use_catboost: bool = True
    ensemble_regime_weighted: bool = True
    ensemble_min_samples_per_regime: int = 150


__all__ = ["MLConfig"]
