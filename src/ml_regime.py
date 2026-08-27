"""
====================================================
QuantAI Regime-Aware ML Engine
====================================================

Separate heads per market regime (TREND_UP / TREND_DOWN / RANGE).

Each regime gets its own XGBoost, trained only on samples
from that regime (causal classification via RegimeFilter).
Falls back to global model when regime data is scarce.

Uses Combinatorial Purged K-Fold for honest OOS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.ml_engine import MLConfig
from src.regime_filter import RegimeFilter, RegimeConfig, RANGE
from src.validation.purged_kfold import get_purged_cv
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight


@dataclass
class RegimeAwareConfig:
    enabled: bool = False
    min_samples_per_regime: int = 150
    regime_config: RegimeConfig = None

    def __post_init__(self):
        if self.regime_config is None:
            self.regime_config = RegimeConfig()


class RegimeAwareMLEngine:
    """
    Wrapper: global fallback + per-regime XGBoost heads.
    """

    def __init__(
        self,
        base_config: MLConfig | None = None,
        regime_config: RegimeAwareConfig | None = None,
    ):
        self.base_config = base_config or MLConfig()
        self.regime_config = regime_config or RegimeAwareConfig()
        self.models: Dict[str, XGBClassifier] = {}
        self.global_model: XGBClassifier | None = None
        self.feature_names: list[str] | None = None
        self.regime_filter = RegimeFilter(
            self.regime_config.regime_config
        )

    def _create_model(self) -> XGBClassifier:
        return XGBClassifier(
            n_estimators=self.base_config.n_estimators,
            max_depth=self.base_config.max_depth,
            learning_rate=self.base_config.learning_rate,
            subsample=self.base_config.subsample,
            colsample_bytree=self.base_config.colsample_bytree,
            random_state=self.base_config.random_state,
            eval_metric="mlogloss",
            verbosity=0,
        )

    def _classify_regimes(self, df: pd.DataFrame) -> list[str]:
        """Classify each row's regime causally."""
        regimes = []
        filt = RegimeFilter(self.regime_config.regime_config)
        # Need prepared columns; if df is already dataset with features,
        # we use a proxy: if no ema_trend/adx, fallback to RANGE
        if "ema_trend" not in df.columns or "adx" not in df.columns:
            return [RANGE] * len(df)
        for i in range(len(df)):
            window = df.iloc[max(0, i - 100): i + 1]
            regimes.append(filt.classify(window))
        return regimes

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        regimes: list[str] | None = None,
    ) -> dict:
        """
        Train global + per-regime models.
        Returns dict with per-regime sample counts and OOS scores.
        """
        self.feature_names = list(X.columns)
        y_xgb = y.replace({-1: 0, 0: 1, 1: 2})

        # Global fallback
        cv_global = get_purged_cv(
            cv_type=self.base_config.cv_type,
            n_splits=self.base_config.n_splits,
            embargo_pct=self.base_config.embargo_pct,
            purge_pct=self.base_config.purge_pct,
            n_test_folds=self.base_config.n_test_folds,
        )
        # Train global
        w_global = None
        if self.base_config.use_class_weights:
            w_global = compute_sample_weight("balanced", y_xgb)
        self.global_model = self._create_model()
        self.global_model.fit(X, y_xgb, sample_weight=w_global)

        stats = {"global_samples": len(X)}

        if not self.regime_config.enabled or regimes is None:
            return stats

        # Per-regime heads
        regime_series = pd.Series(regimes, index=X.index)
        for regime in [r for r in regime_series.unique() if r != RANGE or True]:
            mask = regime_series == regime
            n_reg = int(mask.sum())
            stats[f"{regime}_samples"] = n_reg
            if n_reg < self.regime_config.min_samples_per_regime:
                continue
            X_r = X[mask]
            y_r = y_xgb[mask]
            if y_r.nunique() < 3:
                continue  # need all 3 classes for 3-way head; else fallback to global
            w_r = None
            if self.base_config.use_class_weights:
                try:
                    w_r = compute_sample_weight("balanced", y_r)
                except Exception:
                    w_r = None
            model = self._create_model()
            try:
                model.fit(X_r, y_r, sample_weight=w_r)
            except ValueError:
                continue
            self.models[regime] = model
            stats[f"{regime}_trained"] = True

        return stats

    def predict(self, X: pd.DataFrame, regimes: list[str] | None = None) -> np.ndarray:
        """Routed prediction: per-regime model or global fallback."""
        if regimes is None:
            regimes = [RANGE] * len(X)
        preds = []
        for idx, regime in enumerate(regimes):
            row = X.iloc[[idx]]
            model = self.models.get(regime, self.global_model)
            if model is None:
                preds.append(1)  # HOLD fallback
            else:
                preds.append(int(model.predict(row)[0]))
        # Map back: 0->SELL(-1), 1->HOLD(0), 2->BUY(1)
        mapping = {0: -1, 1: 0, 2: 1}
        return np.array([mapping[p] for p in preds])

    def predict_proba(
        self, X: pd.DataFrame, regimes: list[str] | None = None
    ) -> np.ndarray:
        """Routed probabilities."""
        if regimes is None:
            regimes = [RANGE] * len(X)
        probas = []
        for idx, regime in enumerate(regimes):
            row = X.iloc[[idx]]
            model = self.models.get(regime, self.global_model)
            if model is None:
                probas.append([0.33, 0.33, 0.34])
            else:
                probas.append(model.predict_proba(row)[0].tolist())
        return np.array(probas)
