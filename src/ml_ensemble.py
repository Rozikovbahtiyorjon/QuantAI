"""
====================================================
QuantAI Heterogeneous Ensemble (Stage 4)
====================================================

Ensemble of diverse architectures:
  - XGBoost  (gradient boosting, tree)
  - RandomForest (bagging, tree)
  - HistGradientBoosting (sklearn, proxy for LightGBM/CatBoost)

If LightGBM / CatBoost are installed, they are auto-used
as drop-in replacements (adapter).

Regime-weighted:
  - Per-regime weights learned via CPCV OOS balanced_accuracy
  - TREND_UP/DOWN/RANGE get their own weight vector
  - Falls back to global weights when regime data scarce
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

from src.ml_config import MLConfig
from src.regime_filter import RANGE
from src.validation.purged_kfold import get_purged_cv
from xgboost import XGBClassifier


def _try_import_lightgbm():
    try:
        import lightgbm as lgb

        return lgb.LGBMClassifier
    except ImportError:
        return None


def _try_import_catboost():
    try:
        from catboost import CatBoostClassifier

        return CatBoostClassifier
    except ImportError:
        return None


@dataclass
class EnsembleConfig:
    enabled: bool = True
    use_lightgbm: bool = True
    use_catboost: bool = True
    regime_weighted: bool = True
    min_samples_per_regime: int = 150


class HeterogeneousEnsemble:
    """
    Heterogeneous ensemble with regime-weighted voting.
    """

    def __init__(
        self,
        base_config: MLConfig | None = None,
        ensemble_config: EnsembleConfig | None = None,
    ):
        self.base_config = base_config or MLConfig()
        self.ensemble_config = ensemble_config or EnsembleConfig()
        self.models: Dict[str, Dict[str, object]] = {}  # regime -> {name: model}
        self.weights: Dict[str, Dict[str, float]] = {}  # regime -> {name: weight}
        self.feature_names: list[str] | None = None
        self._global_weights: Dict[str, float] | None = None

    def _create_estimators(self) -> List[tuple[str, object]]:
        estimators = []

        # XGBoost (always)
        estimators.append(
            (
                "xgb",
                XGBClassifier(
                    n_estimators=self.base_config.n_estimators,
                    max_depth=self.base_config.max_depth,
                    learning_rate=self.base_config.learning_rate,
                    subsample=self.base_config.subsample,
                    colsample_bytree=self.base_config.colsample_bytree,
                    random_state=self.base_config.random_state,
                    eval_metric="mlogloss",
                    verbosity=0,
                ),
            )
        )

        # LightGBM or fallback RandomForest
        lgb_cls = _try_import_lightgbm() if self.ensemble_config.use_lightgbm else None
        if lgb_cls is not None:
            estimators.append(
                (
                    "lgb",
                    lgb_cls(
                        n_estimators=200,
                        max_depth=self.base_config.max_depth,
                        learning_rate=self.base_config.learning_rate,
                        subsample=self.base_config.subsample,
                        verbose=-1,
                        random_state=self.base_config.random_state,
                    ),
                )
            )
        else:
            estimators.append(
                (
                    "rf",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=self.base_config.max_depth,
                        random_state=self.base_config.random_state,
                        n_jobs=-1,
                    ),
                )
            )

        # CatBoost or HistGradientBoosting fallback
        cat_cls = _try_import_catboost() if self.ensemble_config.use_catboost else None
        if cat_cls is not None:
            estimators.append(
                (
                    "cat",
                    cat_cls(
                        iterations=200,
                        depth=self.base_config.max_depth,
                        learning_rate=self.base_config.learning_rate,
                        verbose=False,
                        random_seed=self.base_config.random_state,
                    ),
                )
            )
        else:
            estimators.append(
                (
                    "hgb",
                    HistGradientBoostingClassifier(
                        max_depth=self.base_config.max_depth,
                        learning_rate=self.base_config.learning_rate,
                        random_state=self.base_config.random_state,
                    ),
                )
            )

        return estimators

    def _learn_weights_cpcv(
        self, X: pd.DataFrame, y: pd.Series, estimators: List[tuple[str, object]]
    ) -> Dict[str, float]:
        """Learn ensemble weights via CPCV OOS balanced_accuracy."""
        cv = get_purged_cv(
            cv_type="combinatorial",
            n_splits=5,
            embargo_pct=self.base_config.embargo_pct,
            purge_pct=self.base_config.purge_pct,
            n_test_folds=2,
        )
        # Map y: -1,0,1 -> 0,1,2 for sklearn
        y_mapped = y.replace({-1: 0, 0: 1, 1: 2})
        scores: Dict[str, list[float]] = {name: [] for name, _ in estimators}

        for train_idx, test_idx in cv.split(X, y_mapped):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y_mapped.iloc[train_idx], y_mapped.iloc[test_idx]
            for name, est in estimators:
                # Clone estimator
                from sklearn.base import clone

                model = clone(est)
                try:
                    model.fit(X_tr, y_tr)
                    pred = model.predict(X_te)
                    # Map back for scoring consistency (balanced acc is label-invariant)
                    scores[name].append(balanced_accuracy_score(y_te, pred))
                except Exception:
                    scores[name].append(0.0)

        # Average OOS score per estimator -> softmax weights
        avg_scores = {name: float(np.mean(v)) if v else 0.0 for name, v in scores.items()}
        # Softmax with temperature 1.0 (avoid overflow)
        vals = np.array(list(avg_scores.values()))
        exp_vals = np.exp(vals - np.max(vals))
        softmax = exp_vals / exp_vals.sum()
        weights = {name: float(w) for (name, _), w in zip(estimators, softmax)}
        return weights

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        regimes: list[str] | None = None,
    ) -> dict:
        self.feature_names = list(X.columns)
        stats: dict = {"global_samples": len(X)}

        # Learn global weights
        estimators = self._create_estimators()
        self._global_weights = self._learn_weights_cpcv(X, y, estimators)

        # Train global models (for fallback)
        y_mapped = y.replace({-1: 0, 0: 1, 1: 2})
        global_models: Dict[str, object] = {}
        for name, est in estimators:
            from sklearn.base import clone

            m = clone(est)
            m.fit(X, y_mapped)
            global_models[name] = m
        self.models[RANGE] = global_models  # RANGE key holds global fallback
        self.weights[RANGE] = self._global_weights
        stats["global_weights"] = self._global_weights

        if not self.ensemble_config.regime_weighted or regimes is None:
            return stats

        # Per-regime ensembles
        regime_series = pd.Series(regimes, index=X.index)
        for regime in regime_series.unique():
            mask = regime_series == regime
            n_reg = int(mask.sum())
            stats[f"{regime}_samples"] = n_reg
            if n_reg < self.ensemble_config.min_samples_per_regime:
                continue
            X_r = X[mask]
            y_r = y[mask]
            if y_r.nunique() < 2:
                continue
            # Learn regime-specific weights
            w_reg = self._learn_weights_cpcv(X_r, y_r, estimators)
            # Train regime-specific models
            y_r_mapped = y_r.replace({-1: 0, 0: 1, 1: 2})
            reg_models: Dict[str, object] = {}
            for name, est in estimators:
                from sklearn.base import clone

                m = clone(est)
                m.fit(X_r, y_r_mapped)
                reg_models[name] = m
            self.models[regime] = reg_models
            self.weights[regime] = w_reg
            stats[f"{regime}_trained"] = True
            stats[f"{regime}_weights"] = w_reg

        return stats

    def predict_proba(
        self, X: pd.DataFrame, regimes: list[str] | None = None
    ) -> np.ndarray:
        if regimes is None:
            regimes = [RANGE] * len(X)
        probas = []
        for idx, regime in enumerate(regimes):
            row = X.iloc[[idx]]
            models = self.models.get(regime, self.models.get(RANGE, {}))
            weights = self.weights.get(regime, self._global_weights or {})
            if not models:
                probas.append([0.33, 0.33, 0.34])
                continue
            # Weighted average of probas
            avg_proba = np.zeros(3)
            total_w = 0.0
            for name, model in models.items():
                w = weights.get(name, 1.0 / len(models))
                try:
                    p = model.predict_proba(row)[0]
                    # Ensure 3 classes; pad if model saw only 2
                    if len(p) == 2:
                        # Heuristic: assume missing class is middle (HOLD) if not present
                        # For simplicity, pad to 3 with uniform
                        p = np.array([p[0], 0.0, p[1]]) if len(p) == 2 else p
                    if len(p) != 3:
                        p = np.array([0.33, 0.33, 0.34])
                    avg_proba += w * np.array(p)
                    total_w += w
                except Exception:
                    continue
            if total_w > 0:
                avg_proba /= total_w
            else:
                avg_proba = np.array([0.33, 0.33, 0.34])
            probas.append(avg_proba.tolist())
        return np.array(probas)

    def predict(self, X: pd.DataFrame, regimes: list[str] | None = None) -> np.ndarray:
        probas = self.predict_proba(X, regimes)
        # probas are in order [SELL(0), HOLD(1), BUY(2)] -> map to -1,0,1
        idx = probas.argmax(axis=1)
        mapping = {0: -1, 1: 0, 2: 1}
        return np.array([mapping[i] for i in idx])
