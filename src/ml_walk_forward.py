"""
====================================================
QuantAI Professional
ML Walk-Forward Integration
====================================================

Integrates PurgedKFold cross-validation into Walk-Forward
engine for proper time-series ML validation.

Flow per window:
  1. Train window → build dataset → PurgedKFold CV → train model
  2. Model saved to ModelManager
  3. Test window → backtest with ML-enabled strategy
  4. Results aggregated across windows

This prevents look-ahead bias by:
- PurgedKFold within each training window (no shuffle)
- Walk-forward across windows (sequential out-of-sample)
====================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import pandas as pd

from src.backtest_engine import BacktestEngine, BacktestResult
from src.dataset_builder import DatasetBuilder, DatasetConfig
from src.ml_engine import MLEngine, MLConfig, TrainingResult
from src.model_manager import ModelManager
from src.validation.purged_kfold import get_purged_cv
from src.walk.walk_forward_engine import (
    WalkForwardEngine,
    WalkForwardResult,
    WalkForwardWindowResult,
    TrainCallback,
)


@dataclass
class MLWalkForwardWindowResult:
    """Extended window result with ML diagnostics."""
    window_id: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    train_size: int
    test_size: int
    backtest_result: BacktestResult
    training_result: Optional[TrainingResult] = None
    model_path: Optional[str] = None
    cv_scores: dict = field(default_factory=dict)
    feature_importance: Optional[pd.DataFrame] = None
    model_result: dict = field(default_factory=dict)


@dataclass
class MLWalkForwardResult:
    """Complete ML Walk-Forward result."""
    initial_balance: float
    final_balance: float
    net_profit: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    windows: list[MLWalkForwardWindowResult] = field(default_factory=list)
    
    # ML aggregate metrics
    avg_balanced_accuracy: float = 0.0
    avg_f1: float = 0.0
    avg_precision: float = 0.0
    avg_recall: float = 0.0
    models_trained: int = 0
    
    @property
    def total_windows(self) -> int:
        return len(self.windows)


def create_ml_train_callback(
    ml_config: MLConfig,
    dataset_config: DatasetConfig,
    model_manager: Optional[ModelManager] = None,
    save_each_window: bool = True,
    feature_store=None,
) -> TrainCallback:
    """
    Create a train_callback for WalkForwardEngine that:
    1. Builds dataset from train_df
    2. Runs PurgedKFold cross-validation
    3. Trains final model on full train_df
    4. Saves model via ModelManager
    
    Args:
        ml_config: ML engine configuration
        dataset_config: Dataset building configuration
        model_manager: Optional custom ModelManager
        save_each_window: If True, saves model per window
    
    Returns:
        Callable compatible with WalkForwardEngine.train_callback
    """
    mm = model_manager or ModelManager()
    
    def train_callback(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        window_id: int,
    ) -> dict:
        # -------------------------------------------------
        # 1. BUILD DATASET FROM TRAIN WINDOW
        #     (auto-materializes to Feature Store if enabled)
        # -------------------------------------------------
        # Per-window view for versioned tracking of walk-forward folds
        per_window_config = dataset_config
        if dataset_config.feature_store_enabled and feature_store is None:
            # Clone config with window-specific view for versioning
            from dataclasses import replace

            per_window_config = replace(
                dataset_config,
                feature_store_view=f"{dataset_config.feature_store_view}_window_{window_id}",
            )

        builder = DatasetBuilder(
            config=per_window_config, feature_store=feature_store
        )
        
        # Build dataset - this adds indicators, features, targets
        # and auto-materializes to Feature Store if enabled
        dataset = builder.build(train_df)
        
        if dataset.empty:
            raise ValueError(
                f"Window {window_id}: Empty dataset after building. "
                f"Check train_size and warmup_bars."
            )
        
        # -------------------------------------------------
        # 2. PURGED K-FOLD CROSS-VALIDATION
        # -------------------------------------------------
        cv = get_purged_cv(
            cv_type=ml_config.cv_type,
            n_splits=ml_config.n_splits,
            embargo_pct=ml_config.embargo_pct,
            purge_pct=ml_config.purge_pct,
            n_test_folds=ml_config.n_test_folds,
        )
        
        # Prepare features and target
        from src.feature_engine import build_features
        from src.ml_engine import MLEngine
        
        engine = MLEngine(config=ml_config)
        
        # Get features and target
        y = dataset["target"].astype(int)
        
        X = dataset.drop(
            columns=["target", "future_return", "index", "tb_barrier", "tb_t1"],
            errors="ignore",
        )
        # Keep only numeric features (tb_barrier is string diagnostic)
        X = X.select_dtypes(include=["number"])
        
        # QuantAI → XGBoost class mapping
        y_xgb = y.replace({-1: 0, 0: 1, 1: 2})
        
        # Run PurgedKFold CV
        cv_scores = []
        fold_importances = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y_xgb)):
            X_fold_train = X.iloc[train_idx]
            y_fold_train = y_xgb.iloc[train_idx]
            X_fold_val = X.iloc[val_idx]
            y_fold_val = y_xgb.iloc[val_idx]
            
            # Train fold model
            from xgboost import XGBClassifier
            fold_model = XGBClassifier(
                n_estimators=ml_config.n_estimators,
                max_depth=ml_config.max_depth,
                learning_rate=ml_config.learning_rate,
                subsample=ml_config.subsample,
                colsample_bytree=ml_config.colsample_bytree,
                random_state=ml_config.random_state + fold_idx,
                objective="multi:softprob",
                num_class=3,
                eval_metric="mlogloss",
                tree_method="hist",
            )
            
            if ml_config.use_class_weights:
                from sklearn.utils.class_weight import compute_sample_weight
                sample_weight = compute_sample_weight(
                    class_weight="balanced", y=y_fold_train
                )
                fold_model.fit(X_fold_train, y_fold_train, sample_weight=sample_weight)
            else:
                fold_model.fit(X_fold_train, y_fold_train)
            
            # Validate
            from sklearn.metrics import balanced_accuracy_score, f1_score
            fold_pred = fold_model.predict(X_fold_val)
            fold_bal_acc = balanced_accuracy_score(y_fold_val, fold_pred)
            fold_f1 = f1_score(y_fold_val, fold_pred, average="macro", zero_division=0)
            
            cv_scores.append({
                "fold": fold_idx,
                "balanced_accuracy": fold_bal_acc,
                "f1_macro": fold_f1,
            })
            
            # Store feature importance
            fold_importances.append(pd.Series(
                fold_model.feature_importances_,
                index=X.columns,
            ))
        
        # Average feature importance across folds
        if fold_importances:
            avg_importance = pd.concat(fold_importances, axis=1).mean(axis=1).sort_values(ascending=False)
        else:
            avg_importance = pd.Series(dtype=float)
        
        # -------------------------------------------------
        # 3. TRAIN FINAL MODEL ON FULL TRAIN DATA
        # -------------------------------------------------
        training_result = engine.train(dataset)
        
        # -------------------------------------------------
        # 4. SAVE MODEL
        # -------------------------------------------------
        model_path = None
        if save_each_window:
            from pathlib import Path
            window_model_path = Path(f"models/quantai_wf_window_{window_id}.pkl")
            window_model_path.parent.mkdir(parents=True, exist_ok=True)
            mm.save(engine.model, window_model_path)
            model_path = str(window_model_path)
        
        # Also save as latest
        mm.save(engine.model)
        
        # -------------------------------------------------
        # 5. RETURN DIAGNOSTICS
        # -------------------------------------------------
        return {
            "training_result": training_result,
            "cv_scores": cv_scores,
            "avg_cv_balanced_accuracy": sum(s["balanced_accuracy"] for s in cv_scores) / len(cv_scores) if cv_scores else 0,
            "avg_cv_f1": sum(s["f1_macro"] for s in cv_scores) / len(cv_scores) if cv_scores else 0,
            "feature_importance": avg_importance.head(20),
            "model_path": model_path,
            "window_id": window_id,
            "dataset_stats": builder.statistics(dataset),
        }
    
    return train_callback


class MLWalkForwardEngine:
    """
    Walk-Forward engine with integrated ML training and validation.
    
    Extends WalkForwardEngine with:
    - PurgedKFold CV in each training window
    - ML model training per window
    - ML-enabled backtesting in test windows
    - Aggregated ML performance metrics
    """
    
    def __init__(
        self,
        train_size: int = 500,
        test_size: int = 100,
        step_size: Optional[int] = None,
        initial_balance: float = 1000.0,
        ml_config: Optional[MLConfig] = None,
        dataset_config: Optional[DatasetConfig] = None,
        model_manager: Optional[ModelManager] = None,
        save_each_window: bool = True,
        feature_store=None,
    ) -> None:
        self.ml_config = ml_config or MLConfig()
        self.dataset_config = dataset_config or DatasetConfig()
        self.model_manager = model_manager or ModelManager()
        self.save_each_window = save_each_window
        self.feature_store = feature_store
        
        # Create ML train callback (with Feature Store versioning per fold)
        train_callback = create_ml_train_callback(
            ml_config=self.ml_config,
            dataset_config=self.dataset_config,
            model_manager=self.model_manager,
            save_each_window=save_each_window,
            feature_store=feature_store,
        )
        
        # Create base engine with ML callback
        self.base_engine = WalkForwardEngine(
            train_size=train_size,
            test_size=test_size,
            step_size=step_size,
            initial_balance=initial_balance,
            train_callback=train_callback,
        )
        
        self._result: Optional[MLWalkForwardResult] = None
    
    def run(self, df: pd.DataFrame) -> MLWalkForwardResult:
        """Run ML Walk-Forward analysis."""
        # Run base engine (which calls train_callback per window)
        base_result = self.base_engine.run(df)
        
        # Convert results to MLWalkForwardResult
        ml_windows = []
        for w in base_result.windows:
            ml_window = MLWalkForwardWindowResult(
                window_id=w.window_id,
                train_start=w.train_start,
                train_end=w.train_end,
                test_start=w.test_start,
                test_end=w.test_end,
                train_size=w.train_size,
                test_size=w.test_size,
                backtest_result=w.backtest_result,
                training_result=w.model_result.get("training_result") if w.model_result else None,
                model_path=w.model_result.get("model_path") if w.model_result else None,
                cv_scores=w.model_result.get("cv_scores", []) if w.model_result else [],
                feature_importance=w.model_result.get("feature_importance") if w.model_result else None,
            )
            ml_windows.append(ml_window)
        
        # Calculate aggregate ML metrics
        windows_with_ml = [w for w in ml_windows if w.training_result is not None]
        if windows_with_ml:
            avg_bal_acc = sum(w.training_result.balanced_accuracy for w in windows_with_ml) / len(windows_with_ml)
            avg_f1 = sum(w.training_result.f1 for w in windows_with_ml) / len(windows_with_ml)
            avg_prec = sum(w.training_result.precision for w in windows_with_ml) / len(windows_with_ml)
            avg_rec = sum(w.training_result.recall for w in windows_with_ml) / len(windows_with_ml)
        else:
            avg_bal_acc = avg_f1 = avg_prec = avg_rec = 0.0
        
        self._result = MLWalkForwardResult(
            initial_balance=base_result.initial_balance,
            final_balance=base_result.final_balance,
            net_profit=base_result.net_profit,
            total_trades=base_result.total_trades,
            winning_trades=base_result.winning_trades,
            losing_trades=base_result.losing_trades,
            win_rate=base_result.win_rate,
            windows=ml_windows,
            avg_balanced_accuracy=avg_bal_acc,
            avg_f1=avg_f1,
            avg_precision=avg_prec,
            avg_recall=avg_rec,
            models_trained=len(windows_with_ml),
        )
        
        return self._result
    
    @property
    def result(self) -> Optional[MLWalkForwardResult]:
        return self._result
    
    def reset(self) -> None:
        self.base_engine.reset()
        self._result = None
    
    @staticmethod
    def print_report(result: MLWalkForwardResult) -> None:
        """Print extended ML Walk-Forward report."""
        # Print base-style report manually (MLWalkForwardResult is not WalkForwardResult)
        print()
        print("=" * 70)
        print("QUANTAI ML WALK-FORWARD REPORT")
        print("=" * 70)
        print(f"Initial Balance : {result.initial_balance:.2f}")
        print(f"Final Balance   : {result.final_balance:.2f}")
        print(f"Net Profit      : {result.net_profit:.2f}")
        print("-" * 70)
        print(f"Windows         : {result.total_windows}")
        print(f"Total Trades    : {result.total_trades}")
        print(f"Winning Trades  : {result.winning_trades}")
        print(f"Losing Trades   : {result.losing_trades}")
        print(f"Win Rate        : {result.win_rate:.2f}%")
        print("-" * 70)
        
        for window in result.windows:
            bt = window.backtest_result
            model_status = "trained" if window.training_result else "not_trained"
            print(
                f"Window {window.window_id}: "
                f"TRAIN={window.train_start}:{window.train_end} | "
                f"TEST={window.test_start}:{window.test_end} | "
                f"train_size={window.train_size} | "
                f"test_size={window.test_size} | "
                f"trades={bt.total_trades} | "
                f"profit={bt.net_profit:.2f} | "
                f"win_rate={bt.win_rate:.2f}% | "
                f"model={model_status}"
            )
        
        print("-" * 70)
        print("ML WALK-FORWARD METRICS")
        print("=" * 70)
        print(f"Models Trained        : {result.models_trained}")
        print(f"Avg Balanced Accuracy : {result.avg_balanced_accuracy:.4f}")
        print(f"Avg F1 Score          : {result.avg_f1:.4f}")
        print(f"Avg Precision         : {result.avg_precision:.4f}")
        print(f"Avg Recall            : {result.avg_recall:.4f}")
        print("-" * 70)
        
        for window in result.windows:
            if window.training_result:
                tr = window.training_result
                cv_bal_acc = window.model_result.get('avg_cv_balanced_accuracy', 0) if window.model_result else 0
                cv_f1 = window.model_result.get('avg_cv_f1', 0) if window.model_result else 0
                print(
                    f"Window {window.window_id}: "
                    f"BalAcc={tr.balanced_accuracy:.4f} "
                    f"F1={tr.f1:.4f} "
                    f"CV_BalAcc={cv_bal_acc:.4f} "
                    f"CV_F1={cv_f1:.4f}"
                )
        print("=" * 70)


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================

def run_ml_walk_forward(
    df: pd.DataFrame,
    train_size: int = 500,
    test_size: int = 100,
    step_size: Optional[int] = None,
    initial_balance: float = 1000.0,
    ml_config: Optional[MLConfig] = None,
    dataset_config: Optional[DatasetConfig] = None,
    feature_store=None,
) -> MLWalkForwardResult:
    """
    Convenience function for ML Walk-Forward.
    
    Example:
        result = run_ml_walk_forward(
            df,
            train_size=1000,
            test_size=200,
            ml_config=MLConfig(cv_type="purged", n_splits=5, embargo_pct=0.01),
        )
    """
    engine = MLWalkForwardEngine(
        train_size=train_size,
        test_size=test_size,
        step_size=step_size,
        initial_balance=initial_balance,
        ml_config=ml_config,
        dataset_config=dataset_config,
        feature_store=feature_store,
    )
    
    result = engine.run(df)
    engine.print_report(result)
    return result


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "MLWalkForwardWindowResult",
    "MLWalkForwardResult",
    "MLWalkForwardEngine",
    "create_ml_train_callback",
    "run_ml_walk_forward",
]