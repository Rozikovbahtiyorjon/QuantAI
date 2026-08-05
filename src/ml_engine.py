"""
====================================================
QuantAI Professional v5.0
Machine Learning Engine
====================================================

Назначение

Обучение AI-моделей
на исторических данных.

Pipeline

Dataset
    ↓
Train/Test Split
    ↓
Model Training
    ↓
Evaluation
    ↓
Model Saving

====================================================
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from xgboost import XGBClassifier

from src.model_manager import ModelManager

# ====================================================
# CONFIG
# ====================================================

@dataclass
class MLConfig:

    test_size: float = 0.20

    random_state: int = 42

    n_estimators: int = 300

    max_depth: int = 6

    learning_rate: float = 0.05

# ====================================================
# TRAIN RESULT
# ====================================================

@dataclass
class TrainingResult:

    accuracy: float

    precision: float

    recall: float

    f1: float

# ====================================================
# ML ENGINE
# ====================================================

class MLEngine:

    """
    Обучение модели Machine Learning.
    """

    def __init__(
        self,
        config: MLConfig | None = None,
    ) -> None:

        self.config = config or MLConfig()

        self.model_manager = ModelManager()

        loaded = self.model_manager.load()

        if loaded is not None:

            self.model = loaded

        else:

            self.model = XGBClassifier(

                n_estimators=self.config.n_estimators,

                max_depth=self.config.max_depth,

                learning_rate=self.config.learning_rate,

                random_state=self.config.random_state,

                objective="multi:softmax",

                num_class=3,

                eval_metric="mlogloss",

            )

        self.feature_names: list[str] = []

    # ====================================================
    # PREPARE DATA
    # ====================================================

    def prepare_dataset(

        self,

        dataset: pd.DataFrame,

    ):

        data = dataset.copy()

        y = data["target"]

        X = data.drop(

            columns=[

                "target",

                "future_return",

                "index",

            ],

            errors="ignore",

        )

        self.feature_names = list(X.columns)

        #
        # XGBoost требует классы:
        #
        # SELL=-1 → 0
        # HOLD=0  → 1
        # BUY=1   → 2
        #

        y = y.replace({

            -1: 0,

             0: 1,

             1: 2,

        })

        return train_test_split(

            X,

            y,

            test_size=self.config.test_size,

            random_state=self.config.random_state,

            shuffle=True,

        )

        # ====================================================
    # TRAIN MODEL
    # ====================================================

    def train(
        self,
        dataset: pd.DataFrame,
    ) -> TrainingResult:

        (
            X_train,
            X_test,
            y_train,
            y_test,
        ) = self.prepare_dataset(dataset)

        print()

        print("=" * 60)
        print("TRAINING XGBOOST")
        print("=" * 60)

        print(f"Train samples : {len(X_train)}")
        print(f"Test samples  : {len(X_test)}")

        print()

        # ====================================================
        # TRAIN MODEL
        # ====================================================

        self.model.fit(
            X_train,
            y_train,
        )

        # ====================================================
        # PREDICTION
        # ====================================================

        prediction = self.model.predict(
            X_test,
        )

        # ====================================================
        # RESTORE QUANTAI TRADING CLASSES
        # ====================================================
        #
        # XGBoost internal classes:
        # 0 = SELL
        # 1 = HOLD
        # 2 = BUY
        #
        # QuantAI trading classes:
        # -1 = SELL
        #  0 = HOLD
        #  1 = BUY
        # ====================================================

        prediction = pd.Series(
            prediction,
        ).replace({
            0: -1,
            1: 0,
            2: 1,
        }).to_numpy()

        y_test_original = pd.Series(
            y_test,
        ).replace({
            0: -1,
            1: 0,
            2: 1,
        }).to_numpy()

        # ====================================================
        # METRICS
        # ====================================================

        accuracy = accuracy_score(
            y_test_original,
            prediction,
        )

        precision = precision_score(
            y_test_original,
            prediction,
            average="macro",
            zero_division=0,
        )

        recall = recall_score(
            y_test_original,
            prediction,
            average="macro",
            zero_division=0,
        )

        f1 = f1_score(
            y_test_original,
            prediction,
            average="macro",
            zero_division=0,
        )

        # ====================================================
        # TRAINING RESULT
        # ====================================================

        result = TrainingResult(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
        )

        # ====================================================
        # SAVE MODEL
        # ====================================================

        self.model_manager.save(
            self.model,
        )

        return result

    # ====================================================
    # FEATURE IMPORTANCE
    # ====================================================

    def feature_importance(
        self,
        top: int = 20,
    ) -> pd.DataFrame:

        importance = pd.DataFrame({

            "feature": self.feature_names,

            "importance": self.model.feature_importances_,

        })

        importance = importance.sort_values(

            by="importance",

            ascending=False,

        )

        return importance.head(top)

    # ====================================================
    # PRINT REPORT
    # ====================================================

    def print_report(
        self,
        result: TrainingResult,
    ) -> None:

        print()

        print("=" * 60)
        print("MODEL PERFORMANCE")
        print("=" * 60)

        print(f"Accuracy  : {result.accuracy:.4f}")
        print(f"Precision : {result.precision:.4f}")
        print(f"Recall    : {result.recall:.4f}")
        print(f"F1 Score  : {result.f1:.4f}")

        print("=" * 60)

        print()

        print("TOP FEATURES")
        print("-" * 60)

        importance = self.feature_importance()

        print(importance.to_string(index=False))

        print("=" * 60)

# ====================================================
# PUBLIC API
# ====================================================

def train_model(
    dataset: pd.DataFrame,
) -> tuple[MLEngine, TrainingResult]:

    engine = MLEngine()

    result = engine.train(dataset)

    engine.print_report(result)

    return engine, result


__all__ = [
    "MLConfig",
    "TrainingResult",
    "MLEngine",
    "train_model",
]