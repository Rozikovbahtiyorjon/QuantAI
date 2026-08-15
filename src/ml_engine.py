"""
====================================================
QuantAI Professional v5.0
Machine Learning Engine v2.0
====================================================

Назначение

Обучение XGBoost для классификации:

    -1 = SELL
     0 = HOLD
     1 = BUY

Внутренние классы XGBoost:

     0 = SELL
     1 = HOLD
     2 = BUY

Основные возможности:

    • Stratified train/test split
    • Class balancing
    • XGBoost multi-class classification
    • Probability prediction
    • BUY / SELL / HOLD probabilities
    • Balanced Accuracy
    • Precision / Recall / F1
    • Confusion Matrix
    • Feature Importance
    • Model save/load

====================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from sklearn.utils.class_weight import compute_sample_weight

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

    subsample: float = 0.90

    colsample_bytree: float = 0.90

    use_class_weights: bool = True


# ====================================================
# TRAINING RESULT
# ====================================================

@dataclass
class TrainingResult:

    accuracy: float

    precision: float

    recall: float

    f1: float

    balanced_accuracy: float

    confusion_matrix: np.ndarray

    class_distribution: Dict[int, int]


# ====================================================
# ML ENGINE
# ====================================================

class MLEngine:

    """
    QuantAI Machine Learning Engine.

    QuantAI target classes:

        -1 = SELL
         0 = HOLD
         1 = BUY

    XGBoost classes:

         0 = SELL
         1 = HOLD
         2 = BUY
    """

    def __init__(
        self,
        config: MLConfig | None = None,
        load_existing: bool = False,
    ) -> None:

        self.config = (
            config
            or MLConfig()
        )

        self.model_manager = (
            ModelManager()
        )

        self.feature_names: list[str] = []

        self.class_weights: Dict[
            int,
            float,
        ] = {}

        self.last_probabilities = None

        # ------------------------------------------------
        # MODEL
        # ------------------------------------------------

        self.model = None

        if load_existing:

            loaded = (
                self.model_manager.load()
            )

            if loaded is not None:

                self.model = loaded

        if self.model is None:

            self.model = (
                self._create_model()
            )

    # ====================================================
    # CREATE MODEL
    # ====================================================

    def _create_model(self) -> XGBClassifier:

        return XGBClassifier(

            n_estimators=(
                self.config.n_estimators
            ),

            max_depth=(
                self.config.max_depth
            ),

            learning_rate=(
                self.config.learning_rate
            ),

            subsample=(
                self.config.subsample
            ),

            colsample_bytree=(
                self.config.colsample_bytree
            ),

            random_state=(
                self.config.random_state
            ),

            objective="multi:softprob",

            num_class=3,

            eval_metric="mlogloss",

            tree_method="hist",

        )

    # ====================================================
    # PREPARE DATASET
    # ====================================================

    def prepare_dataset(
        self,
        dataset: pd.DataFrame,
    ):

        if not isinstance(
            dataset,
            pd.DataFrame,
        ):

            raise TypeError(
                "dataset must be "
                "a pandas DataFrame."
            )

        if dataset.empty:

            raise ValueError(
                "Dataset is empty."
            )

        if "target" not in dataset.columns:

            raise ValueError(
                "Dataset must contain "
                "'target' column."
            )

        data = (
            dataset
            .copy()
        )

        # ------------------------------------------------
        # Remove infinite values
        # ------------------------------------------------

        data = data.replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )

        # ------------------------------------------------
        # Remove NaN
        # ------------------------------------------------

        data = (
            data
            .dropna()
            .reset_index(
                drop=True,
            )
        )

        if data.empty:

            raise ValueError(
                "Dataset is empty after "
                "NaN cleanup."
            )

        # ------------------------------------------------
        # TARGET
        # ------------------------------------------------

        y = (
            data["target"]
            .astype(int)
        )

        # ------------------------------------------------
        # Validate classes
        # ------------------------------------------------

        valid_classes = {
            -1,
            0,
            1,
        }

        actual_classes = set(
            y.unique()
        )

        invalid = (
            actual_classes
            - valid_classes
        )

        if invalid:

            raise ValueError(
                "Invalid target classes: "
                f"{sorted(invalid)}. "
                "Expected -1, 0, 1."
            )

        # ------------------------------------------------
        # FEATURES
        # ------------------------------------------------

        X = data.drop(

            columns=[
                "target",
                "future_return",
                "index",
            ],

            errors="ignore",

        )

        # ------------------------------------------------
        # Numeric validation
        # ------------------------------------------------

        non_numeric = (
            X.select_dtypes(
                exclude=[
                    np.number
                ]
            ).columns.tolist()
        )

        if non_numeric:

            raise ValueError(
                "Non-numeric features found: "
                f"{non_numeric}"
            )

        self.feature_names = (
            list(X.columns)
        )

        if not self.feature_names:

            raise ValueError(
                "No ML features found."
            )

        # ------------------------------------------------
        # QuantAI → XGBoost
        # ------------------------------------------------

        y_xgb = (
            y.replace(
                {
                    -1: 0,
                     0: 1,
                     1: 2,
                }
            )
        )

        # ------------------------------------------------
        # STRATIFIED SPLIT
        # ------------------------------------------------

        (
            X_train,
            X_test,
            y_train,
            y_test,
        ) = train_test_split(

            X,

            y_xgb,

            test_size=(
                self.config.test_size
            ),

            random_state=(
                self.config.random_state
            ),

            shuffle=True,

            stratify=y_xgb,

        )

        return (
            X_train,
            X_test,
            y_train,
            y_test,
        )

    # ====================================================
    # CLASS WEIGHTS
    # ====================================================

    def calculate_class_weights(
        self,
        y_train: pd.Series,
    ) -> Dict[int, float]:
        """
        Calculate balanced class weights.

        XGBoost classes:

            0 = SELL
            1 = HOLD
            2 = BUY
        """

        sample_weights = (
            compute_sample_weight(
                class_weight="balanced",
                y=y_train,
            )
        )

        weights = {}

        for cls in [
            0,
            1,
            2,
        ]:

            mask = (
                y_train
                .to_numpy()
                == cls
            )

            if mask.any():

                weights[cls] = float(
                    np.mean(
                        sample_weights[
                            mask
                        ]
                    )
                )

            else:

                weights[cls] = 1.0

        self.class_weights = (
            weights
        )

        return weights

    # ====================================================
    # TRAIN
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
        ) = self.prepare_dataset(
            dataset
        )

        print()

        print(
            "=" * 60
        )

        print(
            "TRAINING XGBOOST v2.0"
        )

        print(
            "=" * 60
        )

        print(
            f"Train samples : "
            f"{len(X_train)}"
        )

        print(
            f"Test samples  : "
            f"{len(X_test)}"
        )

        print(
            f"Features      : "
            f"{len(self.feature_names)}"
        )

        print()

        # ------------------------------------------------
        # TRAIN DISTRIBUTION
        # ------------------------------------------------

        train_counts = (
            y_train
            .value_counts()
            .sort_index()
        )

        print(
            "TRAIN CLASS DISTRIBUTION"
        )

        print(
            "-" * 60
        )

        print(
            f"SELL : "
            f"{int(train_counts.get(0, 0))}"
        )

        print(
            f"HOLD : "
            f"{int(train_counts.get(1, 0))}"
        )

        print(
            f"BUY  : "
            f"{int(train_counts.get(2, 0))}"
        )

        print()

        # ------------------------------------------------
        # SAMPLE WEIGHTS
        # ------------------------------------------------

        sample_weight = None

        if self.config.use_class_weights:

            self.calculate_class_weights(
                y_train
            )

            sample_weight = (
                compute_sample_weight(
                    class_weight="balanced",
                    y=y_train,
                )
            )

            print(
                "CLASS WEIGHTS"
            )

            print(
                "-" * 60
            )

            print(
                f"SELL : "
                f"{self.class_weights[0]:.4f}"
            )

            print(
                f"HOLD : "
                f"{self.class_weights[1]:.4f}"
            )

            print(
                f"BUY  : "
                f"{self.class_weights[2]:.4f}"
            )

            print()

        # ------------------------------------------------
        # NEW MODEL
        # ------------------------------------------------

        self.model = (
            self._create_model()
        )

        # ------------------------------------------------
        # FIT
        # ------------------------------------------------

        print(
            "Fitting XGBoost..."
        )

        self.model.fit(

            X_train,

            y_train,

            sample_weight=(
                sample_weight
            ),

        )

        print(
            "Training completed."
        )

        print()

        # ------------------------------------------------
        # PREDICTION
        # ------------------------------------------------

        prediction_xgb = (
            self.model.predict(
                X_test
            )
        )

        # ------------------------------------------------
        # PROBABILITIES
        # ------------------------------------------------

        probabilities = (
            self.model.predict_proba(
                X_test
            )
        )

        self.last_probabilities = (
            probabilities
        )

        # ------------------------------------------------
        # XGBoost → QuantAI
        # ------------------------------------------------

        prediction = (
            pd.Series(
                prediction_xgb
            )
            .replace(
                {
                    0: -1,
                    1: 0,
                    2: 1,
                }
            )
            .to_numpy()
        )

        y_test_original = (
            pd.Series(
                y_test
            )
            .replace(
                {
                    0: -1,
                    1: 0,
                    2: 1,
                }
            )
            .to_numpy()
        )

        # =================================================
        # METRICS
        # =================================================

        accuracy = accuracy_score(
            y_test_original,
            prediction,
        )

        balanced_accuracy = (
            balanced_accuracy_score(
                y_test_original,
                prediction,
            )
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

        matrix = confusion_matrix(

            y_test_original,

            prediction,

            labels=[
                -1,
                0,
                1,
            ],

        )

        # ------------------------------------------------
        # TEST DISTRIBUTION
        # ------------------------------------------------

        class_distribution = {

            -1: int(
                (
                    y_test_original
                    == -1
                ).sum()
            ),

            0: int(
                (
                    y_test_original
                    == 0
                ).sum()
            ),

            1: int(
                (
                    y_test_original
                    == 1
                ).sum()
            ),

        }

        # =================================================
        # RESULT
        # =================================================

        result = TrainingResult(

            accuracy=accuracy,

            precision=precision,

            recall=recall,

            f1=f1,

            balanced_accuracy=(
                balanced_accuracy
            ),

            confusion_matrix=matrix,

            class_distribution=(
                class_distribution
            ),

        )

        # =================================================
        # SAVE
        # =================================================

        self.model_manager.save(
            self.model
        )

        return result

    # ====================================================
    # PREDICT PROBABILITIES
    # ====================================================

    def predict_probabilities(
        self,
        features: pd.DataFrame,
    ) -> Dict[str, float]:

        if self.model is None:

            raise RuntimeError(
                "Model is not loaded."
            )

        if features.empty:

            raise ValueError(
                "Features DataFrame is empty."
            )

        # ------------------------------------------------
        # If feature names are not known,
        # try to get them from the trained model.
        # ------------------------------------------------

        if not self.feature_names:

            if hasattr(
                self.model,
                "feature_names_in_",
            ):

                self.feature_names = (
                    list(
                        self.model
                        .feature_names_in_
                    )
                )

            else:

                raise RuntimeError(
                    "Feature names are unknown."
                )

        # ------------------------------------------------
        # Validate features
        # ------------------------------------------------

        missing = [
            feature
            for feature
            in self.feature_names
            if feature not in features.columns
        ]

        if missing:

            raise ValueError(
                "Missing ML features: "
                f"{missing}"
            )

        X = (
            features[
                self.feature_names
            ]
            .copy()
        )

        X = X.replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )

        X = X.fillna(0.0)

        probabilities = (
            self.model
            .predict_proba(X)
        )

        latest = (
            probabilities[-1]
        )

        return {

            "SELL": float(
                latest[0] * 100
            ),

            "HOLD": float(
                latest[1] * 100
            ),

            "BUY": float(
                latest[2] * 100
            ),

        }

    # ====================================================
    # PREDICT SIGNAL
    # ====================================================

    def predict_signal(
        self,
        features: pd.DataFrame,
    ) -> tuple[str, float]:

        probabilities = (
            self.predict_probabilities(
                features
            )
        )

        signal = max(
            probabilities,
            key=probabilities.get,
        )

        probability = float(
            probabilities[
                signal
            ]
        )

        return (
            signal,
            probability,
        )

    # ====================================================
    # FEATURE IMPORTANCE
    # ====================================================

    def feature_importance(
        self,
        top: int = 20,
    ) -> pd.DataFrame:

        if not hasattr(
            self.model,
            "feature_importances_",
        ):

            raise RuntimeError(
                "Model has not been trained."
            )

        importance = pd.DataFrame({

            "feature":
                self.feature_names,

            "importance":
                self.model.feature_importances_,

        })

        importance = (
            importance
            .sort_values(
                by="importance",
                ascending=False,
            )
            .reset_index(
                drop=True,
            )
        )

        return importance.head(
            top
        )

    # ====================================================
    # CONFUSION MATRIX
    # ====================================================

    @staticmethod
    def print_confusion_matrix(
        matrix: np.ndarray,
    ) -> None:

        print()

        print(
            "CONFUSION MATRIX"
        )

        print(
            "-" * 60
        )

        print(
            "             "
            "SELL     HOLD      BUY"
        )

        labels = [
            "SELL",
            "HOLD",
            "BUY",
        ]

        for i, label in enumerate(
            labels
        ):

            print(
                f"{label:<10}"
                f"{matrix[i][0]:>8}"
                f"{matrix[i][1]:>10}"
                f"{matrix[i][2]:>10}"
            )

        print()

    # ====================================================
    # REPORT
    # ====================================================

    def print_report(
        self,
        result: TrainingResult,
    ) -> None:

        print()

        print(
            "=" * 60
        )

        print(
            "MODEL PERFORMANCE v2.0"
        )

        print(
            "=" * 60
        )

        print(
            f"Accuracy          : "
            f"{result.accuracy:.4f}"
        )

        print(
            f"Balanced Accuracy : "
            f"{result.balanced_accuracy:.4f}"
        )

        print(
            f"Precision         : "
            f"{result.precision:.4f}"
        )

        print(
            f"Recall            : "
            f"{result.recall:.4f}"
        )

        print(
            f"F1 Score          : "
            f"{result.f1:.4f}"
        )

        print(
            "=" * 60
        )

        print()

        print(
            "TEST CLASS DISTRIBUTION"
        )

        print(
            "-" * 60
        )

        print(
            f"SELL : "
            f"{result.class_distribution.get(-1, 0)}"
        )

        print(
            f"HOLD : "
            f"{result.class_distribution.get(0, 0)}"
        )

        print(
            f"BUY  : "
            f"{result.class_distribution.get(1, 0)}"
        )

        self.print_confusion_matrix(
            result.confusion_matrix
        )

        print(
            "TOP FEATURES"
        )

        print(
            "-" * 60
        )

        try:

            importance = (
                self.feature_importance()
            )

            print(
                importance.to_string(
                    index=False
                )
            )

        except Exception as exc:

            print(
                f"Feature importance error: "
                f"{exc}"
            )

        print(
            "=" * 60
        )

    # ====================================================
    # LOAD MODEL
    # ====================================================

    def load_model(self) -> bool:
        """
        Explicitly load saved model.
        """

        loaded = (
            self.model_manager.load()
        )

        if loaded is None:

            return False

        self.model = loaded

        # XGBoost stores feature names
        # after fitting.

        if hasattr(
            self.model,
            "feature_names_in_",
        ):

            self.feature_names = (
                list(
                    self.model
                    .feature_names_in_
                )
            )

        return True


# ====================================================
# PUBLIC API
# ====================================================

def train_model(
    dataset: pd.DataFrame,
) -> tuple[MLEngine, TrainingResult]:

    engine = (
        MLEngine()
    )

    result = (
        engine.train(
            dataset
        )
    )

    engine.print_report(
        result
    )

    return (
        engine,
        result,
    )


# ====================================================
# EXPORTS
# ====================================================

__all__ = [
    "MLConfig",
    "TrainingResult",
    "MLEngine",
    "train_model",
]
