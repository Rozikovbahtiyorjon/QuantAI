from pathlib import Path
import sys

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.data_loader import load_binance_data
from src.indicators import add_indicators
from src.dataset_builder import DatasetBuilder
from src.ml_engine import MLEngine

from config.settings import (
    SYMBOL,
    TIMEFRAME,
    LIMIT,
)


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "quantai_v5.pkl"
)


# ============================================================
# SIGNAL MAP
# ============================================================

XGB_SIGNAL_NAMES = {
    0: "SELL",
    1: "HOLD",
    2: "BUY",
}

TARGET_TO_XGB = {
    -1: 0,
    0: 1,
    1: 2,
}


# ============================================================
# HELPERS
# ============================================================

def print_distribution(
    title: str,
    values,
    mapping: dict,
):
    print()
    print(title)
    print("-" * 70)

    series = pd.Series(values)

    counts = (
        series
        .value_counts()
        .sort_index()
    )

    total = len(series)

    print(
        "CLASS  | SIGNAL | COUNT | PERCENT"
    )

    print("-" * 45)

    for cls in sorted(mapping):

        count = int(
            counts.get(
                cls,
                0,
            )
        )

        percent = (
            count / total * 100.0
            if total
            else 0.0
        )

        print(
            f"{cls:5d} | "
            f"{mapping[cls]:6s} | "
            f"{count:5d} | "
            f"{percent:7.2f}%"
        )


def print_probability_summary(
    probabilities: np.ndarray,
):
    print()
    print(
        "PROBABILITY SUMMARY"
    )
    print("-" * 70)

    for index, signal in XGB_SIGNAL_NAMES.items():

        values = (
            probabilities[:, index]
            * 100.0
        )

        print()
        print(signal)

        print(
            f"  mean   : {values.mean():.2f}%"
        )

        print(
            f"  median : {np.median(values):.2f}%"
        )

        print(
            f"  min    : {values.min():.2f}%"
        )

        print(
            f"  max    : {values.max():.2f}%"
        )

        print(
            f"  p25    : {np.percentile(values, 25):.2f}%"
        )

        print(
            f"  p75    : {np.percentile(values, 75):.2f}%"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 90)
    print("QUANTAI ML QUALITY CHECK")
    print("=" * 90)

    # ========================================================
    # MODEL
    # ========================================================

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    print()
    print("MODEL")
    print("-" * 90)

    print(
        f"Path   : {MODEL_PATH}"
    )

    model = joblib.load(
        MODEL_PATH
    )

    print(
        f"Classes: {model.classes_}"
    )

    # ========================================================
    # MARKET DATA
    # ========================================================

    print()
    print("MARKET DATA")
    print("-" * 90)

    print(
        f"Symbol   : {SYMBOL}"
    )

    print(
        f"Timeframe: {TIMEFRAME}"
    )

    print(
        f"Limit    : {LIMIT}"
    )

    print()
    print(
        "Loading Binance data..."
    )

    df = load_binance_data(
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        limit=LIMIT,
    )

    print(
        f"OHLCV shape: {df.shape}"
    )

    # ========================================================
    # INDICATORS
    # ========================================================

    print()
    print("INDICATORS")
    print("-" * 90)

    df = add_indicators(
        df
    )

    print(
        f"After indicators: {df.shape}"
    )

    # ========================================================
    # DATASET
    # ========================================================

    print()
    print("DATASET")
    print("-" * 90)

    dataset = (
        DatasetBuilder()
        .build(df)
    )

    print(
        f"Dataset shape: {dataset.shape}"
    )

    print(
        f"Dataset rows : {len(dataset)}"
    )

    # ========================================================
    # PREPARE SAME TEST SPLIT
    # AS MLEngine
    # ========================================================

    print()
    print(
        "PREPARING TEST DATA"
    )

    engine = MLEngine()

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = engine.prepare_dataset(
        dataset
    )

    print(
        f"Train samples: {len(X_train)}"
    )

    print(
        f"Test samples : {len(X_test)}"
    )

    print(
        f"Features     : {X_test.shape[1]}"
    )

    # ========================================================
    # MODEL FEATURE VALIDATION
    # ========================================================

    if hasattr(
        model,
        "feature_names_in_",
    ):

        model_features = list(
            model.feature_names_in_
        )

        test_features = list(
            X_test.columns
        )

        if model_features != test_features:

            print()
            print(
                "WARNING: MODEL FEATURE ORDER MISMATCH"
            )

            print(
                f"Model features: {len(model_features)}"
            )

            print(
                f"Test features : {len(test_features)}"
            )

            missing = [
                x
                for x in model_features
                if x not in test_features
            ]

            extra = [
                x
                for x in test_features
                if x not in model_features
            ]

            if missing:
                print(
                    f"Missing features: {missing}"
                )

            if extra:
                print(
                    f"Extra features: {extra}"
                )

            raise ValueError(
                "Model/test feature mismatch."
            )

    # ========================================================
    # PREDICTIONS
    # ========================================================

    print()
    print(
        "RUNNING MODEL PREDICTIONS..."
    )

    probabilities = (
        model.predict_proba(
            X_test
        )
    )

    predictions = (
        model.predict(
            X_test
        )
    )

    # ========================================================
    # PROBABILITY VALIDATION
    # ========================================================

    print()
    print(
        "PROBABILITY VALIDATION"
    )
    print("-" * 70)

    probability_sums = (
        probabilities.sum(
            axis=1
        )
    )

    print(
        f"Minimum probability sum: "
        f"{probability_sums.min():.10f}"
    )

    print(
        f"Maximum probability sum: "
        f"{probability_sums.max():.10f}"
    )

    if not np.allclose(
        probability_sums,
        1.0,
        atol=1e-6,
    ):

        raise ValueError(
            "Model probabilities do not sum to 1."
        )

    print(
        "Probability validation: OK"
    )

    # ========================================================
    # DISTRIBUTIONS
    # ========================================================

    print_distribution(
        "TEST ACTUAL DISTRIBUTION",
        y_test,
        XGB_SIGNAL_NAMES,
    )

    print_distribution(
        "TEST PREDICTION DISTRIBUTION",
        predictions,
        XGB_SIGNAL_NAMES,
    )

    # ========================================================
    # METRICS
    # ========================================================

    accuracy = (
        accuracy_score(
            y_test,
            predictions,
        )
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_test,
            predictions,
        )
    )

    macro_precision = (
        precision_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        )
    )

    macro_recall = (
        recall_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        )
    )

    macro_f1 = (
        f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        )
    )

    weighted_f1 = (
        f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        )
    )

    # ========================================================
    # MAIN METRICS
    # ========================================================

    print()
    print("=" * 90)
    print("OVERALL METRICS")
    print("=" * 90)

    print(
        f"Accuracy           : {accuracy * 100:.2f}%"
    )

    print(
        f"Balanced Accuracy  : "
        f"{balanced_accuracy * 100:.2f}%"
    )

    print(
        f"Macro Precision    : "
        f"{macro_precision * 100:.2f}%"
    )

    print(
        f"Macro Recall       : "
        f"{macro_recall * 100:.2f}%"
    )

    print(
        f"Macro F1           : "
        f"{macro_f1 * 100:.2f}%"
    )

    print(
        f"Weighted F1        : "
        f"{weighted_f1 * 100:.2f}%"
    )

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print()
    print("=" * 90)
    print("CLASSIFICATION REPORT")
    print("=" * 90)

    report = classification_report(
        y_test,
        predictions,
        labels=[
            0,
            1,
            2,
        ],
        target_names=[
            "SELL",
            "HOLD",
            "BUY",
        ],
        digits=4,
        zero_division=0,
    )

    print()
    print(report)

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[
            0,
            1,
            2,
        ],
    )

    print()
    print("=" * 90)
    print("CONFUSION MATRIX")
    print("=" * 90)

    print()
    print(
        "                 PREDICTED"
    )

    print(
        "             SELL   HOLD    BUY"
    )

    print(
        "ACTUAL"
    )

    print(
        f"SELL       {matrix[0, 0]:6d}"
        f"{matrix[0, 1]:7d}"
        f"{matrix[0, 2]:7d}"
    )

    print(
        f"HOLD       {matrix[1, 0]:6d}"
        f"{matrix[1, 1]:7d}"
        f"{matrix[1, 2]:7d}"
    )

    print(
        f"BUY        {matrix[2, 0]:6d}"
        f"{matrix[2, 1]:7d}"
        f"{matrix[2, 2]:7d}"
    )

    # ========================================================
    # CLASS-SPECIFIC METRICS
    # ========================================================

    print()
    print("=" * 90)
    print("CLASS-SPECIFIC PERFORMANCE")
    print("=" * 90)

    for cls, signal in XGB_SIGNAL_NAMES.items():

        actual_mask = (
            y_test == cls
        )

        predicted_mask = (
            predictions == cls
        )

        true_positive = int(
            (
                actual_mask
                & predicted_mask
            ).sum()
        )

        actual_count = int(
            actual_mask.sum()
        )

        predicted_count = int(
            predicted_mask.sum()
        )

        class_recall = (
            true_positive
            / actual_count
            if actual_count
            else 0.0
        )

        class_precision = (
            true_positive
            / predicted_count
            if predicted_count
            else 0.0
        )

        print()
        print(
            f"{signal}"
        )

        print(
            f"  Actual samples    : "
            f"{actual_count}"
        )

        print(
            f"  Predicted samples : "
            f"{predicted_count}"
        )

        print(
            f"  Correct           : "
            f"{true_positive}"
        )

        print(
            f"  Precision         : "
            f"{class_precision * 100:.2f}%"
        )

        print(
            f"  Recall            : "
            f"{class_recall * 100:.2f}%"
        )

    # ========================================================
    # PROBABILITY SUMMARY
    # ========================================================

    print_probability_summary(
        probabilities
    )

    # ========================================================
    # CONFIDENCE DISTRIBUTION
    # ========================================================

    max_probability = (
        probabilities.max(
            axis=1
        )
        * 100.0
    )

    print()
    print("=" * 90)
    print("MODEL CONFIDENCE DISTRIBUTION")
    print("=" * 90)

    print(
        f"Mean   : {max_probability.mean():.2f}%"
    )

    print(
        f"Median : {np.median(max_probability):.2f}%"
    )

    print(
        f"Min    : {max_probability.min():.2f}%"
    )

    print(
        f"Max    : {max_probability.max():.2f}%"
    )

    print(
        f"P25    : "
        f"{np.percentile(max_probability, 25):.2f}%"
    )

    print(
        f"P75    : "
        f"{np.percentile(max_probability, 75):.2f}%"
    )

    # ========================================================
    # HIGH-CONFIDENCE PREDICTIONS
    # ========================================================

    print()
    print("=" * 90)
    print("HIGH-CONFIDENCE PREDICTIONS")
    print("=" * 90)

    for threshold in [
        50,
        60,
        70,
        80,
        90,
        95,
        99,
    ]:

        count = int(
            (
                max_probability
                >= threshold
            ).sum()
        )

        percent = (
            count
            / len(max_probability)
            * 100.0
        )

        print(
            f">= {threshold:2d}% : "
            f"{count:4d} "
            f"({percent:6.2f}%)"
        )

    # ========================================================
    # DIRECTIONAL SIGNAL QUALITY
    # ========================================================

    print()
    print("=" * 90)
    print("DIRECTIONAL SIGNAL QUALITY")
    print("=" * 90)

    directional_mask = (
        predictions != 1
    )

    directional_count = int(
        directional_mask.sum()
    )

    if directional_count:

        directional_actual = (
            y_test[
                directional_mask
            ]
        )

        directional_prediction = (
            predictions[
                directional_mask
            ]
        )

        directional_accuracy = (
            accuracy_score(
                directional_actual,
                directional_prediction,
            )
        )

        print()
        print(
            f"Directional predictions: "
            f"{directional_count}"
        )

        print(
            f"Directional accuracy  : "
            f"{directional_accuracy * 100:.2f}%"
        )

    else:

        print()
        print(
            "No SELL/BUY predictions "
            "were produced."
        )

    # ========================================================
    # SELL / BUY CONFIDENCE
    # ========================================================

    sell_buy_mask = (
        predictions != 1
    )

    if sell_buy_mask.any():

        directional_probabilities = (
            probabilities[
                sell_buy_mask
            ]
        )

        directional_max = (
            directional_probabilities.max(
                axis=1
            )
            * 100.0
        )

        print()
        print(
            "Directional confidence:"
        )

        print(
            f"  Mean   : "
            f"{directional_max.mean():.2f}%"
        )

        print(
            f"  Median : "
            f"{np.median(directional_max):.2f}%"
        )

        print(
            f"  Min    : "
            f"{directional_max.min():.2f}%"
        )

        print(
            f"  Max    : "
            f"{directional_max.max():.2f}%"
        )

    # ========================================================
    # TOP FEATURES
    # ========================================================

    if hasattr(
        model,
        "feature_importances_",
    ):

        print()
        print("=" * 90)
        print("TOP FEATURES")
        print("=" * 90)

        feature_names = list(
            X_test.columns
        )

        importances = (
            model.feature_importances_
        )

        feature_table = (
            pd.DataFrame(
                {
                    "feature": feature_names,
                    "importance": importances,
                }
            )
            .sort_values(
                "importance",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

        print()

        print(
            feature_table
            .head(15)
            .to_string(
                index=False,
                formatters={
                    "importance": (
                        lambda x:
                        f"{x:.6f}"
                    )
                },
            )
        )

    # ========================================================
    # FINAL DIAGNOSIS
    # ========================================================

    print()
    print("=" * 90)
    print("DIAGNOSIS")
    print("=" * 90)

    prediction_distribution = {
        cls: int(
            (
                predictions == cls
            ).sum()
        )
        for cls in [
            0,
            1,
            2,
        ]
    }

    if (
        prediction_distribution[0] == 0
        and prediction_distribution[2] == 0
    ):

        print()
        print(
            "WARNING: MODEL PREDICTS HOLD ONLY."
        )

        print(
            "The model is not producing "
            "directional SELL/BUY signals."
        )

    elif (
        prediction_distribution[1]
        / len(predictions)
        > 0.90
    ):

        print()
        print(
            "WARNING: HOLD DOMINANCE > 90%."
        )

        print(
            "The model may be strongly biased "
            "toward HOLD."
        )

    elif balanced_accuracy < 0.55:

        print()
        print(
            "WARNING: LOW BALANCED ACCURACY."
        )

        print(
            "Class performance is weak "
            "despite possible overall accuracy."
        )

    else:

        print()
        print(
            "Model produces directional signals "
            "and class balance appears usable."
        )

    print()
    print("=" * 90)
    print("QUALITY CHECK COMPLETED")
    print("=" * 90)
    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()