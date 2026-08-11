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
import pandas as pd

from src.data_loader import load_binance_data
from src.indicators import add_indicators
from src.dataset_builder import DatasetBuilder
from src.feature_engine import build_features

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

LAST_ROWS = 20


# ============================================================
# SIGNAL MAPS
# ============================================================

XGB_SIGNAL_NAMES = {
    0: "SELL",
    1: "HOLD",
    2: "BUY",
}

TARGET_SIGNAL_NAMES = {
    -1: "SELL",
    0: "HOLD",
    1: "BUY",
}


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 90)
    print("QUANTAI ML — LATEST DATASET VALIDATION")
    print("=" * 90)

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    print()
    print("MODEL")
    print("-" * 90)
    print(f"Path   : {MODEL_PATH}")
    print("Loading model...")

    model = joblib.load(
        MODEL_PATH
    )

    print(
        f"Classes: {model.classes_}"
    )

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

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
    print("Loading Binance data...")

    df = load_binance_data(
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        limit=LIMIT,
    )

    print(
        f"OHLCV shape: {df.shape}"
    )

    # --------------------------------------------------------
    # INDICATORS
    # --------------------------------------------------------

    print()
    print("INDICATORS")
    print("-" * 90)

    df = add_indicators(
        df
    )

    print(
        f"After indicators: {df.shape}"
    )

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # DATASET DISTRIBUTION
    # --------------------------------------------------------

    print()
    print("FULL DATASET TARGET DISTRIBUTION")
    print("-" * 90)

    target_counts = (
        dataset["target"]
        .value_counts()
        .sort_index()
    )

    target_percent = (
        dataset["target"]
        .value_counts(
            normalize=True
        )
        .sort_index()
        * 100
    )

    print(
        "TARGET | SIGNAL | COUNT | PERCENT"
    )

    print("-" * 45)

    for target in [-1, 0, 1]:

        count = int(
            target_counts.get(
                target,
                0,
            )
        )

        percent = float(
            target_percent.get(
                target,
                0.0,
            )
        )

        signal = (
            TARGET_SIGNAL_NAMES[target]
        )

        print(
            f"{target:6d} | "
            f"{signal:6s} | "
            f"{count:5d} | "
            f"{percent:6.2f}%"
        )

    # --------------------------------------------------------
    # LAST ROWS
    # --------------------------------------------------------

    start = max(
        0,
        len(dataset) - LAST_ROWS,
    )

    print()
    print(
        f"LATEST {len(dataset) - start} DATASET ROWS"
    )

    print("-" * 90)

    print(
        "IDX | SRC | ACTUAL | SELL% | HOLD% | BUY% | PREDICTION"
    )

    print("-" * 90)

    # --------------------------------------------------------
    # DIAGNOSTIC COUNTERS
    # --------------------------------------------------------

    actual_counts = {
        -1: 0,
        0: 0,
        1: 0,
    }

    prediction_counts = {
        0: 0,
        1: 0,
        2: 0,
    }

    correct = 0

    # --------------------------------------------------------
    # PROCESS LAST ROWS
    # --------------------------------------------------------

    for i in range(
        start,
        len(dataset),
    ):

        row = dataset.iloc[i]

        # ----------------------------------------------------
        # ACTUAL TARGET
        # ----------------------------------------------------

        actual = int(
            row["target"]
        )

        actual_counts[actual] += 1

        # ----------------------------------------------------
        # SOURCE INDEX
        # ----------------------------------------------------

        source_index = int(
            row["index"]
        )

        # ----------------------------------------------------
        # HISTORICAL DATA AVAILABLE
        # ONLY UP TO THIS CANDLE
        # ----------------------------------------------------

        history = df.iloc[
            : source_index + 1
        ]

        # ----------------------------------------------------
        # BUILD FEATURES
        # ----------------------------------------------------

        features = build_features(
            history
        )

        X = pd.DataFrame(
            [features]
        )

        # ----------------------------------------------------
        # VALIDATE FEATURE COUNT
        # ----------------------------------------------------

        if len(features) != len(
            model.feature_names_in_
        ):

            raise ValueError(
                "Feature count mismatch: "
                f"features={len(features)}, "
                f"model={len(model.feature_names_in_)}"
            )

        # ----------------------------------------------------
        # PROBABILITIES
        # ----------------------------------------------------

        probabilities = (
            model.predict_proba(X)[0]
        )

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        prediction = int(
            model.predict(X)[0]
        )

        prediction_counts[
            prediction
        ] += 1

        # ----------------------------------------------------
        # CHECK
        # ----------------------------------------------------

        expected_xgb_class = {
            -1: 0,
            0: 1,
            1: 2,
        }[actual]

        if prediction == expected_xgb_class:

            correct += 1

        # ----------------------------------------------------
        # PROBABILITIES
        # ----------------------------------------------------

        sell_probability = (
            probabilities[0] * 100.0
        )

        hold_probability = (
            probabilities[1] * 100.0
        )

        buy_probability = (
            probabilities[2] * 100.0
        )

        # ----------------------------------------------------
        # PRINT
        # ----------------------------------------------------

        actual_signal = (
            TARGET_SIGNAL_NAMES.get(
                actual,
                "UNKNOWN",
            )
        )

        predicted_signal = (
            XGB_SIGNAL_NAMES.get(
                prediction,
                "UNKNOWN",
            )
        )

        print(
            f"{i:3d} | "
            f"{source_index:3d} | "
            f"{actual_signal:6s} | "
            f"{sell_probability:6.2f} | "
            f"{hold_probability:6.2f} | "
            f"{buy_probability:6.2f} | "
            f"{predicted_signal}"
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    total_latest = (
        len(dataset) - start
    )

    accuracy = (
        correct / total_latest * 100.0
        if total_latest > 0
        else 0.0
    )

    print()
    print("=" * 90)
    print("LATEST ROWS SUMMARY")
    print("=" * 90)

    print()
    print("ACTUAL TARGETS")
    print("-" * 90)

    print(
        f"SELL : {actual_counts[-1]}"
    )

    print(
        f"HOLD : {actual_counts[0]}"
    )

    print(
        f"BUY  : {actual_counts[1]}"
    )

    print()
    print("XGBOOST PREDICTIONS")
    print("-" * 90)

    print(
        f"SELL : {prediction_counts[0]}"
    )

    print(
        f"HOLD : {prediction_counts[1]}"
    )

    print(
        f"BUY  : {prediction_counts[2]}"
    )

    print()
    print(
        f"Latest-row accuracy: {accuracy:.2f}%"
    )

    # ========================================================
    # LATEST CANDLE — CURRENT MODEL VIEW
    # ========================================================

    latest_source_index = (
        int(
            dataset.iloc[-1]["index"]
        )
    )

    latest_history = df.iloc[
        : latest_source_index + 1
    ]

    latest_features = build_features(
        latest_history
    )

    latest_X = pd.DataFrame(
        [latest_features]
    )

    latest_probabilities = (
        model.predict_proba(
            latest_X
        )[0]
    )

    latest_prediction = int(
        model.predict(
            latest_X
        )[0]
    )

    latest_signal = (
        XGB_SIGNAL_NAMES.get(
            latest_prediction,
            "UNKNOWN",
        )
    )

    print()
    print("=" * 90)
    print("CURRENT / LATEST MODEL PREDICTION")
    print("=" * 90)

    print()
    print(
        f"SELL probability: "
        f"{latest_probabilities[0] * 100:.2f}%"
    )

    print(
        f"HOLD probability: "
        f"{latest_probabilities[1] * 100:.2f}%"
    )

    print(
        f"BUY  probability: "
        f"{latest_probabilities[2] * 100:.2f}%"
    )

    print()
    print(
        f"Prediction: {latest_signal}"
    )

    print(
        f"Probability: "
        f"{latest_probabilities[latest_prediction] * 100:.2f}%"
    )

    print()
    print("=" * 90)
    print("CHECK COMPLETED")
    print("=" * 90)
    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()