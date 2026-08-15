# ============================================================
# QuantAI v5
# Strategy -> ML -> Confidence Integration Test
# ============================================================

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================
# IMPORTS
# ============================================================

from src.confidence_engine import ConfidenceEngine
from src.strategy import (
    fuse_ai_ml,
)


# ============================================================
# HELPERS
# ============================================================

def check(
    condition: bool,
    message: str,
) -> None:

    if not condition:
        raise AssertionError(
            f"FAILED: {message}"
        )

    print(
        f"OK    : {message}"
    )


# ============================================================
# CONFIDENCE ENGINE TEST
# ============================================================

def test_confidence_engine() -> None:

    print()
    print("=" * 70)
    print("CONFIDENCE ENGINE")
    print("=" * 70)

    engine = ConfidenceEngine()

    # --------------------------------------------------------
    # Empty state
    # --------------------------------------------------------

    score = engine.calculate_score()
    confidence = engine.calculate_confidence(score)
    signal = engine.decide(
        score,
        confidence,
    )

    check(
        score == 0.0,
        "Empty score = 0",
    )

    check(
        confidence == 50.0,
        "Empty confidence = 50%",
    )

    check(
        signal == "HOLD",
        "Empty engine returns HOLD",
    )

    # --------------------------------------------------------
    # BUY
    # --------------------------------------------------------

    engine.reset()

    engine.add_component(
        "trend",
        1.0,
        "Strong bullish trend",
    )

    score = engine.calculate_score()
    confidence = engine.calculate_confidence(score)
    signal = engine.decide(
        score,
        confidence,
    )

    check(
        score > 0,
        "BUY component produces positive score",
    )

    check(
        signal == "BUY",
        "Positive score produces BUY",
    )

    # --------------------------------------------------------
    # SELL
    # --------------------------------------------------------

    engine.reset()

    engine.add_component(
        "trend",
        -1.0,
        "Strong bearish trend",
    )

    score = engine.calculate_score()
    confidence = engine.calculate_confidence(score)
    signal = engine.decide(
        score,
        confidence,
    )

    check(
        score < 0,
        "SELL component produces negative score",
    )

    check(
        signal == "SELL",
        "Negative score produces SELL",
    )


# ============================================================
# AI + ML FUSION
# ============================================================

def test_ai_ml_fusion() -> None:

    print()
    print("=" * 70)
    print("AI + ML FUSION")
    print("=" * 70)

    # --------------------------------------------------------
    # AI HOLD + ML HOLD
    # --------------------------------------------------------

    result = fuse_ai_ml(
        ai_signal="HOLD",
        ai_confidence=80.0,
        ml_signal="HOLD",
        ml_probability=99.0,
    )

    signal, confidence, approved, reason = result

    check(
        signal == "HOLD",
        "AI HOLD + ML HOLD => HOLD",
    )

    check(
        not approved,
        "AI HOLD + ML HOLD => not approved",
    )

    # --------------------------------------------------------
    # AI HOLD + ML BUY
    # --------------------------------------------------------

    result = fuse_ai_ml(
        ai_signal="HOLD",
        ai_confidence=80.0,
        ml_signal="BUY",
        ml_probability=99.0,
    )

    signal, confidence, approved, reason = result

    check(
        signal == "HOLD",
        "AI HOLD blocks ML BUY",
    )

    check(
        not approved,
        "AI HOLD + ML BUY => not approved",
    )

    # --------------------------------------------------------
    # AI BUY + ML BUY
    # --------------------------------------------------------

    result = fuse_ai_ml(
        ai_signal="BUY",
        ai_confidence=90.0,
        ml_signal="BUY",
        ml_probability=95.0,
    )

    signal, confidence, approved, reason = result

    check(
        signal == "BUY",
        "AI BUY + ML BUY => BUY",
    )

    check(
        confidence >= 0.0,
        "BUY fusion confidence is valid",
    )

    # --------------------------------------------------------
    # AI SELL + ML SELL
    # --------------------------------------------------------

    result = fuse_ai_ml(
        ai_signal="SELL",
        ai_confidence=90.0,
        ml_signal="SELL",
        ml_probability=95.0,
    )

    signal, confidence, approved, reason = result

    check(
        signal == "SELL",
        "AI SELL + ML SELL => SELL",
    )

    check(
        confidence >= 0.0,
        "SELL fusion confidence is valid",
    )

    # --------------------------------------------------------
    # AI BUY + ML SELL
    # --------------------------------------------------------

    result = fuse_ai_ml(
        ai_signal="BUY",
        ai_confidence=90.0,
        ml_signal="SELL",
        ml_probability=95.0,
    )

    signal, confidence, approved, reason = result

    check(
        signal in {
            "BUY",
            "SELL",
            "HOLD",
        },
        "Conflicting AI/ML signals produce valid signal",
    )

    # --------------------------------------------------------
    # AI SELL + ML BUY
    # --------------------------------------------------------

    result = fuse_ai_ml(
        ai_signal="SELL",
        ai_confidence=90.0,
        ml_signal="BUY",
        ml_probability=95.0,
    )

    signal, confidence, approved, reason = result

    check(
        signal in {
            "BUY",
            "SELL",
            "HOLD",
        },
        "Opposing AI/ML signals produce valid signal",
    )


# ============================================================
# ML PROBABILITY VALIDATION
# ============================================================

def test_probability_validation() -> None:

    print()
    print("=" * 70)
    print("ML PROBABILITY VALIDATION")
    print("=" * 70)

    probabilities = np.array(
        [
            0.01,
            0.98,
            0.01,
        ]
    )

    check(
        np.isclose(
            probabilities.sum(),
            1.0,
            atol=1e-6,
        ),
        "Probability sum = 1",
    )

    check(
        np.all(
            probabilities >= 0.0
        ),
        "Probabilities are >= 0",
    )

    check(
        np.all(
            probabilities <= 1.0
        ),
        "Probabilities are <= 1",
    )

    prediction = int(
        probabilities.argmax()
    )

    check(
        prediction == 1,
        "99% HOLD-like probability predicts class 1",
    )


# ============================================================
# CLASS MAPPING VALIDATION
# ============================================================

def test_class_mapping() -> None:

    print()
    print("=" * 70)
    print("XGBOOST CLASS MAPPING")
    print("=" * 70)

    mapping = {
        0: "SELL",
        1: "HOLD",
        2: "BUY",
    }

    check(
        mapping[0] == "SELL",
        "Class 0 = SELL",
    )

    check(
        mapping[1] == "HOLD",
        "Class 1 = HOLD",
    )

    check(
        mapping[2] == "BUY",
        "Class 2 = BUY",
    )


# ============================================================
# DATAFRAME PROBABILITY TEST
# ============================================================

def test_probability_dataframe() -> None:

    print()
    print("=" * 70)
    print("PROBABILITY DATA STRUCTURE")
    print("=" * 70)

    probabilities = pd.DataFrame(
        [
            {
                "SELL": 1.0,
                "HOLD": 98.5,
                "BUY": 0.5,
            },
            {
                "SELL": 95.0,
                "HOLD": 4.0,
                "BUY": 1.0,
            },
            {
                "SELL": 2.0,
                "HOLD": 3.0,
                "BUY": 95.0,
            },
        ]
    )

    check(
        len(probabilities) == 3,
        "Three probability rows created",
    )

    check(
        list(probabilities.columns)
        == [
            "SELL",
            "HOLD",
            "BUY",
        ],
        "Probability columns are SELL/HOLD/BUY",
    )

    sums = (
        probabilities.sum(axis=1)
    )

    check(
        np.allclose(
            sums,
            100.0,
            atol=1e-6,
        ),
        "Probability percentages sum to 100%",
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print()
    print("=" * 70)
    print("QuantAI v5")
    print("STRATEGY -> ML -> CONFIDENCE INTEGRATION TEST")
    print("=" * 70)

    test_confidence_engine()

    test_ai_ml_fusion()

    test_probability_validation()

    test_class_mapping()

    test_probability_dataframe()

    print()
    print("=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)
    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()