"""
====================================================
QuantAI
Strategy AI + ML Fusion Diagnostic Test
====================================================

Defines the expected behavior of the AI + ML
confidence fusion layer.

IMPORTANT:
This test does NOT modify production code.

It tests the fusion rule independently from the
full Strategy Engine.
====================================================
"""

from __future__ import annotations


# ====================================================
# FUSION FUNCTION
# ====================================================

def calculate_fusion(
    ai_decision: str,
    ai_confidence: float,
    ml_signal: str,
    ml_probability: float,
) -> float:
    """
    Temporary reference implementation of the
    TARGET fusion behavior.

    This function is intentionally local to the test.

    Production strategy.py is NOT modified yet.
    """

    ai_decision = ai_decision.upper()
    ml_signal = ml_signal.upper()

    ai_confidence = float(ai_confidence)
    ml_probability = float(ml_probability)

    # =================================================
    # ML HOLD
    # =================================================
    #
    # HOLD is not directional confirmation.
    #
    # Therefore ML HOLD must NOT increase the
    # directional AI confidence.
    #

    if ml_signal == "HOLD":

        return round(
            ai_confidence,
            2,
        )

    # =================================================
    # ML AGREES WITH AI
    # =================================================

    if ml_signal == ai_decision:

        combined = (
            ai_confidence * 0.60
            +
            ml_probability * 0.40
        )

        return round(
            combined,
            2,
        )

    # =================================================
    # ML DISAGREES WITH AI
    # =================================================
    #
    # Directional disagreement is a conflict.
    #
    # The ML probability should NOT be added as
    # positive confidence.
    #

    return round(
        ai_confidence * 0.70,
        2,
    )


# ====================================================
# HEADER
# ====================================================

print()
print("=" * 60)
print("QUANTAI AI + ML FUSION TARGET TEST")
print("=" * 60)


# ====================================================
# TEST 1
# AI BUY + ML BUY
# ====================================================

result = calculate_fusion(
    ai_decision="BUY",
    ai_confidence=60.0,
    ml_signal="BUY",
    ml_probability=80.0,
)

expected = 68.0

assert result == expected

print("[1] AI BUY + ML BUY PASSED")


# ====================================================
# TEST 2
# AI SELL + ML SELL
# ====================================================

result = calculate_fusion(
    ai_decision="SELL",
    ai_confidence=60.0,
    ml_signal="SELL",
    ml_probability=80.0,
)

expected = 68.0

assert result == expected

print("[2] AI SELL + ML SELL PASSED")


# ====================================================
# TEST 3
# AI BUY + ML HOLD
# ====================================================

result = calculate_fusion(
    ai_decision="BUY",
    ai_confidence=54.9,
    ml_signal="HOLD",
    ml_probability=98.86,
)

expected = 54.9

assert result == expected

print("[3] AI BUY + ML HOLD PASSED")


# ====================================================
# TEST 4
# AI SELL + ML HOLD
# ====================================================

result = calculate_fusion(
    ai_decision="SELL",
    ai_confidence=55.0,
    ml_signal="HOLD",
    ml_probability=98.0,
)

expected = 55.0

assert result == expected

print("[4] AI SELL + ML HOLD PASSED")


# ====================================================
# TEST 5
# AI BUY + ML SELL
# ====================================================

result = calculate_fusion(
    ai_decision="BUY",
    ai_confidence=70.0,
    ml_signal="SELL",
    ml_probability=80.0,
)

expected = 49.0

assert result == expected

print("[5] AI BUY + ML SELL PASSED")


# ====================================================
# TEST 6
# AI SELL + ML BUY
# ====================================================

result = calculate_fusion(
    ai_decision="SELL",
    ai_confidence=70.0,
    ml_signal="BUY",
    ml_probability=80.0,
)

expected = 49.0

assert result == expected

print("[6] AI SELL + ML BUY PASSED")


# ====================================================
# TEST 7
# HIGH ML HOLD MUST NOT CREATE HIGH CONFIDENCE
# ====================================================

result = calculate_fusion(
    ai_decision="BUY",
    ai_confidence=50.0,
    ml_signal="HOLD",
    ml_probability=99.0,
)

assert result == 50.0

assert result < 60.0

print(
    "[7] High ML HOLD cannot create BUY confidence PASSED"
)


# ====================================================
# TEST 8
# ML AGREEMENT CAN INCREASE CONFIDENCE
# ====================================================

result = calculate_fusion(
    ai_decision="BUY",
    ai_confidence=60.0,
    ml_signal="BUY",
    ml_probability=90.0,
)

assert result > 60.0

print(
    "[8] ML agreement increases confidence PASSED"
)


# ====================================================
# TEST 9
# ML DISAGREEMENT REDUCES CONFIDENCE
# ====================================================

result = calculate_fusion(
    ai_decision="BUY",
    ai_confidence=70.0,
    ml_signal="SELL",
    ml_probability=90.0,
)

assert result < 70.0

print(
    "[9] ML disagreement reduces confidence PASSED"
)


# ====================================================
# TEST 10
# HOLD + HOLD REMAINS NEUTRAL
# ====================================================

result = calculate_fusion(
    ai_decision="HOLD",
    ai_confidence=55.0,
    ml_signal="HOLD",
    ml_probability=99.0,
)

assert result == 55.0

print(
    "[10] HOLD + HOLD remains neutral PASSED"
)


# ====================================================
# FINAL
# ====================================================

print()
print("=" * 60)
print("AI + ML FUSION TARGET TEST PASSED")
print("=" * 60)
print()
print(
    "Production strategy.py has NOT been modified."
)
print(
    "This test defines the target fusion behavior."
)
print("=" * 60)