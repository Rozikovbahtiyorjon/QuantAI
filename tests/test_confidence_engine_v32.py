"""
====================================================
QuantAI
Confidence Engine v3.2 Target Behavior Test
====================================================

This test defines the TARGET behavior for Confidence
Engine v3.2.

IMPORTANT:
The current ConfidenceEngine v3.1 is NOT modified here.

Some tests are expected to FAIL against v3.1.
That is intentional.

The purpose of this file is to define the behavior
we want before changing production code.
====================================================
"""

from src.confidence_engine import ConfidenceEngine


# ====================================================
# HEADER
# ====================================================

print()
print("=" * 60)
print("QUANTAI CONFIDENCE ENGINE v3.2 TARGET TEST")
print("=" * 60)


# ====================================================
# TEST 1
# ZERO SCORE
# ====================================================

engine = ConfidenceEngine()

engine.add_component(
    "trend",
    0.0,
)

result = engine.evaluate()

assert result.total_score == 0.0

assert result.confidence == 50.0

assert result.decision == "HOLD"

print("[1] Zero score behavior PASSED")


# ====================================================
# TEST 2
# POSITIVE DIRECTION
# ====================================================

engine.reset()

engine.add_component(
    "trend",
    2.0,
)

result = engine.evaluate()

assert result.total_score == 2.0

assert result.decision == "BUY"

assert result.confidence >= 60.0

print("[2] Positive BUY direction PASSED")


# ====================================================
# TEST 3
# NEGATIVE DIRECTION
# ====================================================

engine.reset()

engine.add_component(
    "trend",
    -2.0,
)

result = engine.evaluate()

assert result.total_score == -2.0

assert result.decision == "SELL"

assert result.confidence >= 60.0

print("[3] Negative SELL direction PASSED")


# ====================================================
# TEST 4
# WEAK SIGNAL MUST HOLD
# ====================================================

engine.reset()

engine.add_component(
    "trend",
    0.5,
)

result = engine.evaluate()

assert result.total_score == 0.5

assert result.decision == "HOLD"

assert result.confidence < 60.0

print("[4] Weak signal HOLD PASSED")


# ====================================================
# TEST 5
# STRONG SELL MUST NOT HAVE LOW CONFIDENCE
# ====================================================

engine.reset()

engine.add_component(
    "trend",
    -3.0,
)

result = engine.evaluate()

assert result.total_score == -3.0

assert result.decision == "SELL"

assert result.confidence >= 60.0

print("[5] Strong SELL confidence PASSED")


# ====================================================
# TEST 6
# STRONG BUY MUST NOT HAVE LOW CONFIDENCE
# ====================================================

engine.reset()

engine.add_component(
    "trend",
    3.0,
)

result = engine.evaluate()

assert result.total_score == 3.0

assert result.decision == "BUY"

assert result.confidence >= 60.0

print("[6] Strong BUY confidence PASSED")


# ====================================================
# TEST 7
# WEIGHTED SCORE
# ====================================================

engine.reset()

engine.add_component(
    "trend",
    2.0,
)

engine.add_component(
    "momentum",
    1.0,
)

result = engine.evaluate()

expected_score = (
    2.0 * 1.50
    +
    1.0 * 1.20
) / (
    1.50
    +
    1.20
)

expected_score = round(
    expected_score,
    2,
)

assert result.total_score == expected_score

print("[7] Weighted score PASSED")


# ====================================================
# TEST 8
# SCORE SIGN DETERMINES DIRECTION
# ====================================================

engine.reset()

engine.add_component(
    "trend",
    -1.5,
)

engine.add_component(
    "momentum",
    -1.0,
)

result = engine.evaluate()

assert result.total_score < 0

assert result.decision == "SELL"

print("[8] Score sign determines SELL PASSED")


# ====================================================
# TEST 9
# POSITIVE SCORE CANNOT PRODUCE SELL
# ====================================================

engine.reset()

engine.add_component(
    "trend",
    2.0,
)

result = engine.evaluate()

assert result.total_score > 0

assert result.decision != "SELL"

print("[9] Positive score cannot produce SELL PASSED")


# ====================================================
# TEST 10
# NEGATIVE SCORE CANNOT PRODUCE BUY
# ====================================================

engine.reset()

engine.add_component(
    "trend",
    -2.0,
)

result = engine.evaluate()

assert result.total_score < 0

assert result.decision != "BUY"

print("[10] Negative score cannot produce BUY PASSED")


# ====================================================
# FINAL
# ====================================================

print()
print("=" * 60)
print("CONFIDENCE ENGINE v3.2 TARGET TEST COMPLETED")
print("=" * 60)
print()
print(
    "NOTE:"
)
print(
    "This test defines the target behavior for v3.2."
)
print(
    "Failures against v3.1 may be expected."
)
print(
    "Do NOT modify production code yet."
)
print("=" * 60)