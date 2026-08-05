"""
====================================================
QuantAI
Confidence Engine Baseline Test
====================================================
"""

from src.confidence_engine import ConfidenceEngine


# ====================================================
# HEADER
# ====================================================

print()
print("=" * 60)
print("QUANTAI CONFIDENCE ENGINE BASELINE TEST")
print("=" * 60)


# ====================================================
# TEST 1 — EMPTY ENGINE
# ====================================================

engine = ConfidenceEngine()

result = engine.evaluate()

assert result.total_score == 0.0
assert result.confidence == 50.0
assert result.probability == 50.0
assert result.decision == "HOLD"

print()
print("[1] Empty engine test PASSED")


# ====================================================
# TEST 2 — POSITIVE SCORE
# ====================================================

engine.reset()

engine.add_component(
    "trend",
    1.0,
)

result = engine.evaluate()

assert result.total_score == 1.0
assert result.confidence == 60.0
assert result.probability == 60.0
assert result.decision == "BUY"

print("[2] Positive score test PASSED")


# ====================================================
# TEST 3 — NEGATIVE SCORE
# ====================================================

engine.reset()

engine.add_component(
    "trend",
    -1.0,
)

result = engine.evaluate()

assert result.total_score == -1.0
assert result.confidence == 60.0
assert result.probability == 60.0
assert result.decision == "SELL"

print("[3] Negative score test PASSED")


# ====================================================
# TEST 4 — STRONG BUY
# ====================================================

engine.reset()

engine.add_component(
    "trend",
    2.0,
)

result = engine.evaluate()

assert result.total_score == 2.0
assert result.confidence == 70.0
assert result.probability == 70.0
assert result.decision == "BUY"

print("[4] Strong BUY test PASSED")


# ====================================================
# TEST 5 — STRONG SELL
# ====================================================

engine.reset()

engine.add_component(
    "trend",
    -2.0,
)

result = engine.evaluate()

assert result.total_score == -2.0
assert result.confidence == 70.0
assert result.probability == 70.0
assert result.decision == "SELL"

print("[5] Strong SELL test PASSED")


# ====================================================
# TEST 6 — LOW CONFIDENCE
# ====================================================

engine.reset()

engine.add_component(
    "trend",
    0.5,
)

result = engine.evaluate()

assert result.total_score == 0.5
assert result.confidence == 55.0
assert result.probability == 55.0
assert result.decision == "HOLD"

print("[6] Low confidence HOLD test PASSED")


# ====================================================
# TEST 7 — WEIGHTED SCORE
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
    1.50 + 1.20
)

expected_score = round(
    expected_score,
    2,
)

assert result.total_score == expected_score

print("[7] Weighted score test PASSED")


# ====================================================
# FINAL
# ====================================================

print()
print("=" * 60)
print("CONFIDENCE ENGINE BASELINE TEST PASSED")
print("=" * 60)