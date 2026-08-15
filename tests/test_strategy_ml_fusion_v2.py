"""
====================================================
QuantAI AI + ML Fusion v2 Target Test
====================================================

Defines the target behavior for the next
Strategy AI + ML fusion stage.

IMPORTANT:
Production strategy.py is NOT modified by this test.
====================================================
"""

from dataclasses import dataclass


# ====================================================
# TEST FUSION MODEL
# ====================================================

@dataclass
class FusionResult:

    signal: str
    confidence: float
    approved: bool


# ====================================================
# TARGET FUSION LOGIC
# ====================================================

def fuse_signals(
    ai_signal: str,
    ai_confidence: float,
    ml_signal: str,
    ml_probability: float,
) -> FusionResult:

    ai_signal = ai_signal.upper()
    ml_signal = ml_signal.upper()

    ai_confidence = float(ai_confidence)
    ml_probability = float(ml_probability)

    # ------------------------------------------------
    # HOLD + HOLD
    # ------------------------------------------------

    if ai_signal == "HOLD" and ml_signal == "HOLD":

        return FusionResult(
            signal="HOLD",
            confidence=ai_confidence,
            approved=False,
        )

    # ------------------------------------------------
    # AI HOLD
    # ML cannot create a trade by itself
    # ------------------------------------------------

    if ai_signal == "HOLD":

        return FusionResult(
            signal="HOLD",
            confidence=ai_confidence,
            approved=False,
        )

    # ------------------------------------------------
    # ML AGREES WITH AI
    # ------------------------------------------------

    if ml_signal == ai_signal:

        combined = (
            ai_confidence * 0.60
            +
            ml_probability * 0.40
        )

        return FusionResult(
            signal=ai_signal,
            confidence=round(combined, 2),
            approved=combined >= 60.0,
        )

    # ------------------------------------------------
    # ML HOLD
    #
    # High ML HOLD must NOT create a trade.
    #
    # AI direction remains visible, but trade is
    # not approved without ML directional confirmation.
    # ------------------------------------------------

    if ml_signal == "HOLD":

        return FusionResult(
            signal="HOLD",
            confidence=round(ai_confidence, 2),
            approved=False,
        )

    # ------------------------------------------------
    # ML DISAGREEMENT
    #
    # BUY vs SELL conflict.
    # ------------------------------------------------

    return FusionResult(
        signal="HOLD",
        confidence=round(
            ai_confidence * 0.70,
            2,
        ),
        approved=False,
    )


# ====================================================
# ASSERT HELPER
# ====================================================

def check(
    condition: bool,
    message: str,
):

    if not condition:

        raise AssertionError(
            f"FAILED: {message}"
        )

    print(
        f"{message} PASSED"
    )


# ====================================================
# TEST SUITE
# ====================================================

def main():

    print()
    print("=" * 60)
    print("QUANTAI AI + ML FUSION v2 TARGET TEST")
    print("=" * 60)

    # =================================================
    # [1] AI BUY + ML BUY
    # =================================================

    result = fuse_signals(
        "BUY",
        70.0,
        "BUY",
        80.0,
    )

    check(
        result.signal == "BUY",
        "[1] AI BUY + ML BUY direction",
    )

    check(
        result.approved is True,
        "[1] AI BUY + ML BUY approval",
    )

    # =================================================
    # [2] AI SELL + ML SELL
    # =================================================

    result = fuse_signals(
        "SELL",
        70.0,
        "SELL",
        80.0,
    )

    check(
        result.signal == "SELL",
        "[2] AI SELL + ML SELL direction",
    )

    check(
        result.approved is True,
        "[2] AI SELL + ML SELL approval",
    )

    # =================================================
    # [3] AI BUY + ML HOLD
    # =================================================

    result = fuse_signals(
        "BUY",
        75.0,
        "HOLD",
        95.0,
    )

    check(
        result.signal == "HOLD",
        "[3] High ML HOLD blocks BUY",
    )

    check(
        result.approved is False,
        "[3] High ML HOLD cannot approve BUY",
    )

    # =================================================
    # [4] AI SELL + ML HOLD
    # =================================================

    result = fuse_signals(
        "SELL",
        75.0,
        "HOLD",
        95.0,
    )

    check(
        result.signal == "HOLD",
        "[4] High ML HOLD blocks SELL",
    )

    check(
        result.approved is False,
        "[4] High ML HOLD cannot approve SELL",
    )

    # =================================================
    # [5] AI BUY + ML SELL
    # =================================================

    result = fuse_signals(
        "BUY",
        75.0,
        "SELL",
        80.0,
    )

    check(
        result.signal == "HOLD",
        "[5] BUY + SELL conflict becomes HOLD",
    )

    check(
        result.approved is False,
        "[5] BUY + SELL conflict cannot trade",
    )

    # =================================================
    # [6] AI SELL + ML BUY
    # =================================================

    result = fuse_signals(
        "SELL",
        75.0,
        "BUY",
        80.0,
    )

    check(
        result.signal == "HOLD",
        "[6] SELL + BUY conflict becomes HOLD",
    )

    check(
        result.approved is False,
        "[6] SELL + BUY conflict cannot trade",
    )

    # =================================================
    # [7] AI HOLD + ML BUY
    # =================================================

    result = fuse_signals(
        "HOLD",
        55.0,
        "BUY",
        95.0,
    )

    check(
        result.signal == "HOLD",
        "[7] ML BUY cannot create BUY",
    )

    check(
        result.approved is False,
        "[7] ML BUY cannot create trade",
    )

    # =================================================
    # [8] AI HOLD + ML SELL
    # =================================================

    result = fuse_signals(
        "HOLD",
        55.0,
        "SELL",
        95.0,
    )

    check(
        result.signal == "HOLD",
        "[8] ML SELL cannot create SELL",
    )

    check(
        result.approved is False,
        "[8] ML SELL cannot create trade",
    )

    # =================================================
    # [9] Weak AI + ML agreement
    # =================================================

    result = fuse_signals(
        "BUY",
        50.0,
        "BUY",
        55.0,
    )

    check(
        result.signal == "BUY",
        "[9] Weak BUY direction preserved",
    )

    check(
        result.approved is False,
        "[9] Weak BUY cannot trade",
    )

    # =================================================
    # [10] Strong agreement
    # =================================================

    result = fuse_signals(
        "BUY",
        80.0,
        "BUY",
        90.0,
    )

    check(
        result.signal == "BUY",
        "[10] Strong BUY direction",
    )

    check(
        result.approved is True,
        "[10] Strong BUY approved",
    )

    print()
    print("=" * 60)
    print("AI + ML FUSION v2 TARGET TEST PASSED")
    print("=" * 60)

    print()
    print(
        "Production strategy.py has NOT been modified."
    )

    print(
        "This test defines the target behavior for v2."
    )

    print("=" * 60)


# ====================================================
# ENTRY POINT
# ====================================================

if __name__ == "__main__":

    main()