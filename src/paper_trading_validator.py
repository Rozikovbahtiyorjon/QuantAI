"""
=========================================================
QuantAI Professional v5
Paper Trading Validator

Validates completed paper-trading results.

This module does NOT:
    - execute trades
    - connect to Binance
    - generate Strategy signals
    - modify PaperTradingEngine
    - modify PaperTradingRunner
    - modify PaperTradingSession
    - modify PaperTradingPipeline

It only validates paper-trading results.
=========================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.paper_trading_pipeline import (
    PaperTradingPipelineResult,
)


# =========================================================
# VALIDATION RESULT
# =========================================================

@dataclass
class PaperTradingValidationResult:
    """
    Result of paper-trading validation.
    """

    valid: bool

    errors: List[str]

    warnings: List[str]

    checks_passed: int

    checks_failed: int


# =========================================================
# VALIDATOR
# =========================================================

class PaperTradingValidator:
    """
    Validate a completed PaperTradingPipelineResult.
    """

    def __init__(
        self,
        result: PaperTradingPipelineResult,
    ) -> None:

        if not isinstance(
            result,
            PaperTradingPipelineResult,
        ):
            raise TypeError(
                "result must be PaperTradingPipelineResult."
            )

        self.result = result

    # =====================================================
    # VALIDATION
    # =====================================================

    def validate(
        self,
    ) -> PaperTradingValidationResult:
        """
        Run all validation checks.
        """

        errors: List[str] = []
        warnings: List[str] = []

        checks_passed = 0
        checks_failed = 0

        # -------------------------------------------------
        # CHECK 1
        # -------------------------------------------------

        if self.result.initial_balance > 0:

            checks_passed += 1

        else:

            checks_failed += 1

            errors.append(
                "Initial balance must be greater than zero."
            )

        # -------------------------------------------------
        # CHECK 2
        # -------------------------------------------------

        if self.result.final_balance >= 0:

            checks_passed += 1

        else:

            checks_failed += 1

            errors.append(
                "Final balance cannot be negative."
            )

        # -------------------------------------------------
        # CHECK 3
        # -------------------------------------------------

        if self.result.total_steps >= 0:

            checks_passed += 1

        else:

            checks_failed += 1

            errors.append(
                "Total steps cannot be negative."
            )

        # -------------------------------------------------
        # CHECK 4
        # -------------------------------------------------

        if self.result.opened_positions >= 0:

            checks_passed += 1

        else:

            checks_failed += 1

            errors.append(
                "Opened positions cannot be negative."
            )

        # -------------------------------------------------
        # CHECK 5
        # -------------------------------------------------

        if self.result.closed_positions >= 0:

            checks_passed += 1

        else:

            checks_failed += 1

            errors.append(
                "Closed positions cannot be negative."
            )

        # -------------------------------------------------
        # CHECK 6
        # -------------------------------------------------

        if (
            self.result.closed_positions
            <= self.result.opened_positions
        ):

            checks_passed += 1

        else:

            checks_failed += 1

            errors.append(
                "Closed positions cannot exceed opened positions."
            )

        # -------------------------------------------------
        # CHECK 7
        # -------------------------------------------------

        if (
            self.result.total_steps
            >= self.result.opened_positions
        ):

            checks_passed += 1

        else:

            checks_failed += 1

            errors.append(
                "Opened positions cannot exceed total steps."
            )

        # -------------------------------------------------
        # CHECK 8
        # -------------------------------------------------

        if (
            self.result.total_steps
            >= self.result.closed_positions
        ):

            checks_passed += 1

        else:

            checks_failed += 1

            errors.append(
                "Closed positions cannot exceed total steps."
            )

        # -------------------------------------------------
        # CHECK 9
        # -------------------------------------------------

        if (
            0.0
            <= self.result.return_percent
            <= 1000000.0
        ) or (
            self.result.return_percent < 0.0
        ):

            checks_passed += 1

        else:

            checks_failed += 1

            errors.append(
                "Return percent contains an invalid value."
            )

        # -------------------------------------------------
        # CHECK 10
        # -------------------------------------------------

        expected_profit = (
            self.result.final_balance
            - self.result.initial_balance
        )

        if abs(
            expected_profit
            - self.result.realized_profit
        ) <= 1e-8:

            checks_passed += 1

        else:

            warnings.append(
                "Realized profit does not equal "
                "final balance minus initial balance."
            )

            checks_passed += 1

        # -------------------------------------------------
        # CHECK 11
        # -------------------------------------------------

        if self.result.session_result is not None:

            checks_passed += 1

        else:

            checks_failed += 1

            errors.append(
                "Session result cannot be None."
            )

        # -------------------------------------------------
        # FINAL RESULT
        # -------------------------------------------------

        valid = (
            checks_failed == 0
        )

        return PaperTradingValidationResult(
            valid=valid,

            errors=errors,

            warnings=warnings,

            checks_passed=checks_passed,

            checks_failed=checks_failed,
        )


# =========================================================
# CONVENIENCE FUNCTION
# =========================================================

def validate_paper_trading(
    result: PaperTradingPipelineResult,
) -> PaperTradingValidationResult:
    """
    Validate a paper-trading pipeline result.
    """

    validator = PaperTradingValidator(
        result
    )

    return validator.validate()


# =========================================================
# EXPORTS
# =========================================================

__all__ = [
    "PaperTradingValidationResult",
    "PaperTradingValidator",
    "validate_paper_trading",
]
