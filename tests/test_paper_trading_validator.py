"""
=========================================================
QuantAI Professional v5
Paper Trading Validator Tests
=========================================================
"""

from __future__ import annotations

import pytest

from src.paper_trading_pipeline import (
    PaperTradingPipelineResult,
)
from src.paper_trading_session import (
    PaperTradingSessionResult,
)
from src.paper_trading_validator import (
    PaperTradingValidationResult,
    PaperTradingValidator,
    validate_paper_trading,
)


# =========================================================
# HELPERS
# =========================================================

def make_result(
    initial_balance: float = 1000.0,
    final_balance: float = 1050.0,
    realized_profit: float = 50.0,
    total_steps: int = 10,
    opened_positions: int = 2,
    closed_positions: int = 2,
    return_percent: float = 5.0,
) -> PaperTradingPipelineResult:

    session_result = PaperTradingSessionResult(
        steps=[],
        initial_balance=initial_balance,
        final_balance=final_balance,
        realized_profit=realized_profit,
        total_steps=total_steps,
        opened_positions=opened_positions,
        closed_positions=closed_positions,
    )

    return PaperTradingPipelineResult(
        session_result=session_result,
        initial_balance=initial_balance,
        final_balance=final_balance,
        realized_profit=realized_profit,
        total_steps=total_steps,
        opened_positions=opened_positions,
        closed_positions=closed_positions,
        return_percent=return_percent,
    )


# =========================================================
# 1. VALID RESULT
# =========================================================

def test_valid_result():

    result = make_result()

    validator = PaperTradingValidator(
        result
    )

    validation = validator.validate()

    assert isinstance(
        validation,
        PaperTradingValidationResult,
    )

    assert validation.valid is True

    assert validation.checks_failed == 0


# =========================================================
# 2. ALL CHECKS PASS
# =========================================================

def test_checks_passed():

    result = make_result()

    validation = (
        PaperTradingValidator(
            result
        ).validate()
    )

    assert validation.checks_passed > 0

    assert validation.checks_failed == 0


# =========================================================
# 3. INVALID RESULT TYPE
# =========================================================

def test_invalid_result_type():

    with pytest.raises(TypeError):

        PaperTradingValidator(
            None
        )


# =========================================================
# 4. NEGATIVE INITIAL BALANCE
# =========================================================

def test_negative_initial_balance():

    result = make_result(
        initial_balance=-100.0,
        final_balance=50.0,
        realized_profit=150.0,
    )

    validation = (
        PaperTradingValidator(
            result
        ).validate()
    )

    assert validation.valid is False

    assert any(
        "Initial balance"
        in error
        for error in validation.errors
    )


# =========================================================
# 5. NEGATIVE FINAL BALANCE
# =========================================================

def test_negative_final_balance():

    result = make_result(
        final_balance=-10.0,
        realized_profit=-1010.0,
    )

    validation = (
        PaperTradingValidator(
            result
        ).validate()
    )

    assert validation.valid is False

    assert any(
        "Final balance"
        in error
        for error in validation.errors
    )


# =========================================================
# 6. NEGATIVE TOTAL STEPS
# =========================================================

def test_negative_total_steps():

    result = make_result(
        total_steps=-1,
    )

    validation = (
        PaperTradingValidator(
            result
        ).validate()
    )

    assert validation.valid is False

    assert any(
        "Total steps"
        in error
        for error in validation.errors
    )


# =========================================================
# 7. NEGATIVE OPENED POSITIONS
# =========================================================

def test_negative_opened_positions():

    result = make_result(
        opened_positions=-1,
    )

    validation = (
        PaperTradingValidator(
            result
        ).validate()
    )

    assert validation.valid is False

    assert any(
        "Opened positions"
        in error
        for error in validation.errors
    )


# =========================================================
# 8. NEGATIVE CLOSED POSITIONS
# =========================================================

def test_negative_closed_positions():

    result = make_result(
        closed_positions=-1,
    )

    validation = (
        PaperTradingValidator(
            result
        ).validate()
    )

    assert validation.valid is False

    assert any(
        "Closed positions"
        in error
        for error in validation.errors
    )


# =========================================================
# 9. CLOSED > OPENED
# =========================================================

def test_closed_positions_exceed_opened():

    result = make_result(
        opened_positions=1,
        closed_positions=2,
    )

    validation = (
        PaperTradingValidator(
            result
        ).validate()
    )

    assert validation.valid is False

    assert any(
        "Closed positions cannot exceed"
        in error
        for error in validation.errors
    )


# =========================================================
# 10. OPENED > TOTAL STEPS
# =========================================================

def test_opened_positions_exceed_steps():

    result = make_result(
        total_steps=1,
        opened_positions=2,
        closed_positions=0,
    )

    validation = (
        PaperTradingValidator(
            result
        ).validate()
    )

    assert validation.valid is False

    assert any(
        "Opened positions cannot exceed"
        in error
        for error in validation.errors
    )


# =========================================================
# 11. CLOSED > TOTAL STEPS
# =========================================================

def test_closed_positions_exceed_steps():

    result = make_result(
        total_steps=1,
        opened_positions=2,
        closed_positions=2,
    )

    validation = (
        PaperTradingValidator(
            result
        ).validate()
    )

    assert validation.valid is False

    assert any(
        "Closed positions cannot exceed"
        in error
        for error in validation.errors
    )


# =========================================================
# 12. PROFIT MISMATCH IS WARNING
# =========================================================

def test_profit_mismatch_is_warning():

    result = make_result(
        initial_balance=1000.0,
        final_balance=1050.0,
        realized_profit=40.0,
    )

    validation = (
        PaperTradingValidator(
            result
        ).validate()
    )

    assert validation.valid is True

    assert len(
        validation.warnings
    ) >= 1

    assert any(
        "Realized profit"
        in warning
        for warning in validation.warnings
    )


# =========================================================
# 13. CONVENIENCE FUNCTION
# =========================================================

def test_validate_paper_trading():

    result = make_result()

    validation = validate_paper_trading(
        result
    )

    assert isinstance(
        validation,
        PaperTradingValidationResult,
    )

    assert validation.valid is True


# =========================================================
# 14. EMPTY SESSION RESULT
# =========================================================

def test_empty_session():

    result = make_result(
        initial_balance=1000.0,
        final_balance=1000.0,
        realized_profit=0.0,
        total_steps=0,
        opened_positions=0,
        closed_positions=0,
        return_percent=0.0,
    )

    validation = (
        PaperTradingValidator(
            result
        ).validate()
    )

    assert validation.valid is True

    assert validation.checks_failed == 0


# =========================================================
# 15. ZERO BALANCE DIFFERENCE
# =========================================================

def test_zero_profit():

    result = make_result(
        initial_balance=1000.0,
        final_balance=1000.0,
        realized_profit=0.0,
        return_percent=0.0,
    )

    validation = (
        PaperTradingValidator(
            result
        ).validate()
    )

    assert validation.valid is True

    assert validation.warnings == []


# =========================================================
# 16. RESULT OBJECT STRUCTURE
# =========================================================

def test_validation_result_structure():

    result = make_result()

    validation = (
        PaperTradingValidator(
            result
        ).validate()
    )

    assert hasattr(
        validation,
        "valid",
    )

    assert hasattr(
        validation,
        "errors",
    )

    assert hasattr(
        validation,
        "warnings",
    )

    assert hasattr(
        validation,
        "checks_passed",
    )

    assert hasattr(
        validation,
        "checks_failed",
    )
