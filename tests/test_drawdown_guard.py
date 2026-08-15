from __future__ import annotations

import pytest

from src.drawdown_guard import (
    DrawdownGuard,
    DrawdownGuardResult,
)


def test_default_configuration() -> None:
    guard = DrawdownGuard()

    assert guard.max_drawdown_percent == 10.0
    assert guard.peak_equity is None


def test_custom_configuration() -> None:
    guard = DrawdownGuard(
        max_drawdown_percent=5.0,
    )

    assert guard.max_drawdown_percent == 5.0


def test_negative_limit_is_rejected() -> None:
    with pytest.raises(ValueError):
        DrawdownGuard(
            max_drawdown_percent=-1.0,
        )


def test_initial_equity_sets_peak() -> None:
    guard = DrawdownGuard()

    result = guard.evaluate(1000.0)

    assert isinstance(
        result,
        DrawdownGuardResult,
    )
    assert result.peak_equity == 1000.0
    assert result.current_equity == 1000.0
    assert result.drawdown == 0.0
    assert result.drawdown_percent == 0.0
    assert result.allowed is True


def test_equity_growth_updates_peak() -> None:
    guard = DrawdownGuard()

    guard.evaluate(1000.0)
    result = guard.evaluate(1200.0)

    assert result.peak_equity == 1200.0
    assert result.current_equity == 1200.0
    assert result.drawdown == 0.0
    assert result.drawdown_percent == 0.0
    assert result.allowed is True


def test_drawdown_is_calculated_from_peak() -> None:
    guard = DrawdownGuard()

    guard.evaluate(1000.0)
    result = guard.evaluate(900.0)

    assert result.peak_equity == 1000.0
    assert result.current_equity == 900.0
    assert result.drawdown == 100.0
    assert result.drawdown_percent == 10.0
    assert result.allowed is True


def test_drawdown_limit_is_rejected() -> None:
    guard = DrawdownGuard(
        max_drawdown_percent=10.0,
    )

    guard.evaluate(1000.0)
    result = guard.evaluate(899.0)

    assert result.drawdown == 101.0
    assert result.drawdown_percent == 10.1
    assert result.allowed is False


def test_recovery_does_not_reduce_peak() -> None:
    guard = DrawdownGuard()

    guard.evaluate(1000.0)
    guard.evaluate(800.0)
    result = guard.evaluate(900.0)

    assert result.peak_equity == 1000.0
    assert result.current_equity == 900.0
    assert result.drawdown == 100.0
    assert result.drawdown_percent == 10.0


def test_new_high_resets_drawdown_to_zero() -> None:
    guard = DrawdownGuard()

    guard.evaluate(1000.0)
    guard.evaluate(800.0)
    result = guard.evaluate(1100.0)

    assert result.peak_equity == 1100.0
    assert result.current_equity == 1100.0
    assert result.drawdown == 0.0
    assert result.drawdown_percent == 0.0
    assert result.allowed is True


def test_is_allowed_true() -> None:
    guard = DrawdownGuard(
        max_drawdown_percent=10.0,
    )

    guard.evaluate(1000.0)

    assert guard.is_allowed(950.0) is True


def test_is_allowed_false() -> None:
    guard = DrawdownGuard(
        max_drawdown_percent=5.0,
    )

    guard.evaluate(1000.0)

    assert guard.is_allowed(900.0) is False


def test_zero_equity_is_rejected() -> None:
    guard = DrawdownGuard()

    with pytest.raises(ValueError):
        guard.evaluate(0.0)


def test_negative_equity_is_rejected() -> None:
    guard = DrawdownGuard()

    with pytest.raises(ValueError):
        guard.evaluate(-100.0)


def test_precision() -> None:
    guard = DrawdownGuard()

    guard.evaluate(1000.0)

    result = guard.evaluate(
        876.543210987,
    )

    assert result.peak_equity == 1000.0
    assert result.drawdown == 123.45678901
    assert result.drawdown_percent == 12.3456789


def test_multiple_equity_updates() -> None:
    guard = DrawdownGuard(
        max_drawdown_percent=20.0,
    )

    values = [
        1000.0,
        1100.0,
        1050.0,
        900.0,
        950.0,
        1200.0,
        1000.0,
    ]

    results = [
        guard.evaluate(value)
        for value in values
    ]

    assert results[0].peak_equity == 1000.0
    assert results[1].peak_equity == 1100.0
    assert results[2].drawdown == 50.0
    assert results[3].drawdown == 200.0
    assert results[4].drawdown == 150.0
    assert results[5].peak_equity == 1200.0
    assert results[5].drawdown == 0.0
    assert results[6].drawdown == 200.0


def test_reset() -> None:
    guard = DrawdownGuard()

    guard.evaluate(1000.0)
    guard.evaluate(800.0)

    guard.reset()

    assert guard.peak_equity is None

    result = guard.evaluate(900.0)

    assert result.peak_equity == 900.0
    assert result.drawdown == 0.0
    assert result.drawdown_percent == 0.0


def test_exact_limit_is_allowed() -> None:
    guard = DrawdownGuard(
        max_drawdown_percent=10.0,
    )

    guard.evaluate(1000.0)
    result = guard.evaluate(900.0)

    assert result.drawdown_percent == 10.0
    assert result.allowed is True


def test_below_limit_is_allowed() -> None:
    guard = DrawdownGuard(
        max_drawdown_percent=10.0,
    )

    guard.evaluate(1000.0)
    result = guard.evaluate(950.0)

    assert result.drawdown_percent == 5.0
    assert result.allowed is True


def test_result_is_immutable() -> None:
    guard = DrawdownGuard()

    result = guard.evaluate(1000.0)

    with pytest.raises(
        AttributeError
    ):
        result.current_equity = 900.0