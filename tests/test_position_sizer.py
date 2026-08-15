import pytest

from src.position_sizer import (
    PositionSizer,
    PositionSizeResult,
)


def test_basic_position_size() -> None:
    sizer = PositionSizer()

    result = sizer.calculate(
        balance=1000.0,
        risk_percent=1.0,
        entry_price=100.0,
        stop_price=95.0,
        leverage=1.0,
    )

    assert isinstance(
        result,
        PositionSizeResult,
    )

    assert result.risk_amount == 10.0
    assert result.stop_distance == 5.0
    assert result.position_size == 2.0
    assert result.position_notional == 200.0
    assert result.margin_required == 200.0


def test_short_position_uses_absolute_stop_distance() -> None:
    sizer = PositionSizer()

    result = sizer.calculate(
        balance=1000.0,
        risk_percent=2.0,
        entry_price=100.0,
        stop_price=105.0,
        leverage=1.0,
    )

    assert result.risk_amount == 20.0
    assert result.stop_distance == 5.0
    assert result.position_size == 4.0
    assert result.position_notional == 400.0


def test_leverage_reduces_margin_required() -> None:
    sizer = PositionSizer()

    result = sizer.calculate(
        balance=1000.0,
        risk_percent=1.0,
        entry_price=100.0,
        stop_price=95.0,
        leverage=10.0,
    )

    assert result.position_notional == 200.0
    assert result.margin_required == 20.0


def test_stop_percent_long() -> None:
    sizer = PositionSizer()

    result = sizer.calculate_from_stop_percent(
        balance=1000.0,
        risk_percent=1.0,
        entry_price=100.0,
        stop_percent=5.0,
        leverage=5.0,
        side="LONG",
    )

    assert result.stop_price == 95.0
    assert result.stop_distance == 5.0
    assert result.position_size == 2.0


def test_stop_percent_short() -> None:
    sizer = PositionSizer()

    result = sizer.calculate_from_stop_percent(
        balance=1000.0,
        risk_percent=1.0,
        entry_price=100.0,
        stop_percent=5.0,
        leverage=5.0,
        side="SHORT",
    )

    assert result.stop_price == 105.0
    assert result.stop_distance == 5.0
    assert result.position_size == 2.0


def test_stop_distance_percent() -> None:
    sizer = PositionSizer()

    result = sizer.calculate(
        balance=2000.0,
        risk_percent=1.0,
        entry_price=200.0,
        stop_price=190.0,
        leverage=2.0,
    )

    assert result.stop_distance_percent == 5.0


def test_invalid_balance() -> None:
    sizer = PositionSizer()

    with pytest.raises(ValueError):
        sizer.calculate(
            balance=0.0,
            risk_percent=1.0,
            entry_price=100.0,
            stop_price=95.0,
        )


def test_invalid_risk_percent() -> None:
    sizer = PositionSizer()

    with pytest.raises(ValueError):
        sizer.calculate(
            balance=1000.0,
            risk_percent=0.0,
            entry_price=100.0,
            stop_price=95.0,
        )


def test_invalid_entry_price() -> None:
    sizer = PositionSizer()

    with pytest.raises(ValueError):
        sizer.calculate(
            balance=1000.0,
            risk_percent=1.0,
            entry_price=0.0,
            stop_price=95.0,
        )


def test_invalid_stop_price() -> None:
    sizer = PositionSizer()

    with pytest.raises(ValueError):
        sizer.calculate(
            balance=1000.0,
            risk_percent=1.0,
            entry_price=100.0,
            stop_price=0.0,
        )


def test_equal_entry_and_stop_is_invalid() -> None:
    sizer = PositionSizer()

    with pytest.raises(ValueError):
        sizer.calculate(
            balance=1000.0,
            risk_percent=1.0,
            entry_price=100.0,
            stop_price=100.0,
        )


def test_min_leverage_validation() -> None:
    sizer = PositionSizer(
        min_leverage=2.0,
        max_leverage=50.0,
    )

    with pytest.raises(ValueError):
        sizer.calculate(
            balance=1000.0,
            risk_percent=1.0,
            entry_price=100.0,
            stop_price=95.0,
            leverage=1.0,
        )


def test_max_leverage_validation() -> None:
    sizer = PositionSizer(
        min_leverage=1.0,
        max_leverage=10.0,
    )

    with pytest.raises(ValueError):
        sizer.calculate(
            balance=1000.0,
            risk_percent=1.0,
            entry_price=100.0,
            stop_price=95.0,
            leverage=11.0,
        )


def test_invalid_leverage_configuration() -> None:
    with pytest.raises(ValueError):
        PositionSizer(
            min_leverage=10.0,
            max_leverage=5.0,
        )


def test_invalid_stop_percent() -> None:
    sizer = PositionSizer()

    with pytest.raises(ValueError):
        sizer.calculate_from_stop_percent(
            balance=1000.0,
            risk_percent=1.0,
            entry_price=100.0,
            stop_percent=0.0,
        )


def test_invalid_side() -> None:
    sizer = PositionSizer()

    with pytest.raises(ValueError):
        sizer.calculate_from_stop_percent(
            balance=1000.0,
            risk_percent=1.0,
            entry_price=100.0,
            stop_percent=5.0,
            side="INVALID",
        )


def test_long_and_short_have_same_risk_for_equal_stop_distance() -> None:
    sizer = PositionSizer()

    long_result = sizer.calculate_from_stop_percent(
        balance=5000.0,
        risk_percent=2.0,
        entry_price=100.0,
        stop_percent=5.0,
        side="LONG",
    )

    short_result = sizer.calculate_from_stop_percent(
        balance=5000.0,
        risk_percent=2.0,
        entry_price=100.0,
        stop_percent=5.0,
        side="SHORT",
    )

    assert long_result.risk_amount == short_result.risk_amount
    assert long_result.position_size == short_result.position_size


def test_result_is_reproducible() -> None:
    sizer = PositionSizer()

    first = sizer.calculate(
        balance=1000.0,
        risk_percent=1.5,
        entry_price=100.0,
        stop_price=97.5,
        leverage=4.0,
    )

    second = sizer.calculate(
        balance=1000.0,
        risk_percent=1.5,
        entry_price=100.0,
        stop_price=97.5,
        leverage=4.0,
    )

    assert first == second


def test_position_notional_is_independent_of_leverage() -> None:
    sizer = PositionSizer()

    result_1x = sizer.calculate(
        balance=1000.0,
        risk_percent=1.0,
        entry_price=100.0,
        stop_price=95.0,
        leverage=1.0,
    )

    result_10x = sizer.calculate(
        balance=1000.0,
        risk_percent=1.0,
        entry_price=100.0,
        stop_price=95.0,
        leverage=10.0,
    )

    assert result_1x.position_size == result_10x.position_size
    assert result_1x.position_notional == result_10x.position_notional
    assert result_10x.margin_required < result_1x.margin_required


def test_risk_amount_scales_with_balance() -> None:
    sizer = PositionSizer()

    result = sizer.calculate(
        balance=2000.0,
        risk_percent=1.0,
        entry_price=100.0,
        stop_price=90.0,
    )

    assert result.risk_amount == 20.0
    assert result.position_size == 2.0


def test_risk_amount_scales_with_risk_percent() -> None:
    sizer = PositionSizer()

    result = sizer.calculate(
        balance=1000.0,
        risk_percent=2.0,
        entry_price=100.0,
        stop_price=90.0,
    )

    assert result.risk_amount == 20.0
    assert result.position_size == 2.0


def test_stop_percent_is_reflected_in_result() -> None:
    sizer = PositionSizer()

    result = sizer.calculate_from_stop_percent(
        balance=1000.0,
        risk_percent=1.0,
        entry_price=100.0,
        stop_percent=10.0,
        side="LONG",
    )

    assert result.stop_distance_percent == 10.0


def test_default_leverage_is_one() -> None:
    sizer = PositionSizer()

    result = sizer.calculate(
        balance=1000.0,
        risk_percent=1.0,
        entry_price=100.0,
        stop_price=95.0,
    )

    assert result.leverage == 1.0


def test_default_limits_allow_requested_leverage() -> None:
    sizer = PositionSizer()

    result = sizer.calculate(
        balance=1000.0,
        risk_percent=1.0,
        entry_price=100.0,
        stop_price=95.0,
        leverage=50.0,
    )

    assert result.leverage == 50.0
    assert result.margin_required == 4.0