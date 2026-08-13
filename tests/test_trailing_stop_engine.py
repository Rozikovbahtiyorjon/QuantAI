import pytest

from src.trailing_stop_engine import (
    TrailingStopEngine,
    TrailingStopResult,
)


def test_long_trailing_stop() -> None:
    engine = TrailingStopEngine(
        trailing_percent=2.0
    )

    result = engine.calculate(
        current_price=100.0,
        side="LONG",
    )

    assert isinstance(
        result,
        TrailingStopResult,
    )

    assert result.side == "LONG"
    assert result.current_price == 100.0
    assert result.trailing_percent == 2.0
    assert result.stop_price == 98.0
    assert result.previous_stop is None
    assert result.moved is True


def test_short_trailing_stop() -> None:
    engine = TrailingStopEngine(
        trailing_percent=2.0
    )

    result = engine.calculate(
        current_price=100.0,
        side="SHORT",
    )

    assert result.stop_price == 102.0
    assert result.moved is True


def test_long_stop_moves_up() -> None:
    engine = TrailingStopEngine(
        trailing_percent=2.0
    )

    result = engine.calculate(
        current_price=105.0,
        side="LONG",
        previous_stop=98.0,
    )

    assert result.stop_price == 102.9
    assert result.moved is True


def test_long_stop_does_not_move_down() -> None:
    engine = TrailingStopEngine(
        trailing_percent=2.0
    )

    result = engine.calculate(
        current_price=95.0,
        side="LONG",
        previous_stop=98.0,
    )

    assert result.stop_price == 98.0
    assert result.moved is False


def test_short_stop_moves_down() -> None:
    engine = TrailingStopEngine(
        trailing_percent=2.0
    )

    result = engine.calculate(
        current_price=95.0,
        side="SHORT",
        previous_stop=102.0,
    )

    assert result.stop_price == 96.9
    assert result.moved is True


def test_short_stop_does_not_move_up() -> None:
    engine = TrailingStopEngine(
        trailing_percent=2.0
    )

    result = engine.calculate(
        current_price=105.0,
        side="SHORT",
        previous_stop=102.0,
    )

    assert result.stop_price == 102.0
    assert result.moved is False


def test_custom_trailing_percent() -> None:
    engine = TrailingStopEngine(
        trailing_percent=2.0
    )

    result = engine.calculate(
        current_price=100.0,
        side="LONG",
        trailing_percent=5.0,
    )

    assert result.stop_price == 95.0
    assert result.trailing_percent == 5.0


def test_custom_percent_does_not_change_engine_default() -> None:
    engine = TrailingStopEngine(
        trailing_percent=2.0
    )

    engine.calculate(
        current_price=100.0,
        side="LONG",
        trailing_percent=5.0,
    )

    result = engine.calculate(
        current_price=100.0,
        side="LONG",
    )

    assert result.stop_price == 98.0
    assert result.trailing_percent == 2.0


def test_long_exact_same_stop_is_not_moved() -> None:
    engine = TrailingStopEngine(
        trailing_percent=2.0
    )

    result = engine.calculate(
        current_price=100.0,
        side="LONG",
        previous_stop=98.0,
    )

    assert result.stop_price == 98.0
    assert result.moved is False


def test_short_exact_same_stop_is_not_moved() -> None:
    engine = TrailingStopEngine(
        trailing_percent=2.0
    )

    result = engine.calculate(
        current_price=100.0,
        side="SHORT",
        previous_stop=102.0,
    )

    assert result.stop_price == 102.0
    assert result.moved is False


def test_invalid_current_price() -> None:
    engine = TrailingStopEngine()

    with pytest.raises(ValueError):
        engine.calculate(
            current_price=0.0,
            side="LONG",
        )


def test_invalid_negative_current_price() -> None:
    engine = TrailingStopEngine()

    with pytest.raises(ValueError):
        engine.calculate(
            current_price=-100.0,
            side="LONG",
        )


def test_invalid_side() -> None:
    engine = TrailingStopEngine()

    with pytest.raises(ValueError):
        engine.calculate(
            current_price=100.0,
            side="INVALID",
        )


def test_invalid_constructor_percent() -> None:
    with pytest.raises(ValueError):
        TrailingStopEngine(
            trailing_percent=0.0
        )


def test_invalid_negative_constructor_percent() -> None:
    with pytest.raises(ValueError):
        TrailingStopEngine(
            trailing_percent=-1.0
        )


def test_invalid_runtime_percent() -> None:
    engine = TrailingStopEngine()

    with pytest.raises(ValueError):
        engine.calculate(
            current_price=100.0,
            side="LONG",
            trailing_percent=0.0,
        )


def test_invalid_negative_runtime_percent() -> None:
    engine = TrailingStopEngine()

    with pytest.raises(ValueError):
        engine.calculate(
            current_price=100.0,
            side="LONG",
            trailing_percent=-1.0,
        )


def test_result_is_reproducible() -> None:
    engine = TrailingStopEngine(
        trailing_percent=2.0
    )

    first = engine.calculate(
        current_price=100.0,
        side="LONG",
    )

    second = engine.calculate(
        current_price=100.0,
        side="LONG",
    )

    assert first == second


def test_long_trailing_stop_never_decreases() -> None:
    engine = TrailingStopEngine(
        trailing_percent=1.0
    )

    stop = None

    for price in [100.0, 105.0, 103.0, 110.0]:
        result = engine.calculate(
            current_price=price,
            side="LONG",
            previous_stop=stop,
        )
        stop = result.stop_price

    assert stop == 108.9


def test_short_trailing_stop_never_increases() -> None:
    engine = TrailingStopEngine(
        trailing_percent=1.0
    )

    stop = None

    for price in [100.0, 95.0, 98.0, 90.0]:
        result = engine.calculate(
            current_price=price,
            side="SHORT",
            previous_stop=stop,
        )
        stop = result.stop_price

    assert stop == 90.9


def test_long_initialization_with_previous_stop() -> None:
    engine = TrailingStopEngine(
        trailing_percent=2.0
    )

    result = engine.calculate(
        current_price=100.0,
        side="LONG",
        previous_stop=90.0,
    )

    assert result.stop_price == 98.0
    assert result.previous_stop == 90.0


def test_short_initialization_with_previous_stop() -> None:
    engine = TrailingStopEngine(
        trailing_percent=2.0
    )

    result = engine.calculate(
        current_price=100.0,
        side="SHORT",
        previous_stop=110.0,
    )

    assert result.stop_price == 102.0
    assert result.previous_stop == 110.0


def test_reset_does_not_raise() -> None:
    engine = TrailingStopEngine()

    engine.reset()


def test_case_insensitive_long() -> None:
    engine = TrailingStopEngine(
        trailing_percent=2.0
    )

    result = engine.calculate(
        current_price=100.0,
        side="long",
    )

    assert result.side == "LONG"
    assert result.stop_price == 98.0


def test_case_insensitive_short() -> None:
    engine = TrailingStopEngine(
        trailing_percent=2.0
    )

    result = engine.calculate(
        current_price=100.0,
        side="short",
    )

    assert result.side == "SHORT"
    assert result.stop_price == 102.0