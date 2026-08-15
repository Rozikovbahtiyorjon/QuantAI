import pytest

from src.stop_loss_manager import (
    StopLossManager,
    StopLossResult,
)


def test_long_stop_loss() -> None:
    manager = StopLossManager()

    result = manager.calculate(
        entry_price=100.0,
        side="LONG",
        stop_percent=5.0,
    )

    assert isinstance(
        result,
        StopLossResult,
    )

    assert result.side == "LONG"
    assert result.entry_price == 100.0
    assert result.stop_price == 95.0
    assert result.stop_distance == 5.0
    assert result.stop_distance_percent == 5.0
    assert result.trailing_stop_price is None


def test_short_stop_loss() -> None:
    manager = StopLossManager()

    result = manager.calculate(
        entry_price=100.0,
        side="SHORT",
        stop_percent=5.0,
    )

    assert result.side == "SHORT"
    assert result.stop_price == 105.0
    assert result.stop_distance == 5.0
    assert result.stop_distance_percent == 5.0


def test_default_stop_percent() -> None:
    manager = StopLossManager(
        default_stop_percent=3.0
    )

    result = manager.calculate(
        entry_price=100.0,
        side="LONG",
    )

    assert result.stop_price == 97.0


def test_custom_stop_percent_overrides_default() -> None:
    manager = StopLossManager(
        default_stop_percent=3.0
    )

    result = manager.calculate(
        entry_price=100.0,
        side="LONG",
        stop_percent=7.0,
    )

    assert result.stop_price == 93.0


def test_long_trailing_stop() -> None:
    manager = StopLossManager(
        trailing_stop_percent=2.0
    )

    result = manager.calculate_trailing(
        current_price=100.0,
        side="LONG",
    )

    assert result == 98.0


def test_short_trailing_stop() -> None:
    manager = StopLossManager(
        trailing_stop_percent=2.0
    )

    result = manager.calculate_trailing(
        current_price=100.0,
        side="SHORT",
    )

    assert result == 102.0


def test_long_trailing_stop_moves_up() -> None:
    manager = StopLossManager(
        trailing_stop_percent=2.0
    )

    first = manager.update_trailing(
        current_price=100.0,
        side="LONG",
        previous_stop=None,
    )

    second = manager.update_trailing(
        current_price=105.0,
        side="LONG",
        previous_stop=first,
    )

    assert first == 98.0
    assert second == 102.9


def test_long_trailing_stop_never_moves_down() -> None:
    manager = StopLossManager(
        trailing_stop_percent=2.0
    )

    result = manager.update_trailing(
        current_price=95.0,
        side="LONG",
        previous_stop=98.0,
    )

    assert result == 98.0


def test_short_trailing_stop_moves_down() -> None:
    manager = StopLossManager(
        trailing_stop_percent=2.0
    )

    first = manager.update_trailing(
        current_price=100.0,
        side="SHORT",
        previous_stop=None,
    )

    second = manager.update_trailing(
        current_price=95.0,
        side="SHORT",
        previous_stop=first,
    )

    assert first == 102.0
    assert second == 96.9


def test_short_trailing_stop_never_moves_up() -> None:
    manager = StopLossManager(
        trailing_stop_percent=2.0
    )

    result = manager.update_trailing(
        current_price=105.0,
        side="SHORT",
        previous_stop=102.0,
    )

    assert result == 102.0


def test_long_stop_is_hit() -> None:
    manager = StopLossManager()

    assert manager.is_stop_hit(
        current_price=95.0,
        stop_price=95.0,
        side="LONG",
    )


def test_long_stop_is_not_hit() -> None:
    manager = StopLossManager()

    assert not manager.is_stop_hit(
        current_price=96.0,
        stop_price=95.0,
        side="LONG",
    )


def test_short_stop_is_hit() -> None:
    manager = StopLossManager()

    assert manager.is_stop_hit(
        current_price=105.0,
        stop_price=105.0,
        side="SHORT",
    )


def test_short_stop_is_not_hit() -> None:
    manager = StopLossManager()

    assert not manager.is_stop_hit(
        current_price=104.0,
        stop_price=105.0,
        side="SHORT",
    )


def test_invalid_entry_price() -> None:
    manager = StopLossManager()

    with pytest.raises(ValueError):
        manager.calculate(
            entry_price=0.0,
            side="LONG",
        )


def test_invalid_side() -> None:
    manager = StopLossManager()

    with pytest.raises(ValueError):
        manager.calculate(
            entry_price=100.0,
            side="INVALID",
        )


def test_invalid_stop_percent() -> None:
    manager = StopLossManager()

    with pytest.raises(ValueError):
        manager.calculate(
            entry_price=100.0,
            side="LONG",
            stop_percent=0.0,
        )


def test_invalid_trailing_percent() -> None:
    manager = StopLossManager()

    with pytest.raises(ValueError):
        manager.calculate_trailing(
            current_price=100.0,
            side="LONG",
            trailing_percent=0.0,
        )


def test_invalid_current_price() -> None:
    manager = StopLossManager()

    with pytest.raises(ValueError):
        manager.calculate_trailing(
            current_price=0.0,
            side="LONG",
        )


def test_invalid_stop_hit_side() -> None:
    manager = StopLossManager()

    with pytest.raises(ValueError):
        manager.is_stop_hit(
            current_price=100.0,
            stop_price=95.0,
            side="INVALID",
        )


def test_update_trailing_initializes_from_current_price() -> None:
    manager = StopLossManager(
        trailing_stop_percent=1.0
    )

    result = manager.update_trailing(
        current_price=200.0,
        side="LONG",
        previous_stop=None,
    )

    assert result == 198.0


def test_custom_trailing_percent() -> None:
    manager = StopLossManager()

    result = manager.calculate_trailing(
        current_price=100.0,
        side="LONG",
        trailing_percent=5.0,
    )

    assert result == 95.0


def test_stop_distance_is_absolute_for_short() -> None:
    manager = StopLossManager()

    result = manager.calculate(
        entry_price=200.0,
        side="SHORT",
        stop_percent=10.0,
    )

    assert result.stop_price == 220.0
    assert result.stop_distance == 20.0
    assert result.stop_distance_percent == 10.0