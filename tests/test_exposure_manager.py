import pytest

from src.exposure_manager import (
    ExposureManager,
    ExposureResult,
)


def test_default_limits() -> None:
    manager = ExposureManager()

    assert manager.max_total_exposure_percent == 60.0
    assert manager.max_position_exposure_percent == 5.0


def test_custom_limits() -> None:
    manager = ExposureManager(
        max_total_exposure_percent=70.0,
        max_position_exposure_percent=7.0,
    )

    assert manager.max_total_exposure_percent == 70.0
    assert manager.max_position_exposure_percent == 7.0


@pytest.mark.parametrize(
    "total,position",
    [
        (-1.0, 0.0),
        (0.0, -1.0),
    ],
)
def test_negative_exposure_rejected(
    total: float,
    position: float,
) -> None:
    manager = ExposureManager()

    with pytest.raises(ValueError):
        manager.calculate(
            equity=1000.0,
            current_exposure=total,
            position_exposure=position,
        )


@pytest.mark.parametrize(
    "equity",
    [0.0, -1.0],
)
def test_invalid_equity_rejected(equity: float) -> None:
    manager = ExposureManager()

    with pytest.raises(ValueError):
        manager.calculate(
            equity=equity,
            current_exposure=100.0,
        )


def test_calculate_default_exposure() -> None:
    manager = ExposureManager()

    result = manager.calculate(
        equity=1000.0,
        current_exposure=300.0,
        position_exposure=40.0,
    )

    assert isinstance(result, ExposureResult)
    assert result.equity == 1000.0
    assert result.total_exposure == 300.0
    assert result.total_exposure_percent == 30.0
    assert result.available_exposure == 300.0
    assert result.available_exposure_percent == 30.0
    assert result.position_exposure == 40.0
    assert result.position_exposure_percent == 4.0
    assert result.within_limit is True


def test_total_exposure_limit() -> None:
    manager = ExposureManager()

    result = manager.calculate(
        equity=1000.0,
        current_exposure=600.0,
        position_exposure=50.0,
    )

    assert result.total_exposure_percent == 60.0
    assert result.available_exposure == 0.0
    assert result.within_limit is True


def test_total_exposure_above_limit() -> None:
    manager = ExposureManager()

    result = manager.calculate(
        equity=1000.0,
        current_exposure=650.0,
        position_exposure=50.0,
    )

    assert result.total_exposure_percent == 65.0
    assert result.available_exposure == 0.0
    assert result.within_limit is False


def test_position_exposure_above_limit() -> None:
    manager = ExposureManager()

    result = manager.calculate(
        equity=1000.0,
        current_exposure=300.0,
        position_exposure=60.0,
    )

    assert result.position_exposure_percent == 6.0
    assert result.within_limit is False


@pytest.mark.parametrize(
    "current,new_position,expected",
    [
        (0.0, 50.0, True),
        (500.0, 50.0, True),
        (550.0, 50.0, True),
        (550.0, 60.0, False),
        (600.0, 1.0, False),
        (0.0, 51.0, False),
    ],
)
def test_can_open_position(
    current: float,
    new_position: float,
    expected: bool,
) -> None:
    manager = ExposureManager()

    assert (
        manager.can_open_position(
            equity=1000.0,
            current_exposure=current,
            new_position_exposure=new_position,
        )
        is expected
    )


def test_max_position_capital() -> None:
    manager = ExposureManager()

    assert manager.max_position_capital(1000.0) == 50.0
    assert manager.max_position_capital(2000.0) == 100.0


def test_max_total_capital() -> None:
    manager = ExposureManager()

    assert manager.max_total_capital(1000.0) == 600.0
    assert manager.max_total_capital(2000.0) == 1200.0


@pytest.mark.parametrize(
    "total_limit,position_limit",
    [
        (-1.0, 5.0),
        (101.0, 5.0),
        (60.0, 0.0),
        (60.0, -1.0),
        (60.0, 101.0),
    ],
)
def test_invalid_limits_rejected(
    total_limit: float,
    position_limit: float,
) -> None:
    with pytest.raises(ValueError):
        ExposureManager(
            max_total_exposure_percent=total_limit,
            max_position_exposure_percent=position_limit,
        )


def test_zero_current_exposure() -> None:
    manager = ExposureManager()

    result = manager.calculate(
        equity=1000.0,
        current_exposure=0.0,
        position_exposure=0.0,
    )

    assert result.total_exposure_percent == 0.0
    assert result.position_exposure_percent == 0.0
    assert result.available_exposure == 600.0
    assert result.within_limit is True


def test_exact_position_limit() -> None:
    manager = ExposureManager()

    assert (
        manager.can_open_position(
            equity=1000.0,
            current_exposure=0.0,
            new_position_exposure=50.0,
        )
        is True
    )


def test_position_above_limit() -> None:
    manager = ExposureManager()

    assert (
        manager.can_open_position(
            equity=1000.0,
            current_exposure=0.0,
            new_position_exposure=50.01,
        )
        is False
    )


def test_result_is_frozen() -> None:
    manager = ExposureManager()

    result = manager.calculate(
        equity=1000.0,
        current_exposure=100.0,
    )

    with pytest.raises(AttributeError):
        result.equity = 2000.0