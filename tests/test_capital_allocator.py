import pytest

from src.capital_allocator import (
    CapitalAllocationResult,
    CapitalAllocator,
)


def test_default_allocation() -> None:
    allocator = CapitalAllocator()

    result = allocator.allocate(
        equity=1000.0,
        asset_count=4,
    )

    assert isinstance(
        result,
        CapitalAllocationResult,
    )

    assert result.equity == 1000.0
    assert result.reserve == 400.0
    assert result.trading_capital == 600.0
    assert result.allocated_capital == 600.0
    assert result.allocation_percent == 60.0
    assert result.reserve_percent == 40.0
    assert result.per_asset_capital == 150.0
    assert result.asset_count == 4


def test_five_assets() -> None:
    allocator = CapitalAllocator()

    result = allocator.allocate(
        equity=1000.0,
        asset_count=5,
    )

    assert result.reserve == 400.0
    assert result.trading_capital == 600.0
    assert result.per_asset_capital == 120.0


def test_custom_allocation_percent() -> None:
    allocator = CapitalAllocator()

    result = allocator.allocate(
        equity=1000.0,
        asset_count=4,
        allocation_percent=40.0,
    )

    assert result.reserve == 400.0
    assert result.trading_capital == 600.0
    assert result.allocated_capital == 400.0
    assert result.per_asset_capital == 100.0


def test_custom_reserve_and_trading_percent() -> None:
    allocator = CapitalAllocator(
        reserve_percent=30.0,
        trading_percent=70.0,
    )

    result = allocator.allocate(
        equity=2000.0,
        asset_count=5,
    )

    assert result.reserve == 600.0
    assert result.trading_capital == 1400.0
    assert result.per_asset_capital == 280.0


def test_calculate_reserve() -> None:
    allocator = CapitalAllocator()

    assert allocator.calculate_reserve(
        2500.0
    ) == 1000.0


def test_calculate_trading_capital() -> None:
    allocator = CapitalAllocator()

    assert allocator.calculate_trading_capital(
        2500.0
    ) == 1500.0


def test_calculate_asset_capital() -> None:
    allocator = CapitalAllocator()

    assert allocator.calculate_asset_capital(
        equity=2500.0,
        asset_count=5,
    ) == 300.0


def test_zero_equity_rejected() -> None:
    allocator = CapitalAllocator()

    with pytest.raises(ValueError):
        allocator.allocate(
            equity=0.0,
            asset_count=4,
        )


def test_negative_equity_rejected() -> None:
    allocator = CapitalAllocator()

    with pytest.raises(ValueError):
        allocator.allocate(
            equity=-100.0,
            asset_count=4,
        )


def test_zero_asset_count_rejected() -> None:
    allocator = CapitalAllocator()

    with pytest.raises(ValueError):
        allocator.allocate(
            equity=1000.0,
            asset_count=0,
        )


def test_negative_asset_count_rejected() -> None:
    allocator = CapitalAllocator()

    with pytest.raises(ValueError):
        allocator.allocate(
            equity=1000.0,
            asset_count=-1,
        )


def test_negative_reserve_percent_rejected() -> None:
    with pytest.raises(ValueError):
        CapitalAllocator(
            reserve_percent=-1.0,
            trading_percent=101.0,
        )


def test_negative_trading_percent_rejected() -> None:
    with pytest.raises(ValueError):
        CapitalAllocator(
            reserve_percent=101.0,
            trading_percent=-1.0,
        )


def test_percentages_must_equal_100() -> None:
    with pytest.raises(ValueError):
        CapitalAllocator(
            reserve_percent=40.0,
            trading_percent=50.0,
        )


def test_zero_per_asset_percent_rejected() -> None:
    with pytest.raises(ValueError):
        CapitalAllocator(
            per_asset_percent=0.0,
        )


def test_negative_per_asset_percent_rejected() -> None:
    with pytest.raises(ValueError):
        CapitalAllocator(
            per_asset_percent=-1.0,
        )


def test_allocation_cannot_exceed_trading_capital() -> None:
    allocator = CapitalAllocator()

    with pytest.raises(ValueError):
        allocator.allocate(
            equity=1000.0,
            asset_count=4,
            allocation_percent=61.0,
        )


def test_negative_allocation_rejected() -> None:
    allocator = CapitalAllocator()

    with pytest.raises(ValueError):
        allocator.allocate(
            equity=1000.0,
            asset_count=4,
            allocation_percent=-1.0,
        )


def test_allocation_zero_is_allowed() -> None:
    allocator = CapitalAllocator()

    result = allocator.allocate(
        equity=1000.0,
        asset_count=4,
        allocation_percent=0.0,
    )

    assert result.allocated_capital == 0.0
    assert result.per_asset_capital == 0.0


def test_single_asset() -> None:
    allocator = CapitalAllocator()

    result = allocator.allocate(
        equity=1000.0,
        asset_count=1,
    )

    assert result.per_asset_capital == 600.0


def test_large_equity() -> None:
    allocator = CapitalAllocator()

    result = allocator.allocate(
        equity=1_000_000.0,
        asset_count=5,
    )

    assert result.reserve == 400_000.0
    assert result.trading_capital == 600_000.0
    assert result.per_asset_capital == 120_000.0


def test_fractional_asset_count_is_rejected() -> None:
    allocator = CapitalAllocator()

    with pytest.raises(
        (ValueError, TypeError)
    ):
        allocator.allocate(
            equity=1000.0,
            asset_count=2.5,
        )


def test_case_allocation_matches_equity_percentage() -> None:
    allocator = CapitalAllocator(
        reserve_percent=40.0,
        trading_percent=60.0,
    )

    result = allocator.allocate(
        equity=5000.0,
        asset_count=5,
        allocation_percent=50.0,
    )

    assert result.allocated_capital == 2500.0
    assert result.per_asset_capital == 500.0


def test_reserve_plus_trading_equals_equity() -> None:
    allocator = CapitalAllocator()

    result = allocator.allocate(
        equity=12345.67,
        asset_count=5,
    )

    assert (
        result.reserve
        + result.trading_capital
    ) == pytest.approx(
        result.equity,
        abs=1e-7,
    )


def test_per_asset_allocation_sum() -> None:
    allocator = CapitalAllocator()

    result = allocator.allocate(
        equity=10000.0,
        asset_count=5,
        allocation_percent=50.0,
    )

    assert (
        result.per_asset_capital
        * result.asset_count
    ) == pytest.approx(
        result.allocated_capital,
        abs=1e-7,
    )