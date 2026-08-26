from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CapitalAllocationResult:
    equity: float
    reserve: float
    trading_capital: float
    allocated_capital: float
    allocation_percent: float
    reserve_percent: float
    per_asset_capital: float
    asset_count: int


class CapitalAllocator:
    def __init__(
        self,
        reserve_percent: float = 40.0,
        trading_percent: float = 60.0,
        per_asset_percent: float = 5.0,
    ) -> None:
        if reserve_percent < 0:
            raise ValueError(
                "reserve_percent cannot be negative."
            )

        if trading_percent < 0:
            raise ValueError(
                "trading_percent cannot be negative."
            )

        if per_asset_percent <= 0:
            raise ValueError(
                "per_asset_percent must be greater than zero."
            )

        if abs(
            reserve_percent
            + trading_percent
            - 100.0
        ) > 1e-9:
            raise ValueError(
                "reserve_percent + trading_percent "
                "must equal 100."
            )

        self.reserve_percent = float(
            reserve_percent
        )
        self.trading_percent = float(
            trading_percent
        )
        self.per_asset_percent = float(
            per_asset_percent
        )

    def allocate(
        self,
        equity: float,
        asset_count: int,
        allocation_percent: float | None = None,
    ) -> CapitalAllocationResult:
        if equity <= 0:
            raise ValueError(
                "equity must be greater than zero."
            )

        if (
            isinstance(asset_count, bool)
            or not isinstance(asset_count, int)
        ):
            raise TypeError(
                "asset_count must be an integer."
            )

        if asset_count <= 0:
            raise ValueError(
                "asset_count must be greater than zero."
            )

        allocation = (
            self.trading_percent
            if allocation_percent is None
            else float(allocation_percent)
        )

        if allocation < 0:
            raise ValueError(
                "allocation_percent cannot be negative."
            )

        if allocation > self.trading_percent:
            raise ValueError(
                "allocation_percent cannot exceed "
                "trading_percent."
            )

        reserve = (
            equity
            * self.reserve_percent
            / 100.0
        )

        trading_capital = (
            equity
            * self.trading_percent
            / 100.0
        )

        allocated_capital = (
            equity
            * allocation
            / 100.0
        )

        per_asset_capital = (
            allocated_capital
            / asset_count
        )

        return CapitalAllocationResult(
            equity=float(equity),
            reserve=round(reserve, 8),
            trading_capital=round(
                trading_capital,
                8,
            ),
            allocated_capital=round(
                allocated_capital,
                8,
            ),
            allocation_percent=round(
                allocation,
                8,
            ),
            reserve_percent=round(
                self.reserve_percent,
                8,
            ),
            per_asset_capital=round(
                per_asset_capital,
                8,
            ),
            asset_count=asset_count,
        )

    def calculate_asset_capital(
        self,
        equity: float,
        asset_count: int,
    ) -> float:
        return self.allocate(
            equity=equity,
            asset_count=asset_count,
        ).per_asset_capital

    def calculate_reserve(
        self,
        equity: float,
    ) -> float:
        if equity <= 0:
            raise ValueError(
                "equity must be greater than zero."
            )

        return round(
            equity
            * self.reserve_percent
            / 100.0,
            8,
        )

    def calculate_trading_capital(
        self,
        equity: float,
    ) -> float:
        if equity <= 0:
            raise ValueError(
                "equity must be greater than zero."
            )

        return round(
            equity
            * self.trading_percent
            / 100.0,
            8,
        )


__all__ = [
    "CapitalAllocationResult",
    "CapitalAllocator",
]