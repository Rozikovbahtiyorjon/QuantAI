from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


OrderSide = Literal["BUY", "SELL"]
PositionSide = Literal["LONG", "SHORT"]


@dataclass(frozen=True)
class ExecutionResult:
    side: OrderSide
    requested_price: float
    execution_price: float
    quantity: float
    commission: float
    slippage: float
    notional: float


class ExecutionSimulator:
    """
    Deterministic virtual order execution simulator.

    Handles:
    - BUY / SELL
    - slippage
    - commission
    - quantity validation
    - long / short execution pricing
    """

    def __init__(
        self,
        commission: float = 0.0004,
        slippage: float = 0.0,
    ) -> None:
        if commission < 0:
            raise ValueError(
                "commission cannot be negative."
            )

        if slippage < 0:
            raise ValueError(
                "slippage cannot be negative."
            )

        self.commission = float(commission)
        self.slippage = float(slippage)

    def execute(
        self,
        side: OrderSide,
        price: float,
        quantity: float,
    ) -> ExecutionResult:
        """
        Simulate execution of one market order.

        BUY:
            execution_price = price * (1 + slippage)

        SELL:
            execution_price = price * (1 - slippage)
        """

        if side not in ("BUY", "SELL"):
            raise ValueError(
                "side must be BUY or SELL."
            )

        if price <= 0:
            raise ValueError(
                "price must be greater than zero."
            )

        if quantity <= 0:
            raise ValueError(
                "quantity must be greater than zero."
            )

        if side == "BUY":
            execution_price = (
                price * (1.0 + self.slippage)
            )
        else:
            execution_price = (
                price * (1.0 - self.slippage)
            )

        notional = (
            execution_price * quantity
        )

        commission = (
            notional * self.commission
        )

        return ExecutionResult(
            side=side,
            requested_price=float(price),
            execution_price=float(
                execution_price
            ),
            quantity=float(quantity),
            commission=float(commission),
            slippage=float(
                abs(execution_price - price)
            ),
            notional=float(notional),
        )

    def open_long(
        self,
        price: float,
        quantity: float,
    ) -> ExecutionResult:
        return self.execute(
            side="BUY",
            price=price,
            quantity=quantity,
        )

    def close_long(
        self,
        price: float,
        quantity: float,
    ) -> ExecutionResult:
        return self.execute(
            side="SELL",
            price=price,
            quantity=quantity,
        )

    def open_short(
        self,
        price: float,
        quantity: float,
    ) -> ExecutionResult:
        return self.execute(
            side="SELL",
            price=price,
            quantity=quantity,
        )

    def close_short(
        self,
        price: float,
        quantity: float,
    ) -> ExecutionResult:
        return self.execute(
            side="BUY",
            price=price,
            quantity=quantity,
        )

    def calculate_long_pnl(
        self,
        entry: ExecutionResult,
        exit: ExecutionResult,
    ) -> float:
        """
        Calculate net PnL for a completed LONG position.
        """

        if entry.side != "BUY":
            raise ValueError(
                "LONG entry must be a BUY execution."
            )

        if exit.side != "SELL":
            raise ValueError(
                "LONG exit must be a SELL execution."
            )

        if entry.quantity != exit.quantity:
            raise ValueError(
                "Entry and exit quantities must match."
            )

        gross_pnl = (
            exit.execution_price
            - entry.execution_price
        ) * entry.quantity

        return float(
            gross_pnl
            - entry.commission
            - exit.commission
        )

    def calculate_short_pnl(
        self,
        entry: ExecutionResult,
        exit: ExecutionResult,
    ) -> float:
        """
        Calculate net PnL for a completed SHORT position.
        """

        if entry.side != "SELL":
            raise ValueError(
                "SHORT entry must be a SELL execution."
            )

        if exit.side != "BUY":
            raise ValueError(
                "SHORT exit must be a BUY execution."
            )

        if entry.quantity != exit.quantity:
            raise ValueError(
                "Entry and exit quantities must match."
            )

        gross_pnl = (
            entry.execution_price
            - exit.execution_price
        ) * entry.quantity

        return float(
            gross_pnl
            - entry.commission
            - exit.commission
        )


__all__ = [
    "ExecutionResult",
    "ExecutionSimulator",
]