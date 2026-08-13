from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PositionSizeResult:
    balance: float
    risk_percent: float
    risk_amount: float
    entry_price: float
    stop_price: float
    stop_distance: float
    stop_distance_percent: float
    position_size: float
    position_notional: float
    leverage: float
    margin_required: float


class PositionSizer:
    def __init__(
        self,
        min_leverage: float = 1.0,
        max_leverage: float = 50.0,
    ) -> None:
        if min_leverage <= 0:
            raise ValueError(
                "min_leverage must be greater than zero."
            )

        if max_leverage < min_leverage:
            raise ValueError(
                "max_leverage must be greater than or equal "
                "to min_leverage."
            )

        self.min_leverage = float(min_leverage)
        self.max_leverage = float(max_leverage)

    def calculate(
        self,
        balance: float,
        risk_percent: float,
        entry_price: float,
        stop_price: float,
        leverage: float = 1.0,
    ) -> PositionSizeResult:
        if balance <= 0:
            raise ValueError(
                "balance must be greater than zero."
            )

        if risk_percent <= 0:
            raise ValueError(
                "risk_percent must be greater than zero."
            )

        if entry_price <= 0:
            raise ValueError(
                "entry_price must be greater than zero."
            )

        if stop_price <= 0:
            raise ValueError(
                "stop_price must be greater than zero."
            )

        if stop_price == entry_price:
            raise ValueError(
                "stop_price cannot equal entry_price."
            )

        if leverage < self.min_leverage:
            raise ValueError(
                "leverage is below the allowed minimum."
            )

        if leverage > self.max_leverage:
            raise ValueError(
                "leverage is above the allowed maximum."
            )

        risk_amount = (
            float(balance)
            * float(risk_percent)
            / 100.0
        )

        stop_distance = abs(
            float(entry_price) - float(stop_price)
        )

        stop_distance_percent = (
            stop_distance
            / float(entry_price)
            * 100.0
        )

        position_size = (
            risk_amount
            / stop_distance
        )

        position_notional = (
            position_size
            * float(entry_price)
        )

        margin_required = (
            position_notional
            / float(leverage)
        )

        return PositionSizeResult(
            balance=float(balance),
            risk_percent=float(risk_percent),
            risk_amount=round(risk_amount, 8),
            entry_price=float(entry_price),
            stop_price=float(stop_price),
            stop_distance=round(stop_distance, 8),
            stop_distance_percent=round(
                stop_distance_percent,
                8,
            ),
            position_size=round(
                position_size,
                8,
            ),
            position_notional=round(
                position_notional,
                8,
            ),
            leverage=float(leverage),
            margin_required=round(
                margin_required,
                8,
            ),
        )

    def calculate_from_stop_percent(
        self,
        balance: float,
        risk_percent: float,
        entry_price: float,
        stop_percent: float,
        leverage: float = 1.0,
        side: str = "LONG",
    ) -> PositionSizeResult:
        if stop_percent <= 0:
            raise ValueError(
                "stop_percent must be greater than zero."
            )

        normalized_side = str(side).upper()

        if normalized_side == "LONG":
            stop_price = (
                float(entry_price)
                * (1.0 - float(stop_percent) / 100.0)
            )

        elif normalized_side == "SHORT":
            stop_price = (
                float(entry_price)
                * (1.0 + float(stop_percent) / 100.0)
            )

        else:
            raise ValueError(
                "side must be either LONG or SHORT."
            )

        return self.calculate(
            balance=balance,
            risk_percent=risk_percent,
            entry_price=entry_price,
            stop_price=stop_price,
            leverage=leverage,
        )