from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StopLossResult:
    side: str
    entry_price: float
    stop_price: float
    stop_distance: float
    stop_distance_percent: float
    trailing_stop_price: float | None


class StopLossManager:
    def __init__(
        self,
        default_stop_percent: float = 2.0,
        trailing_stop_percent: float = 1.0,
    ) -> None:
        if default_stop_percent <= 0:
            raise ValueError(
                "default_stop_percent must be greater than zero."
            )

        if trailing_stop_percent <= 0:
            raise ValueError(
                "trailing_stop_percent must be greater than zero."
            )

        self.default_stop_percent = float(
            default_stop_percent
        )
        self.trailing_stop_percent = float(
            trailing_stop_percent
        )

    def calculate(
        self,
        entry_price: float,
        side: str,
        stop_percent: float | None = None,
    ) -> StopLossResult:
        if entry_price <= 0:
            raise ValueError(
                "entry_price must be greater than zero."
            )

        normalized_side = str(side).upper()

        if normalized_side not in {"LONG", "SHORT"}:
            raise ValueError(
                "side must be either LONG or SHORT."
            )

        percent = (
            self.default_stop_percent
            if stop_percent is None
            else float(stop_percent)
        )

        if percent <= 0:
            raise ValueError(
                "stop_percent must be greater than zero."
            )

        if normalized_side == "LONG":
            stop_price = (
                entry_price
                * (1.0 - percent / 100.0)
            )
        else:
            stop_price = (
                entry_price
                * (1.0 + percent / 100.0)
            )

        stop_distance = abs(
            entry_price - stop_price
        )

        stop_distance_percent = (
            stop_distance
            / entry_price
            * 100.0
        )

        return StopLossResult(
            side=normalized_side,
            entry_price=float(entry_price),
            stop_price=round(stop_price, 8),
            stop_distance=round(
                stop_distance,
                8,
            ),
            stop_distance_percent=round(
                stop_distance_percent,
                8,
            ),
            trailing_stop_price=None,
        )

    def calculate_trailing(
        self,
        current_price: float,
        side: str,
        trailing_percent: float | None = None,
    ) -> float:
        if current_price <= 0:
            raise ValueError(
                "current_price must be greater than zero."
            )

        normalized_side = str(side).upper()

        if normalized_side not in {"LONG", "SHORT"}:
            raise ValueError(
                "side must be either LONG or SHORT."
            )

        percent = (
            self.trailing_stop_percent
            if trailing_percent is None
            else float(trailing_percent)
        )

        if percent <= 0:
            raise ValueError(
                "trailing_percent must be greater than zero."
            )

        if normalized_side == "LONG":
            stop_price = (
                current_price
                * (1.0 - percent / 100.0)
            )
        else:
            stop_price = (
                current_price
                * (1.0 + percent / 100.0)
            )

        return round(stop_price, 8)

    def update_trailing(
        self,
        current_price: float,
        side: str,
        previous_stop: float | None,
        trailing_percent: float | None = None,
    ) -> float:
        new_stop = self.calculate_trailing(
            current_price=current_price,
            side=side,
            trailing_percent=trailing_percent,
        )

        if previous_stop is None:
            return new_stop

        normalized_side = str(side).upper()

        if normalized_side == "LONG":
            return max(
                float(previous_stop),
                new_stop,
            )

        return min(
            float(previous_stop),
            new_stop,
        )

    def is_stop_hit(
        self,
        current_price: float,
        stop_price: float,
        side: str,
    ) -> bool:
        if current_price <= 0:
            raise ValueError(
                "current_price must be greater than zero."
            )

        if stop_price <= 0:
            raise ValueError(
                "stop_price must be greater than zero."
            )

        normalized_side = str(side).upper()

        if normalized_side == "LONG":
            return current_price <= stop_price

        if normalized_side == "SHORT":
            return current_price >= stop_price

        raise ValueError(
            "side must be either LONG or SHORT."
        )