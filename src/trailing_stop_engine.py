from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrailingStopResult:
    side: str
    current_price: float
    trailing_percent: float
    stop_price: float
    previous_stop: float | None
    moved: bool


class TrailingStopEngine:
    def __init__(
        self,
        trailing_percent: float = 1.0,
    ) -> None:
        if trailing_percent <= 0:
            raise ValueError(
                "trailing_percent must be greater than zero."
            )

        self.trailing_percent = float(
            trailing_percent
        )

    def calculate(
        self,
        current_price: float,
        side: str,
        previous_stop: float | None = None,
        trailing_percent: float | None = None,
    ) -> TrailingStopResult:
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
            self.trailing_percent
            if trailing_percent is None
            else float(trailing_percent)
        )

        if percent <= 0:
            raise ValueError(
                "trailing_percent must be greater than zero."
            )

        if normalized_side == "LONG":
            calculated_stop = (
                current_price
                * (1.0 - percent / 100.0)
            )

            if previous_stop is None:
                stop_price = calculated_stop
            else:
                stop_price = max(
                    float(previous_stop),
                    calculated_stop,
                )

        else:
            calculated_stop = (
                current_price
                * (1.0 + percent / 100.0)
            )

            if previous_stop is None:
                stop_price = calculated_stop
            else:
                stop_price = min(
                    float(previous_stop),
                    calculated_stop,
                )

        moved = (
            previous_stop is None
            or round(stop_price, 8)
            != round(float(previous_stop), 8)
        )

        return TrailingStopResult(
            side=normalized_side,
            current_price=float(current_price),
            trailing_percent=float(percent),
            stop_price=round(
                stop_price,
                8,
            ),
            previous_stop=(
                None
                if previous_stop is None
                else float(previous_stop)
            ),
            moved=moved,
        )

    def reset(self) -> None:
        return None