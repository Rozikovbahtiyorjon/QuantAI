from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExposureResult:
    equity: float
    total_exposure: float
    total_exposure_percent: float
    available_exposure: float
    available_exposure_percent: float
    position_exposure: float
    position_exposure_percent: float
    within_limit: bool


class ExposureManager:
    def __init__(
        self,
        max_total_exposure_percent: float = 60.0,
        max_position_exposure_percent: float = 5.0,
    ) -> None:
        if not 0.0 <= max_total_exposure_percent <= 100.0:
            raise ValueError(
                "max_total_exposure_percent must be between 0 and 100."
            )

        if not 0.0 < max_position_exposure_percent <= 100.0:
            raise ValueError(
                "max_position_exposure_percent must be greater than zero "
                "and no greater than 100."
            )

        self.max_total_exposure_percent = float(
            max_total_exposure_percent
        )
        self.max_position_exposure_percent = float(
            max_position_exposure_percent
        )

    def calculate(
        self,
        equity: float,
        current_exposure: float,
        position_exposure: float = 0.0,
    ) -> ExposureResult:
        if equity <= 0:
            raise ValueError("equity must be greater than zero.")

        if current_exposure < 0:
            raise ValueError("current_exposure cannot be negative.")

        if position_exposure < 0:
            raise ValueError("position_exposure cannot be negative.")

        total_exposure_percent = (
            current_exposure / equity * 100.0
        )

        position_exposure_percent = (
            position_exposure / equity * 100.0
        )

        max_total_exposure = (
            equity
            * self.max_total_exposure_percent
            / 100.0
        )

        available_exposure = max(
            max_total_exposure - current_exposure,
            0.0,
        )

        available_exposure_percent = (
            available_exposure / equity * 100.0
        )

        within_limit = (
            total_exposure_percent
            <= self.max_total_exposure_percent + 1e-9
            and position_exposure_percent
            <= self.max_position_exposure_percent + 1e-9
        )

        return ExposureResult(
            equity=float(equity),
            total_exposure=round(current_exposure, 8),
            total_exposure_percent=round(
                total_exposure_percent,
                8,
            ),
            available_exposure=round(
                available_exposure,
                8,
            ),
            available_exposure_percent=round(
                available_exposure_percent,
                8,
            ),
            position_exposure=round(
                position_exposure,
                8,
            ),
            position_exposure_percent=round(
                position_exposure_percent,
                8,
            ),
            within_limit=within_limit,
        )

    def can_open_position(
        self,
        equity: float,
        current_exposure: float,
        new_position_exposure: float,
    ) -> bool:
        if equity <= 0:
            raise ValueError("equity must be greater than zero.")

        if current_exposure < 0:
            raise ValueError("current_exposure cannot be negative.")

        if new_position_exposure < 0:
            raise ValueError(
                "new_position_exposure cannot be negative."
            )

        total_after_open = (
            current_exposure
            + new_position_exposure
        )

        total_percent = (
            total_after_open / equity * 100.0
        )

        position_percent = (
            new_position_exposure / equity * 100.0
        )

        return (
            total_percent
            <= self.max_total_exposure_percent + 1e-9
            and position_percent
            <= self.max_position_exposure_percent + 1e-9
        )

    def max_position_capital(
        self,
        equity: float,
    ) -> float:
        if equity <= 0:
            raise ValueError("equity must be greater than zero.")

        return round(
            equity
            * self.max_position_exposure_percent
            / 100.0,
            8,
        )

    def max_total_capital(
        self,
        equity: float,
    ) -> float:
        if equity <= 0:
            raise ValueError("equity must be greater than zero.")

        return round(
            equity
            * self.max_total_exposure_percent
            / 100.0,
            8,
        )


__all__ = [
    "ExposureResult",
    "ExposureManager",
]