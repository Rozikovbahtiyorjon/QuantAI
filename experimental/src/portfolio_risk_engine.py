from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Mapping


@dataclass(frozen=True)
class PortfolioRiskResult:
    equity: float
    total_exposure_percent: float
    total_risk_percent: float
    position_count: int
    risk_allowed: bool
    exposure_limit_ok: bool
    risk_limit_ok: bool
    position_limit_ok: bool


class PortfolioRiskEngine:
    def __init__(
        self,
        max_total_exposure_percent: float = 60.0,
        max_total_risk_percent: float = 10.0,
        max_positions: int = 10,
    ) -> None:
        if max_total_exposure_percent < 0:
            raise ValueError(
                "max_total_exposure_percent cannot be negative."
            )

        if max_total_risk_percent < 0:
            raise ValueError(
                "max_total_risk_percent cannot be negative."
            )

        if max_positions <= 0:
            raise ValueError(
                "max_positions must be greater than zero."
            )

        self.max_total_exposure_percent = float(
            max_total_exposure_percent
        )
        self.max_total_risk_percent = float(
            max_total_risk_percent
        )
        self.max_positions = int(max_positions)

    def evaluate(
        self,
        equity: float,
        positions: Mapping[str, Mapping[str, float]],
    ) -> PortfolioRiskResult:
        if equity <= 0:
            raise ValueError(
                "equity must be greater than zero."
            )

        if not isinstance(positions, Mapping):
            raise TypeError(
                "positions must be a mapping."
            )

        total_exposure = Decimal("0")
        total_risk = Decimal("0")
        active_position_count = 0

        for symbol, position in positions.items():
            self._validate_symbol(symbol)

            if not isinstance(position, Mapping):
                raise TypeError(
                    "each position must be a mapping."
                )

            exposure = self._to_decimal(
                position.get("exposure_percent", 0.0)
            )

            risk = self._to_decimal(
                position.get("risk_percent", 0.0)
            )

            if exposure < 0:
                raise ValueError(
                    "exposure_percent cannot be negative."
                )

            if risk < 0:
                raise ValueError(
                    "risk_percent cannot be negative."
                )

            total_exposure += exposure
            total_risk += risk

            if exposure > 0:
                active_position_count += 1

        total_exposure = total_exposure.quantize(
            Decimal("0.00000001")
        )

        total_risk = total_risk.quantize(
            Decimal("0.01")
        )

        total_exposure_float = float(
            total_exposure
        )

        total_risk_float = float(
            total_risk
        )

        exposure_limit_ok = (
            total_exposure_float
            <= self.max_total_exposure_percent
        )

        risk_limit_ok = (
            total_risk_float
            <= self.max_total_risk_percent
        )

        position_limit_ok = (
            active_position_count
            <= self.max_positions
        )

        risk_allowed = (
            exposure_limit_ok
            and risk_limit_ok
            and position_limit_ok
        )

        return PortfolioRiskResult(
            equity=float(equity),
            total_exposure_percent=total_exposure_float,
            total_risk_percent=total_risk_float,
            position_count=active_position_count,
            risk_allowed=risk_allowed,
            exposure_limit_ok=exposure_limit_ok,
            risk_limit_ok=risk_limit_ok,
            position_limit_ok=position_limit_ok,
        )

    def is_allowed(
        self,
        equity: float,
        positions: Mapping[str, Mapping[str, float]],
    ) -> bool:
        return self.evaluate(
            equity=equity,
            positions=positions,
        ).risk_allowed

    @staticmethod
    def _validate_symbol(symbol: object) -> None:
        if not isinstance(symbol, str):
            raise TypeError(
                "position symbol must be a string."
            )

        if not symbol.strip():
            raise ValueError(
                "position symbol cannot be empty."
            )

    @staticmethod
    def _to_decimal(value: object) -> Decimal:
        try:
            result = Decimal(str(value))
        except (InvalidOperation, ValueError, TypeError) as exc:
            raise TypeError(
                "risk and exposure percentages must be numeric."
            ) from exc

        if not result.is_finite():
            raise ValueError(
                "risk and exposure percentages must be finite."
            )

        return result