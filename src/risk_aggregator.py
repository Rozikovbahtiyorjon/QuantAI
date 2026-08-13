from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Mapping


@dataclass(frozen=True)
class RiskAggregationResult:
    total_risk_percent: float
    total_exposure_percent: float
    position_count: int
    allowed: bool


class RiskAggregator:
    def __init__(
        self,
        max_total_risk_percent: float = 10.0,
        max_total_exposure_percent: float = 60.0,
        max_positions: int = 10,
    ) -> None:
        if max_total_risk_percent < 0:
            raise ValueError(
                "max_total_risk_percent cannot be negative."
            )

        if max_total_exposure_percent < 0:
            raise ValueError(
                "max_total_exposure_percent cannot be negative."
            )

        if max_positions <= 0:
            raise ValueError(
                "max_positions must be greater than zero."
            )

        self.max_total_risk_percent = float(
            max_total_risk_percent
        )
        self.max_total_exposure_percent = float(
            max_total_exposure_percent
        )
        self.max_positions = int(max_positions)

    @staticmethod
    def _to_decimal(value: object) -> Decimal:
        try:
            return Decimal(str(value))
        except Exception as exc:
            raise TypeError(
                "risk and exposure values must be numeric."
            ) from exc

    @staticmethod
    def _normalize_percent(value: Decimal) -> float:
        rounded = value.quantize(
            Decimal("0.00000001"),
            rounding=ROUND_HALF_UP,
        )

        nearest_integer = rounded.to_integral_value(
            rounding=ROUND_HALF_UP
        )

        if abs(rounded - nearest_integer) <= Decimal("0.00000001"):
            return float(nearest_integer)

        return float(rounded)

    def aggregate(
        self,
        positions: Mapping[str, Mapping[str, float]],
    ) -> RiskAggregationResult:
        if not isinstance(positions, Mapping):
            raise TypeError(
                "positions must be a mapping."
            )

        total_risk = Decimal("0")
        total_exposure = Decimal("0")
        position_count = 0

        for symbol, position in positions.items():
            if not isinstance(symbol, str):
                raise TypeError(
                    "position symbol must be a string."
                )

            if not symbol.strip():
                raise ValueError(
                    "position symbol cannot be empty."
                )

            if not isinstance(position, Mapping):
                raise TypeError(
                    "each position must be a mapping."
                )

            risk = self._to_decimal(
                position.get("risk_percent", 0.0)
            )

            exposure = self._to_decimal(
                position.get("exposure_percent", 0.0)
            )

            if risk < 0:
                raise ValueError(
                    "risk_percent cannot be negative."
                )

            if exposure < 0:
                raise ValueError(
                    "exposure_percent cannot be negative."
                )

            total_risk += risk
            total_exposure += exposure

            if exposure > 0:
                position_count += 1

        total_risk = total_risk.quantize(
            Decimal("0.00000001"),
            rounding=ROUND_HALF_UP,
        )

        total_exposure = total_exposure.quantize(
            Decimal("0.00000001"),
            rounding=ROUND_HALF_UP,
        )

        max_risk = Decimal(
            str(self.max_total_risk_percent)
        )

        max_exposure = Decimal(
            str(self.max_total_exposure_percent)
        )

        allowed = (
            total_risk <= max_risk
            and total_exposure <= max_exposure
            and position_count <= self.max_positions
        )

        return RiskAggregationResult(
            total_risk_percent=self._normalize_percent(
                total_risk
            ),
            total_exposure_percent=self._normalize_percent(
                total_exposure
            ),
            position_count=position_count,
            allowed=allowed,
        )

    def is_allowed(
        self,
        positions: Mapping[str, Mapping[str, float]],
    ) -> bool:
        return self.aggregate(positions).allowed

    def reset(self) -> None:
        pass