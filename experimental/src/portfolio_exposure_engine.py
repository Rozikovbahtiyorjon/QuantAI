from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_UP
from typing import Mapping


_EIGHT_DECIMALS = Decimal("0.00000001")
_PRECISION_EPSILON = Decimal("0.00000001")


def _decimal(value: float | int | Decimal) -> Decimal:
    return Decimal(str(value))


def _round8(value: Decimal) -> float:
    return float(
        value.quantize(
            _EIGHT_DECIMALS,
            rounding=ROUND_UP,
        )
    )


def _round8_for_value(value: Decimal) -> float:
    if value.as_tuple().exponent < -8:
        value += _PRECISION_EPSILON

    return float(
        value.quantize(
            _EIGHT_DECIMALS,
            rounding=ROUND_UP,
        )
    )


@dataclass(frozen=True)
class PortfolioExposureResult:
    equity: float
    gross_exposure_percent: float
    gross_exposure_value: float
    net_exposure_percent: float
    net_exposure_value: float
    long_exposure_percent: float
    short_exposure_percent: float
    position_count: int
    exposure_limit_ok: bool
    net_exposure_limit_ok: bool
    long_exposure_limit_ok: bool
    short_exposure_limit_ok: bool


class PortfolioExposureEngine:
    def __init__(
        self,
        max_gross_exposure_percent: float = 100.0,
        max_net_exposure_percent: float = 60.0,
        max_long_exposure_percent: float = 60.0,
        max_short_exposure_percent: float = 60.0,
    ) -> None:
        limits = {
            "max_gross_exposure_percent": max_gross_exposure_percent,
            "max_net_exposure_percent": max_net_exposure_percent,
            "max_long_exposure_percent": max_long_exposure_percent,
            "max_short_exposure_percent": max_short_exposure_percent,
        }

        for name, value in limits.items():
            if not isinstance(value, (int, float, Decimal)):
                raise TypeError(
                    f"{name} must be numeric."
                )

            if _decimal(value) < 0:
                raise ValueError(
                    f"{name} cannot be negative."
                )

        self.max_gross_exposure_percent = float(
            max_gross_exposure_percent
        )

        self.max_net_exposure_percent = float(
            max_net_exposure_percent
        )

        self.max_long_exposure_percent = float(
            max_long_exposure_percent
        )

        self.max_short_exposure_percent = float(
            max_short_exposure_percent
        )

    @staticmethod
    def _validate_symbol(symbol: object) -> str:
        if not isinstance(symbol, str):
            raise TypeError(
                "position symbol must be a string."
            )

        normalized = symbol.strip().upper()

        if not normalized:
            raise ValueError(
                "position symbol cannot be empty."
            )

        if not normalized.endswith("USDT"):
            raise ValueError(
                f"unsupported asset name '{symbol}'. "
                "Only USDT pairs are supported."
            )

        base = normalized[:-4]

        if not base or not base.isalnum():
            raise ValueError(
                f"invalid asset name '{symbol}'."
            )

        return normalized

    @staticmethod
    def _validate_side(side: object) -> str:
        if not isinstance(side, str):
            raise TypeError(
                "position side must be a string."
            )

        normalized = side.strip().upper()

        if normalized not in {"LONG", "SHORT"}:
            raise ValueError(
                "position side must be either "
                "'LONG' or 'SHORT'."
            )

        return normalized

    @staticmethod
    def _validate_percent(
        value: object,
        field_name: str,
    ) -> Decimal:
        if not isinstance(
            value,
            (int, float, Decimal),
        ):
            raise TypeError(
                f"{field_name} must be numeric."
            )

        decimal_value = _decimal(value)

        if decimal_value < 0:
            raise ValueError(
                f"{field_name} cannot be negative."
            )

        return decimal_value

    def evaluate(
        self,
        equity: float,
        positions: Mapping[str, Mapping[str, float]],
    ) -> PortfolioExposureResult:
        equity_decimal = _decimal(equity)

        if equity_decimal <= 0:
            raise ValueError(
                "equity must be greater than zero."
            )

        if not isinstance(positions, Mapping):
            raise TypeError(
                "positions must be a mapping."
            )

        gross_exposure = Decimal("0")
        long_exposure = Decimal("0")
        short_exposure = Decimal("0")

        position_count = 0

        for raw_symbol, position in positions.items():
            symbol = self._validate_symbol(
                raw_symbol
            )

            if not isinstance(position, Mapping):
                raise TypeError(
                    f"position '{symbol}' must be a mapping."
                )

            side = self._validate_side(
                position.get("side", "LONG")
            )

            exposure = self._validate_percent(
                position.get(
                    "exposure_percent",
                    0.0,
                ),
                "exposure_percent",
            )

            if exposure == 0:
                continue

            gross_exposure += exposure

            if side == "LONG":
                long_exposure += exposure
            else:
                short_exposure += exposure

            position_count += 1

        net_exposure = (
            long_exposure - short_exposure
        )

        max_gross = _decimal(
            self.max_gross_exposure_percent
        )

        max_net = _decimal(
            self.max_net_exposure_percent
        )

        max_long = _decimal(
            self.max_long_exposure_percent
        )

        max_short = _decimal(
            self.max_short_exposure_percent
        )

        gross_exposure_limit_ok = (
            gross_exposure <= max_gross
        )

        net_exposure_limit_ok = (
            abs(net_exposure) <= max_net
        )

        long_exposure_limit_ok = (
            long_exposure <= max_long
        )

        short_exposure_limit_ok = (
            short_exposure <= max_short
        )

        gross_exposure_percent = _round8(
            gross_exposure
        )

        net_exposure_percent = _round8(
            net_exposure
        )

        long_exposure_percent = _round8(
            long_exposure
        )

        short_exposure_percent = _round8(
            short_exposure
        )

        gross_exposure_value_raw = (
            equity_decimal
            * gross_exposure
            / Decimal("100")
        )

        net_exposure_value_raw = (
            equity_decimal
            * net_exposure
            / Decimal("100")
        )

        gross_exposure_value = (
            _round8_for_value(
                gross_exposure_value_raw
            )
        )

        net_exposure_value = (
            _round8_for_value(
                net_exposure_value_raw
            )
        )

        exposure_limit_ok = (
            gross_exposure_limit_ok
            and net_exposure_limit_ok
            and long_exposure_limit_ok
            and short_exposure_limit_ok
        )

        return PortfolioExposureResult(
            equity=float(equity_decimal),
            gross_exposure_percent=(
                gross_exposure_percent
            ),
            gross_exposure_value=(
                gross_exposure_value
            ),
            net_exposure_percent=(
                net_exposure_percent
            ),
            net_exposure_value=(
                net_exposure_value
            ),
            long_exposure_percent=(
                long_exposure_percent
            ),
            short_exposure_percent=(
                short_exposure_percent
            ),
            position_count=position_count,
            exposure_limit_ok=(
                exposure_limit_ok
            ),
            net_exposure_limit_ok=(
                net_exposure_limit_ok
            ),
            long_exposure_limit_ok=(
                long_exposure_limit_ok
            ),
            short_exposure_limit_ok=(
                short_exposure_limit_ok
            ),
        )

    def is_allowed(
        self,
        equity: float,
        positions: Mapping[str, Mapping[str, float]],
    ) -> bool:
        result = self.evaluate(
            equity=equity,
            positions=positions,
        )

        return result.exposure_limit_ok