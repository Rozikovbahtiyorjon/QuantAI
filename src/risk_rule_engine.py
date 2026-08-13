from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskRuleResult:
    risk_per_trade_percent: float
    total_exposure_percent: float
    losing_trades: int
    total_trades: int
    risk_limit_ok: bool
    exposure_limit_ok: bool
    loss_streak_ok: bool
    allowed: bool


class RiskRuleEngine:
    def __init__(
        self,
        max_risk_per_trade_percent: float = 3.0,
        max_total_exposure_percent: float = 5.0,
        max_losing_trades: int = 7,
    ) -> None:
        if max_risk_per_trade_percent <= 0:
            raise ValueError(
                "max_risk_per_trade_percent must be greater than zero."
            )

        if not 0.0 < max_total_exposure_percent <= 100.0:
            raise ValueError(
                "max_total_exposure_percent must be between 0 and 100."
            )

        if (
            isinstance(max_losing_trades, bool)
            or not isinstance(max_losing_trades, int)
        ):
            raise TypeError(
                "max_losing_trades must be an integer."
            )

        if max_losing_trades <= 0:
            raise ValueError(
                "max_losing_trades must be greater than zero."
            )

        self.max_risk_per_trade_percent = float(
            max_risk_per_trade_percent
        )
        self.max_total_exposure_percent = float(
            max_total_exposure_percent
        )
        self.max_losing_trades = max_losing_trades

    def evaluate(
        self,
        risk_per_trade_percent: float,
        total_exposure_percent: float,
        losing_trades: int = 0,
        total_trades: int = 0,
    ) -> RiskRuleResult:
        if risk_per_trade_percent < 0:
            raise ValueError(
                "risk_per_trade_percent cannot be negative."
            )

        if total_exposure_percent < 0:
            raise ValueError(
                "total_exposure_percent cannot be negative."
            )

        if (
            isinstance(losing_trades, bool)
            or not isinstance(losing_trades, int)
        ):
            raise TypeError(
                "losing_trades must be an integer."
            )

        if (
            isinstance(total_trades, bool)
            or not isinstance(total_trades, int)
        ):
            raise TypeError(
                "total_trades must be an integer."
            )

        if losing_trades < 0:
            raise ValueError(
                "losing_trades cannot be negative."
            )

        if total_trades < 0:
            raise ValueError(
                "total_trades cannot be negative."
            )

        if losing_trades > total_trades:
            raise ValueError(
                "losing_trades cannot exceed total_trades."
            )

        risk_limit_ok = (
            risk_per_trade_percent
            <= self.max_risk_per_trade_percent + 1e-9
        )

        exposure_limit_ok = (
            total_exposure_percent
            <= self.max_total_exposure_percent + 1e-9
        )

        loss_streak_ok = (
            losing_trades < self.max_losing_trades
        )

        allowed = (
            risk_limit_ok
            and exposure_limit_ok
            and loss_streak_ok
        )

        return RiskRuleResult(
            risk_per_trade_percent=round(
                float(risk_per_trade_percent),
                8,
            ),
            total_exposure_percent=round(
                float(total_exposure_percent),
                8,
            ),
            losing_trades=losing_trades,
            total_trades=total_trades,
            risk_limit_ok=risk_limit_ok,
            exposure_limit_ok=exposure_limit_ok,
            loss_streak_ok=loss_streak_ok,
            allowed=allowed,
        )

    def is_allowed(
        self,
        risk_per_trade_percent: float,
        total_exposure_percent: float,
        losing_trades: int = 0,
        total_trades: int = 0,
    ) -> bool:
        return self.evaluate(
            risk_per_trade_percent=risk_per_trade_percent,
            total_exposure_percent=total_exposure_percent,
            losing_trades=losing_trades,
            total_trades=total_trades,
        ).allowed


__all__ = [
    "RiskRuleResult",
    "RiskRuleEngine",
]