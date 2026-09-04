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
        max_total_exposure_percent: float | None = None,
        max_position_exposure_percent: float | None = None,
        max_correlation: float | None = None,
        policy: Any = None,
        # Separated exposures (point 26): cash/notional/margin/risk/factor — all derived from policy
        max_cash_exposure_pct: float | None = None,
        max_notional_exposure_pct: float | None = None,
        max_margin_exposure_pct: float | None = None,
        max_risk_exposure_pct: float | None = None,
        max_factor_exposure_pct: float | None = None,
    ) -> None:
        import warnings
        # P0.6 Strict: No Policy -> No Manager — no hidden 60/5 fallback
        if policy is not None:
            if max_total_exposure_percent is None:
                max_total_exposure_percent = float(getattr(policy, "max_total_exposure_pct", 20.0))
            if max_position_exposure_percent is None:
                max_position_exposure_percent = float(getattr(policy, "max_position_exposure_pct", 3.0))
            if max_correlation is None:
                max_correlation = float(getattr(policy, "max_correlation", 0.70))
            if max_factor_exposure_pct is None:
                max_factor_exposure_pct = float(getattr(policy, "max_factor_exposure_pct", 15.0))
        # Strict: No Policy -> No Manager
        if max_total_exposure_percent is None:
            raise ValueError("ExposureManager requires explicit RiskPolicy (Production/Research) or max_total_exposure_percent — No Policy -> No Manager (P0.6)")
        if max_position_exposure_percent is None:
            raise ValueError("ExposureManager requires explicit RiskPolicy or max_position_exposure_percent — No Policy -> No Manager (P0.6)")
        if max_correlation is None:
            max_correlation = 0.70
        if max_correlation is None:
            max_correlation = 0.70
        # Separated exposures: derive from policy if not given
        if max_cash_exposure_pct is None:
            max_cash_exposure_pct = float(max_total_exposure_percent)
        if max_notional_exposure_pct is None:
            max_notional_exposure_pct = float(max_total_exposure_percent)
        if max_margin_exposure_pct is None:
            # Margin exposure = notional / leverage ; use same cap as total but will be checked via PositionSizer margin
            max_margin_exposure_pct = float(max_total_exposure_percent)
        if max_risk_exposure_pct is None:
            # Risk exposure 1% per trade * max positions
            max_risk_exposure_pct = 5.0
        if max_factor_exposure_pct is None:
            max_factor_exposure_pct = 15.0
        if not 0.0 <= max_total_exposure_percent <= 100.0:
            raise ValueError(
                "max_total_exposure_percent must be between 0 and 100."
            )

        if not 0.0 < max_position_exposure_percent <= 100.0:
            raise ValueError(
                "max_position_exposure_percent must be greater than zero "
                "and no greater than 100."
            )

        if not 0.0 <= max_correlation <= 1.0:
            raise ValueError(
                "max_correlation must be between 0.0 and 1.0"
            )

        self.max_total_exposure_percent = float(
            max_total_exposure_percent
        )
        self.max_position_exposure_percent = float(
            max_position_exposure_percent
        )
        self.max_correlation = float(max_correlation)
        # Separated exposures (persist for RiskOrchestrator diagnostics)
        self.max_cash_exposure_pct = float(max_cash_exposure_pct)
        self.max_notional_exposure_pct = float(max_notional_exposure_pct)
        self.max_margin_exposure_pct = float(max_margin_exposure_pct)
        self.max_risk_exposure_pct = float(max_risk_exposure_pct)
        self.max_factor_exposure_pct = float(max_factor_exposure_pct)

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