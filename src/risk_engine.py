from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RiskAssessment:
    approved: bool
    position_size: float
    risk_amount: float
    stop_distance: float
    risk_percent: float
    exposure: float
    exposure_percent: float
    leverage: float
    reason: str


class RiskEngine:
    """
    QuantAI Professional v5
    Core risk-management engine.

    Responsibilities:
    - position sizing
    - risk-per-trade control
    - maximum exposure control
    - stop-loss distance validation
    - leverage limits
    - risk presets
    - capital reserve protection

    Does not:
    - generate trading signals
    - execute trades
    - connect to exchanges
    - modify TradeEngine
    """

    PRESETS = {
        "AGGRESSIVE": {
            "risk_per_trade": 0.02,
            "max_exposure": 0.60,
            "max_leverage": 50.0,
        },
        "NORMAL": {
            "risk_per_trade": 0.01,
            "max_exposure": 0.60,
            "max_leverage": 20.0,
        },
        "PROTECTIVE": {
            "risk_per_trade": 0.005,
            "max_exposure": 0.40,
            "max_leverage": 5.0,
        },
    }

    def __init__(
        self,
        risk_per_trade: float = 0.01,
        max_exposure: float = 0.60,
        reserve_ratio: float = 0.40,
        max_leverage: float = 20.0,
    ) -> None:
        self._validate_ratio(
            risk_per_trade,
            "risk_per_trade",
        )

        self._validate_ratio(
            max_exposure,
            "max_exposure",
        )

        self._validate_ratio(
            reserve_ratio,
            "reserve_ratio",
        )

        if max_exposure + reserve_ratio > 1.0:
            raise ValueError(
                "max_exposure + reserve_ratio "
                "cannot exceed 1.0."
            )

        if max_leverage < 1.0:
            raise ValueError(
                "max_leverage must be at least 1."
            )

        self.risk_per_trade = float(risk_per_trade)
        self.max_exposure = float(max_exposure)
        self.reserve_ratio = float(reserve_ratio)
        self.max_leverage = float(max_leverage)

    @staticmethod
    def _validate_ratio(
        value: float,
        name: str,
    ) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"{name} must be between 0 and 1."
            )

    @classmethod
    def from_preset(
        cls,
        preset: str,
    ) -> "RiskEngine":
        name = str(preset).upper()

        if name not in cls.PRESETS:
            raise ValueError(
                f"Unknown risk preset: {preset}"
            )

        config = cls.PRESETS[name]

        reserve_ratio = (
            1.0 - config["max_exposure"]
        )

        return cls(
            risk_per_trade=config[
                "risk_per_trade"
            ],
            max_exposure=config[
                "max_exposure"
            ],
            reserve_ratio=reserve_ratio,
            max_leverage=config[
                "max_leverage"
            ],
        )

    @property
    def tradable_capital_ratio(self) -> float:
        return 1.0 - self.reserve_ratio

    def calculate_position_size(
        self,
        balance: float,
        entry_price: float,
        stop_price: float,
        confidence: float = 1.0,
        leverage: Optional[float] = None,
    ) -> float:
        self._validate_positive(
            balance,
            "balance",
        )

        self._validate_positive(
            entry_price,
            "entry_price",
        )

        self._validate_positive(
            stop_price,
            "stop_price",
        )

        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                "confidence must be between 0 and 1."
            )

        stop_distance = abs(
            entry_price - stop_price
        )

        if stop_distance <= 0:
            raise ValueError(
                "entry_price and stop_price "
                "must be different."
            )

        selected_leverage = (
            self.max_leverage
            if leverage is None
            else float(leverage)
        )

        if selected_leverage < 1.0:
            raise ValueError(
                "leverage must be at least 1."
            )

        selected_leverage = min(
            selected_leverage,
            self.max_leverage,
        )

        risk_amount = (
            balance
            * self.risk_per_trade
            * confidence
        )

        quantity = (
            risk_amount / stop_distance
        )

        maximum_notional = (
            balance
            * self.max_exposure
            * selected_leverage
        )

        maximum_quantity = (
            maximum_notional / entry_price
        )

        return max(
            0.0,
            min(
                quantity,
                maximum_quantity,
            ),
        )

    def assess(
        self,
        balance: float,
        entry_price: float,
        stop_price: float,
        confidence: float = 1.0,
        current_exposure: float = 0.0,
        leverage: Optional[float] = None,
    ) -> RiskAssessment:
        self._validate_positive(
            balance,
            "balance",
        )

        self._validate_positive(
            entry_price,
            "entry_price",
        )

        self._validate_positive(
            stop_price,
            "stop_price",
        )

        if current_exposure < 0:
            raise ValueError(
                "current_exposure cannot be negative."
            )

        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                "confidence must be between 0 and 1."
            )

        selected_leverage = (
            self.max_leverage
            if leverage is None
            else float(leverage)
        )

        if selected_leverage < 1.0:
            raise ValueError(
                "leverage must be at least 1."
            )

        if selected_leverage > self.max_leverage:
            return RiskAssessment(
                approved=False,
                position_size=0.0,
                risk_amount=0.0,
                stop_distance=abs(
                    entry_price - stop_price
                ),
                risk_percent=0.0,
                exposure=0.0,
                exposure_percent=0.0,
                leverage=selected_leverage,
                reason="LEVERAGE_EXCEEDED",
            )

        stop_distance = abs(
            entry_price - stop_price
        )

        if stop_distance <= 0:
            return RiskAssessment(
                approved=False,
                position_size=0.0,
                risk_amount=0.0,
                stop_distance=0.0,
                risk_percent=0.0,
                exposure=0.0,
                exposure_percent=0.0,
                leverage=selected_leverage,
                reason="INVALID_STOP_DISTANCE",
            )

        current_exposure_percent = (
            current_exposure / balance
        )

        if (
            current_exposure_percent
            >= self.max_exposure
        ):
            return RiskAssessment(
                approved=False,
                position_size=0.0,
                risk_amount=0.0,
                stop_distance=stop_distance,
                risk_percent=0.0,
                exposure=0.0,
                exposure_percent=(
                    current_exposure_percent * 100.0
                ),
                leverage=selected_leverage,
                reason="MAX_EXPOSURE_REACHED",
            )

        position_size = self.calculate_position_size(
            balance=balance,
            entry_price=entry_price,
            stop_price=stop_price,
            confidence=confidence,
            leverage=selected_leverage,
        )

        exposure = (
            position_size
            * entry_price
        )

        exposure_percent = (
            (
                current_exposure
                + exposure
            )
            / balance
        )

        if exposure_percent > self.max_exposure:
            allowed_exposure = (
                balance * self.max_exposure
                - current_exposure
            )

            exposure = max(
                0.0,
                allowed_exposure,
            )

            position_size = (
                exposure / entry_price
                if entry_price > 0
                else 0.0
            )

            exposure_percent = (
                (
                    current_exposure
                    + exposure
                )
                / balance
            )

        risk_amount = (
            position_size
            * stop_distance
        )

        risk_percent = (
            risk_amount
            / balance
            if balance > 0
            else 0.0
        )

        approved = (
            position_size > 0
            and exposure_percent
            <= self.max_exposure
        )

        reason = (
            "APPROVED"
            if approved
            else "POSITION_SIZE_ZERO"
        )

        return RiskAssessment(
            approved=approved,
            position_size=position_size,
            risk_amount=risk_amount,
            stop_distance=stop_distance,
            risk_percent=risk_percent * 100.0,
            exposure=exposure,
            exposure_percent=exposure_percent * 100.0,
            leverage=selected_leverage,
            reason=reason,
        )

    @staticmethod
    def _validate_positive(
        value: float,
        name: str,
    ) -> None:
        if value <= 0:
            raise ValueError(
                f"{name} must be greater than zero."
            )


__all__ = [
    "RiskAssessment",
    "RiskEngine",
]