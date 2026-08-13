from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskProfile:
    name: str
    risk_per_trade_percent: float
    max_total_exposure_percent: float
    max_positions: int
    max_leverage: float
    stop_loss_percent: float
    trailing_stop_percent: float


class RiskProfileManager:
    _PROFILES = {
        "aggressive": RiskProfile(
            name="aggressive",
            risk_per_trade_percent=2.0,
            max_total_exposure_percent=60.0,
            max_positions=12,
            max_leverage=50.0,
            stop_loss_percent=3.0,
            trailing_stop_percent=1.5,
        ),
        "normal": RiskProfile(
            name="normal",
            risk_per_trade_percent=1.0,
            max_total_exposure_percent=40.0,
            max_positions=8,
            max_leverage=20.0,
            stop_loss_percent=2.0,
            trailing_stop_percent=1.0,
        ),
        "maximum_protection": RiskProfile(
            name="maximum_protection",
            risk_per_trade_percent=0.5,
            max_total_exposure_percent=20.0,
            max_positions=4,
            max_leverage=5.0,
            stop_loss_percent=1.0,
            trailing_stop_percent=0.5,
        ),
    }

    def __init__(
        self,
        profile: str = "normal",
    ) -> None:
        self._validate_profile_name(profile)
        self._profile = self._PROFILES[profile.lower()]

    @classmethod
    def _validate_profile_name(
        cls,
        profile: str,
    ) -> None:
        if not isinstance(profile, str):
            raise TypeError(
                "profile must be a string."
            )

        if profile.lower() not in cls._PROFILES:
            raise ValueError(
                f"Unknown risk profile: {profile}."
            )

    @property
    def profile(self) -> RiskProfile:
        return self._profile

    @property
    def name(self) -> str:
        return self._profile.name

    def set_profile(
        self,
        profile: str,
    ) -> RiskProfile:
        self._validate_profile_name(profile)
        self._profile = self._PROFILES[
            profile.lower()
        ]
        return self._profile

    def get_profile(
        self,
        profile: str,
    ) -> RiskProfile:
        self._validate_profile_name(profile)
        return self._PROFILES[
            profile.lower()
        ]

    @classmethod
    def available_profiles(cls) -> tuple[str, ...]:
        return tuple(cls._PROFILES.keys())

    def calculate_max_risk_amount(
        self,
        equity: float,
    ) -> float:
        if equity <= 0:
            raise ValueError(
                "equity must be greater than zero."
            )

        return round(
            equity
            * self._profile.risk_per_trade_percent
            / 100.0,
            8,
        )

    def calculate_max_exposure(
        self,
        equity: float,
    ) -> float:
        if equity <= 0:
            raise ValueError(
                "equity must be greater than zero."
            )

        return round(
            equity
            * self._profile.max_total_exposure_percent
            / 100.0,
            8,
        )

    def clamp_leverage(
        self,
        requested_leverage: float,
    ) -> float:
        if requested_leverage <= 0:
            raise ValueError(
                "requested_leverage must be greater than zero."
            )

        return round(
            min(
                float(requested_leverage),
                self._profile.max_leverage,
            ),
            8,
        )


__all__ = [
    "RiskProfile",
    "RiskProfileManager",
]