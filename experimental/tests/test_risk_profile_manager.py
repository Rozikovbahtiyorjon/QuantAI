import pytest

from experimental.src.risk_profile_manager import (
    RiskProfile,
    RiskProfileManager,
)


def test_default_profile_is_normal() -> None:
    manager = RiskProfileManager()

    assert manager.name == "normal"
    assert isinstance(
        manager.profile,
        RiskProfile,
    )


def test_available_profiles() -> None:
    profiles = RiskProfileManager.available_profiles()

    assert profiles == (
        "aggressive",
        "normal",
        "maximum_protection",
    )


def test_aggressive_profile() -> None:
    manager = RiskProfileManager("aggressive")

    assert manager.profile.risk_per_trade_percent == 2.0
    assert manager.profile.max_total_exposure_percent == 60.0
    assert manager.profile.max_positions == 12
    assert manager.profile.max_leverage == 50.0


def test_normal_profile() -> None:
    manager = RiskProfileManager("normal")

    assert manager.profile.risk_per_trade_percent == 1.0
    assert manager.profile.max_total_exposure_percent == 40.0
    assert manager.profile.max_positions == 8
    assert manager.profile.max_leverage == 20.0


def test_protection_profile() -> None:
    manager = RiskProfileManager("maximum_protection")

    assert manager.profile.risk_per_trade_percent == 0.5
    assert manager.profile.max_total_exposure_percent == 20.0
    assert manager.profile.max_positions == 4
    assert manager.profile.max_leverage == 5.0


def test_profile_names_are_case_insensitive() -> None:
    manager = RiskProfileManager("AGGRESSIVE")

    assert manager.name == "aggressive"


def test_invalid_profile_type() -> None:
    with pytest.raises(TypeError):
        RiskProfileManager(123)


def test_invalid_profile_name() -> None:
    with pytest.raises(ValueError):
        RiskProfileManager("unknown")


def test_set_profile() -> None:
    manager = RiskProfileManager("normal")

    profile = manager.set_profile("aggressive")

    assert profile.name == "aggressive"
    assert manager.name == "aggressive"


def test_get_profile() -> None:
    manager = RiskProfileManager()

    profile = manager.get_profile(
        "maximum_protection"
    )

    assert profile.name == "maximum_protection"


def test_calculate_max_risk_amount() -> None:
    manager = RiskProfileManager("normal")

    assert manager.calculate_max_risk_amount(
        1000.0
    ) == 10.0


def test_calculate_aggressive_risk_amount() -> None:
    manager = RiskProfileManager("aggressive")

    assert manager.calculate_max_risk_amount(
        1000.0
    ) == 20.0


def test_calculate_protection_risk_amount() -> None:
    manager = RiskProfileManager(
        "maximum_protection"
    )

    assert manager.calculate_max_risk_amount(
        1000.0
    ) == 5.0


def test_calculate_max_exposure() -> None:
    manager = RiskProfileManager("normal")

    assert manager.calculate_max_exposure(
        1000.0
    ) == 400.0


def test_calculate_aggressive_exposure() -> None:
    manager = RiskProfileManager("aggressive")

    assert manager.calculate_max_exposure(
        1000.0
    ) == 600.0


def test_calculate_protection_exposure() -> None:
    manager = RiskProfileManager(
        "maximum_protection"
    )

    assert manager.calculate_max_exposure(
        1000.0
    ) == 200.0


def test_invalid_equity_for_risk_amount() -> None:
    manager = RiskProfileManager()

    with pytest.raises(ValueError):
        manager.calculate_max_risk_amount(0.0)


def test_invalid_equity_for_exposure() -> None:
    manager = RiskProfileManager()

    with pytest.raises(ValueError):
        manager.calculate_max_exposure(-1.0)


def test_leverage_within_limit() -> None:
    manager = RiskProfileManager("normal")

    assert manager.clamp_leverage(10.0) == 10.0


def test_leverage_above_limit_is_clamped() -> None:
    manager = RiskProfileManager("normal")

    assert manager.clamp_leverage(50.0) == 20.0


def test_aggressive_leverage_limit() -> None:
    manager = RiskProfileManager("aggressive")

    assert manager.clamp_leverage(100.0) == 50.0


def test_protection_leverage_limit() -> None:
    manager = RiskProfileManager(
        "maximum_protection"
    )

    assert manager.clamp_leverage(20.0) == 5.0


def test_invalid_leverage() -> None:
    manager = RiskProfileManager()

    with pytest.raises(ValueError):
        manager.clamp_leverage(0.0)


def test_profile_switching() -> None:
    manager = RiskProfileManager()

    manager.set_profile("aggressive")
    assert manager.calculate_max_risk_amount(
        5000.0
    ) == 100.0

    manager.set_profile("maximum_protection")
    assert manager.calculate_max_risk_amount(
        5000.0
    ) == 25.0


def test_profile_is_immutable() -> None:
    manager = RiskProfileManager()

    with pytest.raises(AttributeError):
        manager.profile.name = "custom"