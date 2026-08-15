import pytest

from src.risk_rule_engine import (
    RiskRuleEngine,
    RiskRuleResult,
)


def test_default_configuration() -> None:
    engine = RiskRuleEngine()

    assert engine.max_risk_per_trade_percent == 3.0
    assert engine.max_total_exposure_percent == 5.0
    assert engine.max_losing_trades == 7


def test_custom_configuration() -> None:
    engine = RiskRuleEngine(
        max_risk_per_trade_percent=2.0,
        max_total_exposure_percent=10.0,
        max_losing_trades=5,
    )

    assert engine.max_risk_per_trade_percent == 2.0
    assert engine.max_total_exposure_percent == 10.0
    assert engine.max_losing_trades == 5


def test_allowed_risk() -> None:
    engine = RiskRuleEngine()

    result = engine.evaluate(
        risk_per_trade_percent=2.0,
        total_exposure_percent=4.0,
        losing_trades=2,
        total_trades=10,
    )

    assert isinstance(result, RiskRuleResult)
    assert result.risk_limit_ok is True
    assert result.exposure_limit_ok is True
    assert result.loss_streak_ok is True
    assert result.allowed is True


def test_exact_risk_limit_allowed() -> None:
    engine = RiskRuleEngine()

    result = engine.evaluate(
        risk_per_trade_percent=3.0,
        total_exposure_percent=5.0,
        losing_trades=0,
        total_trades=1,
    )

    assert result.allowed is True


def test_risk_limit_rejected() -> None:
    engine = RiskRuleEngine()

    result = engine.evaluate(
        risk_per_trade_percent=3.01,
        total_exposure_percent=2.0,
    )

    assert result.risk_limit_ok is False
    assert result.exposure_limit_ok is True
    assert result.allowed is False


def test_exposure_limit_rejected() -> None:
    engine = RiskRuleEngine()

    result = engine.evaluate(
        risk_per_trade_percent=1.0,
        total_exposure_percent=5.01,
    )

    assert result.risk_limit_ok is True
    assert result.exposure_limit_ok is False
    assert result.allowed is False


def test_seven_losing_trades_rejected() -> None:
    engine = RiskRuleEngine()

    result = engine.evaluate(
        risk_per_trade_percent=1.0,
        total_exposure_percent=2.0,
        losing_trades=7,
        total_trades=10,
    )

    assert result.loss_streak_ok is False
    assert result.allowed is False


def test_six_losing_trades_allowed() -> None:
    engine = RiskRuleEngine()

    result = engine.evaluate(
        risk_per_trade_percent=1.0,
        total_exposure_percent=2.0,
        losing_trades=6,
        total_trades=10,
    )

    assert result.loss_streak_ok is True
    assert result.allowed is True


def test_multiple_rule_failures() -> None:
    engine = RiskRuleEngine()

    result = engine.evaluate(
        risk_per_trade_percent=4.0,
        total_exposure_percent=10.0,
        losing_trades=7,
        total_trades=10,
    )

    assert result.risk_limit_ok is False
    assert result.exposure_limit_ok is False
    assert result.loss_streak_ok is False
    assert result.allowed is False


def test_is_allowed_true() -> None:
    engine = RiskRuleEngine()

    assert (
        engine.is_allowed(
            risk_per_trade_percent=1.0,
            total_exposure_percent=3.0,
            losing_trades=2,
            total_trades=5,
        )
        is True
    )


def test_is_allowed_false() -> None:
    engine = RiskRuleEngine()

    assert (
        engine.is_allowed(
            risk_per_trade_percent=4.0,
            total_exposure_percent=3.0,
            losing_trades=2,
            total_trades=5,
        )
        is False
    )


@pytest.mark.parametrize(
    "risk",
    [-1.0, -0.01],
)
def test_negative_risk_rejected(
    risk: float,
) -> None:
    engine = RiskRuleEngine()

    with pytest.raises(ValueError):
        engine.evaluate(
            risk_per_trade_percent=risk,
            total_exposure_percent=1.0,
        )


@pytest.mark.parametrize(
    "exposure",
    [-1.0, -0.01],
)
def test_negative_exposure_rejected(
    exposure: float,
) -> None:
    engine = RiskRuleEngine()

    with pytest.raises(ValueError):
        engine.evaluate(
            risk_per_trade_percent=1.0,
            total_exposure_percent=exposure,
        )


def test_negative_losing_trades_rejected() -> None:
    engine = RiskRuleEngine()

    with pytest.raises(ValueError):
        engine.evaluate(
            risk_per_trade_percent=1.0,
            total_exposure_percent=1.0,
            losing_trades=-1,
        )


def test_losing_trades_cannot_exceed_total() -> None:
    engine = RiskRuleEngine()

    with pytest.raises(ValueError):
        engine.evaluate(
            risk_per_trade_percent=1.0,
            total_exposure_percent=1.0,
            losing_trades=6,
            total_trades=5,
        )


def test_fractional_losing_trades_rejected() -> None:
    engine = RiskRuleEngine()

    with pytest.raises(TypeError):
        engine.evaluate(
            risk_per_trade_percent=1.0,
            total_exposure_percent=1.0,
            losing_trades=1.5,
        )


def test_fractional_total_trades_rejected() -> None:
    engine = RiskRuleEngine()

    with pytest.raises(TypeError):
        engine.evaluate(
            risk_per_trade_percent=1.0,
            total_exposure_percent=1.0,
            total_trades=5.5,
        )


def test_boolean_losing_trades_rejected() -> None:
    engine = RiskRuleEngine()

    with pytest.raises(TypeError):
        engine.evaluate(
            risk_per_trade_percent=1.0,
            total_exposure_percent=1.0,
            losing_trades=True,
        )


def test_boolean_total_trades_rejected() -> None:
    engine = RiskRuleEngine()

    with pytest.raises(TypeError):
        engine.evaluate(
            risk_per_trade_percent=1.0,
            total_exposure_percent=1.0,
            total_trades=True,
        )


def test_zero_risk_limit_rejected() -> None:
    with pytest.raises(ValueError):
        RiskRuleEngine(
            max_risk_per_trade_percent=0.0,
            max_total_exposure_percent=5.0,
            max_losing_trades=7,
        )


@pytest.mark.parametrize(
    "risk_limit,exposure_limit,loss_limit",
    [
        (3.0, 100.0, 7),
        (3.0, 5.0, 1),
        (1.0, 50.0, 10),
    ],
)
def test_valid_limits(
    risk_limit: float,
    exposure_limit: float,
    loss_limit: int,
) -> None:
    engine = RiskRuleEngine(
        max_risk_per_trade_percent=risk_limit,
        max_total_exposure_percent=exposure_limit,
        max_losing_trades=loss_limit,
    )

    assert engine.max_risk_per_trade_percent == risk_limit
    assert engine.max_total_exposure_percent == exposure_limit
    assert engine.max_losing_trades == loss_limit


def test_invalid_risk_limit() -> None:
    with pytest.raises(ValueError):
        RiskRuleEngine(
            max_risk_per_trade_percent=-1.0
        )


def test_invalid_exposure_limit() -> None:
    with pytest.raises(ValueError):
        RiskRuleEngine(
            max_total_exposure_percent=101.0
        )


def test_invalid_loss_limit_type() -> None:
    with pytest.raises(TypeError):
        RiskRuleEngine(
            max_losing_trades=7.5
        )


def test_invalid_loss_limit_value() -> None:
    with pytest.raises(ValueError):
        RiskRuleEngine(
            max_losing_trades=0
        )