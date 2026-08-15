import pytest

from src.portfolio_risk_engine import (
    PortfolioRiskEngine,
    PortfolioRiskResult,
)


def test_default_configuration() -> None:
    engine = PortfolioRiskEngine()

    assert engine.max_total_exposure_percent == 60.0
    assert engine.max_total_risk_percent == 10.0
    assert engine.max_positions == 10


def test_custom_configuration() -> None:
    engine = PortfolioRiskEngine(
        max_total_exposure_percent=70.0,
        max_total_risk_percent=12.0,
        max_positions=5,
    )

    assert engine.max_total_exposure_percent == 70.0
    assert engine.max_total_risk_percent == 12.0
    assert engine.max_positions == 5


def test_empty_portfolio_allowed() -> None:
    engine = PortfolioRiskEngine()

    result = engine.evaluate(
        equity=1000.0,
        positions={},
    )

    assert isinstance(result, PortfolioRiskResult)
    assert result.total_exposure_percent == 0.0
    assert result.total_risk_percent == 0.0
    assert result.position_count == 0
    assert result.risk_allowed is True


def test_valid_portfolio_allowed() -> None:
    engine = PortfolioRiskEngine()

    result = engine.evaluate(
        equity=1000.0,
        positions={
            "BTCUSDT": {
                "exposure_percent": 20.0,
                "risk_percent": 2.0,
            },
            "ETHUSDT": {
                "exposure_percent": 15.0,
                "risk_percent": 1.5,
            },
            "SOLUSDT": {
                "exposure_percent": 10.0,
                "risk_percent": 1.0,
            },
        },
    )

    assert result.total_exposure_percent == 45.0
    assert result.total_risk_percent == 4.5
    assert result.position_count == 3
    assert result.exposure_limit_ok is True
    assert result.risk_limit_ok is True
    assert result.position_limit_ok is True
    assert result.risk_allowed is True


def test_exposure_limit_rejected() -> None:
    engine = PortfolioRiskEngine(
        max_total_exposure_percent=60.0,
    )

    result = engine.evaluate(
        equity=1000.0,
        positions={
            "BTCUSDT": {
                "exposure_percent": 40.0,
                "risk_percent": 2.0,
            },
            "ETHUSDT": {
                "exposure_percent": 25.0,
                "risk_percent": 2.0,
            },
        },
    )

    assert result.total_exposure_percent == 65.0
    assert result.exposure_limit_ok is False
    assert result.risk_allowed is False


def test_risk_limit_rejected() -> None:
    engine = PortfolioRiskEngine(
        max_total_risk_percent=10.0,
    )

    result = engine.evaluate(
        equity=1000.0,
        positions={
            "BTCUSDT": {
                "exposure_percent": 20.0,
                "risk_percent": 6.0,
            },
            "ETHUSDT": {
                "exposure_percent": 20.0,
                "risk_percent": 5.0,
            },
        },
    )

    assert result.total_risk_percent == 11.0
    assert result.risk_limit_ok is False
    assert result.risk_allowed is False


def test_position_limit_rejected() -> None:
    engine = PortfolioRiskEngine(
        max_positions=2,
    )

    result = engine.evaluate(
        equity=1000.0,
        positions={
            "BTCUSDT": {
                "exposure_percent": 10.0,
                "risk_percent": 1.0,
            },
            "ETHUSDT": {
                "exposure_percent": 10.0,
                "risk_percent": 1.0,
            },
            "SOLUSDT": {
                "exposure_percent": 10.0,
                "risk_percent": 1.0,
            },
        },
    )

    assert result.position_count == 3
    assert result.position_limit_ok is False
    assert result.risk_allowed is False


def test_exact_limits_allowed() -> None:
    engine = PortfolioRiskEngine(
        max_total_exposure_percent=60.0,
        max_total_risk_percent=10.0,
        max_positions=2,
    )

    result = engine.evaluate(
        equity=1000.0,
        positions={
            "BTCUSDT": {
                "exposure_percent": 30.0,
                "risk_percent": 5.0,
            },
            "ETHUSDT": {
                "exposure_percent": 30.0,
                "risk_percent": 5.0,
            },
        },
    )

    assert result.exposure_limit_ok is True
    assert result.risk_limit_ok is True
    assert result.position_limit_ok is True
    assert result.risk_allowed is True


def test_zero_exposure_position_not_counted() -> None:
    engine = PortfolioRiskEngine(
        max_positions=1,
    )

    result = engine.evaluate(
        equity=1000.0,
        positions={
            "BTCUSDT": {
                "exposure_percent": 0.0,
                "risk_percent": 0.0,
            },
            "ETHUSDT": {
                "exposure_percent": 20.0,
                "risk_percent": 2.0,
            },
        },
    )

    assert result.position_count == 1
    assert result.risk_allowed is True


def test_zero_risk_allowed() -> None:
    engine = PortfolioRiskEngine()

    result = engine.evaluate(
        equity=1000.0,
        positions={
            "BTCUSDT": {
                "exposure_percent": 20.0,
                "risk_percent": 0.0,
            },
        },
    )

    assert result.total_risk_percent == 0.0
    assert result.risk_allowed is True


def test_is_allowed_true() -> None:
    engine = PortfolioRiskEngine()

    assert engine.is_allowed(
        equity=1000.0,
        positions={
            "BTCUSDT": {
                "exposure_percent": 20.0,
                "risk_percent": 2.0,
            },
        },
    ) is True


def test_is_allowed_false() -> None:
    engine = PortfolioRiskEngine(
        max_total_exposure_percent=20.0,
    )

    assert engine.is_allowed(
        equity=1000.0,
        positions={
            "BTCUSDT": {
                "exposure_percent": 25.0,
                "risk_percent": 2.0,
            },
        },
    ) is False


def test_invalid_equity() -> None:
    engine = PortfolioRiskEngine()

    with pytest.raises(ValueError):
        engine.evaluate(
            equity=0.0,
            positions={},
        )


def test_negative_equity() -> None:
    engine = PortfolioRiskEngine()

    with pytest.raises(ValueError):
        engine.evaluate(
            equity=-100.0,
            positions={},
        )


def test_invalid_positions_type() -> None:
    engine = PortfolioRiskEngine()

    with pytest.raises(TypeError):
        engine.evaluate(
            equity=1000.0,
            positions=[],
        )


def test_invalid_asset_name() -> None:
    engine = PortfolioRiskEngine()

    with pytest.raises(ValueError):
        engine.evaluate(
            equity=1000.0,
            positions={
                "": {
                    "exposure_percent": 10.0,
                    "risk_percent": 1.0,
                }
            },
        )


def test_invalid_position_type() -> None:
    engine = PortfolioRiskEngine()

    with pytest.raises(TypeError):
        engine.evaluate(
            equity=1000.0,
            positions={
                "BTCUSDT": 10.0,
            },
        )


def test_negative_exposure_rejected() -> None:
    engine = PortfolioRiskEngine()

    with pytest.raises(ValueError):
        engine.evaluate(
            equity=1000.0,
            positions={
                "BTCUSDT": {
                    "exposure_percent": -1.0,
                    "risk_percent": 1.0,
                }
            },
        )


def test_negative_risk_rejected() -> None:
    engine = PortfolioRiskEngine()

    with pytest.raises(ValueError):
        engine.evaluate(
            equity=1000.0,
            positions={
                "BTCUSDT": {
                    "exposure_percent": 10.0,
                    "risk_percent": -1.0,
                }
            },
        )


def test_missing_values_default_to_zero() -> None:
    engine = PortfolioRiskEngine()

    result = engine.evaluate(
        equity=1000.0,
        positions={
            "BTCUSDT": {},
        },
    )

    assert result.total_exposure_percent == 0.0
    assert result.total_risk_percent == 0.0
    assert result.position_count == 0
    assert result.risk_allowed is True


def test_string_numeric_values_are_accepted() -> None:
    engine = PortfolioRiskEngine()

    result = engine.evaluate(
        equity=1000.0,
        positions={
            "BTCUSDT": {
                "exposure_percent": "20.0",
                "risk_percent": "2.0",
            },
        },
    )

    assert result.total_exposure_percent == 20.0
    assert result.total_risk_percent == 2.0
    assert result.position_count == 1


def test_multiple_failures() -> None:
    engine = PortfolioRiskEngine(
        max_total_exposure_percent=50.0,
        max_total_risk_percent=5.0,
        max_positions=2,
    )

    result = engine.evaluate(
        equity=1000.0,
        positions={
            "BTCUSDT": {
                "exposure_percent": 30.0,
                "risk_percent": 3.0,
            },
            "ETHUSDT": {
                "exposure_percent": 30.0,
                "risk_percent": 3.0,
            },
            "SOLUSDT": {
                "exposure_percent": 10.0,
                "risk_percent": 1.0,
            },
        },
    )

    assert result.exposure_limit_ok is False
    assert result.risk_limit_ok is False
    assert result.position_limit_ok is False
    assert result.risk_allowed is False


def test_precision() -> None:
    engine = PortfolioRiskEngine()

    result = engine.evaluate(
        equity=1000.0,
        positions={
            "BTCUSDT": {
                "exposure_percent": 20.123456789,
                "risk_percent": 1.123456789,
            },
            "ETHUSDT": {
                "exposure_percent": 10.987654321,
                "risk_percent": 0.876543219,
            },
        },
    )

    assert result.total_exposure_percent == 31.11111111
    assert result.total_risk_percent == 2.0