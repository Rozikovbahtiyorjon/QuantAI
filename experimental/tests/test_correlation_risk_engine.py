import pytest

from experimental.src.correlation_risk_engine import (
    CorrelationRiskEngine,
    CorrelationRiskResult,
)


def test_default_configuration() -> None:
    engine = CorrelationRiskEngine()

    assert engine.max_correlation == 0.85
    assert engine.max_correlated_assets == 2


def test_custom_configuration() -> None:
    engine = CorrelationRiskEngine(
        max_correlation=0.90,
        max_correlated_assets=3,
    )

    assert engine.max_correlation == 0.90
    assert engine.max_correlated_assets == 3


def test_no_correlations() -> None:
    engine = CorrelationRiskEngine()

    result = engine.evaluate(
        asset="BTCUSDT",
        correlations={},
    )

    assert isinstance(result, CorrelationRiskResult)
    assert result.correlated_assets == ()
    assert result.correlated_count == 0
    assert result.max_correlation == 0.0
    assert result.risk_allowed is True


def test_low_correlations_allowed() -> None:
    engine = CorrelationRiskEngine()

    result = engine.evaluate(
        asset="BTCUSDT",
        correlations={
            "ETHUSDT": 0.60,
            "SOLUSDT": 0.70,
            "XRPUSDT": -0.50,
        },
    )

    assert result.correlated_assets == ()
    assert result.correlated_count == 0
    assert result.risk_allowed is True


def test_high_positive_correlation_detected() -> None:
    engine = CorrelationRiskEngine()

    result = engine.evaluate(
        asset="BTCUSDT",
        correlations={
            "ETHUSDT": 0.90,
            "SOLUSDT": 0.50,
        },
    )

    assert result.correlated_assets == ("ETHUSDT",)
    assert result.correlated_count == 1
    assert result.max_correlation == 0.90
    assert result.risk_allowed is True


def test_high_negative_correlation_detected() -> None:
    engine = CorrelationRiskEngine()

    result = engine.evaluate(
        asset="BTCUSDT",
        correlations={
            "ETHUSDT": -0.90,
            "SOLUSDT": 0.50,
        },
    )

    assert result.correlated_assets == ("ETHUSDT",)
    assert result.correlated_count == 1
    assert result.max_correlation == 0.90
    assert result.risk_allowed is True


def test_correlation_threshold_is_inclusive() -> None:
    engine = CorrelationRiskEngine(
        max_correlation=0.85,
    )

    result = engine.evaluate(
        asset="BTCUSDT",
        correlations={
            "ETHUSDT": 0.85,
        },
    )

    assert result.correlated_count == 1
    assert result.risk_allowed is True


def test_too_many_correlated_assets_rejected() -> None:
    engine = CorrelationRiskEngine(
        max_correlation=0.85,
        max_correlated_assets=2,
    )

    result = engine.evaluate(
        asset="BTCUSDT",
        correlations={
            "ETHUSDT": 0.90,
            "SOLUSDT": 0.88,
            "XRPUSDT": 0.86,
        },
    )

    assert result.correlated_count == 3
    assert result.risk_allowed is False


def test_exact_correlated_asset_limit_allowed() -> None:
    engine = CorrelationRiskEngine(
        max_correlation=0.85,
        max_correlated_assets=2,
    )

    result = engine.evaluate(
        asset="BTCUSDT",
        correlations={
            "ETHUSDT": 0.90,
            "SOLUSDT": 0.88,
        },
    )

    assert result.correlated_count == 2
    assert result.risk_allowed is True


def test_self_correlation_is_ignored() -> None:
    engine = CorrelationRiskEngine()

    result = engine.evaluate(
        asset="BTCUSDT",
        correlations={
            "BTCUSDT": 1.0,
            "ETHUSDT": 0.90,
        },
    )

    assert result.correlated_assets == ("ETHUSDT",)
    assert result.correlated_count == 1


def test_is_allowed_true() -> None:
    engine = CorrelationRiskEngine()

    assert engine.is_allowed(
        asset="BTCUSDT",
        correlations={
            "ETHUSDT": 0.70,
            "SOLUSDT": 0.80,
        },
    ) is True


def test_is_allowed_false() -> None:
    engine = CorrelationRiskEngine(
        max_correlated_assets=1,
    )

    assert engine.is_allowed(
        asset="BTCUSDT",
        correlations={
            "ETHUSDT": 0.90,
            "SOLUSDT": 0.91,
        },
    ) is False


def test_invalid_max_correlation_low() -> None:
    with pytest.raises(ValueError):
        CorrelationRiskEngine(
            max_correlation=-0.01,
        )


def test_invalid_max_correlation_high() -> None:
    with pytest.raises(ValueError):
        CorrelationRiskEngine(
            max_correlation=1.01,
        )


def test_invalid_correlated_assets_type() -> None:
    with pytest.raises(TypeError):
        CorrelationRiskEngine(
            max_correlated_assets=2.5,
        )


def test_invalid_correlated_assets_value() -> None:
    with pytest.raises(ValueError):
        CorrelationRiskEngine(
            max_correlated_assets=0,
        )


def test_invalid_asset() -> None:
    engine = CorrelationRiskEngine()

    with pytest.raises(ValueError):
        engine.evaluate(
            asset="",
            correlations={},
        )


def test_invalid_correlations_type() -> None:
    engine = CorrelationRiskEngine()

    with pytest.raises(TypeError):
        engine.evaluate(
            asset="BTCUSDT",
            correlations=[],
        )


def test_invalid_correlation_value_high() -> None:
    engine = CorrelationRiskEngine()

    with pytest.raises(ValueError):
        engine.evaluate(
            asset="BTCUSDT",
            correlations={
                "ETHUSDT": 1.01,
            },
        )


def test_invalid_correlation_value_low() -> None:
    engine = CorrelationRiskEngine()

    with pytest.raises(ValueError):
        engine.evaluate(
            asset="BTCUSDT",
            correlations={
                "ETHUSDT": -1.01,
            },
        )


def test_invalid_correlation_asset_name() -> None:
    engine = CorrelationRiskEngine()

    with pytest.raises(ValueError):
        engine.evaluate(
            asset="BTCUSDT",
            correlations={
                "": 0.90,
            },
        )


def test_string_correlation_value_is_accepted_if_numeric() -> None:
    engine = CorrelationRiskEngine()

    result = engine.evaluate(
        asset="BTCUSDT",
        correlations={
            "ETHUSDT": "0.90",
        },
    )

    assert result.correlated_count == 1
    assert result.max_correlation == 0.90


def test_result_precision() -> None:
    engine = CorrelationRiskEngine()

    result = engine.evaluate(
        asset="BTCUSDT",
        correlations={
            "ETHUSDT": 0.876543219,
        },
    )

    assert result.max_correlation == 0.87654322


def test_many_low_correlations_allowed() -> None:
    engine = CorrelationRiskEngine(
        max_correlated_assets=1,
    )

    result = engine.evaluate(
        asset="BTCUSDT",
        correlations={
            "ETHUSDT": 0.20,
            "SOLUSDT": 0.30,
            "XRPUSDT": -0.40,
            "BNBUSDT": 0.50,
        },
    )

    assert result.correlated_count == 0
    assert result.risk_allowed is True