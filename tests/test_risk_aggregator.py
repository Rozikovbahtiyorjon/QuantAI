from __future__ import annotations

import pytest

from src.risk_aggregator import (
    RiskAggregationResult,
    RiskAggregator,
)


def test_default_configuration() -> None:
    aggregator = RiskAggregator()

    assert aggregator.max_total_risk_percent == 10.0
    assert aggregator.max_total_exposure_percent == 60.0
    assert aggregator.max_positions == 10


def test_custom_configuration() -> None:
    aggregator = RiskAggregator(
        max_total_risk_percent=5.0,
        max_total_exposure_percent=40.0,
        max_positions=3,
    )

    assert aggregator.max_total_risk_percent == 5.0
    assert aggregator.max_total_exposure_percent == 40.0
    assert aggregator.max_positions == 3


def test_negative_risk_limit_rejected() -> None:
    with pytest.raises(ValueError):
        RiskAggregator(
            max_total_risk_percent=-1.0,
        )


def test_negative_exposure_limit_rejected() -> None:
    with pytest.raises(ValueError):
        RiskAggregator(
            max_total_exposure_percent=-1.0,
        )


def test_zero_max_positions_rejected() -> None:
    with pytest.raises(ValueError):
        RiskAggregator(
            max_positions=0,
        )


def test_empty_positions() -> None:
    aggregator = RiskAggregator()

    result = aggregator.aggregate({})

    assert isinstance(
        result,
        RiskAggregationResult,
    )
    assert result.total_risk_percent == 0.0
    assert result.total_exposure_percent == 0.0
    assert result.position_count == 0
    assert result.allowed is True


def test_single_position() -> None:
    aggregator = RiskAggregator()

    result = aggregator.aggregate(
        {
            "BTCUSDT": {
                "risk_percent": 2.0,
                "exposure_percent": 20.0,
            }
        }
    )

    assert result.total_risk_percent == 2.0
    assert result.total_exposure_percent == 20.0
    assert result.position_count == 1
    assert result.allowed is True


def test_multiple_positions_are_aggregated() -> None:
    aggregator = RiskAggregator()

    result = aggregator.aggregate(
        {
            "BTCUSDT": {
                "risk_percent": 2.0,
                "exposure_percent": 20.0,
            },
            "ETHUSDT": {
                "risk_percent": 1.5,
                "exposure_percent": 15.0,
            },
            "SOLUSDT": {
                "risk_percent": 1.0,
                "exposure_percent": 10.0,
            },
        }
    )

    assert result.total_risk_percent == 4.5
    assert result.total_exposure_percent == 45.0
    assert result.position_count == 3
    assert result.allowed is True


def test_zero_exposure_position_not_counted() -> None:
    aggregator = RiskAggregator(
        max_positions=1,
    )

    result = aggregator.aggregate(
        {
            "BTCUSDT": {
                "risk_percent": 0.0,
                "exposure_percent": 0.0,
            },
            "ETHUSDT": {
                "risk_percent": 2.0,
                "exposure_percent": 20.0,
            },
        }
    )

    assert result.position_count == 1
    assert result.allowed is True


def test_risk_limit_rejected() -> None:
    aggregator = RiskAggregator(
        max_total_risk_percent=5.0,
    )

    result = aggregator.aggregate(
        {
            "BTCUSDT": {
                "risk_percent": 3.0,
                "exposure_percent": 10.0,
            },
            "ETHUSDT": {
                "risk_percent": 3.0,
                "exposure_percent": 10.0,
            },
        }
    )

    assert result.total_risk_percent == 6.0
    assert result.allowed is False


def test_exposure_limit_rejected() -> None:
    aggregator = RiskAggregator(
        max_total_exposure_percent=30.0,
    )

    result = aggregator.aggregate(
        {
            "BTCUSDT": {
                "risk_percent": 1.0,
                "exposure_percent": 20.0,
            },
            "ETHUSDT": {
                "risk_percent": 1.0,
                "exposure_percent": 20.0,
            },
        }
    )

    assert result.total_exposure_percent == 40.0
    assert result.allowed is False


def test_position_limit_rejected() -> None:
    aggregator = RiskAggregator(
        max_positions=2,
    )

    result = aggregator.aggregate(
        {
            "BTCUSDT": {
                "risk_percent": 1.0,
                "exposure_percent": 10.0,
            },
            "ETHUSDT": {
                "risk_percent": 1.0,
                "exposure_percent": 10.0,
            },
            "SOLUSDT": {
                "risk_percent": 1.0,
                "exposure_percent": 10.0,
            },
        }
    )

    assert result.position_count == 3
    assert result.allowed is False


def test_is_allowed_true() -> None:
    aggregator = RiskAggregator()

    assert aggregator.is_allowed(
        {
            "BTCUSDT": {
                "risk_percent": 2.0,
                "exposure_percent": 20.0,
            }
        }
    ) is True


def test_is_allowed_false() -> None:
    aggregator = RiskAggregator(
        max_total_risk_percent=2.0,
    )

    assert aggregator.is_allowed(
        {
            "BTCUSDT": {
                "risk_percent": 3.0,
                "exposure_percent": 20.0,
            }
        }
    ) is False


def test_missing_values_default_to_zero() -> None:
    aggregator = RiskAggregator()

    result = aggregator.aggregate(
        {
            "BTCUSDT": {},
        }
    )

    assert result.total_risk_percent == 0.0
    assert result.total_exposure_percent == 0.0
    assert result.position_count == 0
    assert result.allowed is True


def test_negative_risk_rejected() -> None:
    aggregator = RiskAggregator()

    with pytest.raises(ValueError):
        aggregator.aggregate(
            {
                "BTCUSDT": {
                    "risk_percent": -1.0,
                    "exposure_percent": 10.0,
                }
            }
        )


def test_negative_exposure_rejected() -> None:
    aggregator = RiskAggregator()

    with pytest.raises(ValueError):
        aggregator.aggregate(
            {
                "BTCUSDT": {
                    "risk_percent": 1.0,
                    "exposure_percent": -10.0,
                }
            }
        )


def test_invalid_positions_type_rejected() -> None:
    aggregator = RiskAggregator()

    with pytest.raises(TypeError):
        aggregator.aggregate([])  # type: ignore[arg-type]


def test_invalid_symbol_type_rejected() -> None:
    aggregator = RiskAggregator()

    with pytest.raises(TypeError):
        aggregator.aggregate(
            {
                123: {
                    "risk_percent": 1.0,
                    "exposure_percent": 10.0,
                }
            }  # type: ignore[dict-item]
        )


def test_empty_symbol_rejected() -> None:
    aggregator = RiskAggregator()

    with pytest.raises(ValueError):
        aggregator.aggregate(
            {
                "": {
                    "risk_percent": 1.0,
                    "exposure_percent": 10.0,
                }
            }
        )


def test_invalid_position_type_rejected() -> None:
    aggregator = RiskAggregator()

    with pytest.raises(TypeError):
        aggregator.aggregate(
            {
                "BTCUSDT": 100,
            }  # type: ignore[dict-item]
        )


def test_precision() -> None:
    aggregator = RiskAggregator()

    result = aggregator.aggregate(
        {
            "BTCUSDT": {
                "risk_percent": 1.123456789,
                "exposure_percent": 20.123456789,
            },
            "ETHUSDT": {
                "risk_percent": 0.876543219,
                "exposure_percent": 10.987654321,
            },
        }
    )

    assert result.total_risk_percent == 2.0
    assert result.total_exposure_percent == 31.11111111


def test_reset() -> None:
    aggregator = RiskAggregator()

    aggregator.reset()

    result = aggregator.aggregate(
        {
            "BTCUSDT": {
                "risk_percent": 1.0,
                "exposure_percent": 10.0,
            }
        }
    )

    assert result.allowed is True