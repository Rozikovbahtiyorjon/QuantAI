import pytest

from experimental.src.portfolio_exposure_engine import (
    PortfolioExposureEngine,
)


def test_default_configuration() -> None:
    engine = PortfolioExposureEngine()

    assert engine.max_gross_exposure_percent == 100.0
    assert engine.max_net_exposure_percent == 60.0
    assert engine.max_long_exposure_percent == 60.0
    assert engine.max_short_exposure_percent == 60.0


def test_basic_long_exposure() -> None:
    engine = PortfolioExposureEngine()

    result = engine.evaluate(
        equity=1000.0,
        positions={
            "BTCUSDT": {
                "side": "LONG",
                "exposure_percent": 20.0,
            },
            "ETHUSDT": {
                "side": "LONG",
                "exposure_percent": 10.0,
            },
        },
    )

    assert result.gross_exposure_percent == 30.0
    assert result.net_exposure_percent == 30.0
    assert result.long_exposure_percent == 30.0
    assert result.short_exposure_percent == 0.0
    assert result.gross_exposure_value == 300.0
    assert result.net_exposure_value == 300.0
    assert result.position_count == 2
    assert result.exposure_limit_ok is True


def test_long_and_short_offset_net_exposure() -> None:
    engine = PortfolioExposureEngine()

    result = engine.evaluate(
        equity=1000.0,
        positions={
            "BTCUSDT": {
                "side": "LONG",
                "exposure_percent": 40.0,
            },
            "ETHUSDT": {
                "side": "SHORT",
                "exposure_percent": 30.0,
            },
        },
    )

    assert result.gross_exposure_percent == 70.0
    assert result.net_exposure_percent == 10.0
    assert result.long_exposure_percent == 40.0
    assert result.short_exposure_percent == 30.0
    assert result.exposure_limit_ok is True


def test_gross_exposure_limit() -> None:
    engine = PortfolioExposureEngine(
        max_gross_exposure_percent=50.0,
    )

    result = engine.evaluate(
        equity=1000.0,
        positions={
            "BTCUSDT": {
                "side": "LONG",
                "exposure_percent": 30.0,
            },
            "ETHUSDT": {
                "side": "SHORT",
                "exposure_percent": 30.0,
            },
        },
    )

    assert result.gross_exposure_percent == 60.0
    assert result.exposure_limit_ok is False


def test_net_exposure_limit() -> None:
    engine = PortfolioExposureEngine(
        max_net_exposure_percent=20.0,
    )

    result = engine.evaluate(
        equity=1000.0,
        positions={
            "BTCUSDT": {
                "side": "LONG",
                "exposure_percent": 35.0,
            },
        },
    )

    assert result.net_exposure_percent == 35.0
    assert result.exposure_limit_ok is False


def test_long_exposure_limit() -> None:
    engine = PortfolioExposureEngine(
        max_long_exposure_percent=40.0,
    )

    result = engine.evaluate(
        equity=1000.0,
        positions={
            "BTCUSDT": {
                "side": "LONG",
                "exposure_percent": 50.0,
            },
        },
    )

    assert result.long_exposure_percent == 50.0
    assert result.exposure_limit_ok is False


def test_short_exposure_limit() -> None:
    engine = PortfolioExposureEngine(
        max_short_exposure_percent=30.0,
    )

    result = engine.evaluate(
        equity=1000.0,
        positions={
            "BTCUSDT": {
                "side": "SHORT",
                "exposure_percent": 40.0,
            },
        },
    )

    assert result.short_exposure_percent == 40.0
    assert result.exposure_limit_ok is False


def test_zero_exposure_not_counted() -> None:
    engine = PortfolioExposureEngine()

    result = engine.evaluate(
        equity=1000.0,
        positions={
            "BTCUSDT": {
                "side": "LONG",
                "exposure_percent": 0.0,
            },
            "ETHUSDT": {
                "side": "LONG",
                "exposure_percent": 20.0,
            },
        },
    )

    assert result.position_count == 1


def test_precision() -> None:
    engine = PortfolioExposureEngine()

    result = engine.evaluate(
        equity=1000.0,
        positions={
            "BTCUSDT": {
                "side": "LONG",
                "exposure_percent": 20.123456789,
            },
            "ETHUSDT": {
                "side": "SHORT",
                "exposure_percent": 10.987654321,
            },
        },
    )

    assert result.gross_exposure_percent == 31.11111111
    assert result.net_exposure_percent == 9.13580247
    assert result.gross_exposure_value == 311.11111111
    assert result.net_exposure_value == 91.35802469


def test_empty_positions() -> None:
    engine = PortfolioExposureEngine()

    result = engine.evaluate(
        equity=1000.0,
        positions={},
    )

    assert result.gross_exposure_percent == 0.0
    assert result.net_exposure_percent == 0.0
    assert result.position_count == 0
    assert result.exposure_limit_ok is True


def test_invalid_equity() -> None:
    engine = PortfolioExposureEngine()

    with pytest.raises(ValueError):
        engine.evaluate(
            equity=0.0,
            positions={},
        )


def test_invalid_positions_type() -> None:
    engine = PortfolioExposureEngine()

    with pytest.raises(TypeError):
        engine.evaluate(
            equity=1000.0,
            positions=[],
        )


def test_negative_exposure_rejected() -> None:
    engine = PortfolioExposureEngine()

    with pytest.raises(ValueError):
        engine.evaluate(
            equity=1000.0,
            positions={
                "BTCUSDT": {
                    "exposure_percent": -10.0,
                },
            },
        )


def test_invalid_side_rejected() -> None:
    engine = PortfolioExposureEngine()

    with pytest.raises(ValueError):
        engine.evaluate(
            equity=1000.0,
            positions={
                "BTCUSDT": {
                    "side": "INVALID",
                    "exposure_percent": 10.0,
                },
            },
        )


def test_empty_symbol_rejected() -> None:
    engine = PortfolioExposureEngine()

    with pytest.raises(ValueError):
        engine.evaluate(
            equity=1000.0,
            positions={
                "": {
                    "exposure_percent": 10.0,
                },
            },
        )


def test_non_string_symbol_rejected() -> None:
    engine = PortfolioExposureEngine()

    with pytest.raises(TypeError):
        engine.evaluate(
            equity=1000.0,
            positions={
                123: {
                    "exposure_percent": 10.0,
                },
            },
        )


def test_non_mapping_position_rejected() -> None:
    engine = PortfolioExposureEngine()

    with pytest.raises(TypeError):
        engine.evaluate(
            equity=1000.0,
            positions={
                "BTCUSDT": 10.0,
            },
        )


def test_is_allowed_true() -> None:
    engine = PortfolioExposureEngine()

    assert engine.is_allowed(
        equity=1000.0,
        positions={
            "BTCUSDT": {
                "side": "LONG",
                "exposure_percent": 20.0,
            },
        },
    ) is True


def test_is_allowed_false() -> None:
    engine = PortfolioExposureEngine(
        max_net_exposure_percent=20.0,
    )

    assert engine.is_allowed(
        equity=1000.0,
        positions={
            "BTCUSDT": {
                "side": "LONG",
                "exposure_percent": 25.0,
            },
        },
    ) is False