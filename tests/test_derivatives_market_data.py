from __future__ import annotations

import pytest

from src.derivatives_market_data import (
    DerivativesMarketData,
    DerivativesMarketDataEngine,
    DerivativesSignal,
)


def make_data(
    price: float = 100.0,
    open_interest: float = 1000.0,
) -> DerivativesMarketData:
    return DerivativesMarketData(
        symbol="BTCUSDT",
        timestamp=1,
        price=price,
        open_interest=open_interest,
        funding_rate=0.0001,
        liquidation_volume=10.0,
        long_short_ratio=1.2,
        spot_price=99.0,
    )


def test_baseline_update() -> None:
    result = DerivativesMarketDataEngine().update(
        make_data()
    )

    assert isinstance(
        result,
        DerivativesSignal,
    )

    assert result.context == "BASELINE"
    assert result.price_change == 0.0
    assert result.open_interest_change == 0.0


def test_price_up_oi_up() -> None:
    engine = DerivativesMarketDataEngine()

    engine.update(make_data())

    result = engine.update(
        make_data(
            110.0,
            1200.0,
        )
    )

    assert result.price_change == pytest.approx(
        0.10
    )

    assert result.open_interest_change == pytest.approx(
        0.20
    )

    assert not result.price_oi_divergence

    assert result.context == "PRICE_UP_OI_UP"


def test_price_down_oi_up_is_divergence() -> None:
    engine = DerivativesMarketDataEngine()

    engine.update(make_data())

    result = engine.update(
        make_data(
            90.0,
            1200.0,
        )
    )

    assert result.price_oi_divergence
    assert result.context == "PRICE_DOWN_OI_UP"


def test_price_up_oi_down_is_divergence() -> None:
    engine = DerivativesMarketDataEngine()

    engine.update(make_data())

    result = engine.update(
        make_data(
            110.0,
            800.0,
        )
    )

    assert result.price_oi_divergence
    assert result.context == "PRICE_UP_OI_DOWN"


def test_price_down_oi_down() -> None:
    engine = DerivativesMarketDataEngine()

    engine.update(make_data())

    result = engine.update(
        make_data(
            90.0,
            800.0,
        )
    )

    assert not result.price_oi_divergence
    assert result.context == "PRICE_DOWN_OI_DOWN"


def test_basis() -> None:
    data = make_data()

    assert data.basis == pytest.approx(
        1.0
    )

    assert data.basis_percent == pytest.approx(
        (100.0 / 99.0 - 1.0) * 100.0
    )


def test_validation() -> None:
    engine = DerivativesMarketDataEngine()

    with pytest.raises(TypeError):
        engine.update(123)

    with pytest.raises(ValueError):
        engine.update(
            make_data(0.0)
        )

    with pytest.raises(ValueError):
        engine.update(
            make_data(
                100.0,
                -1.0,
            )
        )


def test_symbol_mismatch() -> None:
    engine = DerivativesMarketDataEngine()

    engine.update(make_data())

    with pytest.raises(ValueError):
        engine.update(
            DerivativesMarketData(
                symbol="ETHUSDT",
                timestamp=2,
                price=100.0,
                open_interest=1000.0,
                funding_rate=0.0001,
                liquidation_volume=10.0,
                long_short_ratio=1.2,
            )
        )


def test_reset() -> None:
    engine = DerivativesMarketDataEngine()

    engine.update(make_data())

    engine.reset()

    assert engine.previous is None