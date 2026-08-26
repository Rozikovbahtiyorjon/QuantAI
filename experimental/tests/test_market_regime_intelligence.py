from __future__ import annotations

import pytest

from experimental.src.market_regime_intelligence import (
    MarketRegimeIntelligenceEngine,
    MarketRegimeSnapshot,
)


def test_trend_up() -> None:
    engine = MarketRegimeIntelligenceEngine()

    signal = engine.classify(
        "BTC/USDT:USDT",
        1000,
        [
            100.0,
            101.0,
            103.0,
            105.0,
            108.0,
        ],
        baseline_volatility=0.02,
    )

    assert signal.regime == "TREND_UP"
    assert signal.context == "BULLISH_TREND"
    assert signal.confidence > 0.0


def test_trend_down() -> None:
    engine = MarketRegimeIntelligenceEngine()

    signal = engine.classify(
        "BTC/USDT:USDT",
        1000,
        [
            108.0,
            106.0,
            104.0,
            102.0,
            100.0,
        ],
        baseline_volatility=0.02,
    )

    assert signal.regime == "TREND_DOWN"
    assert signal.context == "BEARISH_TREND"


def test_range() -> None:
    engine = MarketRegimeIntelligenceEngine(
        high_volatility_ratio=10.0,
        low_volatility_ratio=0.01,
    )

    signal = engine.classify(
        "BTC/USDT:USDT",
        1000,
        [
            100.0,
            100.5,
            99.8,
            100.3,
            100.0,
        ],
        baseline_volatility=0.01,
    )

    assert signal.regime == "RANGE"
    assert signal.context == "SIDEWAYS_MARKET"


def test_high_volatility() -> None:
    engine = MarketRegimeIntelligenceEngine(
        high_volatility_ratio=1.5,
    )

    signal = engine.classify(
        "BTC/USDT:USDT",
        1000,
        [
            100.0,
            101.0,
            99.0,
            101.0,
            100.0,
        ],
        baseline_volatility=0.005,
    )

    assert signal.regime == "HIGH_VOLATILITY"
    assert signal.context == "ELEVATED_VOLATILITY"


def test_low_volatility() -> None:
    engine = MarketRegimeIntelligenceEngine(
        low_volatility_ratio=0.5,
    )

    signal = engine.classify(
        "BTC/USDT:USDT",
        1000,
        [
            100.0,
            100.01,
            99.99,
            100.01,
            100.0,
        ],
        baseline_volatility=0.02,
    )

    assert signal.regime == "LOW_VOLATILITY"
    assert signal.context == "COMPRESSED_VOLATILITY"


def test_shock() -> None:
    engine = MarketRegimeIntelligenceEngine()

    signal = engine.classify(
        "BTC/USDT:USDT",
        1000,
        [
            100.0,
            100.5,
            101.0,
            102.0,
            106.0,
        ],
        baseline_volatility=0.02,
    )

    assert signal.regime == "SHOCK"
    assert signal.context == "MARKET_SHOCK"


def test_recovery_after_shock() -> None:
    engine = MarketRegimeIntelligenceEngine(
        shock_return_percent=3.0,
        recovery_return_percent=1.0,
    )

    shock = engine.classify(
        "BTC/USDT:USDT",
        1000,
        [
            100.0,
            100.5,
            101.0,
            102.0,
            106.0,
        ],
        baseline_volatility=0.02,
    )

    assert shock.regime == "SHOCK"

    recovery = engine.classify(
        "BTC/USDT:USDT",
        2000,
        [
            107.0,
            107.3,
            107.8,
            108.2,
            108.5,
        ],
        baseline_volatility=0.02,
    )

    assert recovery.regime == "RECOVERY"
    assert recovery.context == "POST_SHOCK_RECOVERY"


def test_snapshot_is_stored() -> None:
    engine = MarketRegimeIntelligenceEngine()

    engine.classify(
        "BTC/USDT:USDT",
        1000,
        [
            100.0,
            101.0,
            102.0,
        ],
        baseline_volatility=0.02,
    )

    assert isinstance(
        engine.previous,
        MarketRegimeSnapshot,
    )

    assert engine.previous is not None
    assert (
        engine.previous.symbol
        == "BTC/USDT:USDT"
    )


def test_reset() -> None:
    engine = MarketRegimeIntelligenceEngine()

    engine.classify(
        "BTC/USDT:USDT",
        1000,
        [
            100.0,
            101.0,
            102.0,
        ],
        baseline_volatility=0.02,
    )

    assert engine.previous is not None

    engine.reset()

    assert engine.previous is None


def test_invalid_prices() -> None:
    engine = MarketRegimeIntelligenceEngine()

    with pytest.raises(ValueError):
        engine.classify(
            "BTC/USDT:USDT",
            1000,
            [
                100.0,
                0.0,
                101.0,
            ],
        )

    with pytest.raises(ValueError):
        engine.classify(
            "BTC/USDT:USDT",
            1000,
            [
                100.0,
                101.0,
            ],
        )


def test_invalid_symbol_and_timestamp() -> None:
    engine = MarketRegimeIntelligenceEngine()

    with pytest.raises(TypeError):
        engine.classify(
            123,
            1000,
            [
                100.0,
                101.0,
                102.0,
            ],
        )

    with pytest.raises(ValueError):
        engine.classify(
            "",
            1000,
            [
                100.0,
                101.0,
                102.0,
            ],
        )

    with pytest.raises(ValueError):
        engine.classify(
            "BTC/USDT:USDT",
            -1,
            [
                100.0,
                101.0,
                102.0,
            ],
        )


def test_invalid_configuration() -> None:
    with pytest.raises(ValueError):
        MarketRegimeIntelligenceEngine(
            trend_threshold=1.1,
        )

    with pytest.raises(ValueError):
        MarketRegimeIntelligenceEngine(
            low_volatility_ratio=2.0,
            high_volatility_ratio=1.0,
        )

    with pytest.raises(ValueError):
        MarketRegimeIntelligenceEngine(
            shock_return_percent=0.0,
        )