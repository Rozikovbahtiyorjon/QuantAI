import pytest

from experimental.src.sentiment_divergence_intelligence import (
    SentimentDivergenceIntelligence,
    SentimentDivergenceSnapshot,
)


def make_snapshot(
    timestamp: int = 1000,
    price_change: float = 0.02,
    sentiment_change: float = 0.01,
    attention_change: float = 0.01,
    volume_change: float = 0.01,
) -> SentimentDivergenceSnapshot:
    return SentimentDivergenceSnapshot(
        symbol="BTC/USDT",
        timestamp=timestamp,
        price_change=price_change,
        sentiment_change=sentiment_change,
        attention_change=attention_change,
        volume_change=volume_change,
    )


def test_snapshot_directions() -> None:
    snapshot = make_snapshot(
        price_change=0.02,
        sentiment_change=-0.01,
        attention_change=0.0,
        volume_change=0.03,
    )

    assert snapshot.price_direction == 1
    assert snapshot.sentiment_direction == -1
    assert snapshot.attention_direction == 0
    assert snapshot.volume_direction == 1


def test_aligned_market_context() -> None:
    engine = SentimentDivergenceIntelligence()

    result = engine.update(
        make_snapshot()
    )

    assert result.divergence_score == 0.0
    assert result.divergence is False
    assert result.context == "ALIGNED"


def test_bearish_sentiment_divergence() -> None:
    engine = SentimentDivergenceIntelligence(
        divergence_threshold=0.5
    )

    result = engine.update(
        make_snapshot(
            price_change=0.03,
            sentiment_change=-0.02,
            attention_change=-0.01,
            volume_change=0.01,
        )
    )

    assert result.divergence_score == pytest.approx(
        2.0 / 3.0
    )

    assert result.divergence is True

    assert (
        result.context
        == "BEARISH_SENTIMENT_DIVERGENCE"
    )


def test_bullish_sentiment_divergence() -> None:
    engine = SentimentDivergenceIntelligence(
        divergence_threshold=0.5
    )

    result = engine.update(
        make_snapshot(
            price_change=-0.03,
            sentiment_change=0.02,
            attention_change=0.01,
            volume_change=-0.01,
        )
    )

    assert result.divergence_score == pytest.approx(
        2.0 / 3.0
    )

    assert result.divergence is True

    assert (
        result.context
        == "BULLISH_SENTIMENT_DIVERGENCE"
    )


def test_neutral_price_divergence() -> None:
    engine = SentimentDivergenceIntelligence()

    result = engine.update(
        make_snapshot(
            price_change=0.0,
            sentiment_change=-0.02,
            attention_change=0.01,
            volume_change=0.01,
        )
    )

    assert result.divergence_score == 0.0
    assert result.divergence is False
    assert result.context == "ALIGNED"


def test_zero_factors() -> None:
    engine = SentimentDivergenceIntelligence()

    result = engine.update(
        make_snapshot(
            price_change=0.0,
            sentiment_change=0.0,
            attention_change=0.0,
            volume_change=0.0,
        )
    )

    assert result.divergence_score == 0.0
    assert result.divergence is False


def test_threshold_boundary() -> None:
    engine = SentimentDivergenceIntelligence(
        divergence_threshold=2.0 / 3.0
    )

    result = engine.update(
        make_snapshot(
            price_change=0.02,
            sentiment_change=-0.01,
            attention_change=-0.01,
            volume_change=0.01,
        )
    )

    assert result.divergence_score == pytest.approx(
        2.0 / 3.0
    )

    assert result.divergence is True


def test_custom_threshold() -> None:
    engine = SentimentDivergenceIntelligence(
        divergence_threshold=1.0
    )

    result = engine.update(
        make_snapshot(
            price_change=0.02,
            sentiment_change=-0.01,
            attention_change=-0.01,
            volume_change=0.01,
        )
    )

    assert result.divergence_score == pytest.approx(
        2.0 / 3.0
    )

    assert result.divergence is False


def test_previous_snapshot() -> None:
    engine = SentimentDivergenceIntelligence()

    snapshot = make_snapshot()

    engine.update(snapshot)

    assert engine.previous == snapshot


def test_reset() -> None:
    engine = SentimentDivergenceIntelligence()

    engine.update(
        make_snapshot()
    )

    assert engine.previous is not None

    engine.reset()

    assert engine.previous is None


def test_symbol_change_rejected() -> None:
    engine = SentimentDivergenceIntelligence()

    engine.update(
        make_snapshot()
    )

    changed = SentimentDivergenceSnapshot(
        symbol="ETH/USDT",
        timestamp=2000,
        price_change=0.01,
        sentiment_change=0.01,
        attention_change=0.01,
        volume_change=0.01,
    )

    with pytest.raises(ValueError):
        engine.update(changed)


def test_timestamp_must_increase() -> None:
    engine = SentimentDivergenceIntelligence()

    engine.update(
        make_snapshot(timestamp=1000)
    )

    with pytest.raises(ValueError):
        engine.update(
            make_snapshot(timestamp=1000)
        )

    with pytest.raises(ValueError):
        engine.update(
            make_snapshot(timestamp=999)
        )


def test_invalid_snapshot_type() -> None:
    engine = SentimentDivergenceIntelligence()

    with pytest.raises(TypeError):
        engine.update("invalid")  # type: ignore[arg-type]


def test_invalid_snapshot_values() -> None:
    engine = SentimentDivergenceIntelligence()

    with pytest.raises(TypeError):
        engine.update(
            make_snapshot(
                price_change="0.1"  # type: ignore[arg-type]
            )
        )

    with pytest.raises(ValueError):
        engine.update(
            make_snapshot(
                sentiment_change=float("nan")
            )
        )

    with pytest.raises(ValueError):
        engine.update(
            make_snapshot(
                attention_change=float("inf")
            )
        )


def test_invalid_constructor() -> None:
    with pytest.raises(TypeError):
        SentimentDivergenceIntelligence(
            divergence_threshold="0.5"  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError):
        SentimentDivergenceIntelligence(
            divergence_threshold=0.0
        )

    with pytest.raises(ValueError):
        SentimentDivergenceIntelligence(
            divergence_threshold=1.1
        )