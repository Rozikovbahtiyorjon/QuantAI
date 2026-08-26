from __future__ import annotations

import pytest

from experimental.src.sentiment_information_intelligence import (
    SentimentInformationEngine,
    SentimentObservation,
    SentimentSnapshot,
)


def make_observations() -> tuple[SentimentObservation, ...]:
    return (
        SentimentObservation(
            source="news",
            timestamp=1000,
            sentiment=0.6,
            attention=10.0,
            reliability=1.0,
        ),
        SentimentObservation(
            source="social",
            timestamp=1000,
            sentiment=0.2,
            attention=5.0,
            reliability=0.8,
        ),
    )


def test_observation_validation_and_values() -> None:
    observation = SentimentObservation(
        source="news",
        timestamp=1000,
        sentiment=0.5,
        attention=10.0,
        reliability=0.9,
    )

    assert observation.source == "news"
    assert observation.timestamp == 1000
    assert observation.sentiment == 0.5
    assert observation.attention == 10.0
    assert observation.reliability == 0.9


def test_engine_aggregates_sentiment() -> None:
    engine = SentimentInformationEngine()

    snapshot = engine.analyze(
        make_observations()
    )

    expected = (
        0.6 * 10.0
        + 0.2 * 5.0 * 0.8
    ) / (
        10.0
        + 5.0 * 0.8
    )

    assert snapshot.timestamp == 1000
    assert snapshot.weighted_sentiment == pytest.approx(
        expected
    )
    assert snapshot.attention == 15.0
    assert snapshot.source_count == 2
    assert snapshot.information_quality == pytest.approx(
        1.0
    )
    assert snapshot.context == "BULLISH"


def test_strong_bullish_context() -> None:
    engine = SentimentInformationEngine()

    snapshot = engine.analyze(
        (
            SentimentObservation(
                source="news",
                timestamp=1000,
                sentiment=0.9,
                attention=1.0,
            ),
        )
    )

    assert snapshot.context == "STRONGLY_BULLISH"


def test_bearish_context() -> None:
    engine = SentimentInformationEngine()

    snapshot = engine.analyze(
        (
            SentimentObservation(
                source="news",
                timestamp=1000,
                sentiment=-0.4,
                attention=1.0,
            ),
        )
    )

    assert snapshot.context == "BEARISH"


def test_strong_bearish_context() -> None:
    engine = SentimentInformationEngine()

    snapshot = engine.analyze(
        (
            SentimentObservation(
                source="news",
                timestamp=1000,
                sentiment=-0.8,
                attention=1.0,
            ),
        )
    )

    assert snapshot.context == "STRONGLY_BEARISH"


def test_neutral_context() -> None:
    engine = SentimentInformationEngine()

    snapshot = engine.analyze(
        (
            SentimentObservation(
                source="news",
                timestamp=1000,
                sentiment=0.05,
                attention=1.0,
            ),
        )
    )

    assert snapshot.context == "NEUTRAL"


def test_low_information_context() -> None:
    engine = SentimentInformationEngine(
        minimum_quality=0.8
    )

    snapshot = engine.analyze(
        (
            SentimentObservation(
                source="weak_source",
                timestamp=1000,
                sentiment=0.9,
                attention=0.1,
                reliability=0.5,
            ),
        )
    )

    assert snapshot.information_quality == pytest.approx(
        0.05
    )
    assert snapshot.context == "LOW_INFORMATION"


def test_empty_observations() -> None:
    engine = SentimentInformationEngine()

    snapshot = engine.analyze(())

    assert snapshot == SentimentSnapshot(
        timestamp=0,
        weighted_sentiment=0.0,
        attention=0.0,
        information_quality=0.0,
        source_count=0,
        context="LOW_INFORMATION",
    )


def test_explicit_timestamp() -> None:
    engine = SentimentInformationEngine()

    snapshot = engine.analyze(
        make_observations(),
        timestamp=2000,
    )

    assert snapshot.timestamp == 2000


def test_update_and_previous() -> None:
    engine = SentimentInformationEngine()

    first = engine.analyze(
        (
            SentimentObservation(
                source="news",
                timestamp=1000,
                sentiment=0.4,
                attention=1.0,
            ),
        )
    )

    second = engine.update(
        SentimentObservation(
            source="social",
            timestamp=2000,
            sentiment=0.8,
            attention=1.0,
        )
    )

    assert first.timestamp == 1000
    assert second.timestamp == 2000
    assert engine.previous == second


def test_reset() -> None:
    engine = SentimentInformationEngine()

    engine.analyze(
        make_observations()
    )

    assert engine.previous is not None

    engine.reset()

    assert engine.previous is None


def test_invalid_observation_values() -> None:
    with pytest.raises(ValueError):
        SentimentObservation(
            source="news",
            timestamp=1000,
            sentiment=1.1,
            attention=1.0,
        )

    with pytest.raises(ValueError):
        SentimentObservation(
            source="news",
            timestamp=1000,
            sentiment=0.0,
            attention=-1.0,
        )

    with pytest.raises(ValueError):
        SentimentObservation(
            source="news",
            timestamp=1000,
            sentiment=0.0,
            attention=1.0,
            reliability=1.1,
        )


def test_invalid_engine_configuration() -> None:
    with pytest.raises(ValueError):
        SentimentInformationEngine(
            minimum_quality=1.1
        )

    with pytest.raises(ValueError):
        SentimentInformationEngine(
            bullish_threshold=0.7,
            strong_bullish_threshold=0.6,
        )

    with pytest.raises(ValueError):
        SentimentInformationEngine(
            bearish_threshold=-0.2,
            strong_bearish_threshold=-0.1,
        )


def test_invalid_input_types() -> None:
    engine = SentimentInformationEngine()

    with pytest.raises(TypeError):
        engine.analyze(None)

    with pytest.raises(TypeError):
        engine.analyze([object()])

    with pytest.raises(TypeError):
        engine.update(object())

    with pytest.raises(ValueError):
        engine.analyze(
            (),
            timestamp=-1,
        )