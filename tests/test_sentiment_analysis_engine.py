import pytest

from src.sentiment_analysis_engine import (
    SentimentAnalysisEngine,
    SentimentResult,
)


def test_default_initialization():
    engine = SentimentAnalysisEngine()

    assert engine.bullish_threshold == pytest.approx(0.15)
    assert engine.bearish_threshold == pytest.approx(-0.15)
    assert engine.min_confidence == pytest.approx(0.20)
    assert engine.negation_window == 3


def test_invalid_threshold_order_is_rejected():
    with pytest.raises(ValueError):
        SentimentAnalysisEngine(
            bullish_threshold=-0.2,
            bearish_threshold=-0.1,
        )


def test_invalid_confidence_is_rejected():
    with pytest.raises(ValueError):
        SentimentAnalysisEngine(min_confidence=1.1)


def test_negative_negation_window_is_rejected():
    with pytest.raises(ValueError):
        SentimentAnalysisEngine(negation_window=-1)


def test_empty_positive_terms_are_rejected():
    with pytest.raises(ValueError):
        SentimentAnalysisEngine(positive_terms={})


def test_invalid_term_weight_is_rejected():
    with pytest.raises(TypeError):
        SentimentAnalysisEngine(
            positive_terms={"bullish": "strong"}
        )


def test_empty_text_is_neutral():
    engine = SentimentAnalysisEngine()

    result = engine.analyze_text("")

    assert isinstance(result, SentimentResult)
    assert result.signal == "NEUTRAL"
    assert result.score == pytest.approx(0.0)
    assert result.confidence == pytest.approx(0.0)


def test_bullish_text_is_detected():
    engine = SentimentAnalysisEngine(
        min_confidence=0.0
    )

    result = engine.analyze_text(
        "Bitcoin bullish breakout rally strong adoption"
    )

    assert result.signal == "BULLISH"
    assert result.score > 0
    assert result.positive_count > 0


def test_bearish_text_is_detected():
    engine = SentimentAnalysisEngine(
        min_confidence=0.0
    )

    result = engine.analyze_text(
        "Bitcoin bearish crash selloff weak rejection"
    )

    assert result.signal == "BEARISH"
    assert result.score < 0
    assert result.negative_count > 0


def test_mixed_text_can_be_neutral():
    engine = SentimentAnalysisEngine(
        min_confidence=0.0
    )

    result = engine.analyze_text(
        "bullish rally but bearish rejection"
    )

    assert result.signal in {
        "BULLISH",
        "BEARISH",
        "NEUTRAL",
    }

    assert -1.0 <= result.score <= 1.0


def test_negation_reverses_sentiment():
    engine = SentimentAnalysisEngine(
        min_confidence=0.0
    )

    result = engine.analyze_text(
        "not bullish"
    )

    assert result.score < 0
    assert result.signal == "BEARISH"


def test_positive_intensifier_increases_signal():
    engine = SentimentAnalysisEngine(
        min_confidence=0.0
    )

    normal = engine.analyze_text("bullish")
    intensified = engine.analyze_text("very bullish")

    assert intensified.score >= normal.score


def test_single_string_iterable_is_rejected():
    engine = SentimentAnalysisEngine()

    with pytest.raises(TypeError):
        engine.analyze("bullish")


def test_empty_collection_is_neutral():
    engine = SentimentAnalysisEngine()

    result = engine.analyze([])

    assert result.signal == "NEUTRAL"
    assert result.score == pytest.approx(0.0)
    assert result.confidence == pytest.approx(0.0)
    assert result.source_count == 0


def test_multiple_sources_are_aggregated():
    engine = SentimentAnalysisEngine(
        min_confidence=0.0
    )

    result = engine.analyze(
        [
            "bullish breakout",
            "strong adoption",
            "rally gains",
        ]
    )

    assert result.source_count == 3
    assert result.signal == "BULLISH"
    assert result.score > 0


def test_multiple_bearish_sources_are_aggregated():
    engine = SentimentAnalysisEngine(
        min_confidence=0.0
    )

    result = engine.analyze(
        [
            "bearish crash",
            "major selloff",
            "weak decline",
        ]
    )

    assert result.source_count == 3
    assert result.signal == "BEARISH"
    assert result.score < 0


def test_non_string_source_is_rejected():
    engine = SentimentAnalysisEngine()

    with pytest.raises(TypeError):
        engine.analyze(
            [
                "bullish breakout",
                123,
            ]
        )


def test_score_is_bounded():
    engine = SentimentAnalysisEngine(
        min_confidence=0.0
    )

    result = engine.analyze(
        [
            "massive extremely bullish breakout rally",
            "strong adoption growth",
        ]
    )

    assert -1.0 <= result.score <= 1.0


def test_confidence_is_bounded():
    engine = SentimentAnalysisEngine(
        min_confidence=0.0
    )

    result = engine.analyze(
        [
            "massive extremely bullish breakout rally",
            "strong adoption growth",
        ]
    )

    assert 0.0 <= result.confidence <= 1.0


def test_signal_returns_string():
    engine = SentimentAnalysisEngine()

    signal = engine.signal(
        [
            "bullish breakout",
        ]
    )

    assert isinstance(signal, str)


def test_compare_is_case_insensitive():
    engine = SentimentAnalysisEngine(
        min_confidence=0.0
    )

    texts = ["bullish breakout"]

    signal = engine.signal(texts)

    assert engine.compare(
        texts,
        signal.lower(),
    ) is True


def test_compare_returns_boolean():
    engine = SentimentAnalysisEngine()

    result = engine.compare(
        ["bullish breakout"],
        "BULLISH",
    )

    assert isinstance(result, bool)


def test_summarize_returns_expected_fields():
    engine = SentimentAnalysisEngine(
        min_confidence=0.0
    )

    result = engine.analyze(
        [
            "bullish breakout",
            "strong adoption",
        ]
    )

    summary = engine.summarize(result)

    assert set(summary) == {
        "score",
        "confidence",
        "signal",
        "positive_count",
        "negative_count",
        "neutral_count",
        "source_count",
    }

    assert summary["signal"] == "BULLISH"
    assert summary["source_count"] == 2


def test_summarize_requires_result_object():
    with pytest.raises(TypeError):
        SentimentAnalysisEngine.summarize(
            "BULLISH"
        )


def test_custom_terms_work():
    engine = SentimentAnalysisEngine(
        positive_terms={
            "moon": 1.0,
        },
        negative_terms={
            "rug": 1.0,
        },
        min_confidence=0.0,
    )

    bullish = engine.analyze_text("moon")
    bearish = engine.analyze_text("rug")

    assert bullish.signal == "BULLISH"
    assert bearish.signal == "BEARISH"


def test_neutral_without_matching_terms():
    engine = SentimentAnalysisEngine(
        min_confidence=0.0
    )

    result = engine.analyze_text(
        "bitcoin market discussion today"
    )

    assert result.signal == "NEUTRAL"
    assert result.score == pytest.approx(0.0)