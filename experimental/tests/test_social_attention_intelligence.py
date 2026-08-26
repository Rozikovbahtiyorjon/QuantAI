import pytest

from experimental.src.social_attention_intelligence import (
    SocialAttentionIntelligence,
    SocialAttentionSnapshot,
)


def make_snapshot(
    timestamp: int = 1000,
    attention: float = 100.0,
    social_volume: float = 50.0,
    engagement: float = 25.0,
    mentions: float = 20.0,
    contributors: float = 5.0,
) -> SocialAttentionSnapshot:
    return SocialAttentionSnapshot(
        symbol="BTC/USDT",
        timestamp=timestamp,
        attention=attention,
        social_volume=social_volume,
        engagement=engagement,
        mentions=mentions,
        contributors=contributors,
    )


def test_snapshot_metrics() -> None:
    snapshot = make_snapshot()

    assert snapshot.attention_per_mention == 5.0
    assert snapshot.engagement_per_mention == 1.25

    zero_mentions = make_snapshot(
        mentions=0.0,
    )

    assert zero_mentions.attention_per_mention == 0.0
    assert zero_mentions.engagement_per_mention == 0.0


def test_first_update() -> None:
    engine = SocialAttentionIntelligence()

    result = engine.update(
        make_snapshot()
    )

    assert result.attention_change == 0.0
    assert result.social_volume_change == 0.0
    assert result.engagement_change == 0.0
    assert result.attention_zscore == 0.0
    assert result.anomaly is False
    assert result.context == "MIXED_ATTENTION"


def test_rising_attention() -> None:
    engine = SocialAttentionIntelligence()

    engine.update(
        make_snapshot()
    )

    result = engine.update(
        make_snapshot(
            timestamp=2000,
            attention=120.0,
            social_volume=60.0,
            engagement=30.0,
        )
    )

    assert result.attention_change == pytest.approx(
        0.20
    )

    assert result.social_volume_change == pytest.approx(
        0.20
    )

    assert result.engagement_change == pytest.approx(
        0.20
    )

    assert result.context == "RISING_ATTENTION"


def test_falling_attention() -> None:
    engine = SocialAttentionIntelligence()

    engine.update(
        make_snapshot()
    )

    result = engine.update(
        make_snapshot(
            timestamp=2000,
            attention=80.0,
            social_volume=40.0,
            engagement=20.0,
        )
    )

    assert result.attention_change == pytest.approx(
        -0.20
    )

    assert result.social_volume_change == pytest.approx(
        -0.20
    )

    assert result.engagement_change == pytest.approx(
        -0.20
    )

    assert result.context == "FALLING_ATTENTION"


def test_mixed_attention() -> None:
    engine = SocialAttentionIntelligence()

    engine.update(
        make_snapshot()
    )

    result = engine.update(
        make_snapshot(
            timestamp=2000,
            attention=120.0,
            social_volume=45.0,
            engagement=30.0,
        )
    )

    assert result.context == "MIXED_ATTENTION"


def test_zero_previous_value() -> None:
    engine = SocialAttentionIntelligence()

    engine.update(
        make_snapshot(
            social_volume=0.0,
            engagement=0.0,
        )
    )

    result = engine.update(
        make_snapshot(
            timestamp=2000,
            social_volume=10.0,
            engagement=5.0,
        )
    )

    assert result.social_volume_change == 1.0
    assert result.engagement_change == 1.0


def test_zero_to_zero_value() -> None:
    engine = SocialAttentionIntelligence()

    engine.update(
        make_snapshot(
            social_volume=0.0,
            engagement=0.0,
        )
    )

    result = engine.update(
        make_snapshot(
            timestamp=2000,
            social_volume=0.0,
            engagement=0.0,
        )
    )

    assert result.social_volume_change == 0.0
    assert result.engagement_change == 0.0


def test_attention_anomaly() -> None:
    engine = SocialAttentionIntelligence(
        anomaly_zscore=2.0,
        min_history=3,
    )

    engine.update(
        make_snapshot(
            timestamp=1000,
            attention=100.0,
        )
    )

    engine.update(
        make_snapshot(
            timestamp=2000,
            attention=101.0,
        )
    )

    engine.update(
        make_snapshot(
            timestamp=3000,
            attention=99.0,
        )
    )

    result = engine.update(
        make_snapshot(
            timestamp=4000,
            attention=150.0,
        )
    )

    assert result.anomaly is True
    assert result.context == "ATTENTION_ANOMALY"
    assert result.attention_zscore > 2.0


def test_anomaly_requires_minimum_history() -> None:
    engine = SocialAttentionIntelligence(
        anomaly_zscore=1.0,
        min_history=3,
    )

    result = engine.update(
        make_snapshot(
            attention=1000.0,
        )
    )

    assert result.anomaly is False


def test_history_and_previous() -> None:
    engine = SocialAttentionIntelligence()

    first = make_snapshot()

    second = make_snapshot(
        timestamp=2000,
        attention=110.0,
    )

    engine.update(first)
    engine.update(second)

    assert engine.previous == second
    assert len(engine.history) == 2
    assert engine.history[0] == first
    assert engine.history[1] == second


def test_reset() -> None:
    engine = SocialAttentionIntelligence()

    engine.update(
        make_snapshot()
    )

    assert engine.previous is not None
    assert len(engine.history) == 1

    engine.reset()

    assert engine.previous is None
    assert engine.history == ()


def test_invalid_snapshot_type() -> None:
    engine = SocialAttentionIntelligence()

    with pytest.raises(TypeError):
        engine.update("invalid")  # type: ignore[arg-type]


def test_invalid_snapshot_values() -> None:
    engine = SocialAttentionIntelligence()

    with pytest.raises(ValueError):
        engine.update(
            make_snapshot(
                attention=-1.0,
            )
        )

    with pytest.raises(ValueError):
        engine.update(
            make_snapshot(
                social_volume=float("nan"),
            )
        )


def test_symbol_and_timestamp_validation() -> None:
    engine = SocialAttentionIntelligence()

    engine.update(
        make_snapshot()
    )

    with pytest.raises(ValueError):
        engine.update(
            SocialAttentionSnapshot(
                symbol="ETH/USDT",
                timestamp=2000,
                attention=1.0,
                social_volume=1.0,
                engagement=1.0,
                mentions=1.0,
            )
        )

    with pytest.raises(ValueError):
        engine.update(
            make_snapshot(
                timestamp=1000,
            )
        )


def test_constructor_validation() -> None:
    with pytest.raises(TypeError):
        SocialAttentionIntelligence(
            anomaly_zscore="2.0",  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError):
        SocialAttentionIntelligence(
            anomaly_zscore=0.0,
        )

    with pytest.raises(TypeError):
        SocialAttentionIntelligence(
            min_history=2.0,  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError):
        SocialAttentionIntelligence(
            min_history=0,
        )