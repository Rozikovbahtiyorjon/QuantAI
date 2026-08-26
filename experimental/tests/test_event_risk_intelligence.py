import pytest

from experimental.src.event_risk_intelligence import (
    EventRiskIntelligence,
    EventRiskSignal,
    MarketEvent,
)


def make_event(
    symbol: str = "BTC/USDT",
    timestamp: int = 1000,
    event_type: str = "MACRO",
    importance: float = 1.0,
    expected_impact: float = 1.0,
    hours_to_event: float = 0.0,
    uncertainty: float = 1.0,
    active: bool = True,
) -> MarketEvent:
    return MarketEvent(
        symbol=symbol,
        timestamp=timestamp,
        event_type=event_type,
        importance=importance,
        expected_impact=expected_impact,
        hours_to_event=hours_to_event,
        uncertainty=uncertainty,
        active=active,
    )


def test_market_event_fields() -> None:
    event = make_event(
        event_type="FOMC",
        importance=0.8,
        expected_impact=0.9,
        hours_to_event=4.0,
        uncertainty=0.7,
    )

    assert event.symbol == "BTC/USDT"
    assert event.event_type == "FOMC"
    assert event.importance == 0.8
    assert event.expected_impact == 0.9
    assert event.hours_to_event == 4.0
    assert event.uncertainty == 0.7
    assert event.active is True


def test_extreme_immediate_event() -> None:
    engine = EventRiskIntelligence()

    result = engine.evaluate(make_event())

    assert isinstance(result, EventRiskSignal)
    assert result.risk_score == pytest.approx(1.0)
    assert result.risk_level == "EXTREME"
    assert result.context == "EVENT_ACTIVE"


def test_high_risk_event() -> None:
    engine = EventRiskIntelligence()

    result = engine.evaluate(
        make_event(
            importance=0.9,
            expected_impact=0.9,
            hours_to_event=12.0,
            uncertainty=0.8,
        )
    )

    assert result.risk_score > 0.70
    assert result.risk_level == "HIGH"
    assert result.context == "NEAR_TERM_EVENT"


def test_elevated_risk_event() -> None:
    engine = EventRiskIntelligence()

    result = engine.evaluate(
        make_event(
            importance=0.5,
            expected_impact=0.5,
            hours_to_event=48.0,
            uncertainty=0.5,
        )
    )

    assert 0.0 < result.risk_score < 0.70
    assert result.risk_level == "ELEVATED"


def test_distant_event_has_lower_risk() -> None:
    engine = EventRiskIntelligence()

    result = engine.evaluate(
        make_event(
            importance=1.0,
            expected_impact=1.0,
            hours_to_event=168.0,
            uncertainty=1.0,
        )
    )

    assert result.risk_score == pytest.approx(0.5)
    assert result.risk_level == "ELEVATED"
    assert result.context == "DISTANT_EVENT"


def test_inactive_event_is_normal() -> None:
    engine = EventRiskIntelligence()

    result = engine.evaluate(
        make_event(active=False)
    )

    assert result.risk_score == 0.0
    assert result.risk_level == "NORMAL"
    assert result.context == "NO_ACTIVE_EVENT"


def test_event_type_is_preserved() -> None:
    engine = EventRiskIntelligence()

    result = engine.evaluate(
        make_event(event_type="REGULATORY")
    )

    assert result.event_type == "REGULATORY"


def test_previous_event_is_stored() -> None:
    engine = EventRiskIntelligence()
    event = make_event()

    engine.evaluate(event)

    assert engine.previous == event


def test_reset() -> None:
    engine = EventRiskIntelligence()

    engine.evaluate(make_event())

    assert engine.previous is not None

    engine.reset()

    assert engine.previous is None


def test_symbol_change_rejected() -> None:
    engine = EventRiskIntelligence()

    engine.evaluate(make_event())

    changed = MarketEvent(
        symbol="ETH/USDT",
        timestamp=1001,
        event_type="MACRO",
        importance=0.5,
        expected_impact=0.5,
        hours_to_event=1.0,
        uncertainty=0.5,
    )

    with pytest.raises(ValueError):
        engine.evaluate(changed)


def test_timestamp_cannot_go_backwards() -> None:
    engine = EventRiskIntelligence()

    engine.evaluate(
        make_event(timestamp=1000)
    )

    with pytest.raises(ValueError):
        engine.evaluate(
            make_event(timestamp=999)
        )


def test_constructor_validation() -> None:
    with pytest.raises(TypeError):
        EventRiskIntelligence(
            high_risk_threshold="0.7"
        )

    with pytest.raises(ValueError):
        EventRiskIntelligence(
            high_risk_threshold=0.0
        )

    with pytest.raises(ValueError):
        EventRiskIntelligence(
            high_risk_threshold=1.1
        )

    with pytest.raises(ValueError):
        EventRiskIntelligence(
            high_risk_threshold=0.8,
            extreme_risk_threshold=0.7,
        )


def test_event_validation() -> None:
    engine = EventRiskIntelligence()

    with pytest.raises(TypeError):
        engine.evaluate("invalid")

    with pytest.raises(ValueError):
        engine.evaluate(
            make_event(symbol="")
        )

    with pytest.raises(ValueError):
        engine.evaluate(
            make_event(timestamp=-1)
        )

    with pytest.raises(TypeError):
        engine.evaluate(
            make_event(active=1)
        )


def test_numeric_validation() -> None:
    engine = EventRiskIntelligence()

    with pytest.raises(ValueError):
        engine.evaluate(
            make_event(importance=1.1)
        )

    with pytest.raises(ValueError):
        engine.evaluate(
            make_event(expected_impact=-0.1)
        )

    with pytest.raises(ValueError):
        engine.evaluate(
            make_event(uncertainty=float("nan"))
        )

    with pytest.raises(TypeError):
        engine.evaluate(
            make_event(hours_to_event="2")
        )

    with pytest.raises(ValueError):
        engine.evaluate(
            make_event(hours_to_event=-1.0)
        )


def test_threshold_classification() -> None:
    engine = EventRiskIntelligence(
        high_risk_threshold=0.60,
        extreme_risk_threshold=0.80,
    )

    extreme = engine.evaluate(
        make_event(
            importance=0.8,
            expected_impact=1.0,
            hours_to_event=0.0,
            uncertainty=1.0,
        )
    )

    assert extreme.risk_score == pytest.approx(0.8)
    assert extreme.risk_level == "EXTREME"

    high = engine.evaluate(
        make_event(
            timestamp=1001,
            importance=0.7,
            expected_impact=1.0,
            hours_to_event=0.0,
            uncertainty=1.0,
        )
    )

    assert high.risk_score == pytest.approx(0.7)
    assert high.risk_level == "HIGH"