import pytest

from experimental.src.risk_engine import (
    RiskAssessment,
    RiskEngine,
)


def test_default_configuration():
    engine = RiskEngine()

    assert engine.risk_per_trade == 0.01
    assert engine.max_exposure == 0.60
    assert engine.reserve_ratio == 0.40
    assert engine.max_leverage == 20.0
    assert engine.tradable_capital_ratio == 0.60


def test_invalid_risk_per_trade():
    with pytest.raises(ValueError):
        RiskEngine(risk_per_trade=-0.01)

    with pytest.raises(ValueError):
        RiskEngine(risk_per_trade=1.01)


def test_invalid_exposure():
    with pytest.raises(ValueError):
        RiskEngine(max_exposure=-0.01)

    with pytest.raises(ValueError):
        RiskEngine(max_exposure=1.01)


def test_exposure_and_reserve_cannot_exceed_balance():
    with pytest.raises(ValueError):
        RiskEngine(
            max_exposure=0.70,
            reserve_ratio=0.40,
        )


def test_invalid_leverage():
    with pytest.raises(ValueError):
        RiskEngine(max_leverage=0.5)


def test_normal_preset():
    engine = RiskEngine.from_preset(
        "NORMAL"
    )

    assert engine.risk_per_trade == 0.01
    assert engine.max_exposure == 0.60
    assert engine.max_leverage == 20.0
    assert engine.reserve_ratio == 0.40


def test_aggressive_preset():
    engine = RiskEngine.from_preset(
        "aggressive"
    )

    assert engine.risk_per_trade == 0.02
    assert engine.max_exposure == 0.60
    assert engine.max_leverage == 50.0


def test_protective_preset():
    engine = RiskEngine.from_preset(
        "PROTECTIVE"
    )

    assert engine.risk_per_trade == 0.005
    assert engine.max_exposure == 0.40
    assert engine.max_leverage == 5.0
    assert engine.reserve_ratio == 0.60


def test_unknown_preset():
    with pytest.raises(ValueError):
        RiskEngine.from_preset("UNKNOWN")


def test_position_size_without_leverage_limit():
    engine = RiskEngine(
        risk_per_trade=0.01,
        max_exposure=0.60,
        reserve_ratio=0.40,
        max_leverage=20.0,
    )

    size = engine.calculate_position_size(
        balance=1000.0,
        entry_price=100.0,
        stop_price=90.0,
        confidence=1.0,
    )

    assert size == pytest.approx(1.0)


def test_position_size_scales_with_confidence():
    engine = RiskEngine()

    full = engine.calculate_position_size(
        balance=1000.0,
        entry_price=100.0,
        stop_price=90.0,
        confidence=1.0,
    )

    half = engine.calculate_position_size(
        balance=1000.0,
        entry_price=100.0,
        stop_price=90.0,
        confidence=0.5,
    )

    assert full == pytest.approx(1.0)
    assert half == pytest.approx(0.5)


def test_position_size_is_limited_by_max_exposure():
    engine = RiskEngine(
        risk_per_trade=0.50,
        max_exposure=0.60,
        reserve_ratio=0.40,
        max_leverage=1.0,
    )

    size = engine.calculate_position_size(
        balance=1000.0,
        entry_price=100.0,
        stop_price=99.0,
        confidence=1.0,
        leverage=1.0,
    )

    assert size == pytest.approx(6.0)


def test_assessment_approved():
    engine = RiskEngine()

    result = engine.assess(
        balance=1000.0,
        entry_price=100.0,
        stop_price=90.0,
        confidence=1.0,
        current_exposure=0.0,
        leverage=10.0,
    )

    assert isinstance(
        result,
        RiskAssessment,
    )

    assert result.approved is True
    assert result.position_size == pytest.approx(
        1.0
    )
    assert result.risk_amount == pytest.approx(
        10.0
    )
    assert result.stop_distance == pytest.approx(
        10.0
    )
    assert result.risk_percent == pytest.approx(
        1.0
    )
    assert result.exposure == pytest.approx(
        100.0
    )
    assert result.exposure_percent == pytest.approx(
        10.0
    )
    assert result.leverage == pytest.approx(
        10.0
    )
    assert result.reason == "APPROVED"


def test_assessment_rejects_excessive_leverage():
    engine = RiskEngine(
        max_leverage=10.0
    )

    result = engine.assess(
        balance=1000.0,
        entry_price=100.0,
        stop_price=90.0,
        confidence=1.0,
        leverage=20.0,
    )

    assert result.approved is False
    assert result.position_size == 0.0
    assert result.reason == (
        "LEVERAGE_EXCEEDED"
    )


def test_assessment_rejects_max_exposure():
    engine = RiskEngine()

    result = engine.assess(
        balance=1000.0,
        entry_price=100.0,
        stop_price=90.0,
        confidence=1.0,
        current_exposure=600.0,
        leverage=10.0,
    )

    assert result.approved is False
    assert result.position_size == 0.0
    assert result.reason == (
        "MAX_EXPOSURE_REACHED"
    )


def test_assessment_respects_remaining_exposure():
    engine = RiskEngine(
        risk_per_trade=0.10,
        max_exposure=0.60,
        reserve_ratio=0.40,
        max_leverage=10.0,
    )

    result = engine.assess(
        balance=1000.0,
        entry_price=100.0,
        stop_price=90.0,
        confidence=1.0,
        current_exposure=550.0,
        leverage=10.0,
    )

    assert result.approved is True
    assert result.exposure == pytest.approx(
        50.0
    )
    assert result.position_size == pytest.approx(
        0.5
    )
    assert result.exposure_percent == pytest.approx(
        60.0
    )


def test_invalid_balance():
    engine = RiskEngine()

    with pytest.raises(ValueError):
        engine.calculate_position_size(
            balance=0.0,
            entry_price=100.0,
            stop_price=90.0,
        )


def test_invalid_entry_price():
    engine = RiskEngine()

    with pytest.raises(ValueError):
        engine.calculate_position_size(
            balance=1000.0,
            entry_price=0.0,
            stop_price=90.0,
        )


def test_invalid_stop_price():
    engine = RiskEngine()

    with pytest.raises(ValueError):
        engine.calculate_position_size(
            balance=1000.0,
            entry_price=100.0,
            stop_price=0.0,
        )


def test_equal_entry_and_stop_are_rejected():
    engine = RiskEngine()

    with pytest.raises(ValueError):
        engine.calculate_position_size(
            balance=1000.0,
            entry_price=100.0,
            stop_price=100.0,
        )


def test_invalid_confidence():
    engine = RiskEngine()

    with pytest.raises(ValueError):
        engine.calculate_position_size(
            balance=1000.0,
            entry_price=100.0,
            stop_price=90.0,
            confidence=1.1,
        )

    with pytest.raises(ValueError):
        engine.assess(
            balance=1000.0,
            entry_price=100.0,
            stop_price=90.0,
            confidence=-0.1,
        )


def test_short_position_uses_absolute_stop_distance():
    engine = RiskEngine()

    result = engine.assess(
        balance=1000.0,
        entry_price=100.0,
        stop_price=110.0,
        confidence=1.0,
        leverage=10.0,
    )

    assert result.approved is True
    assert result.stop_distance == pytest.approx(
        10.0
    )
    assert result.position_size == pytest.approx(
        1.0
    )


def test_leverage_is_capped_by_engine_limit():
    engine = RiskEngine(
        max_leverage=10.0
    )

    size = engine.calculate_position_size(
        balance=1000.0,
        entry_price=100.0,
        stop_price=90.0,
        confidence=1.0,
        leverage=100.0,
    )

    assert size == pytest.approx(
        1.0
    )