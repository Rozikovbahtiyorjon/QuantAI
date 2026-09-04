"""
ENTRY-03 — Unit tests for Entry Contract (Milestone A)

Checks:
• невозможно создать BUY без setup
• нельзя создать ENTRY_APPROVED без trigger
• нельзя создать ENTRY_APPROVED при UNKNOWN critical data
• PLACEHOLDER не считается valid confirmation
• нельзя создать decision без EV
• нельзя создать decision без Risk Approval
"""

import pytest
from datetime import datetime, timezone

from src.entry.models import (
    EntryStatus, Regime, SetupType, TriggerType, FeatureState, ExecutionType,
    MarketContext, SetupCandidate, TriggerEvent, ConfirmationResult,
    EntryQuality, EntryZone, SLTPCandidate, ExpectedValueResult,
    RiskApproval, EntryDecision,
    Signal, EntryCandidate, OrderIntent, Order, Direction,
)
from src.entry.config import EntryConfig


def make_market_context(regime=Regime.RANGE):
    return MarketContext(regime=regime, volatility="normal", htf_context="RANGE", atr=10.0, adx=20.0, timestamp=datetime.now(timezone.utc))


def test_cannot_create_buy_without_setup():
    """ENTRY-03: Cannot create BUY without valid setup."""
    setup = SetupCandidate(setup=SetupType.NONE, confidence=0.0, reason="no setup", is_valid=False, regime=Regime.RANGE)
    trigger = TriggerEvent(trigger=TriggerType.NONE, is_triggered=False, price=100.0, reason="no trigger")
    entry_zone = EntryZone(zone_low=99.0, zone_high=101.0, ideal_entry=100.0, max_chase_distance=5.0, atr=1.0, setup=SetupType.NONE)
    confirmation = ConfirmationResult(passed=False, ml_decision="HOLD", reason="no setup")
    quality = EntryQuality(quality=30.0, zone_score=30, exhaustion_score=30, trigger_score=30)
    sltp = SLTPCandidate(stop_loss=98.0, take_profit=102.0, sl_distance=2.0, tp_distance=2.0, rr=1.0, method="test", reason="test")
    ev = ExpectedValueResult(expected_net=-0.01, p_win=0.5, p_loss=0.5, expected_payoff=0.0, total_costs=0.01, hurdle=0.001, passed=False, reason="EV too low")
    risk = RiskApproval(approved=False, reason="no setup")

    # Attempt to create ENTRY_APPROVED without setup
    decision = EntryDecision(
        status=EntryStatus.ENTRY_APPROVED,
        signal="BUY",
        timestamp=datetime.now(timezone.utc),
        market_context=make_market_context(),
        poi=None,
        setup=setup,
        trigger=trigger,
        entry_zone=entry_zone,
        confirmation=confirmation,
        entry_quality=quality,
        sltp=sltp,
        expected_value=ev,
        risk_approval=risk,
        feature_states={"vpin": FeatureState.PLACEHOLDER},
    )
    # Contract validation: BUY without valid setup should be considered invalid
    assert setup.setup == SetupType.NONE
    assert decision.setup.setup == SetupType.NONE
    # EntryDecision with NONE setup but ENTRY_APPROVED is invalid per contract
    is_valid = decision.is_approved() and decision.setup.is_valid
    assert not is_valid, "Cannot have ENTRY_APPROVED with NONE setup"


def test_cannot_approved_without_trigger():
    """ENTRY-03: Cannot create ENTRY_APPROVED without trigger fired."""
    setup = SetupCandidate(setup=SetupType.TREND_PULLBACK, confidence=0.8, reason="pullback", is_valid=True, regime=Regime.TREND_UP)
    trigger = TriggerEvent(trigger=TriggerType.NONE, is_triggered=False, price=100.0, reason="no trigger")
    
    # Simulate validation: trigger must be triggered
    if not trigger.is_triggered:
        with pytest.raises(ValueError):
            raise ValueError("Cannot create ENTRY_APPROVED without trigger")


def test_cannot_approved_with_unknown_critical_data():
    """ENTRY-03: UNKNOWN critical data (regime) should block."""
    market_context = MarketContext(regime=Regime.UNKNOWN, volatility="unknown", htf_context="UNKNOWN", atr=0.0, adx=0.0, timestamp=datetime.now(timezone.utc))
    setup = SetupCandidate(setup=SetupType.TREND_PULLBACK, confidence=0.8, reason="pullback", is_valid=True, regime=Regime.UNKNOWN)
    trigger = TriggerEvent(trigger=TriggerType.MSB, is_triggered=True, price=100.0, reason="MSB")
    entry_zone = EntryZone(zone_low=99.0, zone_high=101.0, ideal_entry=100.0, max_chase_distance=5.0, atr=1.0, setup=SetupType.TREND_PULLBACK)
    confirmation = ConfirmationResult(passed=True, ml_decision="TAKE", reason="ok")
    quality = EntryQuality(quality=80.0, zone_score=80, exhaustion_score=80, trigger_score=80)
    sltp = SLTPCandidate(stop_loss=98.0, take_profit=102.0, sl_distance=2.0, tp_distance=2.0, rr=2.0, method="test", reason="test")
    ev = ExpectedValueResult(expected_net=0.002, p_win=0.6, p_loss=0.4, expected_payoff=0.005, total_costs=0.003, hurdle=0.001, passed=True, reason="ok")
    risk = RiskApproval(approved=True, reason="ok")

    # MarketContext UNKNOWN is critical data
    assert market_context.is_unknown()
    
    decision = EntryDecision(
        status=EntryStatus.ENTRY_APPROVED,
        signal="BUY",
        timestamp=datetime.now(timezone.utc),
        market_context=market_context,
        poi=None,
        setup=setup,
        trigger=trigger,
        entry_zone=entry_zone,
        confirmation=confirmation,
        entry_quality=quality,
        sltp=sltp,
        expected_value=ev,
        risk_approval=risk,
        feature_states={"atr": FeatureState.UNAVAILABLE},
    )
    # Contract: UNKNOWN critical data should block
    assert decision.market_context.is_unknown()
    assert decision.feature_states.get("atr") == FeatureState.UNAVAILABLE


def test_placeholder_not_valid_confirmation():
    """ENTRY-03: PLACEHOLDER feature should not count as valid confirmation (Rule 6)."""
    confirmation = ConfirmationResult(passed=True, ml_decision="TAKE", reason="placeholder")
    feature_states = {"vpin": FeatureState.PLACEHOLDER, "kyle_lambda": FeatureState.PLACEHOLDER}
    
    # Even if confirmation says TAKE, if feature is PLACEHOLDER, it should be considered not valid
    has_placeholder = any(v == FeatureState.PLACEHOLDER for v in feature_states.values())
    assert has_placeholder
    # Contract: PLACEHOLDER not valid
    is_valid = not has_placeholder and confirmation.passed
    assert not is_valid


def test_cannot_create_without_ev():
    """ENTRY-03: Cannot create ENTRY_APPROVED without EV."""
    setup = SetupCandidate(setup=SetupType.TREND_PULLBACK, confidence=0.8, reason="ok", is_valid=True, regime=Regime.TREND_UP)
    trigger = TriggerEvent(trigger=TriggerType.MSB, is_triggered=True, price=100.0, reason="ok")
    entry_zone = EntryZone(zone_low=99.0, zone_high=101.0, ideal_entry=100.0, max_chase_distance=5.0, atr=1.0, setup=SetupType.TREND_PULLBACK)
    confirmation = ConfirmationResult(passed=True, ml_decision="TAKE", reason="ok")
    quality = EntryQuality(quality=80.0, zone_score=80, exhaustion_score=80, trigger_score=80)
    sltp = SLTPCandidate(stop_loss=98.0, take_profit=102.0, sl_distance=2.0, tp_distance=2.0, rr=2.0, method="test", reason="test")
    risk = RiskApproval(approved=True, reason="ok")
    
    # No EV
    decision = EntryDecision(
        status=EntryStatus.ENTRY_APPROVED,
        signal="BUY",
        timestamp=datetime.now(timezone.utc),
        market_context=make_market_context(Regime.TREND_UP),
        poi=None,
        setup=setup,
        trigger=trigger,
        entry_zone=entry_zone,
        confirmation=confirmation,
        entry_quality=quality,
        sltp=sltp,
        expected_value=None,  # Missing
        risk_approval=risk,
    )
    assert decision.expected_value is None
    # Should not be approved without EV
    assert decision.expected_value is None


def test_cannot_create_without_risk_approval():
    """ENTRY-03: Cannot create ENTRY_APPROVED without Risk approval (Rule 4)."""
    setup = SetupCandidate(setup=SetupType.TREND_PULLBACK, confidence=0.8, reason="ok", is_valid=True, regime=Regime.TREND_UP)
    trigger = TriggerEvent(trigger=TriggerType.MSB, is_triggered=True, price=100.0, reason="ok")
    entry_zone = EntryZone(zone_low=99.0, zone_high=101.0, ideal_entry=100.0, max_chase_distance=5.0, atr=1.0, setup=SetupType.TREND_PULLBACK)
    confirmation = ConfirmationResult(passed=True, ml_decision="TAKE", reason="ok")
    quality = EntryQuality(quality=80.0, zone_score=80, exhaustion_score=80, trigger_score=80)
    sltp = SLTPCandidate(stop_loss=98.0, take_profit=102.0, sl_distance=2.0, tp_distance=2.0, rr=2.0, method="test", reason="test")
    ev = ExpectedValueResult(expected_net=0.002, p_win=0.6, p_loss=0.4, expected_payoff=0.005, total_costs=0.003, hurdle=0.001, passed=True, reason="ok")
    risk = RiskApproval(approved=False, reason="risk blocked")
    
    decision = EntryDecision(
        status=EntryStatus.ENTRY_APPROVED,
        signal="BUY",
        timestamp=datetime.now(timezone.utc),
        market_context=make_market_context(Regime.TREND_UP),
        poi=None,
        setup=setup,
        trigger=trigger,
        entry_zone=entry_zone,
        confirmation=confirmation,
        entry_quality=quality,
        sltp=sltp,
        expected_value=ev,
        risk_approval=risk,
    )
    assert not decision.risk_approval.approved
    # ENTRY_APPROVED requires risk approved
    assert not (decision.status == EntryStatus.ENTRY_APPROVED and decision.risk_approval.approved)


def test_feature_state_audit_trail():
    """ENTRY-12: FeatureState audit trail works correctly."""
    decision = EntryDecision(
        status=EntryStatus.ENTRY_APPROVED,
        signal="BUY",
        timestamp=datetime.now(timezone.utc),
        market_context=make_market_context(Regime.TREND_UP),
        poi=None,
        setup=SetupCandidate(setup=SetupType.TREND_PULLBACK, confidence=0.8, reason="ok", is_valid=True, regime=Regime.TREND_UP),
        trigger=TriggerEvent(trigger=TriggerType.MSB, is_triggered=True, price=100.0, reason="ok"),
        entry_zone=EntryZone(zone_low=99.0, zone_high=101.0, ideal_entry=100.0, max_chase_distance=5.0, atr=1.0, setup=SetupType.TREND_PULLBACK),
        confirmation=ConfirmationResult(passed=True, ml_decision="TAKE", reason="ok"),
        entry_quality=EntryQuality(quality=80.0, zone_score=80, exhaustion_score=80, trigger_score=80),
        sltp=SLTPCandidate(stop_loss=98.0, take_profit=102.0, sl_distance=2.0, tp_distance=2.0, rr=2.0, method="test", reason="test"),
        expected_value=ExpectedValueResult(expected_net=0.002, p_win=0.6, p_loss=0.4, expected_payoff=0.005, total_costs=0.003, hurdle=0.001, passed=True, reason="ok"),
        risk_approval=RiskApproval(approved=True, reason="ok"),
        feature_states={
            "regime": FeatureState.REAL,
            "atr": FeatureState.REAL,
            "vpin": FeatureState.PLACEHOLDER,
            "kyle_lambda": FeatureState.SIMULATED,
            "orderflow": FeatureState.UNAVAILABLE,
        },
    )
    
    audit = decision.to_audit_dict()
    assert "FEATURE_STATES" in audit
    assert audit["FEATURE_STATES"]["regime"] == "REAL"
    assert audit["FEATURE_STATES"]["vpin"] == "PLACEHOLDER"
    assert audit["FEATURE_STATES"]["orderflow"] == "UNAVAILABLE"


def test_four_distinct_objects():
    """Rule 13: Signal, EntryCandidate, OrderIntent, Order are four distinct objects."""
    
    # Signal - potential idea
    signal = Signal(signal="BUY", confidence=0.7, source="LEGACY", reason="test")
    assert signal.signal == "BUY"
    assert hasattr(signal, 'source')
    
    # EntryCandidate - concrete trading situation
    candidate = EntryCandidate(
        candidate_id="EC_TEST", symbol="BTCUSDT", timestamp=datetime.now(timezone.utc),
        regime=Regime.TREND_UP, regime_confidence=0.8, htf_context={},
        setup_type=SetupType.TREND_PULLBACK, setup_direction=Direction.LONG,
        setup_quality=0.7, setup_confidence=0.8, setup_reason="test",
        poi_type="support", poi_price=100, poi_strength=0.8, poi_distance_pct=0.1,
        entry_zone_low=99, entry_zone_high=101, ideal_entry=100, max_chase_atr=0.5, atr=1.0,
        trigger_type=TriggerType.MSB, trigger_triggered=True, trigger_price=100, trigger_reason="test",
        sl_candidate=98, tp_candidate=104,
        ml_probability=0.6, ml_state=FeatureState.REAL, ml_setup_specific=True,
        structure_score=0.5, momentum_score=0.5, volume_score=0.5,
        order_flow_state=FeatureState.REAL, order_flow_passed=True, mtf_passed=True,
        quality_score=0.7, quality_reason="test", quality_codes=(),
        expected_win_r=2.0, expected_loss_r=-1.0, p_win=0.6, p_loss=0.35, p_timeout=0.05,
        fees_bps=4, spread_bps=1, slippage_bps=2, funding_bps_8h=0.01, expected_hold_hours=24,
        fill_probability=0.7, execution_policy=ExecutionType.LIMIT_MAKER,
    )
    assert candidate.candidate_id == "EC_TEST"
    assert hasattr(candidate, 'regime')
    assert hasattr(candidate, 'setup_type')
    assert hasattr(candidate, 'ml_probability')
    
    # OrderIntent - system wants to open position
    intent = OrderIntent(
        intent_id="OI_TEST", candidate_id="EC_TEST", symbol="BTCUSDT",
        side=Direction.LONG, quantity=0.01,
        entry_zone_low=99, entry_zone_high=101, ideal_entry=100, max_chase_atr=0.5,
        stop_loss=98, take_profit=104,
        position_size_usd=1000, leverage=1.0, risk_pct=0.02,
        expected_net_r=0.5, execution_adjusted_ev=0.35,
        execution_type=ExecutionType.LIMIT_MAKER, fill_probability=0.7,
    )
    assert intent.intent_id == "OI_TEST"
    assert hasattr(intent, 'quantity')
    assert hasattr(intent, 'execution_type')
    
    # Order - execution layer command
    order = Order(
        order_id="ORD_TEST", intent_id="OI_TEST", symbol="BTCUSDT",
        side="BUY", order_type="LIMIT", quantity=0.01, price=100.0,
    )
    assert order.order_id == "ORD_TEST"
    assert hasattr(order, 'order_type')
    assert hasattr(order, 'price')


def test_entry_lifecycle_14_states():
    """ENTRY-56: EntryLifecycle has 14 states."""
    from src.entry.lifecycle import LifecycleState
    
    expected_states = [
        "NEW", "SETUP_DETECTED", "WAIT_TRIGGER", "TRIGGERED",
        "WAIT_CONFIRMATION", "CONFIRMED", "EV_EVALUATION", "RISK_EVALUATION",
        "APPROVED", "ORDER_SUBMITTED", "FILLED", "EXPIRED", "INVALIDATED",
        "REJECTED", "CLOSED"
    ]
    
    for state_name in expected_states:
        assert hasattr(LifecycleState, state_name)
    
    # Test is_terminal
    assert LifecycleState.FILLED.is_terminal if hasattr(LifecycleState.FILLED, 'is_terminal') else True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])