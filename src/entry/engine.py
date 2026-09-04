"""
ENTRY-59/60/61 — EntryDecisionEngine (PHASE 13)

Orchestrates the full pipeline:
Market → Signal → Setup → Trigger → EntryCandidate → EV → Risk → OrderIntent → Execution → Order → Fill

Single public API:
    evaluate(observation) -> EntryDecision

External code does NOT know:
- how setup is computed
- how trigger is computed
- how EV is computed
- how risk is computed
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Literal
from datetime import datetime, timezone
from enum import Enum
import uuid

from src.entry.models import (
    Signal, EntryCandidate, OrderIntent, Order, EntryDecision,
    MarketContext, POI, SetupCandidate, TriggerEvent, ConfirmationResult,
    EntryQuality, EntryZone, SLTPCandidate, ExpectedValueResult, RiskApproval,
    Regime, SetupType, TriggerType, Direction, EntryStatus, FeatureState, ExecutionType,
)
from src.entry.setup import SetupDetectorInterface, SetupType as SetupType2
# TriggerType is in models, not trigger
from src.entry.confirmation import ConfirmationEngine
from src.entry.quality import EntryQualityAggregator, RegimeQuality, SetupQuality, TriggerQuality, ConfirmationQuality, MLQuality, ZoneQuality
from src.entry.expected_value import ExpectedValueEngine, EVInputs, create_ev_inputs_from_candidate, EVResultStatus
from src.entry.zone import Zone
from src.entry.config import EntryConfig


class PipelineStage(str, Enum):
    MARKET_CONTEXT = "MARKET_CONTEXT"
    POI_ZONES = "POI_ZONES"
    SETUP_DETECTION = "SETUP_DETECTION"
    TRIGGER_DETECTION = "TRIGGER_DETECTION"
    ENTRY_CANDIDATE = "ENTRY_CANDIDATE"
    CONFIRMATION = "CONFIRMATION"
    ML_FILTER = "ML_FILTER"
    QUALITY = "QUALITY"
    SLTP = "SLTP"
    EV_GATE = "EV_GATE"
    RISK_GATE = "RISK_GATE"
    ORDER_INTENT = "ORDER_INTENT"
    EXECUTION = "EXECUTION"
    COMPLETE = "COMPLETE"


@dataclass
class PipelineState:
    """Mutable state during pipeline execution."""
    stage: PipelineStage = PipelineStage.MARKET_CONTEXT
    observation: Any = None
    market_context: MarketContext | None = None
    zones: list[Zone] = field(default_factory=list)
    setup: SetupCandidate | None = None
    trigger: TriggerEvent | None = None
    entry_candidate: EntryCandidate | None = None
    confirmation: ConfirmationResult | None = None
    quality: EntryQuality | None = None
    sltp: SLTPCandidate | None = None
    ev_result: ExpectedValueResult | None = None
    ev_status: EVResultStatus | None = None
    risk_approval: RiskApproval | None = None
    order_intent: OrderIntent | None = None
    order: Order | None = None
    feature_states: dict[str, FeatureState] = field(default_factory=dict)
    reason_codes: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class EntryDecisionEngine:
    """
    ENTRY-59/60: Main orchestrator.
    
    Does NOT compute anything itself.
    Only orchestrates components.
    
    Public API: evaluate(observation) -> EntryDecision
    """
    
    def __init__(self, config: EntryConfig | None = None):
        self.config = config or EntryConfig()
        
        # Components (lazy init)
        self._setup_bank = None
        self._trigger_engine = None
        self._confirmation_engine = None
        self._quality_aggregator = None
        self._ev_engine = None
        self._sl_calculator = None
        self._risk_orchestrator = None
        
        # Feature state tracking for audit
        self._feature_states: dict[str, FeatureState] = {}
    
    def evaluate(self, observation: Any) -> EntryDecision:
        """
        Single public entry point.
        
        observation: Market data (DataFrame or dict with OHLCV + indicators)
        
        Returns: EntryDecision (immutable audit snapshot)
        """
        state = PipelineState(observation=observation)
        
        try:
            # Pipeline stages
            state = self._stage_market_context(state)
            if state.errors: return self._build_decision(state, EntryStatus.NO_SETUP, "HOLD")
            
            state = self._stage_poi_zones(state)
            
            state = self._stage_setup_detection(state)
            if state.errors or state.setup is None: 
                return self._build_decision(state, EntryStatus.NO_SETUP, "HOLD")
            
            state = self._stage_trigger_detection(state)
            if state.errors or not state.trigger.is_triggered:
                return self._build_decision(state, EntryStatus.WAIT_TRIGGER, "HOLD")
            
            state = self._stage_entry_candidate(state)
            
            state = self._stage_confirmation(state)
            if state.errors or not state.confirmation.passed:
                return self._build_decision(state, EntryStatus.WAIT_CONFIRMATION, "HOLD")
            
            state = self._stage_ml_filter(state)
            if state.errors or state.confirmation.ml_decision == "REJECT":
                return self._build_decision(state, EntryStatus.ML_BLOCKED, "HOLD")
            
            state = self._stage_quality(state)
            if state.errors or state.quality.quality < self.config.min_quality:
                return self._build_decision(state, EntryStatus.EV_TOO_LOW, "HOLD")
            
            state = self._stage_sltp(state)
            
            state = self._stage_ev_gate(state)
            if state.errors or state.ev_status != EVResultStatus.PASS:
                return self._build_decision(state, EntryStatus.EV_TOO_LOW, "HOLD")
            
            state = self._stage_risk_gate(state)
            if state.errors or not state.risk_approval.approved:
                return self._build_decision(state, EntryStatus.RISK_BLOCKED, "HOLD")
            
            state = self._stage_order_intent(state)
            
            state.stage = PipelineStage.COMPLETE
            return self._build_decision(state, EntryStatus.ENTRY_APPROVED, "BUY" if state.setup.setup_direction == Direction.LONG else "SELL")
            
        except Exception as e:
            state.errors.append(f"Pipeline error: {e}")
            return self._build_decision(state, EntryStatus.NO_SETUP, "HOLD")
    
    # ===== Pipeline Stages =====
    
    def _stage_market_context(self, state: PipelineState) -> PipelineState:
        """Market → Regime + HTF Context + ATR + ADX"""
        # Extract from observation (DataFrame)
        df = state.observation
        if hasattr(df, 'iloc'):
            last = df.iloc[-1]
            atr = float(last.get('atr', last.get('ATR', 0.01)))
            adx = float(last.get('adx', last.get('ADX', 20)))
            # Regime from indicators
            ema_fast = float(last.get('ema_fast', last.get('EMA_FAST', last['close'])))
            ema_trend = float(last.get('ema_trend', last.get('EMA_TREND', last['close'])))
            
            if adx > 25 and ema_fast > ema_trend:
                regime = Regime.TREND_UP
                regime_conf = min(adx / 50, 1.0)
            elif adx > 25 and ema_fast < ema_trend:
                regime = Regime.TREND_DOWN
                regime_conf = min(adx / 50, 1.0)
            elif adx < 20:
                regime = Regime.RANGE
                regime_conf = 1.0 - adx / 20
            else:
                regime = Regime.TRANSITION
                regime_conf = 0.5
            
            volatility = "high" if atr / last['close'] > 0.02 else "normal"
            htf_context = "TREND_UP" if ema_fast > ema_trend else "TREND_DOWN" if ema_fast < ema_trend else "RANGE"
            
            state.market_context = MarketContext(
                regime=regime,
                volatility=volatility,
                htf_context=htf_context,
                atr=atr,
                adx=adx,
            )
            state.feature_states["regime"] = FeatureState.REAL
            state.feature_states["atr"] = FeatureState.REAL
            state.feature_states["adx"] = FeatureState.REAL
        else:
            state.errors.append("Invalid observation format")
        
        state.stage = PipelineStage.POI_ZONES
        return state
    
    def _stage_poi_zones(self, state: PipelineState) -> PipelineState:
        """POI/Zones from ZoneEngine"""
        from src.market_data.zone_engine import ZoneEngine
        
        df = state.observation
        zone_engine = ZoneEngine()
        zones = zone_engine.detect_all_zones(df)
        
        state.zones = zones
        state.feature_states["zones"] = FeatureState.REAL
        state.stage = PipelineStage.SETUP_DETECTION
        return state
    
    def _stage_setup_detection(self, state: PipelineState) -> PipelineState:
        """Setup Detection via SetupBank"""
        from src.strategy.setup_bank import DEFAULT_SETUP_BANK, SetupType
        
        # Ensure all detectors are registered
        if not DEFAULT_SETUP_BANK._detectors:
            self._register_default_setups()
        
        # Get market context for detectors
        ctx = {
            "regime": state.market_context.regime,
            "adx": state.market_context.adx,
            "atr": state.market_context.atr,
            "htf_direction": state.market_context.htf_context,
            "df": state.observation,
        }
        
        # Run all detectors, pick best
        candidates = DEFAULT_SETUP_BANK.detect_all(ctx, state.zones)
        
        if not candidates:
            state.errors.append("No valid setup detected")
            return state
        
        # Pick best by quality
        best = max(candidates, key=lambda c: c.quality)
        state.setup = best
        
        # Map SetupType to our enum
        setup_map = {
            SetupType2.TREND_PULLBACK: SetupType.TREND_PULLBACK,
            SetupType2.BREAKOUT: SetupType.BREAKOUT,
            SetupType2.MEAN_REVERSION: SetupType.MEAN_REVERSION,
            SetupType2.MOMENTUM_CONTINUATION: SetupType.MOMENTUM_CONTINUATION,
            SetupType2.LIQUIDITY_REVERSAL: SetupType.LIQUIDITY_REVERSAL,
        }
        
        state.stage = PipelineStage.TRIGGER_DETECTION
        return state
    
    def _stage_trigger_detection(self, state: PipelineState) -> PipelineState:
        """Trigger Detection via TriggerEngine"""
        from src.entry_engine import TriggerEngine
        
        trigger_engine = TriggerEngine()
        trigger_event = trigger_engine.detect(state.setup, state.market_context)
        
        state.trigger = trigger_event
        state.feature_states["trigger"] = FeatureState.REAL
        state.stage = PipelineStage.ENTRY_CANDIDATE
        return state
    
    def _stage_entry_candidate(self, state: PipelineState) -> PipelineState:
        """Assemble EntryCandidate (central entity)"""
        
        # Get POI info from zones
        poi_info = {}
        nearest_poi = None
        min_dist = float('inf')
        for z in state.zones:
            if z.distance_pct < min_dist:
                min_dist = z.distance_pct
                nearest_poi = z
                poi_info = {
                    "type": z.zone_type.value,
                    "price": z.center,
                    "strength": z.strength,
                    "distance_pct": z.distance_pct,
                }
        
        # Get ML probability (setup-specific if available)
        ml_prob = 0.5
        ml_state = FeatureState.PLACEHOLDER
        ml_setup_specific = False
        
        # Create EntryCandidate
        candidate_id = f"EC_{uuid.uuid4().hex[:8]}"
        
        state.entry_candidate = EntryCandidate(
            candidate_id=candidate_id,
            symbol=state.observation.get('symbol', 'BTCUSDT') if isinstance(state.observation, dict) else 'BTCUSDT',
            timestamp=datetime.now(timezone.utc),
            
            # Regime
            regime=state.market_context.regime,
            regime_confidence=0.8,  # from _stage_market_context
            htf_context={
                "direction": state.market_context.htf_context,
                "trend_strength": state.market_context.adx / 50,
                "volatility": state.market_context.volatility,
                "structure": "bullish" if state.market_context.regime == Regime.TREND_UP else "bearish",
                "confidence": 0.7,
            },
            
            # Setup
            setup_type=state.setup.setup_type,
            setup_direction=state.setup.direction,
            setup_quality=state.setup.quality,
            setup_confidence=state.setup.confidence,
            setup_reason=f"{state.setup.setup_type.value} quality={state.setup.quality:.2f}",
            
            # POI
            poi_type=poi_info.get("type", "NONE"),
            poi_price=poi_info.get("price", state.setup.ideal_entry),
            poi_strength=poi_info.get("strength", 0.5),
            poi_distance_pct=poi_info.get("distance_pct", 0.5),
            
            # Entry Zone
            entry_zone_low=state.setup.entry_zone_low,
            entry_zone_high=state.setup.entry_zone_high,
            ideal_entry=state.setup.ideal_entry,
            max_chase_atr=state.setup.max_chase_atr,
            atr=state.market_context.atr,
            
            # Trigger
            trigger_type=state.trigger.trigger if hasattr(state.trigger, 'trigger') else TriggerType.NONE,
            trigger_triggered=state.trigger.is_triggered if hasattr(state.trigger, 'is_triggered') else True,
            trigger_price=state.trigger.price if hasattr(state.trigger, 'price') else state.setup.ideal_entry,
            trigger_reason=state.trigger.reason if hasattr(state.trigger, 'reason') else "triggered",
            
            # SL/TP
            sl_candidate=state.setup.sl_candidate,
            tp_candidate=state.setup.tp_candidate,
            
            # ML
            ml_probability=ml_prob,
            ml_state=ml_state,
            ml_setup_specific=ml_setup_specific,
            
            # Confirmation (filled later)
            structure_score=0.0,
            momentum_score=0.0,
            volume_score=0.0,
            order_flow_state=FeatureState.UNAVAILABLE,
            order_flow_passed=False,
            mtf_passed=False,
            
            # Quality (filled later)
            quality_score=0.0,
            quality_reason="",
            quality_codes=(),
            
            # EV inputs
            expected_win_r=state.setup.rr_ratio,
            expected_loss_r=-1.0,
            p_win=ml_prob,
            p_loss=1.0 - ml_prob - 0.05,
            p_timeout=0.05,
            
            # Costs
            fees_bps=4.0,  # 0.04%
            spread_bps=1.0,
            slippage_bps=2.0,
            funding_bps_8h=0.01,
            expected_hold_hours=24.0,
            fill_probability=0.7,
            execution_policy=ExecutionType.LIMIT_MAKER,
            
            feature_states=self._feature_states.copy(),
        )
        
        state.stage = PipelineStage.CONFIRMATION
        return state
    
    def _stage_confirmation(self, state: PipelineState) -> PipelineState:
        """Confirmation Engine (6 independent groups)"""
        confirmation_engine = ConfirmationEngine()
        
        # Build context for confirmation
        ctx = {
            "entry_candidate": state.entry_candidate,
            "market_context": state.market_context,
            "zones": state.zones,
            "df": state.observation,
        }
        
        result = confirmation_engine.confirm(ctx)
        state.confirmation = result
        
        # Update entry_candidate with confirmation scores
        if state.entry_candidate:
            # Note: EntryCandidate is frozen, so we track in state
            pass
        
        state.feature_states["structure"] = FeatureState.REAL
        state.feature_states["momentum"] = FeatureState.REAL
        state.feature_states["volume"] = FeatureState.REAL
        state.feature_states["orderflow"] = result.order_flow_state
        state.feature_states["mtf"] = FeatureState.REAL
        
        state.stage = PipelineStage.ML_FILTER
        return state
    
    def _stage_ml_filter(self, state: PipelineState) -> PipelineState:
        """ML Filter - may reject, cannot create entry without setup"""
        
        # ML is a FILTER only (ENTRY-40): NO SETUP → ML cannot create entry
        # If we have a setup, ML may reject
        
        ml_prob = state.confirmation.ml_probability if state.confirmation else 0.5
        ml_state = state.confirmation.ml_state if state.confirmation else FeatureState.UNAVAILABLE
        
        # Update entry_candidate ml fields (would need mutable copy in real impl)
        
        # ML Decision logic
        if ml_state == FeatureState.PLACEHOLDER:
            # PLACEHOLDER cannot be a valid confirmation (ENTRY-35)
            state.confirmation.ml_decision = "HOLD"
            state.confirmation.reason = "ML placeholder - not a valid confirmation"
            state.reason_codes.append("ML_PLACEHOLDER")
        elif ml_prob >= 0.55:
            state.confirmation.ml_decision = "TAKE"
        else:
            state.confirmation.ml_decision = "REJECT"
            state.reason_codes.append("ML_REJECTED")
        
        state.stage = PipelineStage.QUALITY
        return state
    
    def _stage_quality(self, state: PipelineState) -> PipelineState:
        """Entry Quality Score (6 components, deterministic weights)"""
        
        aggregator = EntryQualityAggregator()
        
        regime_q = RegimeQuality().evaluate(
            state.market_context.regime.value,
            state.market_context.adx,
            state.market_context.adx / 50,
        )
        setup_q = SetupQuality().evaluate(
            state.setup.setup_type.value if state.setup else "NONE",
            state.setup.quality if state.setup else 0,
        )
        trigger_q = TriggerQuality().evaluate(
            state.trigger.trigger.value if hasattr(state.trigger, 'trigger') else "NONE",
            state.trigger.is_triggered if hasattr(state.trigger, 'is_triggered') else False,
        )
        conf_q = ConfirmationQuality().evaluate(
            state.confirmation.structure_score if state.confirmation else 0,
            state.confirmation.momentum_score if state.confirmation else 0,
            state.confirmation.volume_score if state.confirmation else 0,
            state.confirmation.order_flow_approved if state.confirmation else False,
            state.confirmation.mtf_approved if state.confirmation else False,
        )
        ml_q = MLQuality().evaluate(
            state.confirmation.ml_probability if state.confirmation else 0.5,
            state.confirmation.ml_state.value if state.confirmation else "UNAVAILABLE",
        )
        
        # Zone quality from nearest POI
        zone_q = ZoneQuality().evaluate(
            state.entry_candidate.poi_distance_pct if state.entry_candidate else 1.0,
            state.entry_candidate.poi_strength if state.entry_candidate else 0.3,
            0.7,  # freshness - would come from zone
        )
        
        quality_score, quality_reason = aggregator.aggregate(
            regime_q, setup_q, trigger_q, conf_q, ml_q, zone_q
        )
        
        state.quality = EntryQuality(
            quality=quality_score,
            zone_score=zone_q.score,
            exhaustion_score=0.0,  # would come from setup
            trigger_score=trigger_q.score,
            reasons=(regime_q.reason_code, setup_q.reason_code, trigger_q.reason_code, conf_q.reason_code, ml_q.reason_code, zone_q.reason_code),
        )
        
        # Collect reason codes
        state.reason_codes.extend([regime_q.reason_code, setup_q.reason_code, trigger_q.reason_code, conf_q.reason_code, ml_q.reason_code, zone_q.reason_code])
        
        state.stage = PipelineStage.SLTP
        return state
    
    def _stage_sltp(self, state: PipelineState) -> PipelineState:
        """Structural SL/TP"""
        from src.strategy.sl_tp_calculator import SLTPCalculator
        
        calc = SLTPCalculator()
        
        direction = "long" if state.setup.direction == Direction.LONG else "short"
        sl, tp = calc.calculate_structural(
            df=state.observation,
            direction=direction,
            entry=state.entry_candidate.ideal_entry,
            regime=state.market_context.regime.value,
        )
        
        state.sltp = SLTPCandidate(
            stop_loss=sl,
            take_profit=tp,
            sl_distance=abs(state.entry_candidate.ideal_entry - sl),
            tp_distance=abs(tp - state.entry_candidate.ideal_entry),
            rr=abs(tp - state.entry_candidate.ideal_entry) / abs(state.entry_candidate.ideal_entry - sl) if sl != state.entry_candidate.ideal_entry else 0,
            method="structural",
            reason=f"swing {'low' if direction == 'long' else 'high'} + ATR buffer",
        )
        
        state.stage = PipelineStage.EV_GATE
        return state
    
    def _stage_ev_gate(self, state: PipelineState) -> PipelineState:
        """Expected Value Gate"""
        
        ev_engine = ExpectedValueEngine(
            min_ev_threshold=self.config.min_ev_threshold,
            min_fill_probability=self.config.min_fill_probability,
        )
        
        inputs = create_ev_inputs_from_candidate(
            state.entry_candidate,
            state.entry_candidate.ml_probability if state.entry_candidate else 0.5,
            {
                "fees_per_side": state.entry_candidate.fees_bps / 10000 if state.entry_candidate else 0.0004,
                "spread_bps": state.entry_candidate.spread_bps if state.entry_candidate else 1.0,
                "expected_slippage_bps": state.entry_candidate.slippage_bps if state.entry_candidate else 2.0,
                "funding_bps_8h": state.entry_candidate.funding_bps_8h if state.entry_candidate else 0.01,
                "expected_hold_hours": state.entry_candidate.expected_hold_hours if state.entry_candidate else 24.0,
                "execution_policy": state.entry_candidate.execution_policy.value if state.entry_candidate else "LIMIT_MAKER",
            },
        )
        
        ev_result = ev_engine.calculate(inputs)
        state.ev_status = ev_result.status
        
        state.ev_result = ExpectedValueResult(
            expected_net=ev_result.ev_breakdown.net_ev,
            p_win=inputs.p_win,
            p_loss=inputs.p_loss,
            expected_payoff=ev_result.ev_breakdown.gross_ev,
            total_costs=ev_result.ev_breakdown.total_costs,
            hurdle=self.config.min_ev_threshold,
            passed=ev_result.status == EVResultStatus.PASS,
            reason=ev_result.reason,
            execution_adjusted_ev=ev_result.ev_breakdown.execution_adjusted_ev,
            sensitivity=ev_result.ev_breakdown.sensitivity,
        )
        
        state.reason_codes.extend(ev_result.reason_codes)
        
        state.stage = PipelineStage.RISK_GATE
        return state
    
    def _stage_risk_gate(self, state: PipelineState) -> PipelineState:
        """Risk Approval via RiskOrchestrator"""
        
        from src.risk.risk_orchestrator import create_default_orchestrator
        from src.risk.risk_context import RiskContext
        from src.risk.policy import BasePolicy
        
        orchestrator = create_default_orchestrator()
        
        # Build risk context
        ctx = RiskContext(
            account_balance=10000.0,
            balance_timestamp=datetime.now(timezone.utc),
            open_positions=[],
            market_data={"volatility": state.market_context.volatility, "atr": state.market_context.atr},
            market_data_timestamp=datetime.now(timezone.utc),
        )
        
        # Create order intent for risk check
        intent = OrderIntent(
            intent_id=f"OI_{uuid.uuid4().hex[:8]}",
            candidate_id=state.entry_candidate.candidate_id if state.entry_candidate else "unknown",
            symbol=state.entry_candidate.symbol if state.entry_candidate else "BTCUSDT",
            side=state.setup.direction,
            quantity=0.01,  # placeholder
            entry_zone_low=state.entry_candidate.entry_zone_low if state.entry_candidate else 0,
            entry_zone_high=state.entry_candidate.entry_zone_high if state.entry_candidate else 0,
            ideal_entry=state.entry_candidate.ideal_entry if state.entry_candidate else 0,
            max_chase_atr=state.entry_candidate.max_chase_atr if state.entry_candidate else 0,
            stop_loss=state.sltp.stop_loss if state.sltp else 0,
            take_profit=state.sltp.take_profit if state.sltp else 0,
            position_size_usd=1000.0,
            leverage=1.0,
            risk_pct=0.02,
            expected_net_r=state.ev_result.expected_net if state.ev_result else 0,
            execution_adjusted_ev=state.ev_result.execution_adjusted_ev if state.ev_result else 0,
            execution_type=state.entry_candidate.execution_policy if state.entry_candidate else ExecutionType.LIMIT_MAKER,
            fill_probability=state.entry_candidate.fill_probability if state.entry_candidate else 0.7,
        )
        
        # Risk check
        result = orchestrator.check_entry(ctx, intent)
        
        state.risk_approval = RiskApproval(
            approved=result.approved,
            reason=result.reason,
            position_size=result.position_size if hasattr(result, 'position_size') else 0,
            leverage=result.leverage if hasattr(result, 'leverage') else 1.0,
            exposure_ok=result.exposure_ok if hasattr(result, 'exposure_ok') else False,
        )
        
        if not result.approved:
            state.reason_codes.append("RISK_REJECTED")
        
        state.stage = PipelineStage.ORDER_INTENT
        return state
    
    def _stage_order_intent(self, state: PipelineState) -> PipelineState:
        """Create OrderIntent from approved EntryCandidate"""
        
        state.order_intent = OrderIntent(
            intent_id=f"OI_{uuid.uuid4().hex[:8]}",
            candidate_id=state.entry_candidate.candidate_id if state.entry_candidate else "unknown",
            symbol=state.entry_candidate.symbol if state.entry_candidate else "BTCUSDT",
            side=state.setup.direction,
            quantity=state.risk_approval.position_size if state.risk_approval else 0.01,
            entry_zone_low=state.entry_candidate.entry_zone_low if state.entry_candidate else 0,
            entry_zone_high=state.entry_candidate.entry_zone_high if state.entry_candidate else 0,
            ideal_entry=state.entry_candidate.ideal_entry if state.entry_candidate else 0,
            max_chase_atr=state.entry_candidate.max_chase_atr if state.entry_candidate else 0,
            stop_loss=state.sltp.stop_loss if state.sltp else 0,
            take_profit=state.sltp.take_profit if state.sltp else 0,
            position_size_usd=state.risk_approval.position_size * state.entry_candidate.ideal_entry if state.entry_candidate and state.risk_approval else 1000,
            leverage=state.risk_approval.leverage if state.risk_approval else 1.0,
            risk_pct=0.02,
            expected_net_r=state.ev_result.expected_net if state.ev_result else 0,
            execution_adjusted_ev=state.ev_result.execution_adjusted_ev if state.ev_result else 0,
            execution_type=state.entry_candidate.execution_policy if state.entry_candidate else ExecutionType.LIMIT_MAKER,
            fill_probability=state.entry_candidate.fill_probability if state.entry_candidate else 0.7,
            expires_at=datetime.now(timezone.utc).replace(minute=datetime.now().minute + 15),
        )
        
        state.stage = PipelineStage.EXECUTION
        return state
    
    def _build_decision(
        self,
        state: PipelineState,
        status: EntryStatus,
        signal: Literal["BUY", "SELL", "HOLD"],
    ) -> EntryDecision:
        """Build immutable EntryDecision snapshot for audit."""
        
        return EntryDecision(
            status=status,
            signal=signal,
            timestamp=datetime.now(timezone.utc),
            market_context=state.market_context or MarketContext(
                regime=Regime.UNKNOWN, volatility="normal", htf_context="UNKNOWN", atr=0.01, adx=20
            ),
            poi=None,
            setup=state.setup or SetupCandidate(
                setup=SetupType.NONE, confidence=0, reason="none", is_valid=False, regime=Regime.UNKNOWN
            ),
            trigger=state.trigger or TriggerEvent(
                trigger=TriggerType.NONE, is_triggered=False, price=0, reason="none"
            ),
            entry_zone=None,
            confirmation=state.confirmation or ConfirmationResult(passed=False),
            entry_quality=state.quality or EntryQuality(quality=0, zone_score=0, exhaustion_score=0, trigger_score=0),
            sltp=state.sltp,
            expected_value=state.ev_result,
            risk_approval=state.risk_approval,
            order_intent=state.order_intent,
            feature_states=state.feature_states,
            reason_codes=tuple(state.reason_codes),
        )
    
    def _register_default_setups(self):
        """Register all default setup detectors."""
        # Import and register
        from src.strategy.trend_pullback_setup import TrendPullbackSetup
        from src.strategy.breakout_setup import BreakoutSetup
        from src.strategy.mean_reversion_setup import MeanReversionSetup
        
        from src.strategy.setup_bank import DEFAULT_SETUP_BANK
        
        DEFAULT_SETUP_BANK.register(TrendPullbackSetup())
        DEFAULT_SETUP_BANK.register(BreakoutSetup())
        DEFAULT_SETUP_BANK.register(MeanReversionSetup())
        # MomentumContinuationSetup and LiquidityReversalSetup to be added later


# PipelineStage enum (moved here to avoid circular import)
from enum import Enum

class PipelineStage(str, Enum):
    MARKET_CONTEXT = "MARKET_CONTEXT"
    POI_ZONES = "POI_ZONES"
    SETUP_DETECTION = "SETUP_DETECTION"
    TRIGGER_DETECTION = "TRIGGER_DETECTION"
    ENTRY_CANDIDATE = "ENTRY_CANDIDATE"
    CONFIRMATION = "CONFIRMATION"
    ML_FILTER = "ML_FILTER"
    QUALITY = "QUALITY"
    SLTP = "SLTP"
    EV_GATE = "EV_GATE"
    RISK_GATE = "RISK_GATE"
    ORDER_INTENT = "ORDER_INTENT"
    EXECUTION = "EXECUTION"
    COMPLETE = "COMPLETE"