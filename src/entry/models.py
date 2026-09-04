"""
ENTRY-00/01 — QuantAI Entry Contract v1 (Milestone A)

12 Core Types:
1. MarketContext
2. SetupCandidate
3. TriggerEvent
4. ConfirmationResult
5. EntryZone
6. ExpectedValueResult
7. RiskApproval
8. EntryDecision
9. OrderIntent
10. EntryLifecycle
11. FeatureState
12. Signal (for adapter compatibility)

Four distinct objects (Rule 13):
1. Signal — potential idea (from SignalGenerator)
2. EntryCandidate — concrete trading situation (assembled by EntryEngine)
3. OrderIntent — system wants to open position (after EV+Risk approval)
4. Order — execution layer command (sent to exchange)

EntryDecision = immutable audit snapshot of the full decision chain
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Literal, Any
from datetime import datetime, timezone
from enum import Enum

# Import lifecycle state from lifecycle module
from src.entry.lifecycle import LifecycleState


# ===== ENUMS =====
class Regime(str, Enum):
    TREND_UP = "TREND_UP"
    TREND_DOWN = "TREND_DOWN"
    RANGE = "RANGE"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    TRANSITION = "TRANSITION"
    UNKNOWN = "UNKNOWN"


class SetupType(str, Enum):
    NONE = "NONE"
    TREND_PULLBACK = "TREND_PULLBACK"
    BREAKOUT = "BREAKOUT"
    MEAN_REVERSION = "MEAN_REVERSION"
    MOMENTUM_CONTINUATION = "MOMENTUM_CONTINUATION"
    LIQUIDITY_REVERSAL = "LIQUIDITY_REVERSAL"


class TriggerType(str, Enum):
    NONE = "NONE"
    REJECTION = "REJECTION"
    SWEEP = "SWEEP"
    MSB = "MSB"
    BREAKOUT_CONFIRMATION = "BREAKOUT_CONFIRMATION"
    RETEST = "RETEST"


class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class EntryStatus(str, Enum):
    NO_SETUP = "NO_SETUP"
    WAIT_TRIGGER = "WAIT_TRIGGER"
    WAIT_CONFIRMATION = "WAIT_CONFIRMATION"
    ML_BLOCKED = "ML_BLOCKED"
    EV_TOO_LOW = "EV_TOO_LOW"
    RISK_BLOCKED = "RISK_BLOCKED"
    EXPIRED = "EXPIRED"
    INVALIDATED = "INVALIDATED"
    ENTRY_APPROVED = "ENTRY_APPROVED"


class FeatureState(str, Enum):
    """ENTRY-12: Audit trail for each feature."""
    REAL = "REAL"           # Real exchange data
    SIMULATED = "SIMULATED" # Simulated but calibrated
    PROXY = "PROXY"         # Proxy (e.g., volume → OI)
    PLACEHOLDER = "PLACEHOLDER"  # Not implemented, returns default
    UNAVAILABLE = "UNAVAILABLE"  # Exchange doesn't provide


# ===== 10. ENTRY LIFECYCLE — Immutable snapshot for contract =====
@dataclass(frozen=True)
class EntryLifecycle:
    """
    ENTRY-56/10: Immutable lifecycle snapshot for audit trail.
    Tracks the 14-state lifecycle of an entry candidate.
    """
    candidate_id: str
    state: LifecycleState
    bars_since_setup: int
    bars_since_trigger: int
    created_at: datetime
    invalidated_reason: str = ""
    expired_reason: str = ""
    
    @property
    def is_terminal(self) -> bool:
        return self.state in (
            LifecycleState.APPROVED, LifecycleState.ORDER_SUBMITTED, 
            LifecycleState.FILLED, LifecycleState.EXPIRED, 
            LifecycleState.INVALIDATED, LifecycleState.REJECTED, 
            LifecycleState.CLOSED
        )


class ExecutionType(str, Enum):
    LIMIT_MAKER = "LIMIT_MAKER"  # Post-only, maker fee
    LIMIT = "LIMIT"              # Limit, may take
    MARKET = "MARKET"            # Market, taker fee


# ===== 1. SIGNAL — Potential idea (from SignalGenerator) =====
@dataclass(frozen=True)
class Signal:
    """
    Signal — Output of legacy SignalGenerator or new Setup/Trigger/ML.
    Just a potential idea, NOT a trading decision.
    """
    signal: Literal["BUY", "SELL", "HOLD"]
    confidence: float  # 0-1
    source: str  # "LEGACY_SIGNAL_GENERATOR" or "SETUP_BREAKOUT" etc.
    reason: str
    metadata: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ===== 2. ENTRY CANDIDATE — Concrete trading situation (CENTRAL ENTITY) =====
@dataclass(frozen=True)
class EntryCandidate:
    """
    ENTRY-11: Central entity of the new architecture.
    Contains EVERYTHING needed for a trading decision.
    Same structure for Breakout, Pullback, MeanRev, LiquidityReversal.
    """
    # Identity
    candidate_id: str
    symbol: str
    timestamp: datetime
    
    # Market context
    regime: Regime
    regime_confidence: float
    htf_context: dict  # direction, trend_strength, volatility, structure, confidence
    
    # Setup
    setup_type: SetupType
    setup_direction: Direction
    setup_quality: float  # 0-1
    setup_confidence: float
    setup_reason: str
    
    # POI / Zone
    poi_type: str
    poi_price: float
    poi_strength: float
    poi_distance_pct: float
    
    # Entry Zone
    entry_zone_low: float
    entry_zone_high: float
    ideal_entry: float
    max_chase_atr: float
    atr: float
    
    # Trigger
    trigger_type: TriggerType
    trigger_triggered: bool
    trigger_price: float
    trigger_reason: str
    
    # SL/TP (preliminary)
    sl_candidate: float
    tp_candidate: float
    
    # ML
    ml_probability: float
    ml_state: FeatureState
    ml_setup_specific: bool  # True if P(success|SETUP_TYPE)
    
    # Confirmation
    structure_score: float
    momentum_score: float
    volume_score: float
    order_flow_state: FeatureState
    order_flow_passed: bool
    mtf_passed: bool
    
    # Quality
    quality_score: float
    quality_reason: str
    quality_codes: tuple[str, ...]
    
    # EV (preliminary, before execution adjustment)
    expected_win_r: float
    expected_loss_r: float
    p_win: float
    p_loss: float
    p_timeout: float
    
    # Costs
    fees_bps: float
    spread_bps: float
    slippage_bps: float
    funding_bps_8h: float
    expected_hold_hours: float
    fill_probability: float
    execution_policy: ExecutionType
    
    # Feature states for audit
    feature_states: dict[str, FeatureState] = field(default_factory=dict)
    
    @property
    def risk_distance(self) -> float:
        if self.setup_direction == Direction.LONG:
            return self.ideal_entry - self.sl_candidate
        return self.sl_candidate - self.ideal_entry
    
    @property
    def reward_distance(self) -> float:
        if self.setup_direction == Direction.LONG:
            return self.tp_candidate - self.ideal_entry
        return self.ideal_entry - self.tp_candidate
    
    @property
    def rr_ratio(self) -> float:
        rd = self.risk_distance
        return self.reward_distance / rd if rd > 0 else 0
    
    def to_audit_dict(self) -> dict:
        """ENTRY-12: Full audit trail for Supervisor."""
        return {
            "REGIME": f"{self.regime.value} / confidence {self.regime_confidence:.2f}",
            "SETUP": f"{self.setup_type.value} / quality {self.setup_quality:.2f}",
            "TRIGGER": f"{self.trigger_type.value} / triggered={self.trigger_triggered}",
            "POI": f"{self.poi_type} / strength {self.poi_strength:.2f}",
            "ML": f"P(success)={self.ml_probability:.2f} / state={self.ml_state.value}",
            "ORDERFLOW": f"{self.order_flow_state.value} / passed={self.order_flow_passed}",
            "EV": f"gross={self._gross_ev():.4f}R",
            "RISK": f"RR={self.rr_ratio:.2f}",
            "FEATURE_STATES": {k: v.value for k, v in self.feature_states.items()},
        }
    
    def _gross_ev(self) -> float:
        return (self.p_win * self.expected_win_r +
                self.p_loss * self.expected_loss_r +
                self.p_timeout * -0.2)


# ===== 3. ORDER INTENT — System wants to open position =====
@dataclass(frozen=True)
class OrderIntent:
    """
    ENTRY-13: After EntryCandidate passes EV + Risk gates.
    This is what gets sent to Execution layer.
    """
    intent_id: str
    candidate_id: str
    symbol: str
    side: Direction
    quantity: float
    
    # Entry
    entry_zone_low: float
    entry_zone_high: float
    ideal_entry: float
    max_chase_atr: float
    
    # Exit
    stop_loss: float
    take_profit: float
    
    # Risk
    position_size_usd: float
    leverage: float
    risk_pct: float
    
    # EV
    expected_net_r: float
    execution_adjusted_ev: float
    
    # Execution
    execution_type: ExecutionType
    fill_probability: float
    
    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime | None = None  # e.g., +15 min
    
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at


# ===== 4. ORDER — Execution layer command =====
@dataclass(frozen=True)
class Order:
    """
    Sent to exchange. Minimal fields.
    """
    order_id: str
    intent_id: str
    symbol: str
    side: Literal["BUY", "SELL"]
    order_type: Literal["LIMIT", "MARKET", "STOP_LIMIT"]
    quantity: float
    price: float | None = None  # None for MARKET
    stop_price: float | None = None
    time_in_force: str = "GTC"  # GTC, IOC, FOK
    post_only: bool = False
    reduce_only: bool = False
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ===== SUPPORTING MODELS =====
@dataclass(frozen=True)
class MarketContext:
    regime: Regime
    volatility: str  # high/normal/low
    htf_context: str
    atr: float
    adx: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def is_unknown(self) -> bool:
        return self.regime == Regime.UNKNOWN


@dataclass(frozen=True)
class POI:
    zone_type: str
    price: float
    upper: float
    lower: float
    strength: float
    distance_pct: float


@dataclass(frozen=True)
class SetupCandidate:
    setup: SetupType
    confidence: float
    reason: str
    is_valid: bool
    regime: Regime
    quality: float = 0.0


@dataclass(frozen=True)
class TriggerEvent:
    trigger: TriggerType
    is_triggered: bool
    price: float
    reason: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class ConfirmationResult:
    passed: bool
    structure_score: float = 0.0
    momentum_score: float = 0.0
    volume_score: float = 0.0
    order_flow_approved: bool = False
    order_flow_state: FeatureState = FeatureState.UNAVAILABLE
    mtf_approved: bool = False
    ml_decision: str = "HOLD"
    ml_probability: float = 0.0
    ml_state: FeatureState = FeatureState.UNAVAILABLE
    reason: str = ""


@dataclass(frozen=True)
class EntryQuality:
    quality: float
    zone_score: float
    exhaustion_score: float
    trigger_score: float
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class EntryZone:
    zone_low: float
    zone_high: float
    ideal_entry: float
    max_chase_distance: float
    atr: float
    setup: SetupType
    
    def contains(self, price: float) -> bool:
        return self.zone_low <= price <= self.zone_high


@dataclass(frozen=True)
class SLTPCandidate:
    stop_loss: float
    take_profit: float
    sl_distance: float
    tp_distance: float
    rr: float
    method: str
    reason: str


@dataclass(frozen=True)
class ExpectedValueResult:
    expected_net: float
    p_win: float
    p_loss: float
    expected_payoff: float
    total_costs: float
    hurdle: float
    passed: bool
    reason: str
    execution_adjusted_ev: float = 0.0
    sensitivity: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RiskApproval:
    approved: bool
    reason: str
    position_size: float = 0.0
    leverage: float = 1.0
    exposure_ok: bool = False


# ===== ENTRY DECISION — Immutable audit snapshot =====
@dataclass(frozen=True)
class EntryDecision:
    """
    Immutable snapshot — final decision with full context.
    For audit trail and Supervisor feedback.
    """
    status: EntryStatus
    signal: Literal["BUY", "SELL", "HOLD"]
    timestamp: datetime
    
    # Full chain for audit
    market_context: MarketContext
    poi: Optional[POI]
    setup: SetupCandidate
    trigger: TriggerEvent
    entry_zone: Optional[EntryZone]
    confirmation: ConfirmationResult
    entry_quality: EntryQuality
    sltp: Optional[SLTPCandidate]
    expected_value: Optional[ExpectedValueResult]
    risk_approval: Optional[RiskApproval]
    order_intent: Optional[OrderIntent] = None
    
    # Feature states for audit (especially microstructure placeholders)
    feature_states: dict = field(default_factory=dict)
    
    # Reason codes for Supervisor
    reason_codes: tuple[str, ...] = ()
    
    def is_approved(self) -> bool:
        return self.status == EntryStatus.ENTRY_APPROVED and self.signal in ("BUY", "SELL")
    
    def to_audit_dict(self) -> dict:
        """ENTRY-12: WHY DID THE SYSTEM WANT TO ENTER?"""
        return {
            "status": self.status.value,
            "signal": self.signal,
            "timestamp": self.timestamp.isoformat(),
            "REGIME": f"{self.market_context.regime.value} / confidence {self._get_regime_conf():.2f}",
            "SETUP": f"{self.setup.setup.value} / quality {self.setup.quality:.2f}",
            "TRIGGER": f"{self.trigger.trigger.value} / triggered={self.trigger.is_triggered}",
            "POI": f"{self.poi.zone_type if self.poi else 'NONE'} / strength {self.poi.strength if self.poi else 0:.2f}",
            "ML": f"P(success)={self.confirmation.ml_probability:.2f} / {self.confirmation.ml_state.value}",
            "ORDERFLOW": f"{self.confirmation.order_flow_state.value} / passed={self.confirmation.order_flow_approved}",
            "EV": f"net={self.expected_value.expected_net if self.expected_value else 0:.4f}R",
            "RISK": f"approved={self.risk_approval.approved if self.risk_approval else False}",
            "REASON_CODES": list(self.reason_codes),
            "FEATURE_STATES": {k: v.value for k, v in self.feature_states.items()},
        }
    
    def _get_regime_conf(self) -> float:
        return self.market_context.__dict__.get("regime_confidence", 0.5)
    
    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "signal": self.signal,
            "timestamp": self.timestamp.isoformat(),
            "setup": self.setup.setup.value,
            "trigger": self.trigger.trigger.value,
            "entry_zone": [self.entry_zone.zone_low, self.entry_zone.zone_high] if self.entry_zone else None,
            "sl": self.sltp.stop_loss if self.sltp else None,
            "tp": self.sltp.take_profit if self.sltp else None,
            "ev": self.expected_value.expected_net if self.expected_value else None,
            "risk_approved": self.risk_approval.approved if self.risk_approval else False,
            "feature_states": {k: v.value for k, v in self.feature_states.items()},
        }