"""
ENTRY-32/33/35/36 — Confirmation Engine (PHASE 6)

Confirmation based on independent evidence groups:
  STRUCTURE, MOMENTUM, VOLUME, ORDER_FLOW, ML, MTF

Not RSI + MACD + Stochastic = 3 confirmations (same latent factor)
But Momentum group = 1

ENTRY-35: PLACEHOLDER → NOT_AVAILABLE, if mandatory → BLOCK
ENTRY-36: Real L2/Delta pipeline as sub-project
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict
from enum import Enum

from src.entry.models import FeatureState


class EvidenceGroup(str, Enum):
    STRUCTURE = "STRUCTURE"
    MOMENTUM = "MOMENTUM"
    VOLUME = "VOLUME"
    ORDER_FLOW = "ORDER_FLOW"
    ML = "ML"
    MTF = "MTF"


@dataclass
class ConfirmationInput:
    """Inputs to confirmation engine."""
    structure_score: float  # -1 to 1 (trend alignment, structure break, etc.)
    momentum_score: float   # -1 to 1 (single momentum group)
    volume_score: float     # -1 to 1 (volume confirmation)
    order_flow_approved: bool
    order_flow_state: FeatureState
    ml_probability: float
    ml_state: FeatureState
    mtf_aligned: bool
    mtf_state: FeatureState
    
    # Requirements
    require_order_flow: bool = False
    require_ml: bool = False
    min_groups: int = 3


@dataclass
class ConfirmationResult:
    """Confirmation result with full audit trail."""
    passed: bool
    groups_passed: int
    groups_required: int
    scores: Dict[str, float]
    states: Dict[str, FeatureState]
    reason: str
    blocked_reasons: list[str] = field(default_factory=list)


class ConfirmationEngine:
    """
    ENTRY-32: Confirmation based on independent groups.
    ENTRY-33: Independence — RSI+MACD+Stochastic = 1 group, not 3.
    ENTRY-35: PLACEHOLDER → NOT_AVAILABLE → if mandatory → BLOCK
    ENTRY-36: Real L2/Delta pipeline as sub-project (external)
    """
    
    def __init__(
        self,
        min_groups: int = 3,
        require_order_flow: bool = False,
        require_ml: bool = False,
    ):
        self.default_min_groups = min_groups
        self.default_require_order_flow = require_order_flow
        self.default_require_ml = require_ml
    
    def evaluate(self, inp: ConfirmationInput) -> ConfirmationResult:
        """
        Evaluate 6 independent groups.
        
        Rules:
        - PLACEHOLDER features NEVER count as passed
        - If a mandatory feature is PLACEHOLDER/UNAVAILABLE → BLOCK
        - Groups passed = count of REAL/SIMULATED/PROXY groups that pass
        """
        groups = {}
        states = {}
        passed = 0
        blocked_reasons = []
        
        min_groups = inp.min_groups if inp.min_groups > 0 else self.default_min_groups
        require_of = inp.require_order_flow or self.default_require_order_flow
        require_ml = inp.require_ml or self.default_require_ml
        
        # ===== STRUCTURE GROUP =====
        struct_pass = abs(inp.structure_score) > 0.3
        groups["STRUCTURE"] = float(inp.structure_score)
        states["STRUCTURE"] = FeatureState.REAL
        if struct_pass:
            passed += 1
        
        # ===== MOMENTUM GROUP =====
        # Single group: RSI/MACD/Stochastic already combined into ONE score
        mom_pass = abs(inp.momentum_score) > 0.2
        groups["MOMENTUM"] = float(inp.momentum_score)
        states["MOMENTUM"] = FeatureState.REAL
        if mom_pass:
            passed += 1
        
        # ===== VOLUME GROUP =====
        vol_pass = abs(inp.volume_score) > 0.3
        groups["VOLUME"] = float(inp.volume_score)
        states["VOLUME"] = FeatureState.REAL
        if vol_pass:
            passed += 1
        
        # ===== ORDER_FLOW GROUP =====
        of_state = inp.order_flow_state
        states["ORDER_FLOW"] = of_state
        
        if of_state == FeatureState.PLACEHOLDER:
            # ENTRY-35: PLACEHOLDER → UNAVAILABLE
            states["ORDER_FLOW"] = FeatureState.UNAVAILABLE
            of_state = FeatureState.UNAVAILABLE
        
        if of_state in (FeatureState.PLACEHOLDER, FeatureState.UNAVAILABLE):
            if require_of:
                # MANDATORY but unavailable → BLOCK
                blocked_reasons.append("ORDER_FLOW_MANDATORY_BUT_UNAVAILABLE")
                groups["ORDER_FLOW"] = 0.0
            else:
                # Not required, skip (does not count toward min_groups)
                groups["ORDER_FLOW"] = 0.0
        else:
            # REAL, SIMULATED, or PROXY
            groups["ORDER_FLOW"] = 1.0 if inp.order_flow_approved else 0.0
            if inp.order_flow_approved:
                passed += 1
        
        # ===== ML GROUP =====
        ml_state = inp.ml_state
        states["ML"] = ml_state
        
        if ml_state == FeatureState.PLACEHOLDER:
            # PLACEHOLDER → UNAVAILABLE
            states["ML"] = FeatureState.UNAVAILABLE
            ml_state = FeatureState.UNAVAILABLE
        
        ml_pass = inp.ml_probability >= 0.55
        
        if ml_state in (FeatureState.PLACEHOLDER, FeatureState.UNAVAILABLE):
            if require_ml:
                blocked_reasons.append("ML_MANDATORY_BUT_UNAVAILABLE")
                groups["ML"] = 0.0
            else:
                groups["ML"] = 0.0
        else:
            # REAL or SIMULATED (calibrated)
            groups["ML"] = inp.ml_probability
            if ml_pass:
                passed += 1
        
        # ===== MTF GROUP =====
        mtf_state = inp.mtf_state
        states["MTF"] = mtf_state
        
        mtf_pass = inp.mtf_aligned
        if mtf_pass:
            passed += 1
        groups["MTF"] = 1.0 if mtf_pass else 0.0
        
        # ===== OVERALL DECISION =====
        # If any mandatory feature blocked → overall FAIL
        if blocked_reasons:
            overall = False
            reason = f"BLOCKED: {', '.join(blocked_reasons)}"
        else:
            overall = passed >= min_groups
            reason = f"{passed}/{min_groups} groups passed"
        
        # Add placeholder warnings
        if inp.order_flow_state == FeatureState.PLACEHOLDER:
            reason += " (ORDER_FLOW: PLACEHOLDER→UNAVAILABLE)"
        if inp.ml_state == FeatureState.PLACEHOLDER:
            reason += " (ML: PLACEHOLDER→UNAVAILABLE)"
        
        return ConfirmationResult(
            passed=overall,
            groups_passed=passed,
            groups_required=min_groups,
            scores=groups,
            states=states,
            reason=reason,
            blocked_reasons=blocked_reasons,
        )
    
    def confirm(self, ctx: dict) -> ConfirmationResult:
        """
        Convenience method for EntryEngine.
        Extracts inputs from context dict.
        """
        inp = ConfirmationInput(
            structure_score=ctx.get("structure_score", 0.0),
            momentum_score=ctx.get("momentum_score", 0.0),
            volume_score=ctx.get("volume_score", 0.0),
            order_flow_approved=ctx.get("order_flow_approved", False),
            order_flow_state=ctx.get("order_flow_state", FeatureState.UNAVAILABLE),
            ml_probability=ctx.get("ml_probability", 0.5),
            ml_state=ctx.get("ml_state", FeatureState.UNAVAILABLE),
            mtf_aligned=ctx.get("mtf_aligned", False),
            mtf_state=ctx.get("mtf_state", FeatureState.UNAVAILABLE),
            require_order_flow=ctx.get("require_order_flow", False),
            require_ml=ctx.get("require_ml", False),
            min_groups=ctx.get("min_groups", self.default_min_groups),
        )
        return self.evaluate(inp)