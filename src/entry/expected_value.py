"""
ENTRY-48/49/50/51/52 — Expected Value Engine (PHASE 10)

Net EV = gross outcome - fees - spread - slippage - funding
Execution-adjusted EV: models fill_probability
Sensitivity: 1x, 1.5x, 2x, 3x costs
EV Gate: EV <= threshold → EV_TOO_LOW
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import math


class EVResultStatus(str, Enum):
    PASS = "PASS"
    EV_TOO_LOW = "EV_TOO_LOW"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    NEGATIVE_EV = "NEGATIVE_EV"


@dataclass
class EVInputs:
    """Inputs for EV calculation."""
    # Probabilities (from ML + historical)
    p_win: float  # P(tp hit first)
    p_loss: float  # P(sl hit first)
    p_timeout: float  # P(neither hit within time limit)
    
    # Expected outcomes (R-multiples)
    expected_win: float  # R-multiple if win (e.g., 2.3)
    expected_loss: float  # R-multiple if loss (e.g., -1.0)
    expected_timeout: float  # R-multiple if timeout (usually small negative)
    
    # Costs (in R-multiples or absolute price)
    fees_per_side: float  # e.g., 0.0004 (0.04%)
    spread_bps: float  # e.g., 1.0 bp
    slippage_bps: float  # expected slippage in bps
    funding_bps_per_8h: float  # funding cost per 8h in bps
    expected_hold_hours: float  # expected hold time
    
    # Execution
    fill_probability: float = 1.0  # P(limit order fills)
    execution_policy: str = "LIMIT_MAKER"  # LIMIT_MAKER, LIMIT, MARKET
    
    # Position
    position_size_usd: float = 1000.0  # for absolute cost calc


@dataclass
class EVBreakdown:
    """Detailed EV breakdown for audit."""
    gross_ev: float
    fees_cost: float
    spread_cost: float
    slippage_cost: float
    funding_cost: float
    total_costs: float
    net_ev: float
    execution_adjusted_ev: float  # net_ev * fill_probability
    
    # Sensitivity
    sensitivity: dict[str, float] = field(default_factory=dict)  # "1x", "1.5x", "2x", "3x" -> net_ev


@dataclass
class EVResult:
    """Final EV gate result."""
    status: EVResultStatus
    ev_breakdown: EVBreakdown
    threshold: float
    reason: str
    reason_codes: list[str] = field(default_factory=list)


class ExpectedValueEngine:
    """
    ENTRY-48/49/50/51: Full EV calculation.
    
    Net EV = P(win)*win + P(loss)*loss + P(timeout)*timeout
             - fees - spread - slippage - funding
    
    Execution-adjusted EV = Net EV * fill_probability
    (because good price + no fill != good trade)
    """
    
    def __init__(self, min_ev_threshold: float = 0.05, min_fill_probability: float = 0.3):
        """
        min_ev_threshold: minimum net EV in R-multiples to pass (default 0.05R)
        min_fill_probability: minimum fill probability for limit orders
        """
        self.min_ev_threshold = min_ev_threshold
        self.min_fill_probability = min_fill_probability
    
    def calculate(self, inputs: EVInputs) -> EVResult:
        """Calculate EV with full breakdown and sensitivity."""
        
        # 1. Gross EV (before costs)
        gross_ev = (
            inputs.p_win * inputs.expected_win +
            inputs.p_loss * inputs.expected_loss +
            inputs.p_timeout * inputs.expected_timeout
        )
        
        # 2. Costs in R-multiples
        # Fees: 2 sides (entry + exit)
        fees_cost = 2 * inputs.fees_per_side
        
        # Spread: paid on entry (cross spread) or saved (maker)
        if inputs.execution_policy == "LIMIT_MAKER":
            spread_cost = -inputs.spread_bps / 10000  # earn half spread approx
        elif inputs.execution_policy == "LIMIT":
            spread_cost = 0  # at mid
        else:  # MARKET
            spread_cost = inputs.spread_bps / 10000
        
        # Slippage
        slippage_cost = inputs.slippage_bps / 10000
        
        # Funding
        funding_periods = inputs.expected_hold_hours / 8.0
        funding_cost = funding_periods * inputs.funding_bps_per_8h / 10000
        
        total_costs = fees_cost + spread_cost + slippage_cost + funding_cost
        
        # 3. Net EV
        net_ev = gross_ev - total_costs
        
        # 4. Execution-adjusted EV
        execution_adjusted_ev = net_ev * inputs.fill_probability
        
        # 5. Sensitivity analysis (1x, 1.5x, 2x, 3x costs)
        sensitivity = {}
        for mult in [1.0, 1.5, 2.0, 3.0]:
            sens_costs = fees_cost * mult + spread_cost * mult + slippage_cost * mult + funding_cost
            sensitivity[f"{mult}x"] = gross_ev - sens_costs
        
        breakdown = EVBreakdown(
            gross_ev=gross_ev,
            fees_cost=fees_cost,
            spread_cost=spread_cost,
            slippage_cost=slippage_cost,
            funding_cost=funding_cost,
            total_costs=total_costs,
            net_ev=net_ev,
            execution_adjusted_ev=execution_adjusted_ev,
            sensitivity=sensitivity,
        )
        
        # 6. Gate decision
        reason_codes = []
        if net_ev <= 0:
            status = EVResultStatus.NEGATIVE_EV
            reason = f"Negative net EV: {net_ev:.4f}R"
            reason_codes.append("NEGATIVE_EV")
        elif net_ev < self.min_ev_threshold:
            status = EVResultStatus.EV_TOO_LOW
            reason = f"Net EV {net_ev:.4f}R < threshold {self.min_ev_threshold:.4f}R"
            reason_codes.append("EV_TOO_LOW")
        elif inputs.fill_probability < self.min_fill_probability and inputs.execution_policy != "MARKET":
            status = EVResultStatus.EV_TOO_LOW
            reason = f"Fill probability {inputs.fill_probability:.1%} < min {self.min_fill_probability:.1%}"
            reason_codes.append("LOW_FILL_PROB")
        else:
            status = EVResultStatus.PASS
            reason = f"EV PASS: net={net_ev:.4f}R, exec_adj={execution_adjusted_ev:.4f}R"
            reason_codes.append("EV_PASS")
        
        # Check sensitivity
        if sensitivity["3x"] <= 0:
            reason_codes.append("FRAGILE_AT_3X_COSTS")
        
        return EVResult(
            status=status,
            ev_breakdown=breakdown,
            threshold=self.min_ev_threshold,
            reason=reason,
            reason_codes=reason_codes,
        )


def create_ev_inputs_from_candidate(
    entry_candidate: Any,  # EntryCandidate
    ml_probability: float,
    market_data: dict,
    risk_policy: Any = None,
) -> EVInputs:
    """
    Helper to create EVInputs from EntryCandidate + ML + market data.
    """
    # Win/loss probabilities from ML (calibrated)
    p_win = ml_probability
    p_loss = 1.0 - ml_probability - 0.05  # 5% timeout
    p_timeout = 0.05
    
    # Expected R-multiples from SL/TP
    risk_dist = entry_candidate.risk_distance
    reward_dist = entry_candidate.tp_candidate - entry_candidate.ideal_entry if entry_candidate.direction == "LONG" else entry_candidate.ideal_entry - entry_candidate.tp_candidate
    
    expected_win = reward_dist / risk_dist if risk_dist > 0 else 0
    expected_loss = -1.0  # 1R loss
    expected_timeout = -0.2  # small timeout cost
    
    # Market costs
    fees_per_side = market_data.get("fees_per_side", 0.0004)
    spread_bps = market_data.get("spread_bps", 1.0)
    slippage_bps = market_data.get("expected_slippage_bps", 2.0)
    funding_bps = market_data.get("funding_bps_8h", 0.01)
    expected_hold_hours = market_data.get("expected_hold_hours", 24.0)
    
    # Fill probability based on entry zone and policy
    entry_policy = market_data.get("execution_policy", "LIMIT_MAKER")
    chase_atr = entry_candidate.max_chase_atr
    # Simple model: closer to ideal = higher fill prob
    fill_prob = max(0.2, 1.0 - chase_atr * 0.3)
    
    return EVInputs(
        p_win=p_win,
        p_loss=p_loss,
        p_timeout=p_timeout,
        expected_win=expected_win,
        expected_loss=expected_loss,
        expected_timeout=expected_timeout,
        fees_per_side=fees_per_side,
        spread_bps=spread_bps,
        slippage_bps=slippage_bps,
        funding_bps_per_8h=funding_bps,
        expected_hold_hours=expected_hold_hours,
        fill_probability=fill_prob,
        execution_policy=entry_policy,
        position_size_usd=market_data.get("position_size_usd", 1000.0),
    )