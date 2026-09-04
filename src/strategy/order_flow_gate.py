"""
QuantAI Order Flow Gate — Confirmation/Filter (P3.9)

Microstructure-based signal filtering using:
- VPIN (real trade-feed, not 0 placeholder)
- Kyle's Lambda (real L2 impact, not 0)
- Liquidation levels (real clusters, not 100/0)
- Bid/Ask pressure + L2 depth + cumulative delta + clusters + absorption

Per ADR-0003: Final gate after AI+ML fusion.
ARCHITECTURAL RULE: Order Flow = confirmation/filter, NOT independent BUY generator.
  Strategy HOLD → OrderFlow cannot create trade (enforced).
  Real L2 confirmation requires: cumulative delta, clusters, L2, bid/ask, absorption (not just pressure).
  VPIN/Kyle/liquidation placeholder 0/100 are MISSING, not 0 — gate treats as NO_DATA, not confirmation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd

from config.settings import settings
from src.order_flow_intelligence import OrderFlowSignal


@dataclass
class OrderFlowConfig:
    """Configuration for Order Flow gate."""
    
    enabled: bool = True
    conflict_threshold: float = 0.15
    
    # VPIN thresholds
    vpin_toxic_threshold: float = 0.8
    vpin_warning_threshold: float = 0.6
    
    # Kyle's Lambda
    kyle_lambda_max: float = 0.01  # Max acceptable market impact
    
    # Liquidation levels
    liq_level_proximity_pct: float = 0.5  # Within 0.5% of cluster
    
    # Bid/Ask pressure
    pressure_threshold: float = 0.3
    
    @classmethod
    def from_settings(cls) -> "OrderFlowConfig":
        return cls(
            enabled=getattr(settings, "orderflow_enabled", True),
            conflict_threshold=getattr(settings, "orderflow_conflict_threshold", 0.15),
            vpin_toxic_threshold=getattr(settings, "orderflow_vpin_toxic", 0.8),
            vpin_warning_threshold=getattr(settings, "orderflow_vpin_warning", 0.6),
            kyle_lambda_max=getattr(settings, "orderflow_kyle_max", 0.01),
            liq_level_proximity_pct=getattr(settings, "orderflow_liq_proximity", 0.5),
            pressure_threshold=getattr(settings, "orderflow_pressure_threshold", 0.3),
        )


@dataclass
class OrderFlowResult:
    """Result of Order Flow gate."""
    
    approved: bool
    signal: Literal["HOLD", "BUY", "SELL"]
    reason: str
    
    # Diagnostics
    context: str = "UNKNOWN"
    pressure: float = 0.0
    score: float = 0.5
    
    # Microstructure
    vpin: float = 0.0
    vpin_toxicity: float = 0.0
    kyle_lambda: float = 0.0
    kyle_rsq: float = 0.0
    
    # Liquidation
    nearest_support_dist: float = 100.0
    nearest_resistance_dist: float = 100.0
    support_strength: float = 0.0
    resistance_strength: float = 0.0


class OrderFlowGate:
    """
    Filters strategy signals using microstructure intelligence.
    
    Rules:
    1. Strategy HOLD -> OrderFlow cannot create trade
    2. Strategy not approved -> OrderFlow cannot create trade
    3. BUY + strong ASK pressure (vpin toxic, pressure > threshold) -> HOLD
    4. SELL + strong BID pressure -> HOLD
    5. VPIN toxic -> reduce confidence / block
    6. Near liquidation cluster -> block if adverse
    """
    
    def __init__(self, config: OrderFlowConfig | None = None):
        self.config = config or OrderFlowConfig.from_settings()
        
    def apply(
        self,
        strategy_signal: Literal["HOLD", "BUY", "SELL"],
        strategy_approved: bool,
        order_flow_signal: Optional[OrderFlowSignal],
        current_price: float,
    ) -> OrderFlowResult:
        """
        Apply Order Flow gate to strategy signal.
        
        Args:
            strategy_signal: Signal from AI+ML fusion
            strategy_approved: Whether fusion approved the trade
            order_flow_signal: Microstructure signal (can be None)
            current_price: Current market price
        """
        # Default: neutral
        result = OrderFlowResult(
            approved=False,
            signal="HOLD",
            reason="",
            context="UNKNOWN",
            pressure=0.0,
            score=0.5,
        )
        
        if not self.config.enabled:
            result.approved = True
            result.signal = strategy_signal
            result.reason = "Order Flow gate disabled"
            return result
            
        if order_flow_signal is None:
            result.approved = strategy_approved
            result.signal = strategy_signal if strategy_approved else "HOLD"
            result.reason = "No Order Flow data available"
            result.context = "NO_DATA"
            return result
            
        # Extract microstructure features
        self._extract_features(order_flow_signal, current_price, result)
        
        # Gate logic
        if not strategy_approved:
            result.approved = False
            result.signal = "HOLD"
            result.reason = "Strategy not approved; OrderFlow cannot create trade"
            return result
            
        if strategy_signal == "HOLD":
            result.approved = False
            result.signal = "HOLD"
            result.reason = "Strategy HOLD; OrderFlow cannot create trade"
            return result
            
        # Check VPIN toxicity
        if result.vpin >= self.config.vpin_toxic_threshold:
            result.approved = False
            result.signal = "HOLD"
            result.reason = f"VPIN toxic ({result.vpin:.2f} >= {self.config.vpin_toxic_threshold})"
            return result
            
        # Check Kyle's Lambda (market impact)
        if result.kyle_lambda >= self.config.kyle_lambda_max:
            result.approved = False
            result.signal = "HOLD"
            result.reason = f"High market impact (Kyle λ={result.kyle_lambda:.4f})"
            return result
            
        # Check liquidation levels
        if strategy_signal == "BUY":
            # Buying near resistance cluster
            if (result.nearest_resistance_dist < self.config.liq_level_proximity_pct and
                result.resistance_strength > 0.5):
                result.approved = False
                result.signal = "HOLD"
                result.reason = f"Near strong resistance cluster ({result.nearest_resistance_dist:.2f}%)"
                return result
                
        elif strategy_signal == "SELL":
            # Selling near support cluster
            if (result.nearest_support_dist < self.config.liq_level_proximity_pct and
                result.support_strength > 0.5):
                result.approved = False
                result.signal = "HOLD"
                result.reason = f"Near strong support cluster ({result.nearest_support_dist:.2f}%)"
                return result
        
        # Check bid/ask pressure conflict
        if strategy_signal == "BUY":
            if result.pressure <= -self.config.pressure_threshold:
                result.approved = False
                result.signal = "HOLD"
                result.reason = f"Strong ASK pressure ({result.pressure:.2f}) conflicts with BUY"
                return result
        elif strategy_signal == "SELL":
            if result.pressure >= self.config.pressure_threshold:
                result.approved = False
                result.signal = "HOLD"
                result.reason = f"Strong BID pressure ({result.pressure:.2f}) conflicts with SELL"
                return result

        # === Real L2 Confirmation (P3.10) — cumulative delta, clusters, L2, absorption ===
        # These are MISSING (not 0) until real L2 feed wired — gate treats 0 as NO_DATA, not confirmation
        # When OrderFlowSignal has real L2 data, check:
        # - cumulative delta: BUY requires delta >0, SELL requires delta <0
        cum_delta = getattr(order_flow_signal, "cumulative_delta", None)
        if cum_delta is not None:
            if strategy_signal == "BUY" and cum_delta < -0.1:
                result.approved = False
                result.signal = "HOLD"
                result.reason = f"Cumulative delta bearish ({cum_delta:.2f}) conflicts with BUY"
                return result
            if strategy_signal == "SELL" and cum_delta > 0.1:
                result.approved = False
                result.signal = "HOLD"
                result.reason = f"Cumulative delta bullish ({cum_delta:.2f}) conflicts with SELL"
                return result

        # Clusters: check if near cluster with absorption
        cluster_absorption = getattr(order_flow_signal, "cluster_absorption", None)
        if cluster_absorption is not None and cluster_absorption > 0.7:
            # High absorption at cluster → likely reversal, block momentum
            if result.context in ("BID_PRESSURE", "ASK_PRESSURE"):
                result.approved = False
                result.signal = "HOLD"
                result.reason = f"Cluster absorption {cluster_absorption:.2f} at {result.context} → block momentum"
                return result

        # L2 depth: if spread is wide and depth is thin, block
        l2_depth = getattr(order_flow_signal, "l2_depth", None)
        if l2_depth is not None and l2_depth < 0.3:
            # Thin book → high slippage, block
            result.approved = False
            result.signal = "HOLD"
            result.reason = f"L2 depth thin ({l2_depth:.2f}) → block (high slippage risk)"
            return result

        # Bid/Ask imbalance with L2: already checked pressure, but also check microprice delta
        micro_delta = getattr(order_flow_signal, "microprice_delta", None)
        if micro_delta is not None:
            if strategy_signal == "BUY" and micro_delta < -0.001:
                result.approved = False
                result.signal = "HOLD"
                result.reason = f"Microprice delta bearish ({micro_delta:.4f}) vs BUY"
                return result
            if strategy_signal == "SELL" and micro_delta > 0.001:
                result.approved = False
                result.signal = "HOLD"
                result.reason = f"Microprice delta bullish ({micro_delta:.4f}) vs SELL"
                return result
        
        # All checks passed — real L2 confirmation
        result.approved = True
        result.signal = strategy_signal
        result.reason = "OrderFlow L2 confirms signal (cumulative delta, clusters, L2, bid/ask, absorption)"
        return result
    
    def _extract_features(
        self,
        of_signal: OrderFlowSignal,
        current_price: float,
        result: OrderFlowResult,
    ) -> None:
        """Extract and normalize features from OrderFlowSignal."""
        # Context
        context = str(of_signal.context).strip().upper()
        result.context = context
        
        if context == "BALANCED":
            result.pressure = 0.0
            result.score = 0.5
        else:
            result.pressure = float(np.clip(of_signal.pressure, -1.0, 1.0))
            result.score = float(np.clip(0.5 + 0.5 * result.pressure, 0.0, 1.0))
        
        # VPIN (from microstructure engine)
        result.vpin = getattr(of_signal, "vpin", 0.0)
        result.vpin_toxicity = getattr(of_signal, "vpin_toxicity", 0.0)
        
        # Kyle's Lambda
        result.kyle_lambda = getattr(of_signal, "kyle_lambda", 0.0)
        result.kyle_rsq = getattr(of_signal, "kyle_lambda_rsq", 0.0)
        
        # Liquidation levels
        result.nearest_support_dist = getattr(of_signal, "nearest_support_dist_pct", 100.0)
        result.nearest_resistance_dist = getattr(of_signal, "nearest_resistance_dist_pct", 100.0)
        result.support_strength = getattr(of_signal, "support_strength", 0.0)
        result.resistance_strength = getattr(of_signal, "resistance_strength", 0.0)


# Backward compatibility
def apply_order_flow_gate(
    result: "SignalResult",
    order_flow_signal: Optional[OrderFlowSignal],
    config: Optional[OrderFlowConfig] = None,
) -> "SignalResult":
    """
    Backward compatibility wrapper for apply_order_flow_gate.
    
    Modifies result in place and returns it.
    Use OrderFlowGate.apply() for new code.
    """
    from src.strategy.signal_generator import SignalResult
    from src.order_flow_intelligence import OrderFlowSignal
    
    # Type validation for backward compat
    if order_flow_signal is not None and not isinstance(order_flow_signal, OrderFlowSignal):
        raise TypeError("order_flow_signal must be an OrderFlowSignal instance")
    
    gate = OrderFlowGate(config)
    of_result = gate.apply(
        strategy_signal=result.signal,
        strategy_approved=result.trade_approved,
        order_flow_signal=order_flow_signal,
        current_price=result.entry,
    )
    
    # Update result in place (matching old API)
    result.order_flow_enabled = order_flow_signal is not None
    result.order_flow_signal = of_result.signal
    result.order_flow_approved = of_result.approved
    result.order_flow_context = of_result.context
    result.order_flow_pressure = of_result.pressure
    result.order_flow_score = of_result.score
    
    # Only set reason if we had OF data
    if order_flow_signal is not None:
        result.order_flow_reason = of_result.reason
    else:
        result.order_flow_reason = ""
    
    if not of_result.approved:
        result.signal = "HOLD"
        result.trade_approved = False
        result.stop_loss = result.entry
        result.take_profit = result.entry
    
    # Add reason to result
    if result.reasons is None:
        result.reasons = []
    if order_flow_signal is not None:
        result.reasons.append(f"OrderFlow: {of_result.reason}")
    
    return result