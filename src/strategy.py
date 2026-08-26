"""
QuantAI Professional v5.1
Strategy Engine - Modular Pipeline

Pipeline:
    Market Data
        ↓
    AI Analyzer (Technical Analysis)
        ↓
    Confidence Engine (AI Signal)
        ↓
    ML Engine (ML Prediction)
        ↓
    ML Fusion (AI + ML Combination)
        ↓
    Order Flow Gate (Microstructure Filter)
        ↓
    SL/TP Calculator (Regime-Adaptive)
        ↓
    Final Signal

This module provides backward-compatible API.
Internal implementation uses modular components in src/strategy/.
"""

from __future__ import annotations

# Re-export all public API from modular components
from src.strategy.ai_analyzer import AIAnalyzer, MarketComponents
from src.strategy.ml_fusion import MLFusion, FusionConfig, FusionResult
from src.strategy.order_flow_gate import OrderFlowGate, OrderFlowConfig, OrderFlowResult
from src.strategy.signal_generator import SignalGenerator, SignalConfig, SignalResult, generate_signal_result
from src.strategy.sl_tp_calculator import SLTPCalculator, SLTPConfig, SLTPResult

# Backward compatibility constants (deprecated - use config)
MIN_CONFIDENCE = 60.0
AI_WEIGHT = 0.60
ML_WEIGHT = 0.40
CONFLICT_PENALTY = 0.70
ORDER_FLOW_CONFLICT_THRESHOLD = 0.15

__all__ = [
    # Constants (deprecated)
    "MIN_CONFIDENCE",
    "AI_WEIGHT",
    "ML_WEIGHT",
    "CONFLICT_PENALTY",
    "ORDER_FLOW_CONFLICT_THRESHOLD",
    
    # Modular components
    "AIAnalyzer",
    "MarketComponents",
    "MLFusion",
    "FusionConfig",
    "FusionResult",
    "OrderFlowGate",
    "OrderFlowConfig",
    "OrderFlowResult",
    "SignalGenerator",
    "SignalConfig",
    "SignalResult",
    "generate_signal_result",
    "SLTPCalculator",
    "SLTPConfig",
    "SLTPResult",
]