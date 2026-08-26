"""
QuantAI Strategy Package v2.0

Modular strategy pipeline:
- AI Analyzer: Technical analysis (trend, momentum, volume, volatility)
- ML Fusion: AI + ML signal combination with configurable rules
- Order Flow Gate: Microstructure filtering
- Signal Generator: Final signal assembly
- SL/TP Calculator: Regime-adaptive stop loss / take profit
"""

from .ai_analyzer import AIAnalyzer, MarketComponents
from .ml_fusion import MLFusion, FusionConfig, FusionResult
from .order_flow_gate import OrderFlowGate, OrderFlowConfig, OrderFlowResult, apply_order_flow_gate
from .signal_generator import SignalGenerator, SignalConfig, SignalResult, generate_signal_result
from .sl_tp_calculator import SLTPCalculator, SLTPConfig, SLTPResult


# Backward compatibility constants (deprecated - use config)
MIN_CONFIDENCE = 60.0
AI_WEIGHT = 0.60
ML_WEIGHT = 0.40
CONFLICT_PENALTY = 0.70
ORDER_FLOW_CONFLICT_THRESHOLD = 0.15


# Backward compatibility: build_features (from src.feature_engine)
from src.feature_engine import build_features


# Backward compatibility: AI_MODEL (used in tests)
AI_MODEL = None


# Backward compatibility: MarketEngine (used in tests)
from .signal_generator import SignalConfig as MarketEngine


# Backward compatibility: fuse_ai_ml function (returns tuple for old API)
from .ml_fusion import fuse_ai_ml as _fuse_ai_ml

def fuse_ai_ml(
    ai_signal: str,
    ai_confidence: float,
    ml_signal: str,
    ml_probability: float,
) -> tuple[str, float, bool, str]:
    """
    Backward compatibility: fuse_ai_ml returning tuple.
    
    Old API returned: (signal, confidence, approved, reason)
    New API returns FusionResult object.
    """
    from .ml_fusion import fuse_ai_ml as _fuse, FusionConfig
    result = _fuse(
        ai_signal=ai_signal,
        ai_confidence=ai_confidence,
        ml_signal=ml_signal,
        ml_probability=ml_probability,
        config=None,
    )
    return (result.signal, result.combined_confidence, result.approved, result.reason)


# Backward compatibility: evaluate_market function
from .ai_analyzer import AIAnalyzer as _AIAnalyzer
def evaluate_market(df):
    """Backward compatibility: evaluate_market -> AIAnalyzer.analyze()"""
    analyzer = _AIAnalyzer()
    return analyzer.analyze(df)


# Backward compatibility: predict_ml function
def predict_ml(df, model=None):
    """Backward compatibility: predict_ml(df) -> (signal, probability, probabilities)"""
    from src.ml_engine import MLEngine, MLConfig
    from src.strategy import build_features
    
    if model is None:
        return ("HOLD", 0.0, {})
    
    try:
        features = build_features(df)
        import pandas as pd
        X = pd.DataFrame([features])
        
        # Create a temporary engine with the model
        engine = MLEngine(MLConfig())
        engine.model = model
        engine.feature_names = list(features.keys())
        
        probas = engine.predict_probabilities(X)
        signal = max(probas, key=probas.get)
        prob = probas[signal]
        
        return (signal, prob, probas)
    except Exception:
        return ("HOLD", 0.0, {})


# Backward compatibility: MarketEngine
from .signal_generator import SignalConfig as MarketEngine


__all__ = [
    "AIAnalyzer",
    "MarketComponents",
    "MLFusion",
    "FusionConfig",
    "FusionResult",
    "OrderFlowGate",
    "OrderFlowConfig",
    "OrderFlowResult",
    "apply_order_flow_gate",
    "SignalGenerator",
    "SignalConfig",
    "SignalResult",
    "generate_signal_result",
    "SLTPCalculator",
    "SLTPConfig",
    "SLTPResult",
    # Backward compat
    "MarketEngine",
    "fuse_ai_ml",
    "evaluate_market",
    "predict_ml",
    "build_features",
    "MIN_CONFIDENCE",
    "AI_WEIGHT",
    "ML_WEIGHT",
    "CONFLICT_PENALTY",
    "ORDER_FLOW_CONFLICT_THRESHOLD",
]