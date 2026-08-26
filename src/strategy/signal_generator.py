"""
QuantAI Signal Generator

Assembles final SignalResult from all strategy components:
- AI Analyzer (technical components)
- Confidence Engine (AI signal + confidence)
- ML Engine (ML prediction)
- ML Fusion (AI+ML combination)
- Order Flow Gate (microstructure filter)
- SL/TP Calculator (regime-adaptive levels)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import pandas as pd

from config.settings import settings
from src.confidence_engine import ConfidenceEngine, WeightedGate, WeightedGateConfig
from src.ml_engine import MLEngine, MLConfig
from src.model_manager import ModelManager
from src.order_flow_intelligence import OrderFlowSignal
from src.risk_manager import calculate_sl_tp
from src.feature_engine import build_features
from src.liquidation_intelligence import LiquidationIntelligenceEngine, LiquidationSignal
from src.strategy.ai_analyzer import AIAnalyzer, MarketComponents
from src.strategy.ml_fusion import MLFusion, FusionConfig, FusionResult
from src.strategy.order_flow_gate import OrderFlowGate, OrderFlowConfig, OrderFlowResult
from src.strategy.sl_tp_calculator import SLTPCalculator, SLTPConfig, SLTPResult


@dataclass
class SignalConfig:
    """Configuration for signal generation."""
    
    min_confidence: float = 60.0
    use_ml: bool = False
    use_order_flow: bool = True
    use_weighted_gate: bool = True
    ml_model_path: str = "models/quantai_v5.pkl"
    
    # Confidence engine weights
    trend_weight: float = 1.50
    momentum_weight: float = 1.20
    volume_weight: float = 1.10
    volatility_weight: float = 1.00
    
    # Weighted Gate config
    weighted_gate_threshold: float = 0.75
    weighted_gate_min_confidence: float = 60.0
    weighted_gate_long_threshold: float = 0.55
    weighted_gate_short_threshold: float = 0.55
    
    @classmethod
    def from_settings(cls) -> "SignalConfig":
        return cls(
            min_confidence=getattr(settings, "strategy_min_confidence", 60.0),
            use_ml=getattr(settings, "ml_enabled", False),
            use_order_flow=getattr(settings, "orderflow_enabled", True),
            use_weighted_gate=getattr(settings, "strategy_use_weighted_gate", True),
            ml_model_path=getattr(settings, "ml_model_path", "models/quantai_v5.pkl"),
            trend_weight=getattr(settings, "confidence_trend_weight", 1.50),
            momentum_weight=getattr(settings, "confidence_momentum_weight", 1.20),
            volume_weight=getattr(settings, "confidence_volume_weight", 1.10),
            volatility_weight=getattr(settings, "confidence_volatility_weight", 1.00),
            weighted_gate_threshold=getattr(settings, "strategy_weighted_gate_threshold", 0.75),
            weighted_gate_min_confidence=getattr(settings, "strategy_weighted_gate_min_confidence", 60.0),
            weighted_gate_long_threshold=getattr(settings, "strategy_weighted_gate_long_threshold", 0.55),
            weighted_gate_short_threshold=getattr(settings, "strategy_weighted_gate_short_threshold", 0.55),
        )

    # Regime gate (Phase 1 Stage A). Default OFF until validated
    # on long multi-regime out-of-sample data (roadmap Phase 3).
    use_regime_gate: bool = False

    # Optional SLTPConfig override (Phase 3A exit experiments):
    # None -> calculator defaults from settings.
    sl_tp_config: Any = None


@dataclass
class SignalResult:
    """Complete strategy signal with full diagnostics."""
    
    # Final signal
    signal: Literal["HOLD", "BUY", "SELL"] = "HOLD"
    confidence: float = 0.0
    score: float = 0.0
    
    # Entry/Exit
    entry: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    
    # Diagnostics
    reasons: list[str] = field(default_factory=list)
    
    # AI Diagnostics
    ai_signal: Literal["HOLD", "BUY", "SELL"] = "HOLD"
    ai_confidence: float = 0.0
    
    # ML Diagnostics
    ml_signal: Literal["HOLD", "BUY", "SELL"] = "HOLD"
    ml_probability: float = 0.0
    ml_buy_probability: float = 0.0
    ml_sell_probability: float = 0.0
    ml_hold_probability: float = 0.0
    
    # Fusion Diagnostics
    fusion_signal: Literal["HOLD", "BUY", "SELL"] = "HOLD"
    combined_confidence: float = 0.0
    trade_approved: bool = False
    fusion_reason: str = ""
    
    # Order Flow Diagnostics
    order_flow_enabled: bool = False
    order_flow_signal: Literal["HOLD", "BUY", "SELL"] = "HOLD"
    order_flow_approved: bool = False
    order_flow_context: str = "UNKNOWN"
    order_flow_score: float = 0.5
    order_flow_pressure: float = 0.0
    order_flow_reason: str = ""
    
    # SL/TP Diagnostics
    sl_tp_method: str = "atr"
    sl_tp_reason: str = ""
    
    # Metadata
    timestamp: Optional[pd.Timestamp] = None
    window_id: Optional[int] = None
    
    # Market Components
    market_components: Optional[MarketComponents] = None
    fusion_result: Optional[object] = None
    order_flow_result: Optional[object] = None
    sl_tp_result: Optional[object] = None


class SignalGenerator:
    """
    Main signal generation pipeline.
    
    Flow:
    1. AIAnalyzer -> MarketComponents
    2. ConfidenceEngine -> AI signal + confidence + continuous probability
    3. MLEngine -> ML signal + probabilities (if enabled)
    4. WeightedGate -> Continuous probability -> LONG/SHORT/HOLD with approval
    5. OrderFlowGate -> Microstructure filter
    6. LiquidationIntelligence -> Liquidation signal
    7. SLTPCalculator -> Regime-adaptive SL/TP (with liquidation data)
    8. Final SignalResult assembly
    """
    
    def __init__(self, config: SignalConfig | None = None):
        self.config = config or SignalConfig.from_settings()
        
        # Initialize components
        self.ai_analyzer = AIAnalyzer()
        self.confidence_engine = ConfidenceEngine()
        self.ml_fusion = MLFusion()
        self.weighted_gate = WeightedGate(WeightedGateConfig(
            threshold=self.config.weighted_gate_threshold,
            min_confidence=self.config.weighted_gate_min_confidence,
            long_threshold=self.config.weighted_gate_long_threshold,
            short_threshold=self.config.weighted_gate_short_threshold,
        ))
        self.order_flow_gate = OrderFlowGate()
        self.liquidation_engine = LiquidationIntelligenceEngine()
        self.sl_tp_calculator = SLTPCalculator(
            self.config.sl_tp_config
        )

        # Regime filter (Stage A), active only when enabled in config.
        self._regime_filter = None
        if self.config.use_regime_gate:
            from src.regime_filter import RegimeFilter
            self._regime_filter = RegimeFilter()
        
        # ML Engine (lazy loaded)
        self._ml_engine: Optional[MLEngine] = None
        self._ml_model = None
        
        # Set custom weights on confidence engine
        self.confidence_engine.weights = {
            "trend": self.config.trend_weight,
            "momentum": self.config.momentum_weight,
            "volume": self.config.volume_weight,
            "volatility": self.config.volatility_weight,
        }
    
    def _get_ml_model(self):
        """Lazy load ML model."""
        if self._ml_model is None and self.config.use_ml:
            try:
                manager = ModelManager()
                manager.model_path = self.config.ml_model_path
                self._ml_model = manager.load()
                if self._ml_model is not None:
                    print(f"[SignalGenerator] ML model loaded from {self.config.ml_model_path}")
            except Exception as e:
                print(f"[SignalGenerator] Failed to load ML model: {e}")
                self._ml_model = None
        return self._ml_model
    
    def generate(
        self,
        df: pd.DataFrame,
        order_flow_signal: Optional[OrderFlowSignal] = None,
    ) -> SignalResult:
        """
        Generate complete strategy signal.
        
        Args:
            df: DataFrame with OHLCV and indicators (ema_fast, ema_slow, ema_trend, rsi, atr, volume_ratio)
            order_flow_signal: Optional OrderFlowSignal for microstructure filtering
        """
        if len(df) < 2:
            raise ValueError("Need at least 2 candles")
            
        row = df.iloc[-1]
        result = SignalResult()
        result.timestamp = row.get("timestamp")
        result.entry = float(row["close"])
        
        # === 1. AI ANALYZER ===
        market_components = self.ai_analyzer.analyze(df)
        result.market_components = market_components
        result.reasons.extend(market_components.reasons)
        
        # === 2. CONFIDENCE ENGINE (AI SIGNAL) ===
        self.confidence_engine.reset()
        self.confidence_engine.add_component("trend", market_components.trend_score)
        self.confidence_engine.add_component("momentum", market_components.component_scores.get("momentum", 0))
        self.confidence_engine.add_component("volume", market_components.component_scores.get("volume", 0))
        self.confidence_engine.add_component("volatility", market_components.component_scores.get("volatility", 0))
        
        confidence_result = self.confidence_engine.evaluate()
        
        result.ai_signal = confidence_result.decision
        result.ai_confidence = confidence_result.confidence
        result.score = confidence_result.total_score
        result.confidence = confidence_result.confidence
        result.reasons.extend(confidence_result.reasons)
        
        # === 3. ML PREDICTION (if enabled) ===
        ml_signal = "HOLD"
        ml_probability = 0.0
        ml_probabilities = {0: 0.0, 1: 0.0, 2: 0.0}
        
        if self.config.use_ml:
            ml_model = self._get_ml_model()
            if ml_model is not None:
                try:
                    features = build_features(df)
                    ml_engine = MLEngine(MLConfig())
                    ml_engine.model = ml_model
                    ml_engine.feature_names = list(features.keys())
                    
                    probas = ml_engine.predict_probabilities(pd.DataFrame([features]))
                    ml_signal = max(probas, key=probas.get)
                    ml_probability = probas[ml_signal]
                    
                    result.ml_buy_probability = probas.get("BUY", 0.0)
                    result.ml_sell_probability = probas.get("SELL", 0.0)
                    result.ml_hold_probability = probas.get("HOLD", 0.0)
                except Exception as e:
                    print(f"[SignalGenerator] ML prediction error: {e}")
        
        result.ml_signal = ml_signal
        result.ml_probability = ml_probability
        
        # === 4. WEIGHTED GATE (replaces ML Fusion) ===
        # Get continuous probability from AI
        ai_continuous_prob = self.confidence_engine.get_continuous_probability(confidence_result.total_score)
        
        # Combine AI and ML probabilities if ML is enabled
        if self.config.use_ml and ml_probability > 0:
            # Weighted combination: 60% AI, 40% ML
            combined_prob = ai_continuous_prob * 0.6 + ml_probability * 0.4
        else:
            combined_prob = ai_continuous_prob
        
        # Evaluate through Weighted Gate
        weighted_gate_result = self.weighted_gate.evaluate(
            probability=combined_prob,
            confidence=result.ai_confidence,
            ai_signal=result.ai_signal,
        )
        
        # Store diagnostics
        result.fusion_signal = weighted_gate_result.action
        result.combined_confidence = result.ai_confidence
        result.trade_approved = weighted_gate_result.approved
        result.fusion_reason = f"WeightedGate: {weighted_gate_result.reason} (prob={combined_prob:.2%})"
        result.reasons.append(f"WeightedGate: {weighted_gate_result.reason}")
        
        # === 5. ORDER FLOW GATE ===
        # Set order_flow_enabled based on whether we have OF data
        result.order_flow_enabled = order_flow_signal is not None
        
        of_result = self.order_flow_gate.apply(
            strategy_signal=weighted_gate_result.action,
            strategy_approved=weighted_gate_result.approved,
            order_flow_signal=order_flow_signal,
            current_price=result.entry,
        )
        
        result.order_flow_result = of_result
        result.order_flow_signal = of_result.signal
        result.order_flow_approved = of_result.approved
        result.order_flow_context = of_result.context
        result.order_flow_score = of_result.score
        result.order_flow_pressure = of_result.pressure
        result.order_flow_reason = of_result.reason
        result.reasons.append(f"OrderFlow: {of_result.reason}")
        
        # === 6. FINAL SIGNAL DECISION ===

        # Stage-A regime classification: exactly one call per bar so
        # hysteresis state advances sequentially and causally.
        regime = None
        if self._regime_filter is not None:
            regime = self._regime_filter.classify(df)

        if not of_result.approved:
            result.signal = "HOLD"
            result.confidence = result.combined_confidence
            result.stop_loss = result.entry
            result.take_profit = result.entry
            result.trade_approved = False
        elif of_result.signal == "HOLD":
            result.signal = "HOLD"
            result.confidence = result.combined_confidence
            result.stop_loss = result.entry
            result.take_profit = result.entry
            result.trade_approved = False
        elif (
            self._regime_filter is not None
            and not self._regime_filter.allows(of_result.signal)
        ):
            # Stage-A regime gate: block counter-trend entries.
            result.signal = "HOLD"
            result.confidence = result.combined_confidence
            result.stop_loss = result.entry
            result.take_profit = result.entry
            result.trade_approved = False
            result.reasons.append(
                f"RegimeGate: blocked {of_result.signal} while regime={regime}"
            )
        else:
            # Approved trade
            result.signal = of_result.signal
            result.confidence = result.combined_confidence
            result.trade_approved = True
            
            # === 6. LIQUIDATION INTELLIGENCE ===
            liquidation_signal = None
            try:
                # Create snapshot from current market data
                from src.liquidation_intelligence import LiquidationSnapshot, LiquidationEvent
                # We need actual liquidation events - for now use a placeholder
                # In production, this would come from exchange liquidation feed
                liquidation_signal = self.liquidation_engine.previous
            except Exception as e:
                print(f"[SignalGenerator] Liquidation intelligence error: {e}")
            
            # === 7. SL/TP CALCULATOR ===
            atr = float(row["atr"])
            sl_tp_result = self.sl_tp_calculator.calculate(
                entry_price=result.entry,
                atr=atr,
                signal=result.signal,
                volatility_regime=market_components.volatility_regime,
                trend_alignment=market_components.trend_alignment,
                liquidation_signal=liquidation_signal,
            )
            
            result.sl_tp_result = sl_tp_result
            result.stop_loss = sl_tp_result.stop_loss
            result.take_profit = sl_tp_result.take_profit
            result.sl_tp_method = sl_tp_result.method
            result.sl_tp_reason = sl_tp_result.reason
            result.reasons.append(f"SL/TP: {sl_tp_result.reason}")
            
        return result


# Backward compatibility function
def generate_signal_result(
    df: pd.DataFrame,
    order_flow_signal: Optional[OrderFlowSignal] = None,
    model: Optional[object] = None,
) -> SignalResult:
    """
    Backward compatible signal generation.
    
    Note: model parameter is deprecated; use config.use_ml instead.
    """
    generator = SignalGenerator()
    if model is not None:
        # Inject model if provided (for backward compat)
        generator._ml_model = model
    return generator.generate(df, order_flow_signal)