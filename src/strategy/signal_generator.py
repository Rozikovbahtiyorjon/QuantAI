"""
QuantAI Signal Generator

New ML scheme (user requirement: ML does not guess direction):

    Rule Strategy  -> Candidate Trade (BUY/SELL/HOLD)
          ↓
    ML Meta-Labeler -> TAKE / REJECT
          ↓
    "Will this candidate have positive net expectancy?"

ML is a binary filter P(win) on rule candidates, not a 3-class direction predictor.
This aligns ML with real trading objective (expectancy) and avoids
ML vs Rule conflict (old MLFusion). Old 3-class path is deprecated
and kept only for backward compat (use_ml_legacy).

Components:
- AI Analyzer (technical components) -> Rule candidate
- Confidence Engine + WeightedGate -> Rule candidate + confidence
- ML Meta-Labeler (FilteredGenerator / MetaLabelModel) -> TAKE/REJECT
- Order Flow Gate
- SL/TP Calculator
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Any

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
    use_ml: bool = False  # legacy 3-class direction predictor (deprecated)
    use_ml_legacy: bool = False  # explicit legacy flag
    use_meta_labeler: bool = False  # ML as TAKE/REJECT filter on rule candidates
    meta_model_path: str = "models/meta_label.pkl"
    meta_threshold: float = 0.55  # P(win) >= threshold => TAKE (binary)
    use_expected_return: bool = True  # True: predict E[net return], False: P(win)
    expected_return_hurdle: float = 0.0  # required edge, e.g. 0.001 = 0.1% net
    expected_return_costs: Any = None  # CostConfig for net calc, None => defaults
    use_order_flow: bool = True
    use_weighted_gate: bool = True
    ml_model_path: str = "models/quantai_v5.pkl"  # legacy
    
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
    
    # === Regime-Adaptive Dual Strategy (Phase 1) ===
    use_regime_adaptive: bool = True
    
    # Trend regime params (ADX > 25)
    trend_adx_min: float = 25.0
    trend_confidence_boost: float = 1.2
    trend_weighted_gate_threshold: float = 0.70
    
    # Range regime params (ADX <= 25)
    range_adx_max: float = 25.0
    range_confidence_boost: float = 1.0
    range_weighted_gate_threshold: float = 0.65
    
    # BB Squeeze params for range
    bb_squeeze_width: float = 0.02
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    
    @classmethod
    def from_settings(cls) -> "SignalConfig":
        # P1 FIX: Canonical fail-fast config — no getattr fallback for critical trading params.
        # Previously used getattr(settings, "ml_enabled", False) instead of settings.ml.ml_enabled,
        # causing silent fallback to 0.60 even when user set strategy.ai_weight=0.8.
        s = settings.strategy
        ml = settings.ml
        # min_confidence: handle legacy 60.0 vs 0.60 — canonical is 0.60 (0..1)
        raw_conf = s.min_confidence
        min_conf = raw_conf / 100.0 if raw_conf > 1.0 else raw_conf
        # New meta-labeler config: use_meta_labeler from settings if present
        use_meta = bool(getattr(s, "use_meta_labeler", False) or getattr(ml, "use_meta_labeler", False))
        use_exp = bool(getattr(s, "use_expected_return", True) if hasattr(s, "use_expected_return") else getattr(ml, "use_expected_return", True))
        return cls(
            min_confidence=min_conf,
            use_ml=ml.ml_enabled,
            use_ml_legacy=bool(getattr(ml, "use_ml_legacy", False)),
            use_meta_labeler=use_meta,
            meta_model_path=getattr(ml, "meta_model_path", "models/meta_label.pkl"),
            meta_threshold=float(getattr(s, "meta_threshold", 0.55)),
            use_expected_return=use_exp,
            expected_return_hurdle=float(getattr(s, "expected_return_hurdle", 0.0) if hasattr(s, "expected_return_hurdle") else getattr(ml, "expected_return_hurdle", 0.0)),
            use_order_flow=s.orderflow_enabled,
            use_weighted_gate=True,  # Controlled by strategy.weighted_gate_threshold
            ml_model_path=ml.model_path,
            trend_weight=s.confidence_trend_weight,
            momentum_weight=s.confidence_momentum_weight,
            volume_weight=s.confidence_volume_weight,
            volatility_weight=s.confidence_volatility_weight,
            weighted_gate_threshold=s.weighted_gate_threshold,
            weighted_gate_min_confidence=s.weighted_gate_min_confidence,
            weighted_gate_long_threshold=s.weighted_gate_long_threshold,
            weighted_gate_short_threshold=s.weighted_gate_short_threshold,
            # Regime-adaptive
            use_regime_adaptive=getattr(s, "use_regime_adaptive", True) if s else True,
            trend_adx_min=getattr(s, "trend_adx_min", 25.0),
            trend_confidence_boost=getattr(s, "trend_confidence_boost", 1.2),
            trend_weighted_gate_threshold=getattr(s, "trend_weighted_gate_threshold", 0.70),
            range_adx_max=getattr(s, "range_adx_max", 25.0),
            range_confidence_boost=getattr(s, "range_confidence_boost", 1.0),
            range_weighted_gate_threshold=getattr(s, "range_weighted_gate_threshold", 0.65),
            bb_squeeze_width=getattr(s, "bb_squeeze_width", 0.02),
            rsi_oversold=getattr(s, "rsi_oversold", 30.0),
            rsi_overbought=getattr(s, "rsi_overbought", 70.0),
        )
    
    # REGIME ENGINE — keep OFF in legacy SignalGenerator for shadow mode (P3.1)
    # New Entry Engine has it ON as required. See docs/roadmap/phase-gates.md PHASE 3.
    # Do not change legacy here until shadow validation passes (ENTRY-05).
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
    
    # ML Diagnostics (legacy 3-class)
    ml_signal: Literal["HOLD", "BUY", "SELL"] = "HOLD"
    ml_probability: float = 0.0
    ml_buy_probability: float = 0.0
    ml_sell_probability: float = 0.0
    ml_hold_probability: float = 0.0

    # Meta-Labeler Diagnostics (TAKE/REJECT)
    meta_enabled: bool = False
    meta_probability: float = 0.0  # P(win | candidate)
    meta_decision: Literal["TAKE", "REJECT", "HOLD"] = "HOLD"
    meta_threshold: float = 0.55
    meta_reason: str = ""
    
    # Fusion Diagnostics (legacy)
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
    
    # Setup Diagnostics (P3 universal Setup Engine)
    setup: str = "NONE"
    setup_reason: str = ""
    setup_confidence: float = 0.0
    pullback_setup: str = "NONE"
    pullback_quality: float = 0.0
    mean_reversion_setup: str = "NONE"
    poi_result: Any = None

    # Entry Quality (P3.13 distinct from Confidence)
    entry_quality: float = 0.0
    entry_quality_reasons: list = field(default_factory=list)

    # Expected Value (P3.14 mandatory)
    expected_value: float = 0.0
    expected_value_reason: str = ""

    # Market Components
    market_components: Optional[MarketComponents] = None
    fusion_result: Optional[object] = None
    order_flow_result: Optional[object] = None
    sl_tp_result: Optional[object] = None


class SignalGenerator:
    """
    Main signal generation pipeline (Meta-Labeler scheme).

    New flow (ML as TAKE/REJECT, not direction):
    1. AIAnalyzer -> MarketComponents
    2. ConfidenceEngine -> Rule candidate (BUY/SELL/HOLD) + confidence
    3. Regime-specific strategy -> candidate signal (BUY/SELL/HOLD)
    4. ML Meta-Labeler (if enabled): P(win | candidate, context) -> TAKE/REJECT
       - Rule generates candidate, Meta evaluates "Will it have positive expectancy?"
       - REJECT => HOLD (filters false positives, aligns with trading objective)
    5. OrderFlowGate -> Microstructure filter
    6. SLTPCalculator -> Regime-adaptive SL/TP
    7. Final SignalResult assembly

    Legacy 3-class ML (MLFusion) kept only if use_ml_legacy=True (deprecated).
    """
    
    def __init__(
        self,
        config: SignalConfig | None = None,
    ) -> None:
        self.config = config or SignalConfig.from_settings()
        
        # Initialize components
        self.ai_analyzer = AIAnalyzer()
        self.confidence_engine = ConfidenceEngine()
        self.ml_fusion = MLFusion()
        
        # Three weighted gates for different regimes
        self.weighted_gate_default = WeightedGate(WeightedGateConfig(
            threshold=self.config.weighted_gate_threshold,
            min_confidence=self.config.weighted_gate_min_confidence,
            long_threshold=self.config.weighted_gate_long_threshold,
            short_threshold=self.config.weighted_gate_short_threshold,
        ))
        self.weighted_gate_trend = WeightedGate(WeightedGateConfig(
            threshold=self.config.trend_weighted_gate_threshold,
            min_confidence=self.config.weighted_gate_min_confidence,
            long_threshold=self.config.weighted_gate_long_threshold,
            short_threshold=self.config.weighted_gate_short_threshold,
        ))
        self.weighted_gate_range = WeightedGate(WeightedGateConfig(
            threshold=self.config.range_weighted_gate_threshold,
            min_confidence=self.config.weighted_gate_min_confidence,
            long_threshold=self.config.weighted_gate_long_threshold,
            short_threshold=self.config.weighted_gate_short_threshold,
        ))
        
        self.order_flow_gate = OrderFlowGate()
        self.liquidation_engine = LiquidationIntelligenceEngine()
        self.sl_tp_calculator = SLTPCalculator(
            self.config.sl_tp_config
        )
  
        # Regime filter - wired from SignalConfig
        # Legacy SignalGenerator keeps use_regime_gate False for shadow mode (P3.1 new EntryEngine has it True)
        # Do not add Setup/Entry Quality/EV here — they live in New Entry Engine (src/entry_engine.py) in SHADOW
        from src.regime_filter import RegimeFilter, RegimeConfig
        regime_config = RegimeConfig(
            adx_enter=self.config.trend_adx_min,
            adx_exit=max(1.0, self.config.trend_adx_min - 3.0),  # hysteresis
            slope_bars=8,
            min_bars=60,
        )
        self._regime_filter = RegimeFilter(regime_config)
        
        # ML Engine (legacy 3-class, lazy loaded)
        self._ml_engine: Optional[MLEngine] = None
        self._ml_model = None
        # ML Meta-Labeler (TAKE/REJECT, P(win | candidate))
        self._meta_model = None
        self._meta_threshold = float(self.config.meta_threshold)
        
        # Set custom weights on confidence engine
        self.confidence_engine.weights = {
            "trend": self.config.trend_weight,
            "momentum": self.config.momentum_weight,
            "volume": self.config.volume_weight,
            "volatility": self.config.volatility_weight,
        }

    def _get_ml_model(self):
        """Lazy load legacy ML model (3-class direction). Deprecated."""
        if self._ml_model is None and (self.config.use_ml or self.config.use_ml_legacy):
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

    def _get_meta_model(self):
        """Lazy load Meta-Labeler model (TAKE/REJECT, P(win|candidate))."""
        if self._meta_model is None and self.config.use_meta_labeler:
            try:
                # Meta model is a MetaLabelModel pickled via joblib or ModelManager
                import joblib
                from pathlib import Path
                p = Path(self.config.meta_model_path)
                if p.exists():
                    self._meta_model = joblib.load(p)
                    print(f"[SignalGenerator] Meta model loaded from {self.config.meta_model_path}")
                else:
                    # Fallback to ModelManager
                    manager = ModelManager()
                    manager.model_path = self.config.meta_model_path
                    self._meta_model = manager.load()
                    if self._meta_model is not None:
                        print(f"[SignalGenerator] Meta model loaded from {self.config.meta_model_path}")
            except Exception as e:
                print(f"[SignalGenerator] Failed to load Meta model: {e}")
                self._meta_model = None
        return self._meta_model
    
    def generate(
        self,
        df: pd.DataFrame,
        order_flow_signal: Optional[OrderFlowSignal] = None,
    ) -> SignalResult:
        """
        Generate complete strategy signal with regime-adaptive dual strategy.
        
        Regime-based logic:
        - TREND_UP: trend-following long (EMA alignment + ADX confirmation)
        - TREND_DOWN: trend-following short (EMA alignment + ADX confirmation)
        - RANGE: mean-reversion (BB squeeze + RSI extremes)
        
        Args:
            df: DataFrame with OHLCV and indicators
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
        
        # === 3. ML PREDICTION (legacy 3-class, deprecated) ===
        ml_signal = "HOLD"
        ml_probability = 0.0
        # Legacy path: ML guesses direction (BUY/SELL/HOLD) and fuses with Rule
        # Deprecated: use_meta_labeler is the recommended TAKE/REJECT filter.
        if self.config.use_ml_legacy or (self.config.use_ml and not self.config.use_meta_labeler):
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
        # Expose legacy fusion diagnostics as HOLD when not in legacy mode
        result.meta_enabled = bool(self.config.use_meta_labeler)
        result.meta_threshold = float(self.config.meta_threshold)
        
        # === REGIME CLASSIFICATION ===
        regime = self._regime_filter.classify(df)
        result.reasons.append(f"Regime: {regime}")
        
        # === REGIME-SPECIFIC STRATEGY (Rule candidate) ===
        # For meta-labeler scheme, Rule generates candidate WITHOUT ML influence
        # (ml_signal=HOLD), then Meta decides TAKE/REJECT based on P(win|candidate).
        # For legacy scheme, Rule already fused with ML direction.
        if self.config.use_meta_labeler:
            # Rule-only candidate
            if regime == "TREND_UP":
                result = self._generate_trend_long(df, row, result, confidence_result, "HOLD", 0.0, market_components)
            elif regime == "TREND_DOWN":
                result = self._generate_trend_short(df, row, result, confidence_result, "HOLD", 0.0, market_components)
            else:
                result = self._generate_range_reversion(df, row, result, confidence_result, "HOLD", 0.0, market_components)
        else:
            if regime == "TREND_UP":
                result = self._generate_trend_long(df, row, result, confidence_result, ml_signal, ml_probability, market_components)
            elif regime == "TREND_DOWN":
                result = self._generate_trend_short(df, row, result, confidence_result, ml_signal, ml_probability, market_components)
            else:
                result = self._generate_range_reversion(df, row, result, confidence_result, ml_signal, ml_probability, market_components)

        # === 3b. ML META-LABELER (TAKE/REJECT) ===
        # Note: REGIME as required entry gate is in New Entry Engine (src/entry_engine.py), not legacy SignalGenerator (shadow mode)
        # New scheme: E[net return | features] where net = price outcome - commission - slippage - spread - funding
        # Decision: expected edge > hurdle => TAKE, else REJECT. More natural for trading engine than P(win).
        # Legacy binary P(win) is kept only if use_expected_return=False.
        if self.config.use_meta_labeler and result.signal != "HOLD":
            result.meta_enabled = True
            try:
                from src.strategy.meta_label import entry_features
                meta_model = self._get_meta_model()
                if meta_model is not None:
                    feats = entry_features(df, result.signal)
                    # Preferred: ExpectedReturnModel (regression) -> E[net]
                    if getattr(self.config, "use_expected_return", True) and hasattr(meta_model, "predict_expected"):
                        # Regression path: E[net return | features]
                        exp = float(meta_model.predict_expected(feats))  # type: ignore
                        hurdle = float(getattr(meta_model, "hurdle", self.config.expected_return_hurdle))
                        result.meta_probability = exp  # reuse field for expected net
                        result.meta_threshold = hurdle
                        if exp > hurdle:
                            result.meta_decision = "TAKE"
                            result.meta_reason = f"ExpectedReturn TAKE E[net]={exp:.4f} > hurdle {hurdle:.4f}"
                            result.reasons.append(result.meta_reason)
                        else:
                            result.meta_decision = "REJECT"
                            result.meta_reason = f"ExpectedReturn REJECT E[net]={exp:.4f} <= hurdle {hurdle:.4f} -> HOLD"
                            result.reasons.append(result.meta_reason)
                            result.signal = "HOLD"
                            result.trade_approved = False
                            result.reasons.append(f"Candidate {result.ai_signal} rejected by ExpectedReturn (edge {exp:.4f} <= {hurdle:.4f})")
                    elif hasattr(meta_model, "approve"):
                        # Legacy binary P(win)
                        try:
                            import pandas as pd
                            from src.strategy.meta_label import FEATURE_NAMES
                            proba = float(meta_model.model.predict_proba(pd.DataFrame([feats])[FEATURE_NAMES])[0][1]) if hasattr(meta_model, "model") else 0.5
                        except Exception:
                            proba = 1.0 if meta_model.approve(feats) else 0.0  # type: ignore
                        result.meta_probability = proba
                        result.meta_threshold = float(getattr(meta_model, "threshold", self.config.meta_threshold))
                        if proba >= result.meta_threshold:
                            result.meta_decision = "TAKE"
                            result.meta_reason = f"Meta TAKE P(win)={proba:.2f} >= {result.meta_threshold:.2f}"
                            result.reasons.append(result.meta_reason)
                        else:
                            result.meta_decision = "REJECT"
                            result.meta_reason = f"Meta REJECT P(win)={proba:.2f} < {result.meta_threshold:.2f} -> HOLD"
                            result.reasons.append(result.meta_reason)
                            result.signal = "HOLD"
                            result.trade_approved = False
                            result.reasons.append(f"Candidate {result.ai_signal} rejected by Meta")
                    else:
                        import pandas as pd
                        proba = float(meta_model.predict_proba(pd.DataFrame([feats]))[0][1]) if hasattr(meta_model, "predict_proba") else 0.5
                        result.meta_probability = proba
                        if proba < self.config.meta_threshold:
                            result.meta_decision = "REJECT"
                            result.signal = "HOLD"
                            result.trade_approved = False
                            result.meta_reason = f"Meta REJECT {proba:.2f}"
                        else:
                            result.meta_decision = "TAKE"
                            result.meta_reason = f"Meta TAKE {proba:.2f}"
                else:
                    result.meta_decision = "TAKE"
                    result.meta_reason = "Meta no model -> TAKE (permissive, per-window training in research branch)"
            except Exception as e:
                print(f"[SignalGenerator] Meta prediction error: {e}")
                result.meta_decision = "TAKE"
                result.meta_reason = f"Meta error -> TAKE: {e}"
        elif self.config.use_meta_labeler:
            result.meta_enabled = True
            result.meta_decision = "HOLD"
            result.meta_reason = "No candidate -> Meta not evaluated"
        
        # === ORDER FLOW GATE ===
        result.order_flow_enabled = order_flow_signal is not None
        
        of_result = self.order_flow_gate.apply(
            strategy_signal=result.signal,
            strategy_approved=result.trade_approved,
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
        
        # Final signal decision with order flow
        if not of_result.approved or of_result.signal == "HOLD":
            result.signal = "HOLD"
            result.confidence = result.combined_confidence
            result.stop_loss = result.entry
            result.take_profit = result.entry
            result.trade_approved = False
        else:
            result.signal = of_result.signal
            result.confidence = result.combined_confidence
            result.trade_approved = True
            
            # SL/TP CALCULATOR
            atr = float(row["atr"])
            sl_tp_result = self.sl_tp_calculator.calculate(
                entry_price=result.entry,
                atr=atr,
                signal=result.signal,
                volatility_regime=market_components.volatility_regime,
                trend_alignment=market_components.trend_alignment,
                liquidation_signal=None,
            )
            
            result.sl_tp_result = sl_tp_result
            result.stop_loss = sl_tp_result.stop_loss
            result.take_profit = sl_tp_result.take_profit
            result.sl_tp_method = sl_tp_result.method
            result.sl_tp_reason = sl_tp_result.reason
            result.reasons.append(f"SL/TP: {sl_tp_result.reason}")
        
        return result
  
    # ============================================================
    # REGIME-SPECIFIC STRATEGY METHODS
    # ============================================================
  
    def _generate_trend_long(
        self,
        df: pd.DataFrame,
        row: pd.Series,
        result: SignalResult,
        confidence_result,
        ml_signal: str,
        ml_probability: float,
        market_components,
    ) -> SignalResult:
        """Trend-following LONG for TREND_UP regime."""
        
        if not self.config.use_regime_adaptive:
            return self._default_strategy(df, row, result, confidence_result, ml_signal, ml_probability, market_components)
        
        # TREND_LONG: use trend-weighted gate with lower threshold
        ai_continuous_prob = self.confidence_engine.get_continuous_probability(confidence_result.total_score)
        
        # ML probability guard
        if ml_probability > 1.0:
            ml_probability = ml_probability / 100.0
        
        if self.config.use_ml and ml_probability > 0:
            combined_prob = ai_continuous_prob * 0.6 + ml_probability * 0.4
            combined_prob = max(0.0, min(1.0, combined_prob))
        else:
            combined_prob = ai_continuous_prob
        
        # Apply trend confidence boost
        boosted_confidence = min(100.0, confidence_result.confidence * self.config.trend_confidence_boost)
        
        # Use trend-weighted gate (lower threshold for trend-following)
        weighted_gate_result = self.weighted_gate_trend.evaluate(
            probability=combined_prob,
            confidence=boosted_confidence,
            ai_signal="BUY",
        )
        
        result.fusion_signal = weighted_gate_result.action
        result.combined_confidence = boosted_confidence
        result.trade_approved = weighted_gate_result.approved
        result.fusion_reason = f"TrendGate LONG: {weighted_gate_result.reason} (prob={combined_prob:.2%})"
        result.reasons.append(f"TrendGate LONG: {weighted_gate_result.reason}")
        
        # Signal decision
        if not weighted_gate_result.approved:
            result.signal = "HOLD"
            result.confidence = boosted_confidence
            result.trade_approved = False
        else:
            result.signal = "BUY"
            result.confidence = boosted_confidence
            result.trade_approved = True
            result.reasons.append("TREND_UP: trend-following LONG")
        
        return result
  
    def _generate_trend_short(
        self,
        df: pd.DataFrame,
        row: pd.Series,
        result: SignalResult,
        confidence_result,
        ml_signal: str,
        ml_probability: float,
        market_components,
    ) -> SignalResult:
        """Trend-following SHORT for TREND_DOWN regime."""
        
        if not self.config.use_regime_adaptive:
            return self._default_strategy(df, row, result, confidence_result, ml_signal, ml_probability, market_components)
        
        # TREND_SHORT: trend-following short
        ai_continuous_prob = self.confidence_engine.get_continuous_probability(confidence_result.total_score)
        
        if ml_probability > 1.0:
            ml_probability = ml_probability / 100.0
        
        if self.config.use_ml and ml_probability > 0:
            combined_prob = ai_continuous_prob * 0.6 + ml_probability * 0.4
            combined_prob = max(0.0, min(1.0, combined_prob))
        else:
            combined_prob = ai_continuous_prob
        
        boosted_confidence = min(100.0, confidence_result.confidence * self.config.trend_confidence_boost)
        
        # Use trend-weighted gate for short
        weighted_gate_result = self.weighted_gate_trend.evaluate(
            probability=combined_prob,
            confidence=boosted_confidence,
            ai_signal="SELL",
        )
        
        result.fusion_signal = weighted_gate_result.action
        result.combined_confidence = boosted_confidence
        result.trade_approved = weighted_gate_result.approved
        result.fusion_reason = f"TrendGate SHORT: {weighted_gate_result.reason} (prob={combined_prob:.2%})"
        result.reasons.append(f"TrendGate SHORT: {weighted_gate_result.reason}")
        
        if not weighted_gate_result.approved:
            result.signal = "HOLD"
            result.confidence = boosted_confidence
            result.trade_approved = False
        else:
            result.signal = "SELL"
            result.confidence = boosted_confidence
            result.trade_approved = True
            result.reasons.append("TREND_DOWN: trend-following SHORT")
        
        return result
  
    def _generate_range_reversion(
        self,
        df: pd.DataFrame,
        row: pd.Series,
        result: SignalResult,
        confidence_result,
        ml_signal: str,
        ml_probability: float,
        market_components,
    ) -> SignalResult:
        """Mean-reversion for RANGE regime (BB squeeze + RSI extremes)."""
        
        if not self.config.use_regime_adaptive:
            return self._default_strategy(df, row, result, confidence_result, ml_signal, ml_probability, market_components)
        
        # RANGE: mean-reversion with BB + RSI
        ai_continuous_prob = self.confidence_engine.get_continuous_probability(confidence_result.total_score)
        
        if ml_probability > 1.0:
            ml_probability = ml_probability / 100.0
        
        if self.config.use_ml and ml_probability > 0:
            combined_prob = ai_continuous_prob * 0.6 + ml_probability * 0.4
            combined_prob = max(0.0, min(1.0, combined_prob))
        else:
            combined_prob = ai_continuous_prob
        
        boosted_confidence = min(100.0, confidence_result.confidence * self.config.range_confidence_boost)
        
        # Range-specific logic: BB position + RSI
        bb_pos = 0.5
        rsi = float(row.get("rsi", 50))
        bb_width = 1.0
        
        if "bb_position" in row and pd.notna(row["bb_position"]):
            bb_pos = float(row["bb_position"]) + 0.5  # -0.5..0.5 -> 0..1
        if "bb_width" in row and pd.notna(row["bb_width"]):
            bb_width = float(row["bb_width"])
        
        # Range mean-reversion logic
        rsi_oversold = self.config.rsi_oversold
        rsi_overbought = self.config.rsi_overbought
        bb_squeeze = bb_width < self.config.bb_squeeze_width
        
        range_signal = "HOLD"
        range_reason = ""
        
        # Mean-reversion: buy at lower BB + oversold RSI, sell at upper BB + overbought RSI
        if bb_pos <= 0.2 and rsi <= rsi_oversold:
            range_signal = "BUY"
            range_reason = f"Range mean-rev: BB low ({bb_pos:.2f}) + RSI oversold ({rsi:.0f})"
        elif bb_pos >= 0.8 and rsi >= rsi_overbought:
            range_signal = "SELL"
            range_reason = f"Range mean-rev: BB high ({bb_pos:.2f}) + RSI overbought ({rsi:.0f})"
        elif bb_squeeze:
            # BB squeeze: wait for breakout, don't trade
            range_signal = "HOLD"
            range_reason = f"BB squeeze (width={bb_width:.3f}) - no trade"
        else:
            range_signal = "HOLD"
            range_reason = f"Range neutral: BB pos {bb_pos:.2f}, RSI {rsi:.0f}"
        
        result.reasons.append(f"Range: {range_reason}")
        
        # Apply range confidence boost
        boosted_confidence = min(100.0, confidence_result.confidence * self.config.range_confidence_boost)
        
        # Evaluate through range-weighted gate
        weighted_gate_result = self.weighted_gate_range.evaluate(
            probability=combined_prob,
            confidence=boosted_confidence,
            ai_signal=range_signal,
        )
        
        result.fusion_signal = weighted_gate_result.action
        result.combined_confidence = boosted_confidence
        result.trade_approved = weighted_gate_result.approved
        result.fusion_reason = f"RangeGate: {weighted_gate_result.reason} (prob={combined_prob:.2%})"
        result.reasons.append(f"RangeGate: {weighted_gate_result.reason}")
        
        if not weighted_gate_result.approved or range_signal == "HOLD":
            result.signal = "HOLD"
            result.confidence = boosted_confidence
            result.trade_approved = False
        else:
            result.signal = range_signal
            result.confidence = boosted_confidence
            result.trade_approved = True
            result.reasons.append(f"RANGE: {range_signal} mean-reversion")
        
        return result
  
    def _default_strategy(
        self,
        df: pd.DataFrame,
        row: pd.Series,
        result: SignalResult,
        confidence_result,
        ml_signal: str,
        ml_probability: float,
        market_components,
    ) -> SignalResult:
        """Fallback to original single-strategy logic."""
        
        ai_continuous_prob = self.confidence_engine.get_continuous_probability(confidence_result.total_score)
        
        if ml_probability > 1.0:
            ml_probability = ml_probability / 100.0
        
        if self.config.use_ml and ml_probability > 0:
            combined_prob = ai_continuous_prob * 0.6 + ml_probability * 0.4
            combined_prob = max(0.0, min(1.0, combined_prob))
        else:
            combined_prob = ai_continuous_prob
        
        weighted_gate_result = self.weighted_gate_default.evaluate(
            probability=combined_prob,
            confidence=confidence_result.confidence,
            ai_signal=result.ai_signal,
        )
        
        result.fusion_signal = weighted_gate_result.action
        result.combined_confidence = confidence_result.confidence
        result.trade_approved = weighted_gate_result.approved
        result.fusion_reason = f"DefaultGate: {weighted_gate_result.reason} (prob={combined_prob:.2%})"
        result.reasons.append(f"DefaultGate: {weighted_gate_result.reason}")
        
        if not weighted_gate_result.approved:
            result.signal = "HOLD"
            result.confidence = confidence_result.confidence
            result.trade_approved = False
        elif weighted_gate_result.action == "HOLD":
            result.signal = "HOLD"
            result.confidence = confidence_result.confidence
            result.trade_approved = False
        else:
            result.signal = weighted_gate_result.action
            result.confidence = confidence_result.confidence
            result.trade_approved = True
        
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