"""
QuantAI ML Fusion

Combines AI (technical analysis) and ML (XGBoost) signals
with configurable fusion rules per ADR-0003.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from config.settings import settings


@dataclass
class FusionConfig:
    """Configuration for AI+ML fusion rules."""
    
    min_confidence: float = 60.0
    ai_weight: float = 0.60
    ml_weight: float = 0.40
    conflict_penalty: float = 0.70
    
    # ML HOLD behavior
    ml_hold_blocks_ai: bool = True
    
    # AI HOLD behavior  
    ai_hold_blocks_all: bool = True
    
    @classmethod
    def from_settings(cls) -> "FusionConfig":
        return cls(
            min_confidence=getattr(settings, "strategy_min_confidence", 60.0),
            ai_weight=getattr(settings, "strategy_ai_weight", 0.60),
            ml_weight=getattr(settings, "strategy_ml_weight", 0.40),
            conflict_penalty=getattr(settings, "strategy_conflict_penalty", 0.70),
            ml_hold_blocks_ai=getattr(settings, "strategy_ml_hold_blocks_ai", True),
            ai_hold_blocks_all=getattr(settings, "strategy_ai_hold_blocks_all", True),
        )


@dataclass
class FusionResult:
    """Result of AI+ML fusion."""
    
    signal: Literal["HOLD", "BUY", "SELL"]
    combined_confidence: float
    approved: bool
    reason: str
    
    # Diagnostics
    ai_signal: Literal["HOLD", "BUY", "SELL"] = "HOLD"
    ai_confidence: float = 0.0
    ml_signal: Literal["HOLD", "BUY", "SELL"] = "HOLD"
    ml_probability: float = 0.0
    
    # Rule that was applied
    rule_applied: str = ""


class MLFusion:
    """
    Fuses AI and ML signals per configured rules.
    
    Rules (ADR-0003):
    1. AI HOLD + ML HOLD -> HOLD
    2. AI HOLD + ML BUY/SELL -> HOLD (if ai_hold_blocks_all)
    3. AI BUY/SELL + ML HOLD -> HOLD (if ml_hold_blocks_ai)
    4. AI + ML agree -> weighted confidence
    5. AI + ML conflict -> penalized AI confidence, HOLD
    """
    
    def __init__(self, config: FusionConfig | None = None):
        self.config = config or FusionConfig.from_settings()
        
    def fuse(
        self,
        ai_signal: Literal["HOLD", "BUY", "SELL"],
        ai_confidence: float,
        ml_signal: Literal["HOLD", "BUY", "SELL"],
        ml_probability: float,
    ) -> FusionResult:
        """
        Fuse AI and ML signals.
        
        Args:
            ai_signal: AI decision (HOLD/BUY/SELL)
            ai_confidence: AI confidence 0-100
            ml_signal: ML prediction (HOLD/BUY/SELL)
            ml_probability: ML max class probability 0-100
        """
        # Normalize
        ai_signal = self._normalize(ai_signal)
        ml_signal = self._normalize(ml_signal)
        ai_conf = self._clamp(ai_confidence)
        ml_prob = self._clamp(ml_probability)
        
        # Rule 1: Both HOLD
        if ai_signal == "HOLD" and ml_signal == "HOLD":
            return FusionResult(
                signal="HOLD",
                combined_confidence=ai_conf,
                approved=False,
                reason="AI HOLD + ML HOLD",
                ai_signal=ai_signal,
                ai_confidence=ai_conf,
                ml_signal=ml_signal,
                ml_probability=ml_prob,
                rule_applied="both_hold",
            )
        
        # Rule 2: AI HOLD blocks all
        if ai_signal == "HOLD" and self.config.ai_hold_blocks_all:
            return FusionResult(
                signal="HOLD",
                combined_confidence=ai_conf,
                approved=False,
                reason=f"AI HOLD blocks ML {ml_signal}",
                ai_signal=ai_signal,
                ai_confidence=ai_conf,
                ml_signal=ml_signal,
                ml_probability=ml_prob,
                rule_applied="ai_hold_blocks",
            )
        
        # Rule 3: ML HOLD blocks AI
        if ml_signal == "HOLD" and self.config.ml_hold_blocks_ai:
            return FusionResult(
                signal="HOLD",
                combined_confidence=ai_conf,
                approved=False,
                reason=f"ML HOLD blocks AI {ai_signal}",
                ai_signal=ai_signal,
                ai_confidence=ai_conf,
                ml_signal=ml_signal,
                ml_probability=ml_prob,
                rule_applied="ml_hold_blocks",
            )
        
        # Rule 4: Agreement
        if ai_signal == ml_signal:
            combined = (
                ai_conf * self.config.ai_weight +
                ml_prob * self.config.ml_weight
            )
            approved = combined >= self.config.min_confidence
            
            return FusionResult(
                signal=ai_signal,
                combined_confidence=round(combined, 2),
                approved=approved,
                reason=f"ML confirms {ml_signal} ({ml_prob:.1f}%)",
                ai_signal=ai_signal,
                ai_confidence=ai_conf,
                ml_signal=ml_signal,
                ml_probability=ml_prob,
                rule_applied="agreement",
            )
        
        # Rule 5: Conflict
        penalized = ai_conf * self.config.conflict_penalty
        
        return FusionResult(
            signal="HOLD",
            combined_confidence=round(penalized, 2),
            approved=False,
            reason=f"Conflict: AI={ai_signal}, ML={ml_signal} (ML={ml_prob:.1f}%)",
            ai_signal=ai_signal,
            ai_confidence=ai_conf,
            ml_signal=ml_signal,
            ml_probability=ml_prob,
            rule_applied="conflict",
        )
    
    @staticmethod
    def _normalize(signal: str) -> Literal["HOLD", "BUY", "SELL"]:
        """Normalize signal names."""
        if signal is None:
            return "HOLD"
        value = str(signal).strip().upper()
        aliases = {
            "LONG": "BUY",
            "SHORT": "SELL",
            "NEUTRAL": "HOLD",
            "WAIT": "HOLD",
            "NONE": "HOLD",
        }
        return aliases.get(value, value)  # type: ignore
    
    @staticmethod
    def _clamp(value: float) -> float:
        """Clamp to [0, 100]."""
        if value is None:
            return 0.0
        try:
            num = float(value)
        except (TypeError, ValueError):
            return 0.0
        if 0.0 <= num <= 1.0:
            num *= 100.0
        return max(0.0, min(100.0, num))


# Backward compatibility: standalone function
def fuse_ai_ml(
    ai_signal: Literal["HOLD", "BUY", "SELL"],
    ai_confidence: float,
    ml_signal: Literal["HOLD", "BUY", "SELL"],
    ml_probability: float,
    config: FusionConfig | None = None,
) -> FusionResult:
    """
    Backward compatibility wrapper for fuse_ai_ml.
    
    Use MLFusion().fuse() for new code.
    """
    fusion = MLFusion(config)
    return fusion.fuse(ai_signal, ai_confidence, ml_signal, ml_probability)