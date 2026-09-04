"""
====================================================
QuantAI Professional AI Trading System
Confidence Engine v3.1
====================================================

Central engine for combining analytical scores
into a directional trading confidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


# ====================================================
# SCORE COMPONENT
# ====================================================

@dataclass
class ScoreComponent:

    name: str

    score: float

    weight: float = 1.0

    description: str = ""


# ====================================================
# CONFIDENCE RESULT
# ====================================================

@dataclass
class ConfidenceResult:

    total_score: float

    confidence: float

    probability: float

    decision: str

    components: List[ScoreComponent] = field(
        default_factory=list
    )

    reasons: List[str] = field(
        default_factory=list
    )


# ====================================================
# CONFIDENCE ENGINE
# ====================================================

class ConfidenceEngine:

    """
    Central QuantAI Confidence Engine.

    Converts analytical component scores into:

        - Total Score
        - Confidence
        - Probability
        - Decision
    """

    def __init__(self):

        self.components: List[ScoreComponent] = []

        self.weights: Dict[str, float] = {

            "trend": 1.50,

            "momentum": 1.20,

            "volume": 1.10,

            "volatility": 1.00,

            "liquidity": 1.40,

            "structure": 1.30,

            "regime": 1.50,

        }

    # ====================================================
    # RESET
    # ====================================================

    def reset(self):

        self.components.clear()

    # ====================================================
    # ADD COMPONENT
    # ====================================================

    def add_component(
        self,
        name: str,
        score: float,
        description: str = "",
    ):

        weight = self.weights.get(
            name.lower(),
            1.0,
        )

        self.components.append(
            ScoreComponent(
                name=name,
                score=float(score),
                weight=weight,
                description=description,
            )
        )

    # ====================================================
    # CALCULATE SCORE
    # ====================================================

    def calculate_score(self) -> float:

        if not self.components:

            return 0.0

        weighted_sum = 0.0

        total_weight = 0.0

        for component in self.components:

            weighted_sum += (
                component.score
                * component.weight
            )

            total_weight += component.weight

        if total_weight == 0:

            return 0.0

        return round(
            weighted_sum / total_weight,
            2,
        )

    # ====================================================
    # CALCULATE DIRECTIONAL CONFIDENCE
    # ====================================================

    def calculate_confidence(
        self,
        score: float,
    ) -> float:

        """
        Converts directional Score into confidence.

        Important:
        Confidence represents the strength of the
        directional signal, not whether the score
        is positive or negative.

        Example:

            score =  0.0 -> 50%
            score = +1.0 -> 60%
            score = -1.0 -> 60%
            score = +2.0 -> 70%
            score = -2.0 -> 70%

        Therefore a strong SELL score is not treated
        as low confidence.
        """

        strength = abs(float(score))

        confidence = 50.0 + (
            strength * 10.0
        )

        confidence = max(
            0.0,
            confidence,
        )

        confidence = min(
            100.0,
            confidence,
        )

        return round(
            confidence,
            2,
        )

    # ====================================================
    # CALCULATE PROBABILITY
    # ====================================================

    def calculate_probability(
        self,
        confidence: float,
    ) -> float:

        return round(
            confidence,
            2,
        )

    # ====================================================
    # GET CONTINUOUS PROBABILITY (0.0 - 1.0)
    # ====================================================

    def get_continuous_probability(self, score: float) -> float:
        """
        Returns continuous probability in [0.0, 1.0] range.
        
        Uses sigmoid function centered at 0:
        - score = 0.0  -> 0.5 (neutral)
        - score > 0    -> > 0.5 (bullish bias)
        - score < 0    -> < 0.5 (bearish bias)
        
        The steepness parameter controls sensitivity.
        """
        import math
        
        steepness = 1.5  # Configurable sensitivity
        
        # Sigmoid centered at 0
        probability = 1.0 / (1.0 + math.exp(-score * steepness))
        
        return round(probability, 4)

    # ====================================================
    # DECIDE
    # ====================================================

    def decide(
        self,
        score: float,
        confidence: float,
    ) -> str:

        """
        Determines directional AI decision.

        Score determines direction.

        Confidence determines whether the signal
        is strong enough to trade.
        """

        if confidence < 60.0:

            return "HOLD"

        if score >= 1.0:

            return "BUY"

        if score <= -1.0:

            return "SELL"

        return "HOLD"

    # ====================================================
    # EVALUATE
    # ====================================================

    def evaluate(self) -> ConfidenceResult:

        score = self.calculate_score()

        confidence = self.calculate_confidence(
            score
        )

        probability = self.calculate_probability(
            confidence
        )

        decision = self.decide(
            score,
            confidence,
        )

        reasons = []

        for component in self.components:

            reasons.append(
                f"{component.name}: "
                f"{component.score:.2f}"
            )

        return ConfidenceResult(

            total_score=score,

            confidence=confidence,

            probability=probability,

            decision=decision,

            components=self.components.copy(),

            reasons=reasons,

        )

    # ====================================================
    # SUMMARY
    # ====================================================

    def summary(self) -> str:

        result = self.evaluate()

        return (
            f"Decision={result.decision} | "
            f"Score={result.total_score:.2f} | "
            f"Confidence="
            f"{result.confidence:.2f}%"
        )

    # ====================================================
    # PRINT REPORT
    # ====================================================

    def print_report(self):

        result = self.evaluate()

        print()

        print("=" * 60)

        print("CONFIDENCE ENGINE")

        print("=" * 60)

        print(
            f"Decision      : "
            f"{result.decision}"
        )

        print(
            f"Score         : "
            f"{result.total_score:.2f}"
        )

        print(
            f"Confidence    : "
            f"{result.confidence:.2f}%"
        )

        print(
            f"Probability   : "
            f"{result.probability:.2f}%"
        )

        print()

        print("Components:")

        for component in result.components:

            print(
                f"{component.name:<15}"
                f"{component.score:>7.2f}"
                f"   w={component.weight:.2f}"
            )

        print("=" * 60)


# ====================================================
# WEIGHTED GATE
# ====================================================

@dataclass
class WeightedGateConfig:
    """Configuration for Weighted Gate."""
    
    threshold: float = 0.75  # Minimum probability to take action
    min_confidence: float = 60.0  # Minimum confidence % to consider
    long_threshold: float = 0.5  # Probability threshold for LONG
    short_threshold: float = 0.5  # Probability threshold for SHORT


@dataclass
class WeightedGateResult:
    """Result of Weighted Gate evaluation."""
    
    action: str  # "LONG", "SHORT", "HOLD"
    probability: float
    confidence: float
    approved: bool
    reason: str


class WeightedGate:
    """
    Weighted Gate for probabilistic signal filtering.
    
    Replaces binary HOLD/BUY/SELL with continuous probability.
    Action is taken only when probability exceeds threshold.
    
    Logic:
    - probability > long_threshold  -> LONG (if approved)
    - probability < (1 - short_threshold) -> SHORT (if approved)
    - otherwise -> HOLD
    """
    
    def __init__(self, config: WeightedGateConfig | None = None):
        self.config = config or WeightedGateConfig()
    
    def evaluate(
        self,
        probability: float,
        confidence: float,
        ai_signal: str = "HOLD",
    ) -> WeightedGateResult:
        """
        Evaluate continuous probability through weighted gate.
        
        Args:
            probability: Continuous probability from AI (0.0-1.0)
            confidence: Confidence percentage (0-100)
            ai_signal: Original AI directional signal
        """
        # Check minimum confidence
        if confidence < self.config.min_confidence:
            return WeightedGateResult(
                action="HOLD",
                probability=probability,
                confidence=confidence,
                approved=False,
                reason=f"Confidence {confidence:.1f}% below minimum {self.config.min_confidence:.1f}%",
            )
        
        # Determine action based on probability thresholds
        long_thresh = self.config.long_threshold
        short_thresh = 1.0 - self.config.short_threshold
        
        if probability > long_thresh:
            action = "BUY"
            approved = probability >= self.config.threshold
            reason = f"Probability {probability:.2%} > LONG threshold {long_thresh:.2%}"
        elif probability < short_thresh:
            action = "SELL"
            approved = probability <= (1.0 - self.config.threshold)
            reason = f"Probability {probability:.2%} < SHORT threshold {short_thresh:.2%}"
        else:
            action = "HOLD"
            approved = False
            reason = f"Probability {probability:.2%} in neutral zone [{short_thresh:.2%}, {long_thresh:.2%}]"
        
        # If not approved, convert to HOLD
        if not approved:
            action = "HOLD"
            reason = f"Below approval threshold {self.config.threshold:.2%}: {reason}"
        
        return WeightedGateResult(
            action=action,
            probability=probability,
            confidence=confidence,
            approved=approved,
            reason=reason,
        )


# ====================================================
# ENTRY QUALITY SCORE — distinct from Confidence
# ====================================================

@dataclass
class EntryQualityScore:
    """Quality of the specific entry point, not general confidence.
    
    Confidence = strength of directional signal (trend/momentum)
    Quality = quality of the entry location (pullback zone, exhaustion, trigger)
    
    Example: strong trend (confidence 80%) but poor entry (chasing top, quality 30%)
    vs weak trend (confidence 55%) but perfect pullback entry (quality 85%)
    """
    
    quality: float  # 0-100
    zone_score: float = 0.0  # distance to zone, 0-100
    exhaustion_score: float = 0.0  # RSI/volume/wick exhaustion
    trigger_score: float = 0.0  # trigger quality (engulfing, reclaim)
    reasons: List[str] = field(default_factory=list)


class EntryQualityEngine:
    """
    Calculates Entry Quality Score — distinct from Confidence.
    
    Quality = f(zone, exhaustion, trigger) — how good is this specific entry point
    Confidence = f(trend, momentum, volume, volatility) — how strong is the direction
    """
    
    def calculate(
        self,
        setup_result: Any = None,  # SetupResult from SetupDetector
        pullback_result: Any = None,  # TrendPullbackResult
        mean_reversion_result: Any = None,  # MeanReversionResult
        zone_result: Any = None,  # POIResult
        row: Any = None,
    ) -> EntryQualityScore:
        """
        Calculate quality of the specific entry point.
        
        Args:
            setup_result: SetupResult with setup, confidence, zone
            pullback_result: TrendPullbackResult with quality, zone, invalidated
            mean_reversion_result: MeanReversionResult with stages
            zone_result: POIResult with nearest zones
            row: current bar row for fallback
        """
        reasons: List[str] = []
        zone_score = 50.0
        exhaustion_score = 50.0
        trigger_score = 50.0
        
        # Zone quality: distance to ideal zone (0 ATR = perfect)
        if pullback_result and hasattr(pullback_result, 'zone') and pullback_result.zone:
            try:
                dist = abs(pullback_result.pullback_dist_atr)
                # Ideal 0.5-1.5 ATR pullback
                if 0.8 <= dist <= 1.2:
                    zone_score = 90.0
                    reasons.append(f"zone ideal {dist:.2f} ATR")
                elif 0.5 <= dist <= 2.0:
                    zone_score = 70.0
                    reasons.append(f"zone good {dist:.2f} ATR")
                else:
                    zone_score = 40.0
                    reasons.append(f"zone poor {dist:.2f} ATR")
            except Exception:
                zone_score = 50.0
        elif setup_result and hasattr(setup_result, 'bb_position'):
            try:
                bb_pos = float(getattr(setup_result, 'bb_position', 0.5))
                # For pullback, ideal BB 0.3-0.5, for mean reversion BB 0.0-0.2 or 0.8-1.0
                if setup_result.setup in ("LONG_PULLBACK", "SHORT_PULLBACK"):
                    if 0.3 <= bb_pos <= 0.6:
                        zone_score = 80.0
                    else:
                        zone_score = 50.0
                elif "MEAN_REVERSION" in str(setup_result.setup):
                    if bb_pos <= 0.2 or bb_pos >= 0.8:
                        zone_score = 85.0
                reasons.append(f"setup {setup_result.setup} BB {bb_pos:.2f}")
            except Exception:
                pass
        
        # Exhaustion quality: RSI, volume, wick
        if setup_result and hasattr(setup_result, 'rsi'):
            try:
                rsi = float(getattr(setup_result, 'rsi', 50))
                if 30 <= rsi <= 70:
                    if 45 <= rsi <= 55:
                        exhaustion_score = 85.0
                        reasons.append(f"exhaustion RSI ideal {rsi:.0f}")
                    else:
                        exhaustion_score = 65.0
                elif rsi < 30 or rsi > 70:
                    exhaustion_score = 75.0
                    reasons.append(f"exhaustion RSI extreme {rsi:.0f}")
            except Exception:
                pass
        
        # Trigger quality: from pullback or mean reversion result
        if pullback_result and hasattr(pullback_result, 'quality'):
            try:
                trigger_score = float(pullback_result.quality) * 100
                reasons.append(f"trigger pullback quality {pullback_result.quality:.2f}")
            except Exception:
                pass
        elif setup_result and hasattr(setup_result, 'confidence'):
            try:
                trigger_score = float(setup_result.confidence) * 100 if setup_result.confidence <= 1 else float(setup_result.confidence)
                reasons.append(f"trigger setup confidence {trigger_score:.0f}")
            except Exception:
                pass
        
        # Zone proximity bonus from POI
        if zone_result and hasattr(zone_result, 'nearest_support'):
            try:
                # If near strong support for LONG, quality up
                if zone_result.nearest_support and zone_result.nearest_support.strength > 0.6:
                    if zone_result.nearest_support.distance_pct < 0.5:
                        zone_score = min(100, zone_score + 10)
                        reasons.append(f"POI support near {zone_result.nearest_support.distance_pct:.2f}% strength {zone_result.nearest_support.strength:.2f}")
            except Exception:
                pass
        
        quality = (zone_score * 0.4 + exhaustion_score * 0.3 + trigger_score * 0.3)
        
        return EntryQualityScore(
            quality=round(quality, 1),
            zone_score=round(zone_score, 1),
            exhaustion_score=round(exhaustion_score, 1),
            trigger_score=round(trigger_score, 1),
            reasons=reasons,
        )


# ====================================================
# MODULE EXPORT
# ====================================================