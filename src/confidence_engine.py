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
# MODULE EXPORT
# ====================================================