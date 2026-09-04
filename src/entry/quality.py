"""
ENTRY-41 — Quality (PHASE 8)

Split:
  RegimeQuality, SetupQuality, TriggerQuality, ConfirmationQuality, MLQuality, ZoneQuality

ENTRY-42 — Entry Quality Score: deterministic weights, diagnostic not magic maximizer
ENTRY-43 — Quality reason codes: REGIME_WEAK, SETUP_HIGH_QUALITY, etc.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class QualityResult:
    score: float  # 0.0-1.0
    reason_code: str
    reason: str


class RegimeQuality:
    def evaluate(self, regime: str, adx: float, trend_score: float) -> QualityResult:
        if regime in ("TREND_UP", "TREND_DOWN") and adx > 25:
            return QualityResult(score=0.9, reason_code="REGIME_STRONG", reason=f"{regime} ADX {adx:.0f}")
        if regime == "RANGE" and adx < 20:
            return QualityResult(score=0.7, reason_code="REGIME_RANGE_OK", reason=f"RANGE ADX {adx:.0f}")
        return QualityResult(score=0.3, reason_code="REGIME_WEAK", reason=f"{regime} ADX {adx:.0f} weak")


class SetupQuality:
    def evaluate(self, setup: str, confidence: float) -> QualityResult:
        if setup == "NONE":
            return QualityResult(score=0.0, reason_code="NO_SETUP", reason="no setup")
        if confidence >= 0.75:
            return QualityResult(score=0.9, reason_code="SETUP_HIGH_QUALITY", reason=f"{setup} {confidence:.2f}")
        if confidence >= 0.6:
            return QualityResult(score=0.7, reason_code="SETUP_MEDIUM", reason=f"{setup} {confidence:.2f}")
        return QualityResult(score=0.4, reason_code="SETUP_LOW_QUALITY", reason=f"{setup} low {confidence:.2f}")


class TriggerQuality:
    def evaluate(self, trigger_type: str, is_triggered: bool) -> QualityResult:
        if not is_triggered:
            return QualityResult(score=0.0, reason_code="TRIGGER_NOT_FIRED", reason=f"{trigger_type} not triggered")
        if trigger_type in ("MSB", "Sweep"):
            return QualityResult(score=0.85, reason_code="TRIGGER_CONFIRMED", reason=trigger_type)
        if trigger_type == "Retest":
            return QualityResult(score=0.75, reason_code="TRIGGER_CONFIRMED", reason=trigger_type)
        return QualityResult(score=0.6, reason_code="TRIGGER_WEAK", reason=trigger_type)


class ConfirmationQuality:
    def evaluate(self, structure: float, momentum: float, volume: float, order_flow: bool, mtf: bool) -> QualityResult:
        # Independent groups: structure, momentum, volume are one group? No, each is separate but we count groups
        passed = sum([abs(structure) > 0.3, abs(momentum) > 0.2, abs(volume) > 0.3, order_flow, mtf])
        score = passed / 5.0
        if passed >= 3:
            return QualityResult(score=score, reason_code="CONFIRMATION_STRONG", reason=f"{passed}/5 groups")
        if passed >= 2:
            return QualityResult(score=score, reason_code="CONFIRMATION_MEDIUM", reason=f"{passed}/5")
        return QualityResult(score=score, reason_code="CONFIRMATION_WEAK", reason=f"{passed}/5")


class MLQuality:
    def evaluate(self, ml_prob: float, ml_state: str) -> QualityResult:
        if ml_state == "PLACEHOLDER":
            return QualityResult(score=0.0, reason_code="ML_UNAVAILABLE", reason="ML placeholder")
        if ml_prob >= 0.65:
            return QualityResult(score=0.9, reason_code="ML_STRONG", reason=f"P {ml_prob:.2f}")
        if ml_prob >= 0.55:
            return QualityResult(score=0.6, reason_code="ML_MEDIUM", reason=f"P {ml_prob:.2f}")
        return QualityResult(score=0.3, reason_code="ML_WEAK", reason=f"P {ml_prob:.2f}")


class ZoneQuality:
    def evaluate(self, distance_pct: float, strength: float, freshness: float) -> QualityResult:
        # Near strong fresh zone is high quality
        if distance_pct < 0.3 and strength > 0.6 and freshness > 0.7:
            return QualityResult(score=0.9, reason_code="ZONE_FRESH", reason=f"dist {distance_pct:.2f}% strength {strength:.2f}")
        if distance_pct < 0.5 and strength > 0.4:
            return QualityResult(score=0.7, reason_code="ZONE_OK", reason=f"dist {distance_pct:.2f}%")
        return QualityResult(score=0.4, reason_code="ZONE_WEAK", reason=f"dist {distance_pct:.2f}% strength {strength:.2f}")


class EntryQualityAggregator:
    """ENTRY-42: Deterministic weights, diagnostic score."""

    def aggregate(self, regime_q, setup_q, trigger_q, confirmation_q, ml_q, zone_q) -> tuple[float, str]:
        # Fixed deterministic weights (no optimization)
        weights = {
            "regime": 0.15,
            "setup": 0.25,
            "trigger": 0.20,
            "confirmation": 0.15,
            "ml": 0.10,
            "zone": 0.15,
        }
        score = (
            regime_q.score * weights["regime"]
            + setup_q.score * weights["setup"]
            + trigger_q.score * weights["trigger"]
            + confirmation_q.score * weights["confirmation"]
            + ml_q.score * weights["ml"]
            + zone_q.score * weights["zone"]
        )
        # Reason codes for Supervisor
        codes = [regime_q.reason_code, setup_q.reason_code, trigger_q.reason_code, confirmation_q.reason_code, ml_q.reason_code, zone_q.reason_code]
        reason = f"Quality {score:.2f} [{', '.join(codes)}]"
        return round(score, 3), reason
