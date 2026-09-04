"""
Signal Decision Funnel — P2.10/P2.11

Tracks 9 stages for 0-trades diagnostics:
bars → candidate signals → AI accepted → ML accepted → confidence accepted
→ risk accepted → orders → fills → closed trades

Classifies 0-trades into 6 spec reasons:
NO_SIGNAL / ML_REJECTED / CONFIDENCE_REJECTED / RISK_REJECTED / EXECUTION_REJECTED / NO_FILL
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class FunnelCounts:
    """P2.10 9-stage funnel. bars is external, rest are pipeline counts."""
    raw_signals: int = 0  # candidate signals generated (BUY/SELL attempts)
    ai_accepted: int = 0  # after AI filter
    ml_accepted: int = 0  # after ML filter
    confidence_accepted: int = 0  # after confidence threshold
    risk_accepted: int = 0  # after risk (exposure/drawdown)
    orders_submitted: int = 0  # orders sent to execution
    orders_filled: int = 0  # fills received
    trades_closed: int = 0  # closed trades (PnL realized)
    # detailed blocking counters for diagnostics
    blocked_by_regime: int = 0
    blocked_by_ml: int = 0
    blocked_by_risk: int = 0
    blocked_by_confidence: int = 0
    blocked_by_execution: int = 0

    # P2.11 spec 6-way classification for 0 trades
    def classify_zero(self) -> str:
        """P2.11: Separate 6 reasons for 0 trades."""
        if self.raw_signals == 0:
            return "NO_SIGNAL"
        if self.ml_accepted == 0:
            return "ML_REJECTED"
        if self.confidence_accepted == 0:
            return "CONFIDENCE_REJECTED"
        if self.risk_accepted == 0:
            return "RISK_REJECTED"
        if self.orders_submitted == 0:
            return "EXECUTION_REJECTED"
        if self.orders_filled == 0:
            return "NO_FILL"
        if self.trades_closed == 0:
            return "NO_FILL"  # fills but no closed trades → fill without close
        if self.trades_closed < 30:
            return "LOW_ACTIVITY"
        return "BAD_PERFORMANCE"

    def classify_zero_detailed(self) -> Dict[str, str]:
        """Detailed blocking reason with counts."""
        reason = self.classify_zero()
        details = {
            "NO_SIGNAL": f"raw_signals=0 (no BUY/SELL candidate in {self.raw_signals} bars)",
            "ML_REJECTED": f"ML filtered all: ai={self.ai_accepted} -> ml=0",
            "CONFIDENCE_REJECTED": f"confidence filtered: ml={self.ml_accepted} -> conf=0",
            "RISK_REJECTED": f"risk blocked: conf={self.confidence_accepted} -> risk=0 (exposure/drawdown)",
            "EXECUTION_REJECTED": f"execution rejected: risk={self.risk_accepted} -> orders=0 (dedup/rate_limit/validation)",
            "NO_FILL": f"no fills: orders={self.orders_submitted} -> fills=0 (queue/latency/spread)",
        }
        return {"reason": reason, "detail": details.get(reason, ""), "funnel": self.to_dict()}

    def to_dict(self) -> Dict[str, int]:
        return {
            "raw_signals": self.raw_signals,
            "ai_accepted": self.ai_accepted,
            "ml_accepted": self.ml_accepted,
            "confidence_accepted": self.confidence_accepted,
            "risk_accepted": self.risk_accepted,
            "orders_submitted": self.orders_submitted,
            "orders_filled": self.orders_filled,
            "trades_closed": self.trades_closed,
            "blocked_by_ml": self.blocked_by_ml,
            "blocked_by_confidence": self.blocked_by_confidence,
            "blocked_by_risk": self.blocked_by_risk,
            "blocked_by_execution": self.blocked_by_execution,
            "classification": self.classify_zero(),
        }

    def render(self, bars: int) -> str:
        d = self.to_dict()
        lines = [f"{bars} market bars"]
        stages = [
            ("raw_signals", "candidate signals"),
            ("ai_accepted", "AI accepted"),
            ("ml_accepted", "ML accepted"),
            ("confidence_accepted", "confidence accepted"),
            ("risk_accepted", "risk accepted"),
            ("orders_submitted", "orders"),
            ("orders_filled", "fills"),
            ("trades_closed", "closed trades"),
        ]
        for k, label in stages:
            lines.append(f"  -> {label}: {d[k]}")
        det = self.classify_zero_detailed()
        lines.append(f"Classification: {det['reason']} -- {det['detail']}")
        return "\n".join(lines)

    # Backward compat aliases for old code
    @property
    def classification(self) -> str:
        return self.classify_zero()


__all__ = ["FunnelCounts"]
