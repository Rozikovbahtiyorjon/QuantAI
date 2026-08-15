from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from src.trading_activity_optimizer import (
    ActivityAction,
    ActivityOptimization,
)


@dataclass(frozen=True)
class ActivityPolicyDecision:
    action: ActivityAction
    approved: bool
    reason: str
    adjustments: Mapping[str, float]
    confidence: float


class TradingActivityPolicy:
    def __init__(
        self,
        *,
        minimum_confidence: float = 0.70,
        maximum_adjustment: float = 0.02,
    ) -> None:
        if not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError(
                "minimum_confidence must be between 0.0 and 1.0."
            )

        if maximum_adjustment <= 0.0:
            raise ValueError(
                "maximum_adjustment must be positive."
            )

        self.minimum_confidence = float(
            minimum_confidence
        )
        self.maximum_adjustment = float(
            maximum_adjustment
        )

    def evaluate(
        self,
        optimization: ActivityOptimization,
    ) -> ActivityPolicyDecision:
        if not isinstance(
            optimization,
            ActivityOptimization,
        ):
            raise TypeError(
                "optimization must be ActivityOptimization."
            )

        if (
            optimization.action is ActivityAction.HOLD
        ):
            return ActivityPolicyDecision(
                action=ActivityAction.HOLD,
                approved=False,
                reason="No activity adjustment is required.",
                adjustments={
                    "entry_threshold": 0.0,
                    "confidence_threshold": 0.0,
                },
                confidence=optimization.confidence,
            )

        if (
            optimization.confidence
            < self.minimum_confidence
        ):
            return ActivityPolicyDecision(
                action=ActivityAction.HOLD,
                approved=False,
                reason=(
                    "Optimization confidence is below "
                    "the policy threshold."
                ),
                adjustments={
                    "entry_threshold": 0.0,
                    "confidence_threshold": 0.0,
                },
                confidence=optimization.confidence,
            )

        adjustments = {}

        for name, value in optimization.adjustments.items():
            numeric_value = float(value)

            if (
                abs(numeric_value)
                > self.maximum_adjustment
            ):
                numeric_value = (
                    self.maximum_adjustment
                    if numeric_value > 0
                    else -self.maximum_adjustment
                )

            adjustments[name] = numeric_value

        return ActivityPolicyDecision(
            action=optimization.action,
            approved=True,
            reason=(
                "Activity optimization passed "
                "the policy safety checks."
            ),
            adjustments=adjustments,
            confidence=optimization.confidence,
        )

    def evaluate_from_mapping(
        self,
        data: Mapping[str, object],
    ) -> ActivityPolicyDecision:
        if not isinstance(data, Mapping):
            raise TypeError("data must be a mapping.")

        required = (
            "action",
            "confidence",
            "adjustments",
            "reason",
        )

        missing = [
            name
            for name in required
            if name not in data
        ]

        if missing:
            raise ValueError(
                "Missing fields: "
                + ", ".join(missing)
                + "."
            )

        action = data["action"]

        if not isinstance(action, ActivityAction):
            action = ActivityAction(str(action))

        adjustments = data["adjustments"]

        if not isinstance(adjustments, Mapping):
            raise TypeError(
                "adjustments must be a mapping."
            )

        optimization = ActivityOptimization(
            action=action,
            trade_target=int(
                data.get("trade_target", 0)
            ),
            confidence=float(
                data["confidence"]
            ),
            reason=str(data["reason"]),
            adjustments={
                str(key): float(value)
                for key, value in adjustments.items()
            },
        )

        return self.evaluate(optimization)