from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping


class ActivityAction(str, Enum):
    INCREASE = "INCREASE"
    DECREASE = "DECREASE"
    HOLD = "HOLD"


@dataclass(frozen=True)
class ActivitySnapshot:
    trades: int
    min_trades: int
    max_trades: int
    win_rate: float = 0.0
    average_quality: float = 0.0

    def __post_init__(self) -> None:
        for name in ("trades", "min_trades", "max_trades"):
            value = getattr(self, name)

            if not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")

            if value < 0:
                raise ValueError(f"{name} must be non-negative.")

        if self.min_trades > self.max_trades:
            raise ValueError("min_trades cannot exceed max_trades.")

        for name in ("win_rate", "average_quality"):
            value = getattr(self, name)

            if not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric.")

            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(
                    f"{name} must be between 0.0 and 1.0."
                )


@dataclass(frozen=True)
class ActivityOptimization:
    action: ActivityAction
    trade_target: int
    confidence: float
    reason: str
    adjustments: Mapping[str, float]


class TradingActivityOptimizer:
    def __init__(
        self,
        *,
        max_target_step: int = 2,
        min_confidence: float = 0.6,
        quality_floor: float = 0.45,
    ) -> None:
        if not isinstance(max_target_step, int):
            raise TypeError("max_target_step must be an integer.")

        if max_target_step < 1:
            raise ValueError(
                "max_target_step must be at least 1."
            )

        if not isinstance(min_confidence, (int, float)):
            raise TypeError(
                "min_confidence must be numeric."
            )

        if not 0.0 <= float(min_confidence) <= 1.0:
            raise ValueError(
                "min_confidence must be between 0.0 and 1.0."
            )

        if not isinstance(quality_floor, (int, float)):
            raise TypeError(
                "quality_floor must be numeric."
            )

        if not 0.0 <= float(quality_floor) <= 1.0:
            raise ValueError(
                "quality_floor must be between 0.0 and 1.0."
            )

        self.max_target_step = max_target_step
        self.min_confidence = float(min_confidence)
        self.quality_floor = float(quality_floor)

    @staticmethod
    def _clamp(
        value: float,
        lower: float,
        upper: float,
    ) -> float:
        return max(lower, min(upper, value))

    @staticmethod
    def _validate_mapping(
        diagnostics: Mapping[str, object],
    ) -> None:
        if not isinstance(diagnostics, Mapping):
            raise TypeError(
                "diagnostics must be a mapping."
            )

    def from_diagnostics(
        self,
        diagnostics: Mapping[str, object],
    ) -> ActivitySnapshot:
        self._validate_mapping(diagnostics)

        required = (
            "trades",
            "min_trades",
            "max_trades",
        )

        missing = [
            name
            for name in required
            if name not in diagnostics
        ]

        if missing:
            raise ValueError(
                "Missing diagnostic fields: "
                + ", ".join(missing)
                + "."
            )

        return ActivitySnapshot(
            trades=int(diagnostics["trades"]),
            min_trades=int(diagnostics["min_trades"]),
            max_trades=int(diagnostics["max_trades"]),
            win_rate=float(
                diagnostics.get("win_rate", 0.0)
            ),
            average_quality=float(
                diagnostics.get("average_quality", 0.0)
            ),
        )

    def optimize(
        self,
        snapshot: ActivitySnapshot,
    ) -> ActivityOptimization:
        if not isinstance(snapshot, ActivitySnapshot):
            raise TypeError(
                "snapshot must be ActivitySnapshot."
            )

        midpoint = (
            snapshot.min_trades
            + snapshot.max_trades
        ) / 2.0

        if snapshot.trades < snapshot.min_trades:
            gap = snapshot.min_trades - snapshot.trades

            step = min(
                self.max_target_step,
                max(1, gap),
            )

            target = min(
                snapshot.max_trades,
                snapshot.trades + step,
            )

            confidence = self._clamp(
                0.6
                + min(
                    gap / max(snapshot.min_trades, 1),
                    0.4,
                ),
                0.0,
                1.0,
            )

            if snapshot.average_quality < self.quality_floor:
                return ActivityOptimization(
                    action=ActivityAction.HOLD,
                    trade_target=snapshot.trades,
                    confidence=self.min_confidence,
                    reason=(
                        "Activity is low but signal quality "
                        "is below the safety floor."
                    ),
                    adjustments={
                        "entry_threshold": 0.0,
                        "confidence_threshold": 0.0,
                    },
                )

            return ActivityOptimization(
                action=ActivityAction.INCREASE,
                trade_target=target,
                confidence=confidence,
                reason=(
                    "Trading activity is below the "
                    "configured minimum."
                ),
                adjustments={
                    "entry_threshold": -0.01,
                    "confidence_threshold": -0.01,
                },
            )

        if snapshot.trades > snapshot.max_trades:
            excess = (
                snapshot.trades
                - snapshot.max_trades
            )

            step = min(
                self.max_target_step,
                max(1, excess),
            )

            target = max(
                snapshot.min_trades,
                snapshot.trades - step,
            )

            confidence = self._clamp(
                0.6
                + min(
                    excess / max(snapshot.max_trades, 1),
                    0.4,
                ),
                0.0,
                1.0,
            )

            return ActivityOptimization(
                action=ActivityAction.DECREASE,
                trade_target=target,
                confidence=confidence,
                reason=(
                    "Trading activity is above the "
                    "configured maximum."
                ),
                adjustments={
                    "entry_threshold": 0.01,
                    "confidence_threshold": 0.01,
                },
            )

        balance = abs(
            snapshot.trades - midpoint
        ) / max(
            snapshot.max_trades
            - snapshot.min_trades,
            1,
        )

        confidence = self._clamp(
            0.9 - balance,
            0.0,
            1.0,
        )

        if snapshot.average_quality < self.quality_floor:
            return ActivityOptimization(
                action=ActivityAction.HOLD,
                trade_target=snapshot.trades,
                confidence=max(
                    confidence,
                    self.min_confidence,
                ),
                reason=(
                    "Activity is within range but "
                    "signal quality is below the safety floor."
                ),
                adjustments={
                    "entry_threshold": 0.0,
                    "confidence_threshold": 0.0,
                },
            )

        return ActivityOptimization(
            action=ActivityAction.HOLD,
            trade_target=snapshot.trades,
            confidence=max(
                confidence,
                self.min_confidence,
            ),
            reason=(
                "Trading activity is within "
                "the configured range."
            ),
            adjustments={
                "entry_threshold": 0.0,
                "confidence_threshold": 0.0,
            },
        )

    def optimize_from_diagnostics(
        self,
        diagnostics: Mapping[str, object],
    ) -> ActivityOptimization:
        return self.optimize(
            self.from_diagnostics(diagnostics)
        )