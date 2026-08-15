from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class StabilityThresholds:
    min_win_rate: float = 0.45
    max_drawdown: float = 0.20
    min_profit_factor: float = 1.0
    min_trade_count: int = 10
    max_return_drop: float = 0.10


@dataclass(frozen=True)
class StabilitySnapshot:
    status: str
    reasons: tuple[str, ...]
    return_value: float
    win_rate: float
    drawdown: float
    profit_factor: float
    trade_count: int


class ChampionStabilityMonitor:
    def __init__(
        self,
        thresholds: StabilityThresholds | None = None,
    ) -> None:
        self.thresholds = thresholds or StabilityThresholds()

    @staticmethod
    def _value(
        metrics: Mapping[str, Any],
        key: str,
        default: float = 0.0,
    ) -> float:
        value = metrics.get(key, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _trade_count(metrics: Mapping[str, Any]) -> int:
        value = metrics.get("trade_count", 0)
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 0

    def analyze(
        self,
        current: Mapping[str, Any],
        baseline: Mapping[str, Any] | None = None,
    ) -> StabilitySnapshot:
        baseline = baseline or {}

        if not current:
            return StabilitySnapshot(
                status="WARNING",
                reasons=("insufficient_data",),
                return_value=0.0,
                win_rate=0.0,
                drawdown=0.0,
                profit_factor=0.0,
                trade_count=0,
            )

        return_value = self._value(
            current,
            "return",
            self._value(current, "total_return"),
        )
        win_rate = self._value(current, "win_rate")
        drawdown = abs(self._value(current, "drawdown"))
        profit_factor = self._value(current, "profit_factor")
        trade_count = self._trade_count(current)

        baseline_return = self._value(
            baseline,
            "return",
            self._value(baseline, "total_return"),
        )

        reasons: list[str] = []

        if win_rate < self.thresholds.min_win_rate:
            reasons.append("low_win_rate")

        if drawdown > self.thresholds.max_drawdown:
            reasons.append("high_drawdown")

        if profit_factor < self.thresholds.min_profit_factor:
            reasons.append("low_profit_factor")

        if trade_count < self.thresholds.min_trade_count:
            reasons.append("low_trade_count")

        if baseline and baseline_return > 0:
            return_drop = (baseline_return - return_value) / baseline_return
            if return_drop > self.thresholds.max_return_drop:
                reasons.append("return_degradation")

        if "high_drawdown" in reasons or "low_profit_factor" in reasons:
            status = "DEGRADED"
        elif reasons:
            status = "WARNING"
        else:
            status = "STABLE"

        return StabilitySnapshot(
            status=status,
            reasons=tuple(reasons),
            return_value=return_value,
            win_rate=win_rate,
            drawdown=drawdown,
            profit_factor=profit_factor,
            trade_count=trade_count,
        )

    def is_stable(
        self,
        current: Mapping[str, Any],
        baseline: Mapping[str, Any] | None = None,
    ) -> bool:
        return self.analyze(current, baseline).status == "STABLE"