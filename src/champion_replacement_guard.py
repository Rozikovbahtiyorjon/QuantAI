from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ReplacementGuardConfig:
    min_improvement: float = 0.0
    min_profit_factor: float = 1.0
    max_drawdown: float = 0.20
    min_win_rate: float = 0.45
    min_trade_count: int = 10


@dataclass(frozen=True)
class ReplacementDecision:
    approved: bool
    reason: str
    failed_checks: tuple[str, ...]


class ChampionReplacementGuard:
    def __init__(
        self,
        config: ReplacementGuardConfig | None = None,
    ) -> None:
        self.config = config or ReplacementGuardConfig()

    @staticmethod
    def _float(
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
    def _int(
        metrics: Mapping[str, Any],
        key: str,
        default: int = 0,
    ) -> int:
        value = metrics.get(key, default)
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def evaluate(
        self,
        champion: Mapping[str, Any],
        challenger: Mapping[str, Any],
    ) -> ReplacementDecision:
        if not champion or not challenger:
            return ReplacementDecision(
                approved=False,
                reason="insufficient_data",
                failed_checks=("insufficient_data",),
            )

        champion_return = self._float(
            champion,
            "return",
            self._float(champion, "total_return"),
        )
        challenger_return = self._float(
            challenger,
            "return",
            self._float(challenger, "total_return"),
        )
        challenger_pf = self._float(challenger, "profit_factor")
        challenger_dd = abs(self._float(challenger, "drawdown"))
        challenger_wr = self._float(challenger, "win_rate")
        challenger_trades = self._int(challenger, "trade_count")

        failed: list[str] = []

        if challenger_return <= champion_return + self.config.min_improvement:
            failed.append("return_not_improved")

        if challenger_pf < self.config.min_profit_factor:
            failed.append("profit_factor_below_minimum")

        if challenger_dd > self.config.max_drawdown:
            failed.append("drawdown_above_maximum")

        if challenger_wr < self.config.min_win_rate:
            failed.append("win_rate_below_minimum")

        if challenger_trades < self.config.min_trade_count:
            failed.append("trade_count_below_minimum")

        if failed:
            return ReplacementDecision(
                approved=False,
                reason="challenger_rejected",
                failed_checks=tuple(failed),
            )

        return ReplacementDecision(
            approved=True,
            reason="challenger_approved",
            failed_checks=(),
        )

    def should_replace(
        self,
        champion: Mapping[str, Any],
        challenger: Mapping[str, Any],
    ) -> bool:
        return self.evaluate(champion, challenger).approved