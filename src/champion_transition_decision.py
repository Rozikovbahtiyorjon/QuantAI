from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ChampionTransitionDecision:
    action: str
    reason: str
    score: float
    margin: float
    stable: bool


class ChampionTransitionDecisionEngine:
    def __init__(
        self,
        min_score_margin: float = 0.0,
        min_stability_score: float = 0.5,
    ) -> None:
        if min_score_margin < 0:
            raise ValueError("min_score_margin must be non-negative")
        if not 0.0 <= min_stability_score <= 1.0:
            raise ValueError("min_stability_score must be between 0 and 1")

        self.min_score_margin = float(min_score_margin)
        self.min_stability_score = float(min_stability_score)

    def decide(
        self,
        champion: Mapping[str, Any] | None,
        candidate: Mapping[str, Any] | None,
    ) -> ChampionTransitionDecision:
        champion = champion or {}
        candidate = candidate or {}

        champion_score = self._score(champion)
        candidate_score = self._score(candidate)
        margin = candidate_score - champion_score

        stability = self._stability(candidate)
        stable = stability >= self.min_stability_score

        if not candidate:
            return ChampionTransitionDecision(
                "REJECT",
                "candidate_missing",
                candidate_score,
                margin,
                False,
            )

        if not champion:
            if stable:
                return ChampionTransitionDecision(
                    "PROMOTE",
                    "no_current_champion",
                    candidate_score,
                    margin,
                    True,
                )

            return ChampionTransitionDecision(
                "HOLD",
                "candidate_not_stable",
                candidate_score,
                margin,
                False,
            )

        if not stable:
            return ChampionTransitionDecision(
                "HOLD",
                "candidate_not_stable",
                candidate_score,
                margin,
                False,
            )

        if margin >= self.min_score_margin:
            return ChampionTransitionDecision(
                "REPLACE",
                "candidate_outperforms_champion",
                candidate_score,
                margin,
                True,
            )

        return ChampionTransitionDecision(
            "KEEP",
            "champion_remains_superior",
            candidate_score,
            margin,
            True,
        )

    @staticmethod
    def _score(metrics: Mapping[str, Any]) -> float:
        if "score" in metrics:
            return float(metrics["score"])

        profitability = float(metrics.get("profitability", 0.0))
        return_rate = float(
            metrics.get("return", metrics.get("return_rate", 0.0))
        )
        win_rate = float(metrics.get("win_rate", 0.0))
        drawdown = float(metrics.get("drawdown", 0.0))

        return (
            profitability
            + return_rate
            + win_rate
            - max(0.0, drawdown)
        )

    @staticmethod
    def _stability(metrics: Mapping[str, Any]) -> float:
        value = metrics.get(
            "stability",
            metrics.get("stability_score", 1.0),
        )

        return max(0.0, min(1.0, float(value)))