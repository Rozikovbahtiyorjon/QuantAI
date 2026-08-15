from __future__ import annotations

from dataclasses import dataclass

from src.champion_evolution import (
    ChampionComparison,
    ChampionDecision,
    ChampionEvaluation,
    ChampionEvolution,
)


@dataclass(frozen=True)
class ChampionState:
    strategy_id: str
    score: float


class StrategyChampion:
    def __init__(
        self,
        *,
        minimum_improvement: float = 0.01,
        minimum_robustness: float = 0.5,
        minimum_evidence: float = 0.5,
    ) -> None:
        self._evolution = ChampionEvolution(
            minimum_improvement=minimum_improvement,
            minimum_robustness=minimum_robustness,
            minimum_evidence=minimum_evidence,
        )
        self._champion: ChampionEvaluation | None = None

    @property
    def champion(self) -> ChampionEvaluation | None:
        return self._champion

    def state(self) -> ChampionState | None:
        if self._champion is None:
            return None

        return ChampionState(
            strategy_id=self._champion.strategy_id,
            score=self._evolution.score(self._champion),
        )

    def set_initial(
        self,
        evaluation: ChampionEvaluation,
    ) -> ChampionState:
        self._validate(evaluation)

        if self._champion is not None:
            raise RuntimeError("champion is already set.")

        self._champion = evaluation
        return self.state()

    def evaluate(
        self,
        candidate: ChampionEvaluation,
    ) -> ChampionComparison:
        self._validate(candidate)

        if self._champion is None:
            raise RuntimeError("champion is not set.")

        return self._evolution.evaluate_candidate(
            self._champion,
            candidate,
        )

    def promote(
        self,
        candidate: ChampionEvaluation,
    ) -> ChampionComparison:
        comparison = self.evaluate(candidate)

        if comparison.decision is ChampionDecision.ACCEPT:
            self._champion = candidate

        return comparison

    @staticmethod
    def _validate(
        evaluation: ChampionEvaluation,
    ) -> None:
        if not isinstance(evaluation, ChampionEvaluation):
            raise TypeError(
                "evaluation must be ChampionEvaluation."
            )


__all__ = [
    "ChampionState",
    "StrategyChampion",
]