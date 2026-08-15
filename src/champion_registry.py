from __future__ import annotations

from dataclasses import dataclass

from src.champion_evolution import ChampionComparison, ChampionDecision


@dataclass(frozen=True)
class ChampionState:
    strategy_id: str
    score: float
    version: int = 1


class ChampionRegistry:
    def __init__(self, champion: ChampionState) -> None:
        if not isinstance(champion, ChampionState):
            raise TypeError("champion must be ChampionState.")
        self._champion = champion

    @property
    def champion(self) -> ChampionState:
        return self._champion

    def consider(self, comparison: ChampionComparison) -> bool:
        if not isinstance(comparison, ChampionComparison):
            raise TypeError("comparison must be ChampionComparison.")

        if comparison.champion_id != self._champion.strategy_id:
            raise ValueError(
                "comparison champion_id does not match active champion."
            )

        if comparison.decision is not ChampionDecision.ACCEPT:
            return False

        if comparison.candidate_score <= self._champion.score:
            return False

        self._champion = ChampionState(
            strategy_id=comparison.candidate_id,
            score=comparison.candidate_score,
            version=self._champion.version + 1,
        )
        return True

    def snapshot(self) -> dict[str, int | float | str]:
        return {
            "strategy_id": self._champion.strategy_id,
            "score": self._champion.score,
            "version": self._champion.version,
        }


__all__ = ["ChampionRegistry", "ChampionState"]