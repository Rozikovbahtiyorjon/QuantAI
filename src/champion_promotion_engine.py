from dataclasses import dataclass
from typing import Mapping

from src.champion_evaluator import ChampionEvaluator, EvaluationResult


@dataclass(frozen=True)
class PromotionResult:
    promoted: bool
    reason: str
    evaluation: EvaluationResult


class ChampionPromotionEngine:
    def __init__(
        self,
        evaluator: ChampionEvaluator | None = None,
    ) -> None:
        self.evaluator = evaluator or ChampionEvaluator()

    def evaluate(
        self,
        candidate: Mapping[str, float],
        champion: Mapping[str, float],
    ) -> PromotionResult:
        evaluation = self.evaluator.evaluate(candidate, champion)

        if evaluation.qualified:
            return PromotionResult(
                promoted=True,
                reason="candidate_outperforms_champion",
                evaluation=evaluation,
            )

        return PromotionResult(
            promoted=False,
            reason="candidate_does_not_outperform_champion",
            evaluation=evaluation,
        )

    def should_promote(
        self,
        candidate: Mapping[str, float],
        champion: Mapping[str, float],
    ) -> bool:
        return self.evaluate(candidate, champion).promoted    