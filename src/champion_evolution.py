from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class ChampionDecision(str, Enum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"


@dataclass(frozen=True)
class ChampionEvaluation:
    strategy_id: str
    performance_score: float
    robustness_score: float
    risk_adjusted_score: float
    evidence_score: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.strategy_id, str):
            raise TypeError("strategy_id must be a string.")

        if not self.strategy_id.strip():
            raise ValueError("strategy_id cannot be empty.")

        for name in (
            "performance_score",
            "robustness_score",
            "risk_adjusted_score",
            "evidence_score",
        ):
            value = getattr(self, name)

            if not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric.")

            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(
                    f"{name} must be between 0.0 and 1.0."
                )


@dataclass(frozen=True)
class ChampionComparison:
    candidate_id: str
    champion_id: str
    candidate_score: float
    champion_score: float
    improvement: float
    decision: ChampionDecision
    reason: str


class ChampionEvolution:
    """
    Evaluates candidate strategies against the current champion.

    A candidate can replace the champion only when:
    - robustness satisfies the minimum threshold;
    - evidence satisfies the minimum threshold;
    - risk-adjusted composite score improves enough.
    """

    def __init__(
        self,
        *,
        minimum_improvement: float = 0.0,
        minimum_robustness: float = 0.5,
        minimum_evidence: float = 0.5,
    ) -> None:
        for name, value in (
            ("minimum_improvement", minimum_improvement),
            ("minimum_robustness", minimum_robustness),
            ("minimum_evidence", minimum_evidence),
        ):
            if not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric.")

            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(
                    f"{name} must be between 0.0 and 1.0."
                )

        self.minimum_improvement = float(minimum_improvement)
        self.minimum_robustness = float(minimum_robustness)
        self.minimum_evidence = float(minimum_evidence)

    @staticmethod
    def _validate_evaluation(
        evaluation: ChampionEvaluation,
    ) -> ChampionEvaluation:
        if not isinstance(evaluation, ChampionEvaluation):
            raise TypeError(
                "evaluation must be ChampionEvaluation."
            )

        return evaluation

    @staticmethod
    def _score(
        evaluation: ChampionEvaluation,
    ) -> float:
        return (
            0.35 * evaluation.performance_score
            + 0.25 * evaluation.robustness_score
            + 0.25 * evaluation.risk_adjusted_score
            + 0.15 * evaluation.evidence_score
        )

    def score(
        self,
        evaluation: ChampionEvaluation,
    ) -> float:
        evaluation = self._validate_evaluation(evaluation)
        return self._score(evaluation)

    def evaluate_candidate(
        self,
        champion: ChampionEvaluation,
        candidate: ChampionEvaluation,
    ) -> ChampionComparison:
        champion = self._validate_evaluation(champion)
        candidate = self._validate_evaluation(candidate)

        if champion.strategy_id == candidate.strategy_id:
            raise ValueError(
                "candidate and champion must have different "
                "strategy_id values."
            )

        champion_score = self._score(champion)
        candidate_score = self._score(candidate)
        improvement = candidate_score - champion_score

        if candidate.robustness_score < self.minimum_robustness:
            return ChampionComparison(
                candidate_id=candidate.strategy_id,
                champion_id=champion.strategy_id,
                candidate_score=candidate_score,
                champion_score=champion_score,
                improvement=improvement,
                decision=ChampionDecision.REJECT,
                reason=(
                    "Candidate robustness is below "
                    "the minimum threshold."
                ),
            )

        if candidate.evidence_score < self.minimum_evidence:
            return ChampionComparison(
                candidate_id=candidate.strategy_id,
                champion_id=champion.strategy_id,
                candidate_score=candidate_score,
                champion_score=champion_score,
                improvement=improvement,
                decision=ChampionDecision.REJECT,
                reason=(
                    "Candidate evidence score is below "
                    "the minimum threshold."
                ),
            )

        if improvement < self.minimum_improvement:
            return ChampionComparison(
                candidate_id=candidate.strategy_id,
                champion_id=champion.strategy_id,
                candidate_score=candidate_score,
                champion_score=champion_score,
                improvement=improvement,
                decision=ChampionDecision.REJECT,
                reason=(
                    "Candidate does not improve the current "
                    "champion enough."
                ),
            )

        return ChampionComparison(
            candidate_id=candidate.strategy_id,
            champion_id=champion.strategy_id,
            candidate_score=candidate_score,
            champion_score=champion_score,
            improvement=improvement,
            decision=ChampionDecision.ACCEPT,
            reason=(
                "Candidate robustly improves the "
                "current champion."
            ),
        )

    def select_champion(
        self,
        evaluations: Iterable[ChampionEvaluation],
    ) -> ChampionEvaluation:
        items = list(evaluations)

        if not items:
            raise ValueError(
                "evaluations cannot be empty."
            )

        for evaluation in items:
            self._validate_evaluation(evaluation)

        eligible = [
            evaluation
            for evaluation in items
            if (
                evaluation.robustness_score
                >= self.minimum_robustness
                and evaluation.evidence_score
                >= self.minimum_evidence
            )
        ]

        pool = eligible or items

        return max(
            pool,
            key=self._score,
        )