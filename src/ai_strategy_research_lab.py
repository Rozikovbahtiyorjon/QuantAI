from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping


@dataclass(frozen=True)
class ResearchCandidate:
    strategy_id: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.strategy_id, str):
            raise TypeError("strategy_id must be a string.")

        if not self.strategy_id.strip():
            raise ValueError("strategy_id cannot be empty.")

        if not isinstance(self.parameters, Mapping):
            raise TypeError("parameters must be a mapping.")


@dataclass(frozen=True)
class ResearchEvidence:
    backtest_score: float
    walk_forward_score: float
    robustness_score: float
    monte_carlo_score: float
    stress_score: float

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric.")

            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(
                    f"{name} must be between 0 and 1."
                )


@dataclass(frozen=True)
class ResearchResult:
    strategy_id: str
    evidence: ResearchEvidence
    research_score: float
    accepted: bool
    rejection_reason: str | None = None


class AIStrategyResearchLaboratory:
    DEFAULT_WEIGHTS = {
        "backtest_score": 0.20,
        "walk_forward_score": 0.25,
        "robustness_score": 0.20,
        "monte_carlo_score": 0.15,
        "stress_score": 0.20,
    }

    def __init__(
        self,
        evaluator: Callable[[ResearchCandidate], ResearchEvidence],
        acceptance_threshold: float = 0.70,
    ) -> None:
        if not callable(evaluator):
            raise TypeError(
                "evaluator must be callable."
            )

        if not isinstance(
            acceptance_threshold,
            (int, float),
        ):
            raise TypeError(
                "acceptance_threshold must be numeric."
            )

        if not 0.0 <= float(acceptance_threshold) <= 1.0:
            raise ValueError(
                "acceptance_threshold must be between 0 and 1."
            )

        self._evaluator = evaluator
        self._acceptance_threshold = float(
            acceptance_threshold
        )

    @property
    def acceptance_threshold(self) -> float:
        return self._acceptance_threshold

    def evaluate(
        self,
        candidate: ResearchCandidate,
    ) -> ResearchResult:
        if not isinstance(
            candidate,
            ResearchCandidate,
        ):
            raise TypeError(
                "candidate must be a ResearchCandidate."
            )

        evidence = self._evaluator(candidate)

        if not isinstance(
            evidence,
            ResearchEvidence,
        ):
            raise TypeError(
                "evaluator must return ResearchEvidence."
            )

        score = self._calculate_score(evidence)
        accepted = score >= self._acceptance_threshold

        rejection_reason = None

        if not accepted:
            rejection_reason = (
                "Research score is below the "
                "acceptance threshold."
            )

        return ResearchResult(
            strategy_id=candidate.strategy_id,
            evidence=evidence,
            research_score=score,
            accepted=accepted,
            rejection_reason=rejection_reason,
        )

    def evaluate_many(
        self,
        candidates: Iterable[ResearchCandidate],
    ) -> list[ResearchResult]:
        if isinstance(candidates, (str, bytes)):
            raise TypeError(
                "candidates must be an iterable "
                "of ResearchCandidate."
            )

        return [
            self.evaluate(candidate)
            for candidate in candidates
        ]

    def _calculate_score(
        self,
        evidence: ResearchEvidence,
    ) -> float:
        values = {
            "backtest_score": evidence.backtest_score,
            "walk_forward_score": (
                evidence.walk_forward_score
            ),
            "robustness_score": (
                evidence.robustness_score
            ),
            "monte_carlo_score": (
                evidence.monte_carlo_score
            ),
            "stress_score": evidence.stress_score,
        }

        return sum(
            self.DEFAULT_WEIGHTS[name]
            * float(value)
            for name, value in values.items()
        )