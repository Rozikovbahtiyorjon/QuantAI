from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, List, Optional


@dataclass(frozen=True)
class ModelCandidateResult:
    name: str
    validation_score: float
    test_score: float
    stability_score: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def performance_score(self) -> float:
        return (
            self.validation_score
            + self.test_score
            + self.stability_score
        ) / 3.0


@dataclass(frozen=True)
class ModelSelectionDecision:
    champion: Optional[ModelCandidateResult]
    challengers: tuple[ModelCandidateResult, ...]
    selected: bool
    reason: str


@dataclass
class ModelSelectionResult:
    passed: bool
    decision: ModelSelectionDecision
    candidates: List[ModelCandidateResult] = field(
        default_factory=list
    )
    errors: List[str] = field(
        default_factory=list
    )
    warnings: List[str] = field(
        default_factory=list
    )

    @property
    def champion_name(self) -> Optional[str]:
        if self.decision.champion is None:
            return None

        return self.decision.champion.name

    @property
    def total_candidates(self) -> int:
        return len(self.candidates)


class QuantAIModelSelectionEngine:
    """
    Deterministic Champion-Challenger model selection layer.

    Responsibilities:

        - validate candidate models
        - apply validation/test/stability thresholds
        - rank eligible candidates
        - preserve an existing champion when appropriate
        - replace a champion only when improvement is sufficient
        - expose a deterministic selection result

    This module does not implement model training.
    """

    def __init__(
        self,
        minimum_validation_score: float = 0.0,
        minimum_test_score: float = 0.0,
        minimum_stability_score: float = 0.0,
        minimum_improvement: float = 0.0,
    ) -> None:
        for name, value in (
            (
                "minimum_validation_score",
                minimum_validation_score,
            ),
            (
                "minimum_test_score",
                minimum_test_score,
            ),
            (
                "minimum_stability_score",
                minimum_stability_score,
            ),
            (
                "minimum_improvement",
                minimum_improvement,
            ),
        ):
            if isinstance(value, bool) or not isinstance(
                value,
                (int, float),
            ):
                raise TypeError(
                    f"{name} must be numeric."
                )

            if value < 0.0 or value > 1.0:
                raise ValueError(
                    f"{name} must be between 0 and 1."
                )

        self.minimum_validation_score = float(
            minimum_validation_score
        )

        self.minimum_test_score = float(
            minimum_test_score
        )

        self.minimum_stability_score = float(
            minimum_stability_score
        )

        self.minimum_improvement = float(
            minimum_improvement
        )

    def validate_candidates(
        self,
        candidates: Iterable[ModelCandidateResult],
    ) -> List[ModelCandidateResult]:
        if isinstance(
            candidates,
            (str, bytes),
        ):
            raise TypeError(
                "candidates must be an iterable of "
                "ModelCandidateResult."
            )

        try:
            values = list(candidates)
        except TypeError as exc:
            raise TypeError(
                "candidates must be iterable."
            ) from exc

        if not values:
            raise ValueError(
                "candidates cannot be empty."
            )

        names = set()

        for candidate in values:
            if not isinstance(
                candidate,
                ModelCandidateResult,
            ):
                raise TypeError(
                    "All candidates must be "
                    "ModelCandidateResult instances."
                )

            if (
                not isinstance(candidate.name, str)
                or not candidate.name.strip()
            ):
                raise ValueError(
                    "Candidate names must be "
                    "non-empty strings."
                )

            if candidate.name in names:
                raise ValueError(
                    "Candidate names must be unique."
                )

            names.add(candidate.name)

            for field_name in (
                "validation_score",
                "test_score",
                "stability_score",
            ):
                value = getattr(
                    candidate,
                    field_name,
                )

                if isinstance(value, bool) or not isinstance(
                    value,
                    (int, float),
                ):
                    raise TypeError(
                        f"{field_name} must be numeric."
                    )

                if value < 0.0 or value > 1.0:
                    raise ValueError(
                        f"{field_name} must be between 0 and 1."
                    )

        return values

    def eligible_candidates(
        self,
        candidates: Iterable[ModelCandidateResult],
    ) -> List[ModelCandidateResult]:
        values = self.validate_candidates(
            candidates
        )

        return [
            candidate
            for candidate in values
            if (
                candidate.validation_score
                >= self.minimum_validation_score
                and candidate.test_score
                >= self.minimum_test_score
                and candidate.stability_score
                >= self.minimum_stability_score
            )
        ]

    def select(
        self,
        candidates: Iterable[ModelCandidateResult],
        current_champion: Optional[
            ModelCandidateResult
        ] = None,
    ) -> ModelSelectionResult:
        values = self.validate_candidates(
            candidates
        )

        if current_champion is not None:
            if not isinstance(
                current_champion,
                ModelCandidateResult,
            ):
                raise TypeError(
                    "current_champion must be "
                    "ModelCandidateResult or None."
                )

        eligible = self.eligible_candidates(
            values
        )

        if not eligible:
            decision = ModelSelectionDecision(
                champion=current_champion,
                challengers=tuple(values),
                selected=False,
                reason=(
                    "No candidate satisfies the configured "
                    "validation, test, and stability thresholds."
                ),
            )

            return ModelSelectionResult(
                passed=False,
                decision=decision,
                candidates=values,
                errors=[
                    decision.reason
                ],
            )

        ranked = sorted(
            eligible,
            key=lambda candidate: (
                candidate.performance_score,
                candidate.test_score,
                candidate.validation_score,
                candidate.stability_score,
            ),
            reverse=True,
        )

        best = ranked[0]

        if current_champion is None:
            decision = ModelSelectionDecision(
                champion=best,
                challengers=tuple(
                    candidate
                    for candidate in ranked
                    if candidate.name != best.name
                ),
                selected=True,
                reason=(
                    f"Model '{best.name}' selected as champion "
                    "because it has the highest eligible "
                    "performance score."
                ),
            )

            return ModelSelectionResult(
                passed=True,
                decision=decision,
                candidates=values,
            )

        current_eligible = (
            current_champion.validation_score
            >= self.minimum_validation_score
            and current_champion.test_score
            >= self.minimum_test_score
            and current_champion.stability_score
            >= self.minimum_stability_score
        )

        if not current_eligible:
            decision = ModelSelectionDecision(
                champion=best,
                challengers=tuple(
                    ranked[1:]
                ),
                selected=True,
                reason=(
                    f"Current champion "
                    f"'{current_champion.name}' is below "
                    "the configured thresholds; "
                    f"'{best.name}' replaces it."
                ),
            )

            return ModelSelectionResult(
                passed=True,
                decision=decision,
                candidates=values,
            )

        improvement = (
            best.performance_score
            - current_champion.performance_score
        )

        if (
            best.name != current_champion.name
            and improvement >= self.minimum_improvement
        ):
            decision = ModelSelectionDecision(
                champion=best,
                challengers=tuple(
                    [current_champion]
                    + [
                        candidate
                        for candidate in ranked
                        if candidate.name != best.name
                    ]
                ),
                selected=True,
                reason=(
                    f"Challenger '{best.name}' selected over "
                    f"champion '{current_champion.name}' with "
                    f"performance improvement of "
                    f"{improvement:.6f}."
                ),
            )

            return ModelSelectionResult(
                passed=True,
                decision=decision,
                candidates=values,
            )

        decision = ModelSelectionDecision(
            champion=current_champion,
            challengers=tuple(ranked),
            selected=False,
            reason=(
                f"Current champion '{current_champion.name}' "
                "remains selected because no challenger "
                "meets the minimum improvement threshold."
            ),
        )

        return ModelSelectionResult(
            passed=True,
            decision=decision,
            candidates=values,
            warnings=[
                "No challenger provided sufficient improvement."
            ],
        )

    def select_with_evaluator(
        self,
        candidates: Iterable[Any],
        evaluator: Callable[
            [Any],
            ModelCandidateResult,
        ],
        current_champion: Optional[
            ModelCandidateResult
        ] = None,
    ) -> ModelSelectionResult:
        if not callable(evaluator):
            raise TypeError(
                "evaluator must be callable."
            )

        if isinstance(
            candidates,
            (str, bytes),
        ):
            raise TypeError(
                "candidates must be an iterable."
            )

        try:
            candidate_values = list(candidates)
        except TypeError as exc:
            raise TypeError(
                "candidates must be iterable."
            ) from exc

        evaluated = [
            evaluator(candidate)
            for candidate in candidate_values
        ]

        return self.select(
            evaluated,
            current_champion=current_champion,
        )


def select_champion_model(
    candidates: Iterable[ModelCandidateResult],
    current_champion: Optional[
        ModelCandidateResult
    ] = None,
    minimum_validation_score: float = 0.0,
    minimum_test_score: float = 0.0,
    minimum_stability_score: float = 0.0,
    minimum_improvement: float = 0.0,
) -> ModelSelectionResult:
    engine = QuantAIModelSelectionEngine(
        minimum_validation_score=minimum_validation_score,
        minimum_test_score=minimum_test_score,
        minimum_stability_score=minimum_stability_score,
        minimum_improvement=minimum_improvement,
    )

    return engine.select(
        candidates,
        current_champion=current_champion,
    )


__all__ = [
    "ModelCandidateResult",
    "ModelSelectionDecision",
    "ModelSelectionResult",
    "QuantAIModelSelectionEngine",
    "select_champion_model",
]