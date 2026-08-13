from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src.strategy_bank import StrategyRecord, StrategyRegistry


@dataclass(frozen=True)
class StrategyEvaluation:
    strategy_id: str
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    walk_forward_score: float
    robustness_score: float
    monte_carlo_score: float
    stress_score: float

    def __post_init__(self) -> None:
        if not isinstance(self.strategy_id, str):
            raise TypeError("strategy_id must be a string.")

        if not self.strategy_id.strip():
            raise ValueError("strategy_id cannot be empty.")

        numeric_fields = (
            "total_return",
            "sharpe_ratio",
            "max_drawdown",
            "win_rate",
            "profit_factor",
            "walk_forward_score",
            "robustness_score",
            "monte_carlo_score",
            "stress_score",
        )

        for field_name in numeric_fields:
            value = getattr(self, field_name)

            if not isinstance(value, (int, float)):
                raise TypeError(
                    f"{field_name} must be numeric."
                )

        if self.max_drawdown < 0:
            raise ValueError(
                "max_drawdown must be non-negative."
            )

        bounded_fields = (
            "win_rate",
            "walk_forward_score",
            "robustness_score",
            "monte_carlo_score",
            "stress_score",
        )

        for field_name in bounded_fields:
            value = getattr(self, field_name)

            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{field_name} must be between 0 and 1."
                )

        if self.profit_factor < 0:
            raise ValueError(
                "profit_factor must be non-negative."
            )


@dataclass(frozen=True)
class TournamentResult:
    strategy_id: str
    score: float
    rank: int
    evaluation: StrategyEvaluation


@dataclass(frozen=True)
class TournamentRanking:
    results: tuple[TournamentResult, ...]
    champion_strategy_id: str | None

    @property
    def champion(self) -> TournamentResult | None:
        if self.champion_strategy_id is None:
            return None

        for result in self.results:
            if result.strategy_id == self.champion_strategy_id:
                return result

        return None


class StrategyTournament:
    DEFAULT_WEIGHTS = {
        "total_return": 0.15,
        "sharpe_ratio": 0.15,
        "max_drawdown": 0.10,
        "win_rate": 0.10,
        "profit_factor": 0.10,
        "walk_forward_score": 0.15,
        "robustness_score": 0.10,
        "monte_carlo_score": 0.075,
        "stress_score": 0.075,
    }

    def __init__(
        self,
        weights: dict[str, float] | None = None,
    ) -> None:
        self._weights = self._validate_weights(
            weights
        )

    @property
    def weights(self) -> dict[str, float]:
        return dict(self._weights)

    def evaluate(
        self,
        evaluation: StrategyEvaluation,
    ) -> float:
        if not isinstance(
            evaluation,
            StrategyEvaluation,
        ):
            raise TypeError(
                "evaluation must be a StrategyEvaluation."
            )

        normalized = self._normalize_evaluation(
            evaluation
        )

        score = 0.0

        for field_name, weight in self._weights.items():
            score += normalized[field_name] * weight

        return round(score, 10)

    def rank(
        self,
        evaluations: Iterable[StrategyEvaluation],
    ) -> TournamentRanking:
        evaluations = tuple(evaluations)

        if not evaluations:
            raise ValueError(
                "at least one evaluation is required."
            )

        strategy_ids = [
            evaluation.strategy_id
            for evaluation in evaluations
        ]

        if len(strategy_ids) != len(set(strategy_ids)):
            raise ValueError(
                "duplicate strategy_id values are not allowed."
            )

        scored = [
            TournamentResult(
                strategy_id=evaluation.strategy_id,
                score=self.evaluate(evaluation),
                rank=0,
                evaluation=evaluation,
            )
            for evaluation in evaluations
        ]

        scored.sort(
            key=lambda result: (
                result.score,
                result.evaluation.total_return,
                result.evaluation.sharpe_ratio,
                -result.evaluation.max_drawdown,
            ),
            reverse=True,
        )

        ranked: list[TournamentResult] = []

        for index, result in enumerate(
            scored,
            start=1,
        ):
            ranked.append(
                TournamentResult(
                    strategy_id=result.strategy_id,
                    score=result.score,
                    rank=index,
                    evaluation=result.evaluation,
                )
            )

        return TournamentRanking(
            results=tuple(ranked),
            champion_strategy_id=ranked[0].strategy_id,
        )

    def select_champion(
        self,
        evaluations: Iterable[StrategyEvaluation],
    ) -> TournamentResult:
        ranking = self.rank(evaluations)

        champion = ranking.champion

        if champion is None:
            raise RuntimeError(
                "tournament produced no champion."
            )

        return champion

    def compare_with_champion(
        self,
        candidate: StrategyEvaluation,
        champion: StrategyEvaluation,
    ) -> bool:
        if not isinstance(
            candidate,
            StrategyEvaluation,
        ):
            raise TypeError(
                "candidate must be a StrategyEvaluation."
            )

        if not isinstance(
            champion,
            StrategyEvaluation,
        ):
            raise TypeError(
                "champion must be a StrategyEvaluation."
            )

        if candidate.strategy_id == champion.strategy_id:
            raise ValueError(
                "candidate and champion must be different."
            )

        candidate_score = self.evaluate(candidate)
        champion_score = self.evaluate(champion)

        return candidate_score > champion_score

    def promote_champion(
        self,
        registry: StrategyRegistry,
        evaluations: Iterable[StrategyEvaluation],
    ) -> StrategyRecord:
        if not isinstance(
            registry,
            StrategyRegistry,
        ):
            raise TypeError(
                "registry must be a StrategyRegistry."
            )

        ranking = self.rank(evaluations)
        champion = ranking.champion

        if champion is None:
            raise RuntimeError(
                "tournament produced no champion."
            )

        record = registry.get(
            champion.strategy_id
        )

        return registry.set_champion(
            record.genome.strategy_id
        )

    @staticmethod
    def _normalize_evaluation(
        evaluation: StrategyEvaluation,
    ) -> dict[str, float]:
        return {
            "total_return": StrategyTournament._normalize_return(
                evaluation.total_return
            ),
            "sharpe_ratio": StrategyTournament._normalize_sharpe(
                evaluation.sharpe_ratio
            ),
            "max_drawdown": StrategyTournament._normalize_drawdown(
                evaluation.max_drawdown
            ),
            "win_rate": evaluation.win_rate,
            "profit_factor": StrategyTournament._normalize_profit_factor(
                evaluation.profit_factor
            ),
            "walk_forward_score": evaluation.walk_forward_score,
            "robustness_score": evaluation.robustness_score,
            "monte_carlo_score": evaluation.monte_carlo_score,
            "stress_score": evaluation.stress_score,
        }

    @staticmethod
    def _normalize_return(value: float) -> float:
        return max(
            0.0,
            min(1.0, (value + 1.0) / 2.0),
        )

    @staticmethod
    def _normalize_sharpe(value: float) -> float:
        return max(
            0.0,
            min(1.0, (value + 2.0) / 4.0),
        )

    @staticmethod
    def _normalize_drawdown(value: float) -> float:
        return max(
            0.0,
            min(1.0, 1.0 - value),
        )

    @staticmethod
    def _normalize_profit_factor(value: float) -> float:
        return max(
            0.0,
            min(1.0, value / 3.0),
        )

    @classmethod
    def _validate_weights(
        cls,
        weights: dict[str, float] | None,
    ) -> dict[str, float]:
        if weights is None:
            selected = dict(cls.DEFAULT_WEIGHTS)
        else:
            if not isinstance(weights, dict):
                raise TypeError(
                    "weights must be a dictionary or None."
                )
    
            selected = dict(weights)
    
        allowed_names = set(cls.DEFAULT_WEIGHTS)
    
        for name, value in selected.items():
            if not isinstance(name, str):
                raise TypeError(
                    "weight names must be strings."
                )
    
            if name not in allowed_names:
                raise ValueError(
                    f"Unknown weight name: {name!r}."
                )
    
            if not isinstance(value, (int, float)):
                raise TypeError(
                    f"Weight '{name}' must be numeric."
                )
    
            if value < 0:
                raise ValueError(
                    f"Weight '{name}' cannot be negative."
                )
    
        total = sum(
            float(value)
            for value in selected.values()
        )
    
        if total <= 0:
            raise ValueError(
                "The sum of weights must be greater than zero."
            )
    
        return {
            name: float(value) / total
            for name, value in selected.items()
        }