from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class EvaluationResult:
    candidate_score: float
    champion_score: float
    improvement: float
    qualified: bool
    metrics: dict[str, float]


class ChampionEvaluator:
    DEFAULT_WEIGHTS = {
        "profit_factor": 0.30,
        "net_profit": 0.25,
        "win_rate": 0.15,
        "sharpe_ratio": 0.15,
        "max_drawdown": 0.15,
    }

    def __init__(
        self,
        weights: Mapping[str, float] | None = None,
        min_improvement: float = 0.0,
    ) -> None:
        self.weights = dict(weights or self.DEFAULT_WEIGHTS)
        self.min_improvement = float(min_improvement)
        self._validate_weights()

    def _validate_weights(self) -> None:
        required = set(self.DEFAULT_WEIGHTS)
        if set(self.weights) != required:
            raise ValueError("Weights must contain exactly the required metrics.")
        if any(value < 0 for value in self.weights.values()):
            raise ValueError("Weights must be non-negative.")
        total = sum(self.weights.values())
        if total <= 0:
            raise ValueError("Weight sum must be positive.")

        self.weights = {
            key: value / total for key, value in self.weights.items()
        }

    @staticmethod
    def _validate_metrics(metrics: Mapping[str, float]) -> None:
        required = set(ChampionEvaluator.DEFAULT_WEIGHTS)
        if set(metrics) != required:
            raise ValueError("Metrics must contain exactly the required fields.")

        for value in metrics.values():
            if not isinstance(value, (int, float)):
                raise TypeError("Metric values must be numeric.")

    @staticmethod
    def _normalize(
        candidate: Mapping[str, float],
        champion: Mapping[str, float],
    ) -> dict[str, float]:
        normalized = {}

        for metric in ChampionEvaluator.DEFAULT_WEIGHTS:
            candidate_value = float(candidate[metric])
            champion_value = float(champion[metric])

            if metric == "max_drawdown":
                candidate_value = -candidate_value
                champion_value = -champion_value

            denominator = max(abs(champion_value), 1e-12)
            normalized[metric] = (
                candidate_value - champion_value
            ) / denominator

        return normalized

    def evaluate(
        self,
        candidate: Mapping[str, float],
        champion: Mapping[str, float],
    ) -> EvaluationResult:
        self._validate_metrics(candidate)
        self._validate_metrics(champion)

        normalized = self._normalize(candidate, champion)

        candidate_score = sum(
            self.weights[metric] * normalized[metric]
            for metric in self.weights
        )

        champion_score = 0.0
        improvement = candidate_score - champion_score
        qualified = improvement > self.min_improvement

        return EvaluationResult(
            candidate_score=candidate_score,
            champion_score=champion_score,
            improvement=improvement,
            qualified=qualified,
            metrics=normalized,
        )

    def compare(
        self,
        candidate: Mapping[str, float],
        champion: Mapping[str, float],
    ) -> bool:
        return self.evaluate(candidate, champion).qualified