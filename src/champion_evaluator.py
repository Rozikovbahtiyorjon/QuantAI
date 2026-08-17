from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
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
        self.weights = dict(
            weights or self.DEFAULT_WEIGHTS
        )

        self.min_improvement = float(
            min_improvement
        )

        if not isfinite(
            self.min_improvement
        ):
            raise ValueError(
                "min_improvement must be finite."
            )

        self._validate_weights()

    def _validate_weights(self) -> None:
        required = set(
            self.DEFAULT_WEIGHTS
        )

        if set(self.weights) != required:
            raise ValueError(
                "Weights must contain exactly "
                "the required metrics."
            )

        normalized: dict[str, float] = {}

        for metric, value in self.weights.items():
            if (
                isinstance(value, bool)
                or not isinstance(
                    value,
                    (int, float),
                )
            ):
                raise TypeError(
                    f"Weight '{metric}' must be numeric."
                )

            value = float(value)

            if not isfinite(value):
                raise ValueError(
                    f"Weight '{metric}' must be finite."
                )

            if value < 0:
                raise ValueError(
                    "Weights must be non-negative."
                )

            normalized[metric] = value

        total = sum(
            normalized.values()
        )

        if total <= 0:
            raise ValueError(
                "Weight sum must be positive."
            )

        self.weights = {
            key: value / total
            for key, value in normalized.items()
        }

    @classmethod
    def _validate_metrics(
        cls,
        metrics: Mapping[str, float],
    ) -> None:
        if not isinstance(
            metrics,
            Mapping,
        ):
            raise TypeError(
                "Metrics must be a mapping."
            )

        required = set(
            cls.DEFAULT_WEIGHTS
        )

        if set(metrics) != required:
            raise ValueError(
                "Metrics must contain exactly "
                "the required fields."
            )

        for metric, value in metrics.items():
            if (
                isinstance(value, bool)
                or not isinstance(
                    value,
                    (int, float),
                )
            ):
                raise TypeError(
                    f"Metric '{metric}' must be numeric."
                )

            if not isfinite(
                float(value)
            ):
                raise ValueError(
                    f"Metric '{metric}' must be finite."
                )

    @staticmethod
    def _pairwise_delta(
        candidate_value: float,
        champion_value: float,
        *,
        higher_is_better: bool,
    ) -> float:
        if not higher_is_better:
            candidate_value = -candidate_value
            champion_value = -champion_value

        scale = (
            abs(candidate_value)
            + abs(champion_value)
            + 1e-12
        )

        return (
            candidate_value
            - champion_value
        ) / scale

    def _normalize(
        self,
        candidate: Mapping[str, float],
        champion: Mapping[str, float],
    ) -> dict[str, float]:
        self._validate_metrics(
            candidate
        )

        self._validate_metrics(
            champion
        )

        normalized: dict[str, float] = {}

        for metric in self.DEFAULT_WEIGHTS:
            normalized[metric] = (
                self._pairwise_delta(
                    float(candidate[metric]),
                    float(champion[metric]),
                    higher_is_better=(
                        metric != "max_drawdown"
                    ),
                )
            )

        return normalized

    def evaluate(
        self,
        candidate: Mapping[str, float],
        champion: Mapping[str, float],
    ) -> EvaluationResult:
        normalized = self._normalize(
            candidate,
            champion,
        )

        improvement = sum(
            self.weights[metric]
            * normalized[metric]
            for metric in self.weights
        )

        candidate_score = (
            0.5
            + 0.5 * improvement
        )

        champion_score = (
            0.5
            - 0.5 * improvement
        )

        candidate_score = max(
            0.0,
            min(
                1.0,
                candidate_score,
            ),
        )

        champion_score = max(
            0.0,
            min(
                1.0,
                champion_score,
            ),
        )

        qualified = (
            improvement
            > self.min_improvement
        )

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
        return self.evaluate(
            candidate,
            champion,
        ).qualified