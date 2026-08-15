from dataclasses import dataclass
from typing import Mapping, Optional


@dataclass(frozen=True)
class RollbackDecision:
    action: str
    reason: str
    current_score: float
    baseline_score: float
    degradation: float


class ChampionRollbackGuard:
    def __init__(self, min_degradation: float = 0.10, min_samples: int = 20):
        if min_degradation < 0:
            raise ValueError("min_degradation must be non-negative")
        if min_samples < 1:
            raise ValueError("min_samples must be positive")

        self.min_degradation = float(min_degradation)
        self.min_samples = int(min_samples)

    @staticmethod
    def _score(metrics: Mapping[str, float]) -> float:
        if not metrics:
            return 0.0

        return float(
            metrics.get(
                "score",
                metrics.get("profitability", 0.0),
            )
        )

    def evaluate(
        self,
        current_metrics: Mapping[str, float],
        baseline_metrics: Mapping[str, float],
        samples: Optional[int] = None,
    ) -> RollbackDecision:
        current = self._score(current_metrics)
        baseline = self._score(baseline_metrics)

        if samples is not None and samples < 0:
            raise ValueError("samples must be non-negative")

        if not baseline_metrics:
            return RollbackDecision(
                "HOLD",
                "NO_BASELINE",
                current,
                baseline,
                0.0,
            )

        if samples is not None and samples < self.min_samples:
            return RollbackDecision(
                "HOLD",
                "INSUFFICIENT_SAMPLES",
                current,
                baseline,
                0.0,
            )

        if baseline == 0:
            degradation = 0.0 if current >= 0 else 1.0
        else:
            degradation = max(
                0.0,
                (baseline - current) / abs(baseline),
            )

        if degradation >= self.min_degradation:
            return RollbackDecision(
                "ROLLBACK",
                "PERFORMANCE_DEGRADATION",
                current,
                baseline,
                degradation,
            )

        return RollbackDecision(
            "KEEP",
            "STABLE",
            current,
            baseline,
            degradation,
        )