from dataclasses import dataclass
from typing import Mapping, Optional


@dataclass(frozen=True)
class AdmissionDecision:
    action: str
    reason: str
    candidate_score: float
    champion_score: float
    improvement: float
    samples: int


class ChampionAdmissionController:
    def __init__(
        self,
        min_improvement: float = 0.05,
        min_samples: int = 20,
    ):
        if min_improvement < 0:
            raise ValueError("min_improvement must be non-negative")
        if min_samples < 1:
            raise ValueError("min_samples must be positive")

        self.min_improvement = float(min_improvement)
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
        candidate_metrics: Mapping[str, float],
        champion_metrics: Mapping[str, float],
        samples: Optional[int] = None,
    ) -> AdmissionDecision:
        candidate = self._score(candidate_metrics)
        champion = self._score(champion_metrics)
        sample_count = 0 if samples is None else int(samples)

        if sample_count < 0:
            raise ValueError("samples must be non-negative")

        if not candidate_metrics:
            return AdmissionDecision(
                "REJECT",
                "NO_CANDIDATE_METRICS",
                candidate,
                champion,
                0.0,
                sample_count,
            )

        if not champion_metrics:
            if sample_count < self.min_samples:
                return AdmissionDecision(
                    "HOLD",
                    "INSUFFICIENT_SAMPLES",
                    candidate,
                    champion,
                    0.0,
                    sample_count,
                )

            return AdmissionDecision(
                "ADMIT",
                "NO_EXISTING_CHAMPION",
                candidate,
                champion,
                0.0,
                sample_count,
            )

        if sample_count < self.min_samples:
            return AdmissionDecision(
                "HOLD",
                "INSUFFICIENT_SAMPLES",
                candidate,
                champion,
                0.0,
                sample_count,
            )

        if champion == 0:
            improvement = candidate if candidate > 0 else 0.0
        else:
            improvement = (candidate - champion) / abs(champion)

        if improvement >= self.min_improvement:
            return AdmissionDecision(
                "ADMIT",
                "CANDIDATE_OUTPERFORMS_CHAMPION",
                candidate,
                champion,
                improvement,
                sample_count,
            )

        return AdmissionDecision(
            "REJECT",
            "CANDIDATE_NOT_SUPERIOR",
            candidate,
            champion,
            improvement,
            sample_count,
        )