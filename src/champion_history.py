from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src.champion_evolution import (
    ChampionComparison,
    ChampionDecision,
)


@dataclass(frozen=True)
class ChampionHistoryEntry:
    strategy_id: str
    score: float
    decision: ChampionDecision
    improvement: float


class ChampionHistory:
    def __init__(self) -> None:
        self._entries: list[ChampionHistoryEntry] = []

    @property
    def entries(self) -> tuple[ChampionHistoryEntry, ...]:
        return tuple(self._entries)

    def record(
        self,
        comparison: ChampionComparison,
    ) -> ChampionHistoryEntry:
        if not isinstance(comparison, ChampionComparison):
            raise TypeError(
                "comparison must be ChampionComparison."
            )

        entry = ChampionHistoryEntry(
            strategy_id=comparison.candidate_id,
            score=comparison.candidate_score,
            decision=comparison.decision,
            improvement=comparison.improvement,
        )
        self._entries.append(entry)
        return entry

    def accepted(self) -> tuple[ChampionHistoryEntry, ...]:
        return tuple(
            entry
            for entry in self._entries
            if entry.decision is ChampionDecision.ACCEPT
        )

    def rejected(self) -> tuple[ChampionHistoryEntry, ...]:
        return tuple(
            entry
            for entry in self._entries
            if entry.decision is ChampionDecision.REJECT
        )

    def best(self) -> ChampionHistoryEntry | None:
        if not self._entries:
            return None

        return max(
            self._entries,
            key=lambda entry: entry.score,
        )

    def acceptance_rate(self) -> float:
        if not self._entries:
            return 0.0

        return len(self.accepted()) / len(self._entries)

    def summary(self) -> dict[str, float | int | str | None]:
        best = self.best()

        return {
            "total_candidates": len(self._entries),
            "accepted": len(self.accepted()),
            "rejected": len(self.rejected()),
            "acceptance_rate": self.acceptance_rate(),
            "best_strategy_id": (
                best.strategy_id if best is not None else None
            ),
            "best_score": (
                best.score if best is not None else None
            ),
        }

    def extend(
        self,
        comparisons: Iterable[ChampionComparison],
    ) -> None:
        for comparison in comparisons:
            self.record(comparison)


__all__ = [
    "ChampionHistory",
    "ChampionHistoryEntry",
]