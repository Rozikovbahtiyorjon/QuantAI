from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from src.strategy_genome import StrategyGenome


VALID_STATUSES = frozenset(
    {
        "candidate",
        "experimental",
        "validated",
        "champion",
        "deprecated",
        "failed",
    }
)


@dataclass(frozen=True)
class StrategyRecord:
    genome: StrategyGenome
    status: str = "candidate"

    def __post_init__(self) -> None:
        if not isinstance(self.genome, StrategyGenome):
            raise TypeError("genome must be a StrategyGenome.")

        if not isinstance(self.status, str):
            raise TypeError("status must be a string.")

        if self.status not in VALID_STATUSES:
            raise ValueError(
                f"invalid strategy status: {self.status}"
            )


class StrategyRegistry:
    def __init__(self) -> None:
        self._records: dict[str, StrategyRecord] = {}

    def register(
        self,
        genome: StrategyGenome,
        status: str = "candidate",
    ) -> StrategyRecord:
        if not isinstance(genome, StrategyGenome):
            raise TypeError("genome must be a StrategyGenome.")

        if genome.strategy_id in self._records:
            raise ValueError(
                f"strategy already registered: {genome.strategy_id}"
            )

        record = StrategyRecord(
            genome=genome,
            status=status,
        )

        self._records[genome.strategy_id] = record
        return record

    def get(self, strategy_id: str) -> StrategyRecord:
        self._validate_strategy_id(strategy_id)

        try:
            return self._records[strategy_id]
        except KeyError as exc:
            raise KeyError(
                f"strategy not found: {strategy_id}"
            ) from exc

    def contains(self, strategy_id: str) -> bool:
        self._validate_strategy_id(strategy_id)
        return strategy_id in self._records

    def update_status(
        self,
        strategy_id: str,
        status: str,
    ) -> StrategyRecord:
        record = self.get(strategy_id)

        updated = StrategyRecord(
            genome=record.genome,
            status=status,
        )

        self._records[strategy_id] = updated
        return updated

    def remove(self, strategy_id: str) -> StrategyRecord:
        self._validate_strategy_id(strategy_id)

        try:
            return self._records.pop(strategy_id)
        except KeyError as exc:
            raise KeyError(
                f"strategy not found: {strategy_id}"
            ) from exc

    def list(
        self,
        status: str | None = None,
    ) -> tuple[StrategyRecord, ...]:
        if status is not None:
            self._validate_status(status)

        records = tuple(self._records.values())

        if status is None:
            return records

        return tuple(
            record
            for record in records
            if record.status == status
        )

    def champion(self) -> StrategyRecord | None:
        champions = self.list(status="champion")

        if not champions:
            return None

        if len(champions) > 1:
            raise RuntimeError(
                "multiple champion strategies are registered."
            )

        return champions[0]

    def set_champion(
        self,
        strategy_id: str,
    ) -> StrategyRecord:
        self.get(strategy_id)

        current = self.champion()

        if (
            current is not None
            and current.genome.strategy_id != strategy_id
        ):
            self.update_status(
                current.genome.strategy_id,
                "validated",
            )

        return self.update_status(
            strategy_id,
            "champion",
        )

    def count(self, status: str | None = None) -> int:
        return len(self.list(status=status))

    def to_dict(self) -> dict[str, dict[str, Any]]:
        return {
            strategy_id: {
                "genome": record.genome.to_dict(),
                "status": record.status,
            }
            for strategy_id, record in self._records.items()
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Mapping[str, Any]],
    ) -> StrategyRegistry:
        if not isinstance(data, Mapping):
            raise TypeError("data must be a mapping.")

        registry = cls()

        for strategy_id, payload in data.items():
            if not isinstance(strategy_id, str):
                raise TypeError(
                    "strategy_id must be a string."
                )

            if not isinstance(payload, Mapping):
                raise TypeError(
                    "strategy payload must be a mapping."
                )

            if "genome" not in payload:
                raise ValueError(
                    f"missing genome for strategy: {strategy_id}"
                )

            genome = StrategyGenome.from_dict(
                payload["genome"]
            )

            if genome.strategy_id != strategy_id:
                raise ValueError(
                    "strategy_id does not match "
                    "genome.strategy_id."
                )

            registry.register(
                genome=genome,
                status=payload.get(
                    "status",
                    "candidate",
                ),
            )

        return registry

    @staticmethod
    def _validate_strategy_id(
        strategy_id: str,
    ) -> None:
        if not isinstance(strategy_id, str):
            raise TypeError(
                "strategy_id must be a string."
            )

        if not strategy_id.strip():
            raise ValueError(
                "strategy_id cannot be empty."
            )

    @staticmethod
    def _validate_status(status: str) -> None:
        if not isinstance(status, str):
            raise TypeError(
                "status must be a string."
            )

        if status not in VALID_STATUSES:
            raise ValueError(
                f"invalid strategy status: {status}"
            )


__all__ = [
    "VALID_STATUSES",
    "StrategyRecord",
    "StrategyRegistry",
]