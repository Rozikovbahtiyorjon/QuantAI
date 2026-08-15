from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable


MEMORY_LAYERS = (
    "permanent",
    "strategy",
    "asset",
    "regime",
    "experiment",
)

MEMORY_STATUSES = (
    "active",
    "archived",
)


@dataclass(frozen=True)
class MemoryRecord:
    memory_id: str
    layer: str
    content: str
    tags: tuple[str, ...] = ()
    strategy_id: str | None = None
    asset: str | None = None
    regime: str | None = None
    confidence: float = 0.5
    importance: float = 0.5
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    status: str = "active"

    def __post_init__(self) -> None:
        if not isinstance(self.memory_id, str) or not self.memory_id.strip():
            raise ValueError("memory_id must be a non-empty string.")

        if self.layer not in MEMORY_LAYERS:
            raise ValueError(
                f"Unsupported memory layer: {self.layer!r}."
            )

        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("content must be a non-empty string.")

        if not isinstance(self.tags, tuple):
            raise TypeError("tags must be a tuple.")

        if not all(
            isinstance(tag, str) and tag.strip()
            for tag in self.tags
        ):
            raise ValueError(
                "tags must contain non-empty strings."
            )

        for name, value in (
            ("confidence", self.confidence),
            ("importance", self.importance),
        ):
            if not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric.")

            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(
                    f"{name} must be between 0 and 1."
                )

        if self.status not in MEMORY_STATUSES:
            raise ValueError(
                f"Unsupported memory status: {self.status!r}."
            )

    @property
    def score(self) -> float:
        return round(
            0.6 * float(self.confidence)
            + 0.4 * float(self.importance),
            10,
        )


class AIMemory:
    """Deterministic in-memory knowledge store for QuantAI research."""

    def __init__(self) -> None:
        self._records: dict[str, MemoryRecord] = {}

    @property
    def size(self) -> int:
        return len(self._records)

    def add(
        self,
        *,
        memory_id: str,
        layer: str,
        content: str,
        tags: Iterable[str] = (),
        strategy_id: str | None = None,
        asset: str | None = None,
        regime: str | None = None,
        confidence: float = 0.5,
        importance: float = 0.5,
    ) -> MemoryRecord:
        if memory_id in self._records:
            raise ValueError(
                f"Memory already exists: {memory_id!r}."
            )

        normalized_tags = tuple(dict.fromkeys(tags))

        record = MemoryRecord(
            memory_id=memory_id,
            layer=layer,
            content=content,
            tags=normalized_tags,
            strategy_id=strategy_id,
            asset=asset,
            regime=regime,
            confidence=confidence,
            importance=importance,
        )

        self._records[memory_id] = record
        return record

    def get(self, memory_id: str) -> MemoryRecord | None:
        return self._records.get(memory_id)

    def remove(self, memory_id: str) -> bool:
        return self._records.pop(memory_id, None) is not None

    def archive(self, memory_id: str) -> MemoryRecord:
        record = self._require(memory_id)

        archived = MemoryRecord(
            memory_id=record.memory_id,
            layer=record.layer,
            content=record.content,
            tags=record.tags,
            strategy_id=record.strategy_id,
            asset=record.asset,
            regime=record.regime,
            confidence=record.confidence,
            importance=record.importance,
            created_at=record.created_at,
            updated_at=datetime.now(timezone.utc),
            status="archived",
        )

        self._records[memory_id] = archived
        return archived

    def search(
        self,
        query: str | None = None,
        *,
        layer: str | None = None,
        strategy_id: str | None = None,
        asset: str | None = None,
        regime: str | None = None,
        tags: Iterable[str] = (),
        include_archived: bool = False,
        limit: int | None = None,
    ) -> list[MemoryRecord]:
        if query is not None and not isinstance(query, str):
            raise TypeError("query must be a string or None.")

        if layer is not None and layer not in MEMORY_LAYERS:
            raise ValueError(
                f"Unsupported memory layer: {layer!r}."
            )

        if limit is not None:
            if not isinstance(limit, int):
                raise TypeError(
                    "limit must be an integer or None."
                )

            if limit <= 0:
                raise ValueError(
                    "limit must be greater than zero."
                )

        required_tags = {
            tag
            for tag in tags
            if isinstance(tag, str) and tag.strip()
        }

        query_tokens = set(
            (query or "").lower().split()
        )

        matches: list[MemoryRecord] = []

        for record in self._records.values():
            if (
                not include_archived
                and record.status != "active"
            ):
                continue

            if layer is not None and record.layer != layer:
                continue

            if (
                strategy_id is not None
                and record.strategy_id != strategy_id
            ):
                continue

            if asset is not None and record.asset != asset:
                continue

            if regime is not None and record.regime != regime:
                continue

            if (
                required_tags
                and not required_tags.issubset(set(record.tags))
            ):
                continue

            haystack = " ".join(
                (
                    record.content,
                    record.memory_id,
                    *record.tags,
                    record.strategy_id or "",
                    record.asset or "",
                    record.regime or "",
                )
            ).lower()

            if (
                query_tokens
                and not query_tokens.issubset(
                    set(haystack.split())
                )
            ):
                continue

            matches.append(record)

        matches.sort(
            key=lambda item: (
                item.score,
                item.updated_at,
                item.memory_id,
            ),
            reverse=True,
        )

        if limit is not None:
            return matches[:limit]

        return matches

    def best(
        self,
        query: str | None = None,
        **filters: Any,
    ) -> MemoryRecord | None:
        results = self.search(
            query,
            limit=1,
            **filters,
        )

        return results[0] if results else None

    def export(self) -> list[dict[str, Any]]:
        return [
            {
                "memory_id": record.memory_id,
                "layer": record.layer,
                "content": record.content,
                "tags": list(record.tags),
                "strategy_id": record.strategy_id,
                "asset": record.asset,
                "regime": record.regime,
                "confidence": record.confidence,
                "importance": record.importance,
                "created_at": record.created_at.isoformat(),
                "updated_at": record.updated_at.isoformat(),
                "status": record.status,
            }
            for record in self._records.values()
        ]

    def clear(self) -> None:
        self._records.clear()

    def _require(self, memory_id: str) -> MemoryRecord:
        record = self.get(memory_id)

        if record is None:
            raise KeyError(
                f"Unknown memory_id: {memory_id!r}."
            )

        return record