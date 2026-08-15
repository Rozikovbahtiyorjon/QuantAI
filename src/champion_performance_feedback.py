from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


REQUIRED_METRICS = (
    "net_profit",
    "win_rate",
    "trade_count",
    "max_drawdown",
    "signal_quality",
    "stability",
)


@dataclass(frozen=True)
class ChampionPerformanceSnapshot:
    champion_id: str
    net_profit: float
    win_rate: float
    trade_count: int
    max_drawdown: float
    signal_quality: float
    stability: float
    profit_factor: float | None = None
    expectancy: float | None = None


class ChampionPerformanceFeedback:
    """
    Objective performance feedback for the Champion lifecycle.

    This module measures and stores performance facts.

    It does not:
    - generate trading signals;
    - decide promotion;
    - replace a Champion;
    - modify strategy parameters.
    """

    def __init__(self) -> None:
        self._snapshots: dict[str, ChampionPerformanceSnapshot] = {}

    @staticmethod
    def _to_float(value: Any, field: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{field} must be numeric.")

        result = float(value)

        if result != result:
            raise ValueError(f"{field} must not be NaN.")

        return result

    @classmethod
    def _validate_metrics(
        cls,
        metrics: Mapping[str, Any],
    ) -> None:
        missing = [
            name
            for name in REQUIRED_METRICS
            if name not in metrics
        ]

        if missing:
            raise ValueError(
                "Missing required metrics: "
                + ", ".join(missing)
            )

        for name in REQUIRED_METRICS:
            cls._to_float(
                metrics[name],
                name,
            )

        if float(metrics["trade_count"]) < 0:
            raise ValueError(
                "trade_count must be non-negative."
            )

        if not 0 <= float(metrics["win_rate"]) <= 100:
            raise ValueError(
                "win_rate must be between 0 and 100."
            )

        if float(metrics["max_drawdown"]) < 0:
            raise ValueError(
                "max_drawdown must be non-negative."
            )

        if not 0 <= float(metrics["signal_quality"]) <= 1:
            raise ValueError(
                "signal_quality must be between 0 and 1."
            )

        if not 0 <= float(metrics["stability"]) <= 1:
            raise ValueError(
                "stability must be between 0 and 1."
            )

        for name in (
            "profit_factor",
            "expectancy",
        ):
            if name in metrics and metrics[name] is not None:
                cls._to_float(
                    metrics[name],
                    name,
                )

    @classmethod
    def _build_snapshot(
        cls,
        champion_id: str,
        metrics: Mapping[str, Any],
    ) -> ChampionPerformanceSnapshot:
        cls._validate_metrics(metrics)

        profit_factor = (
            None
            if metrics.get("profit_factor") is None
            else cls._to_float(
                metrics["profit_factor"],
                "profit_factor",
            )
        )

        expectancy = (
            None
            if metrics.get("expectancy") is None
            else cls._to_float(
                metrics["expectancy"],
                "expectancy",
            )
        )

        return ChampionPerformanceSnapshot(
            champion_id=champion_id,
            net_profit=cls._to_float(
                metrics["net_profit"],
                "net_profit",
            ),
            win_rate=cls._to_float(
                metrics["win_rate"],
                "win_rate",
            ),
            trade_count=int(
                cls._to_float(
                    metrics["trade_count"],
                    "trade_count",
                )
            ),
            max_drawdown=cls._to_float(
                metrics["max_drawdown"],
                "max_drawdown",
            ),
            signal_quality=cls._to_float(
                metrics["signal_quality"],
                "signal_quality",
            ),
            stability=cls._to_float(
                metrics["stability"],
                "stability",
            ),
            profit_factor=profit_factor,
            expectancy=expectancy,
        )

    def record(
        self,
        champion_id: str,
        metrics: Mapping[str, Any],
    ) -> ChampionPerformanceSnapshot:
        if not champion_id:
            raise ValueError(
                "champion_id must not be empty."
            )

        snapshot = self._build_snapshot(
            champion_id,
            metrics,
        )

        self._snapshots[champion_id] = snapshot

        return snapshot

    def update(
        self,
        champion_id: str,
        metrics: Mapping[str, Any],
    ) -> ChampionPerformanceSnapshot:
        return self.record(
            champion_id,
            metrics,
        )

    def get(
        self,
        champion_id: str,
    ) -> ChampionPerformanceSnapshot | None:
        return self._snapshots.get(champion_id)

    def snapshot(
        self,
    ) -> dict[str, ChampionPerformanceSnapshot]:
        return dict(self._snapshots)

    @staticmethod
    def compare(
        candidate: ChampionPerformanceSnapshot | Mapping[str, Any],
        champion: ChampionPerformanceSnapshot | Mapping[str, Any],
    ) -> dict[str, float]:
        def read(
            source: Any,
            name: str,
        ) -> float:
            if isinstance(source, Mapping):
                return float(source[name])

            return float(
                getattr(source, name)
            )

        return {
            "net_profit_delta": (
                read(candidate, "net_profit")
                - read(champion, "net_profit")
            ),
            "win_rate_delta": (
                read(candidate, "win_rate")
                - read(champion, "win_rate")
            ),
            "trade_count_delta": (
                read(candidate, "trade_count")
                - read(champion, "trade_count")
            ),
            "max_drawdown_delta": (
                read(candidate, "max_drawdown")
                - read(champion, "max_drawdown")
            ),
            "signal_quality_delta": (
                read(candidate, "signal_quality")
                - read(champion, "signal_quality")
            ),
            "stability_delta": (
                read(candidate, "stability")
                - read(champion, "stability")
            ),
        }