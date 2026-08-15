from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class StrategyGenome:
    strategy_id: str
    version: str
    market: str
    timeframes: tuple[str, ...]
    features: tuple[str, ...]
    indicators: tuple[str, ...]
    ml_model: str
    regime_filters: tuple[str, ...]
    entry_logic: Mapping[str, Any]
    exit_logic: Mapping[str, Any]
    risk_profile: str
    position_sizing: Mapping[str, Any]
    portfolio_constraints: Mapping[str, Any]
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._validate_text(self.strategy_id, "strategy_id")
        self._validate_text(self.version, "version")
        self._validate_text(self.market, "market")
        self._validate_text(self.ml_model, "ml_model")
        self._validate_text(self.risk_profile, "risk_profile")

        self._validate_sequence(self.timeframes, "timeframes")
        self._validate_sequence(self.features, "features")
        self._validate_sequence(self.indicators, "indicators")
        self._validate_sequence(self.regime_filters, "regime_filters")

        self._validate_mapping(self.entry_logic, "entry_logic")
        self._validate_mapping(self.exit_logic, "exit_logic")
        self._validate_mapping(self.position_sizing, "position_sizing")
        self._validate_mapping(
            self.portfolio_constraints,
            "portfolio_constraints",
        )
        self._validate_mapping(self.parameters, "parameters")

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "version": self.version,
            "market": self.market,
            "timeframes": list(self.timeframes),
            "features": list(self.features),
            "indicators": list(self.indicators),
            "ml_model": self.ml_model,
            "regime_filters": list(self.regime_filters),
            "entry_logic": dict(self.entry_logic),
            "exit_logic": dict(self.exit_logic),
            "risk_profile": self.risk_profile,
            "position_sizing": dict(self.position_sizing),
            "portfolio_constraints": dict(
                self.portfolio_constraints
            ),
            "parameters": dict(self.parameters),
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> StrategyGenome:
        if not isinstance(data, Mapping):
            raise TypeError("data must be a mapping.")

        required = (
            "strategy_id",
            "version",
            "market",
            "timeframes",
            "features",
            "indicators",
            "ml_model",
            "regime_filters",
            "entry_logic",
            "exit_logic",
            "risk_profile",
            "position_sizing",
            "portfolio_constraints",
        )

        missing = [
            key
            for key in required
            if key not in data
        ]

        if missing:
            raise ValueError(
                "missing required genome fields: "
                + ", ".join(missing)
            )

        return cls(
            strategy_id=data["strategy_id"],
            version=data["version"],
            market=data["market"],
            timeframes=tuple(data["timeframes"]),
            features=tuple(data["features"]),
            indicators=tuple(data["indicators"]),
            ml_model=data["ml_model"],
            regime_filters=tuple(data["regime_filters"]),
            entry_logic=dict(data["entry_logic"]),
            exit_logic=dict(data["exit_logic"]),
            risk_profile=data["risk_profile"],
            position_sizing=dict(data["position_sizing"]),
            portfolio_constraints=dict(
                data["portfolio_constraints"]
            ),
            parameters=dict(data.get("parameters", {})),
        )

    def evolve(
        self,
        **changes: Any,
    ) -> StrategyGenome:
        allowed = {
            "version",
            "market",
            "timeframes",
            "features",
            "indicators",
            "ml_model",
            "regime_filters",
            "entry_logic",
            "exit_logic",
            "risk_profile",
            "position_sizing",
            "portfolio_constraints",
            "parameters",
        }

        unknown = set(changes) - allowed

        if unknown:
            raise ValueError(
                "unknown genome fields: "
                + ", ".join(sorted(unknown))
            )

        data = self.to_dict()
        data.update(changes)

        if "timeframes" in changes:
            data["timeframes"] = tuple(
                changes["timeframes"]
            )

        if "features" in changes:
            data["features"] = tuple(
                changes["features"]
            )

        if "indicators" in changes:
            data["indicators"] = tuple(
                changes["indicators"]
            )

        if "regime_filters" in changes:
            data["regime_filters"] = tuple(
                changes["regime_filters"]
            )

        return StrategyGenome(
            strategy_id=data["strategy_id"],
            version=data["version"],
            market=data["market"],
            timeframes=tuple(data["timeframes"]),
            features=tuple(data["features"]),
            indicators=tuple(data["indicators"]),
            ml_model=data["ml_model"],
            regime_filters=tuple(data["regime_filters"]),
            entry_logic=dict(data["entry_logic"]),
            exit_logic=dict(data["exit_logic"]),
            risk_profile=data["risk_profile"],
            position_sizing=dict(data["position_sizing"]),
            portfolio_constraints=dict(
                data["portfolio_constraints"]
            ),
            parameters=dict(data["parameters"]),
        )

    @staticmethod
    def _validate_text(
        value: Any,
        field_name: str,
    ) -> None:
        if not isinstance(value, str):
            raise TypeError(
                f"{field_name} must be a string."
            )

        if not value.strip():
            raise ValueError(
                f"{field_name} cannot be empty."
            )

    @classmethod
    def _validate_sequence(
        cls,
        value: Any,
        field_name: str,
    ) -> None:
        if not isinstance(value, tuple):
            raise TypeError(
                f"{field_name} must be a tuple."
            )

        if not value:
            raise ValueError(
                f"{field_name} cannot be empty."
            )

        for item in value:
            cls._validate_text(
                item,
                f"{field_name} item",
            )

    @staticmethod
    def _validate_mapping(
        value: Any,
        field_name: str,
    ) -> None:
        if not isinstance(value, Mapping):
            raise TypeError(
                f"{field_name} must be a mapping."
            )


__all__ = ["StrategyGenome"]