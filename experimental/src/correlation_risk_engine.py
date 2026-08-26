from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class CorrelationRiskResult:
    asset: str
    correlated_assets: tuple[str, ...]
    max_correlation: float
    correlated_count: int
    risk_allowed: bool


class CorrelationRiskEngine:
    def __init__(
        self,
        max_correlation: float = 0.85,
        max_correlated_assets: int = 2,
    ) -> None:
        if not 0.0 <= max_correlation <= 1.0:
            raise ValueError(
                "max_correlation must be between 0 and 1."
            )

        if (
            isinstance(max_correlated_assets, bool)
            or not isinstance(max_correlated_assets, int)
        ):
            raise TypeError(
                "max_correlated_assets must be an integer."
            )

        if max_correlated_assets <= 0:
            raise ValueError(
                "max_correlated_assets must be greater than zero."
            )

        self.max_correlation = float(max_correlation)
        self.max_correlated_assets = max_correlated_assets

    def evaluate(
        self,
        asset: str,
        correlations: Mapping[str, float],
    ) -> CorrelationRiskResult:
        if not isinstance(asset, str) or not asset.strip():
            raise ValueError(
                "asset must be a non-empty string."
            )

        if not isinstance(correlations, Mapping):
            raise TypeError(
                "correlations must be a mapping."
            )

        normalized: dict[str, float] = {}

        for name, value in correlations.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    "correlation asset names must be non-empty strings."
                )

            if name == asset:
                continue

            correlation = float(value)

            if not -1.0 <= correlation <= 1.0:
                raise ValueError(
                    "correlation values must be between -1 and 1."
                )

            normalized[name] = correlation

        correlated = tuple(
            name
            for name, correlation in normalized.items()
            if abs(correlation) >= self.max_correlation
        )

        if normalized:
            max_correlation = max(
                abs(value)
                for value in normalized.values()
            )
        else:
            max_correlation = 0.0

        risk_allowed = (
            len(correlated)
            <= self.max_correlated_assets
        )

        return CorrelationRiskResult(
            asset=asset,
            correlated_assets=correlated,
            max_correlation=round(
                max_correlation,
                8,
            ),
            correlated_count=len(correlated),
            risk_allowed=risk_allowed,
        )

    def is_allowed(
        self,
        asset: str,
        correlations: Mapping[str, float],
    ) -> bool:
        return self.evaluate(
            asset=asset,
            correlations=correlations,
        ).risk_allowed


__all__ = [
    "CorrelationRiskResult",
    "CorrelationRiskEngine",
]