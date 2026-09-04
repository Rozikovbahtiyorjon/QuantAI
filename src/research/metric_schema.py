"""
Metric Schema — Point 38-39

Strict unit contract for QuantAI metrics to prevent drawdown = -20 vs -0.20
confusion between Tournament / Evaluator / RobustOOS.

Every metric object must declare name, value, unit, source, period, sample_size.
Units are enforced at runtime; mismatched units raise.

Usage:
    from src.research.metric_schema import MetricSchema, Metric
    m = Metric(name="drawdown", value=0.187, unit="fraction", source="OOS", period="2024-01:2024-06", sample_size=42)
    MetricSchema.validate(m)
    # or dict
    MetricSchema.validate_dict({"name": "profit_factor", "value": 1.4, "unit": "ratio", ...})
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Canonical units per metric family
CANONICAL_UNITS: dict[str, set[str]] = {
    "return": {"fraction", "decimal"},  # 0.05 = 5%
    "drawdown": {"fraction", "decimal"},  # positive fraction e.g. 0.15 = 15%
    "drawdown_pct": {"percent"},  # -15.0
    "win_rate": {"fraction"},  # 0..1
    "profit_factor": {"ratio"},  # raw ratio
    "sharpe": {"ratio"},
    "pf": {"ratio"},
    "expectancy": {"fraction", "decimal", "currency"},
}

# Aliases mapping value field -> family
FAMILY_ALIASES = {
    "return": {"return", "total_return", "net_return", "net_profit"},
    "drawdown": {"drawdown", "max_drawdown", "maxdd", "mdd"},
    "win_rate": {"win_rate", "winrate"},
    "profit_factor": {"profit_factor", "pf", "pf_median", "oos_pf"},
    "sharpe": {"sharpe", "sharpe_ratio", "sharpe_median"},
}


@dataclass(frozen=True)
class Metric:
    name: str
    value: float
    unit: str
    source: str  # OOS, IS, paper, etc.
    period: str  # e.g. "2024-01:2024-06" or window range
    sample_size: int  # trades/windows

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "source": self.source,
            "period": self.period,
            "sample_size": self.sample_size,
        }


class MetricSchema:
    @staticmethod
    def _family_for(name: str) -> str | None:
        low = name.lower().strip()
        for family, aliases in FAMILY_ALIASES.items():
            if low in aliases or low == family:
                return family
        # also check contains
        for family, aliases in FAMILY_ALIASES.items():
            for a in aliases:
                if a in low:
                    return family
        return None

    @staticmethod
    def validate(metric: Metric) -> Metric:
        if not isinstance(metric, Metric):
            raise TypeError("MetricSchema.validate expects Metric dataclass")
        # name required
        if not metric.name or not metric.name.strip():
            raise ValueError("Metric name required")
        # unit required
        if not metric.unit or not metric.unit.strip():
            raise ValueError(f"Metric {metric.name} unit required")
        # sample_size must be >=0
        if metric.sample_size < 0:
            raise ValueError("sample_size must be >=0")
        # family unit check
        family = MetricSchema._family_for(metric.name)
        if family and family in CANONICAL_UNITS:
            allowed = CANONICAL_UNITS[family]
            if metric.unit not in allowed:
                raise ValueError(
                    f"Metric {metric.name} unit {metric.unit!r} not in {allowed} for family {family}"
                )
        # value ranges
        low_name = metric.name.lower()
        if "win_rate" in low_name:
            if not 0.0 <= metric.value <= 1.0:
                raise ValueError(f"win_rate {metric.value} must be in [0,1] (fraction), got {metric.value}")
        if "drawdown" in low_name and metric.unit == "fraction":
            if not 0.0 <= metric.value <= 1.0:
                raise ValueError(f"drawdown fraction {metric.value} must be in [0,1], got {metric.value}")
        if "drawdown" in low_name and metric.unit == "decimal":
            if metric.value < -1.0 or metric.value > 0:
                # allow negative or positive fraction? For DD we expect positive magnitude or negative signed
                pass
        return metric

    @staticmethod
    def validate_dict(d: dict) -> dict:
        required = {"name", "value", "unit", "source", "period", "sample_size"}
        missing = required - set(d.keys())
        if missing:
            raise ValueError(f"Metric dict missing keys: {missing}")
        m = Metric(
            name=str(d["name"]),
            value=float(d["value"]),
            unit=str(d["unit"]),
            source=str(d["source"]),
            period=str(d["period"]),
            sample_size=int(d["sample_size"]),
        )
        MetricSchema.validate(m)
        return m.to_dict()

    @staticmethod
    def coerce_drawdown(value: float, unit: str) -> float:
        """
        Normalize drawdown to canonical fraction 0..1 positive.
        Accepts percent (-15.0 or 15.0), fraction (0.15), or signed decimal.
        """
        v = float(value)
        if unit == "percent":
            return abs(v) / 100.0
        if unit == "fraction":
            return abs(v) if v <= 1.0 else abs(v) / 100.0
        if unit == "decimal":
            # signed fraction -0.15
            return abs(v) if abs(v) <= 1.0 else abs(v) / 100.0
        return abs(v)

    @staticmethod
    def normalize_pf(value: float) -> float:
        """Cap PF infinities to 99.0 per contract."""
        import math
        if not math.isfinite(float(value)):
            return 99.0
        return min(float(value), 99.0)


__all__ = ["Metric", "MetricSchema", "CANONICAL_UNITS"]
