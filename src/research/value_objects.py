"""QuantAI Value Objects — P1 MetricSchema types (Probability01, ReturnFraction, DrawdownFraction)"""
from __future__ import annotations
from dataclasses import dataclass

class ValueObjectError(ValueError):
    pass

@dataclass(frozen=True)
class Probability01:
    """0..1 inclusive, not 0..100. Prevents 0.75+75 confusion."""
    value: float
    def __post_init__(self):
        v = float(self.value)
        if not 0.0 <= v <= 1.0:
            raise ValueObjectError(f"Probability01 must be 0..1, got {v}")
        object.__setattr__(self, 'value', v)
    def to_percent(self) -> float:
        return self.value * 100.0
    def __float__(self): return float(self.value)

@dataclass(frozen=True)
class ReturnFraction:
    """Return as fraction, e.g. 0.05 = 5% (not 5)."""
    value: float
    def __post_init__(self):
        v = float(self.value)
        if not -1.0 <= v <= 5.0:
            # allow -100% .. +500%
            raise ValueObjectError(f"ReturnFraction out of range -1..5, got {v}")
        object.__setattr__(self, 'value', v)
    def to_percent(self) -> float:
        return self.value * 100.0
    def __float__(self): return float(self.value)

@dataclass(frozen=True)
class DrawdownFraction:
    """Drawdown as positive fraction 0..1, e.g. 0.15 = 15% DD."""
    value: float
    def __post_init__(self):
        v = float(self.value)
        if not 0.0 <= v <= 1.0:
            raise ValueObjectError(f"DrawdownFraction must be 0..1, got {v}")
        object.__setattr__(self, 'value', v)
    def to_percent(self) -> float:
        return self.value * 100.0
    def __float__(self): return float(self.value)

@dataclass(frozen=True)
class Price:
    value: float
    def __post_init__(self):
        v = float(self.value)
        if v <= 0:
            raise ValueObjectError(f"Price must be >0, got {v}")
        object.__setattr__(self, 'value', v)
    def __float__(self): return float(self.value)

@dataclass(frozen=True)
class Quantity:
    value: float
    def __post_init__(self):
        v = float(self.value)
        if v < 0:
            raise ValueObjectError(f"Quantity must be >=0, got {v}")
        object.__setattr__(self, 'value', v)
    def __float__(self): return float(self.value)

__all__ = ["Probability01","ReturnFraction","DrawdownFraction","Price","Quantity","ValueObjectError"]
