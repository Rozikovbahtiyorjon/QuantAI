"""
QuantAI Champion Feedback Loop (R4)

Closes the cycle:

    Champion -> paper/live telemetry -> FeedbackReport
             -> parameter mutations  -> next-gen CandidateSpecs

Mutations are deterministic perturbations within explicit bounds —
no random search, no curve fitting: the WF harness remains the judge.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# =====================================================
# TELEMETRY FEEDBACK
# =====================================================

@dataclass
class FeedbackReport:
    trades: int = 0
    net_pnl: float = 0.0
    wins: int = 0
    win_rate: float = 0.0
    balance: float | None = None

    @property
    def healthy(self) -> bool:
        return self.trades > 0 and self.net_pnl > 0 and self.win_rate >= 35.0

    def summary(self) -> str:
        return (
            f"trades={self.trades} net={self.net_pnl:.2f} "
            f"wr={self.win_rate:.1f}% balance={self.balance}"
        )


def feedback_from_long_run(directory: Path) -> FeedbackReport:
    """
    Read the most recent long-run journal (R3 artifact format):
        close_time,side,entry,exit,qty,gross,fees,net,balance
    """
    journal = Path(directory) / "journal.csv"
    if not journal.exists():
        raise FileNotFoundError(f"no journal.csv in {directory}")

    with open(journal, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return FeedbackReport()

    nets = [float(r["net"]) for r in rows]
    wins = sum(1 for n in nets if n > 0)
    balance_raw = rows[-1].get("balance", "").strip()

    return FeedbackReport(
        trades=len(rows),
        net_pnl=sum(nets),
        wins=wins,
        win_rate=100.0 * wins / len(rows),
        balance=float(balance_raw) if balance_raw else None,
    )


# =====================================================
# MUTATIONS
# =====================================================

@dataclass
class MutationBounds:
    """
    Numeric parameter space for deterministic ±step exploration.

    bounds: param -> (low, high, step)
    """
    bounds: dict[str, tuple[float, float, float]] = field(default_factory=dict)


def suggest_mutations(
    base_params: dict[str, Any],
    bounds: MutationBounds,
    max_variants: int = 6,
) -> list[dict]:
    """
    Deterministic variants: for each bounded numeric param produce
    one -step and one +step mutation of the base set (capped).
    Order is stable; no randomness.
    """
    variants: list[dict] = []

    for name, (low, high, step) in bounds.bounds.items():
        base_val = float(base_params.get(name, low))

        for direction in (-1.0, +1.0):
            v = round(min(high, max(low, base_val + direction * step)), 10)
            if v == base_val:
                continue
            mutated = dict(base_params)
            mutated[name] = v
            variants.append(mutated)
            if len(variants) >= max_variants:
                return variants

    return variants


def params_differ(a: dict, b: dict) -> bool:
    keys = set(a) | set(b)
    return any(a.get(k) != b.get(k) for k in keys)


__all__ = [
    "FeedbackReport",
    "feedback_from_long_run",
    "MutationBounds",
    "suggest_mutations",
    "params_differ",
]
