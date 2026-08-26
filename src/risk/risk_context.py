"""
QuantAI RiskContext (R0.1)

Single canonical input object for RiskOrchestrator.

Eliminates the FLIP ambiguity: on an opposite-signal flip the
orchestrator receives BOTH the current exposure (old position still
open) and the projected exposure AFTER the planned close, so the
exposure decision is made against the state that will actually exist
if the trade is committed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RiskContext:
    """
    Immutable snapshot describing a pending entry decision.
    """

    equity: float
    balance: float

    # Notional exposure of positions open RIGHT NOW.
    current_exposure: float = 0.0

    # Notional exposure AFTER the planned action
    # (flip => old position closed => typically 0.0).
    projected_exposure: float = 0.0

    position_side: str | None = None      # side of the open position, if any
    requested_side: str = ""              # "LONG" / "SHORT"

    is_flip: bool = False                 # opposite-signal replacement

    entry: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0

    risk_percent: float = 1.0
    leverage: float = 1.0

    def __post_init__(self) -> None:
        numeric = (
            self.equity,
            self.balance,
            self.current_exposure,
            self.projected_exposure,
            self.entry,
            self.stop_loss,
            self.take_profit,
            self.risk_percent,
            self.leverage,
        )
        for v in numeric:
            if not math.isfinite(float(v)):
                raise ValueError("RiskContext requires finite numeric values")

        if self.equity < 0 or self.balance < 0:
            raise ValueError("RiskContext equity/balance must be non-negative")

        if self.current_exposure < 0 or self.projected_exposure < 0:
            raise ValueError("RiskContext exposures must be non-negative")

        if self.requested_side not in ("LONG", "SHORT"):
            raise ValueError("requested_side must be LONG or SHORT")

    @property
    def effective_exposure(self) -> float:
        """
        Exposure baseline for limit checks: what will exist if the
        planned action commits (old position closed on flip).
        """
        return self.projected_exposure


__all__ = ["RiskContext"]
