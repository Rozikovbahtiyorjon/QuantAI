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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional
import pandas as pd


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

    # Task 7: factor risk context (optional, for correlation-adjusted gate)
    open_positions: Dict[str, Any] = field(default_factory=dict, compare=False, hash=False)  # symbol -> {notional} or Position
    correlation_matrix: Optional[pd.DataFrame] = field(default=None, compare=False, hash=False)
    factor_map: Optional[Dict[str, str]] = field(default=None, compare=False, hash=False)
    betas: Optional[Dict[str, float]] = field(default=None, compare=False, hash=False)
    max_factor_concentration: float = 0.70
    correlation_adjusted_limit: float = 0.15
    max_herfindahl: float = 0.60
    # P0.3: staleness / missing data — UNKNOWN → REJECT
    balance_timestamp: Optional[datetime] = field(default=None, compare=False, hash=False)
    market_data_timestamp: Optional[datetime] = field(default=None, compare=False, hash=False)
    position_state_version: Optional[int] = field(default=None, compare=False, hash=False)
    # Max age for balances/market data (seconds)
    max_balance_age_sec: float = 5.0
    max_market_data_age_sec: float = 5.0

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
