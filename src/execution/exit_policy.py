"""
ExitPolicy — extracted from src/trade_engine.py:80 (Audit §43 split)
Canonical location. TradeEngine re-exports for backward compat.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExitPolicy:
    """
    Configurable exit behavior.
    Defaults reproduce the historical engine exactly:
        trail 2.0 x ATR, break-even after 1.0 x ATR,
        fixed TP active, no time-based exit.
    """

    trail_atr_mult: float = 2.0
    break_even_atr: float | None = 1.0
    use_take_profit: bool = True
    time_exit_bars: int | None = None
