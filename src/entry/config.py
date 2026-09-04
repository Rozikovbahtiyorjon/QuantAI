"""
ENTRY-02 — Entry Configuration Contract (PHASE 0)

No Optuna yet. Only fixed parameters.
Optimization later inside Nested WF.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EntryConfig:
    """Fixed configuration — no optimization until Shadow validation passes."""

    # Setup
    minimum_setup_quality: float = 0.60  # quality 0-1

    # Trigger
    minimum_trigger_quality: float = 0.55

    # Confirmation
    minimum_confirmation_quality: float = 0.60  # structure+momentum+volume avg
    minimum_ml_probability: float = 0.55  # P(win) threshold for TAKE
    minimum_ev: float = 0.001  # 0.1% net edge hurdle

    # Entry zone
    entry_expiration_bars: int = 20  # max bars to wait after setup -> EXPIRED
    max_chase_distance_atr: float = 0.8  # max distance from ideal still chase
    zone_atr_mult: float = 0.4  # half-width of entry zone

    # SL/TP
    sl_buffer_atr: float = 0.3  # structural buffer
    min_rr: float = 2.0  # will be validated via expectancy, not just changed to 2.3
    max_rr: float = 3.0

    # Risk
    risk_per_trade: float = 0.01  # 1%
    max_leverage: float = 3.0  # production

    # Feature states: microstructure placeholders are UNAVAILABLE until live feed
    allow_placeholder_features: bool = False  # if False, PLACEHOLDER features block ENTRY_APPROVED


DEFAULT_ENTRY_CONFIG = EntryConfig()
