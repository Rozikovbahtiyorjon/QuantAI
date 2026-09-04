"""
ENTRY-56 — Lifecycle (PHASE 12)

States: NEW, SETUP_DETECTED, WAIT_TRIGGER, TRIGGERED, WAIT_CONFIRMATION, CONFIRMED, EV_EVALUATION, RISK_EVALUATION, APPROVED, ORDER_SUBMITTED, FILLED, EXPIRED, INVALIDATED, REJECTED, CLOSED
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum


class LifecycleState(str, Enum):
    NEW = "NEW"
    SETUP_DETECTED = "SETUP_DETECTED"
    WAIT_TRIGGER = "WAIT_TRIGGER"
    TRIGGERED = "TRIGGERED"
    WAIT_CONFIRMATION = "WAIT_CONFIRMATION"
    CONFIRMED = "CONFIRMED"
    EV_EVALUATION = "EV_EVALUATION"
    RISK_EVALUATION = "RISK_EVALUATION"
    APPROVED = "APPROVED"
    ORDER_SUBMITTED = "ORDER_SUBMITTED"
    FILLED = "FILLED"
    EXPIRED = "EXPIRED"
    INVALIDATED = "INVALIDATED"
    REJECTED = "REJECTED"
    CLOSED = "CLOSED"


@dataclass
class LifecycleConfig:
    setup_expiry_bars: int = 20
    trigger_expiry_bars: int = 10
    entry_zone_expiry_bars: int = 15
    max_age_bars: int = 30


@dataclass
class EntryLifecycleState:
    state: LifecycleState = LifecycleState.NEW
    bars_since_setup: int = 0
    bars_since_trigger: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    invalidated_reason: str = ""
    expired_reason: str = ""

    def is_terminal(self) -> bool:
        return self.state in (LifecycleState.APPROVED, LifecycleState.ORDER_SUBMITTED, LifecycleState.FILLED, LifecycleState.EXPIRED, LifecycleState.INVALIDATED, LifecycleState.REJECTED, LifecycleState.CLOSED)

    def should_expire(self, config: LifecycleConfig) -> tuple[bool, str]:
        if self.state == LifecycleState.SETUP_DETECTED and self.bars_since_setup > config.setup_expiry_bars:
            return True, f"SETUP lifetime {self.bars_since_setup} > {config.setup_expiry_bars}"
        if self.state == LifecycleState.WAIT_TRIGGER and self.bars_since_setup > config.setup_expiry_bars:
            return True, f"TRIGGER lifetime {self.bars_since_setup} > {config.setup_expiry_bars}"
        if self.state in (LifecycleState.TRIGGERED, LifecycleState.WAIT_CONFIRMATION) and self.bars_since_trigger > config.trigger_expiry_bars:
            return True, f"TRIGGER expiry {self.bars_since_trigger} > {config.trigger_expiry_bars}"
        if self.bars_since_setup > config.max_age_bars:
            return True, f"max age {self.bars_since_setup} > {config.max_age_bars}"
        return False, ""

    def invalidate(self, reason: str):
        self.state = LifecycleState.INVALIDATED
        self.invalidated_reason = reason

    def expire(self, reason: str):
        self.state = LifecycleState.EXPIRED
        self.expired_reason = reason
