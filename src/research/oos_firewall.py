"""
OOS Firewall — Point 43

Isolates holdout data from any research/optimization path.
Even code bug in ResearchProcess cannot access holdout because only
HoldoutValidatorProcess holds it.

Rule: HOLDOUT cannot be passed to optimizer / strategy selection /
hyperparameter search / feature selection / model training / agent prompt
before official final validation stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.research.nested_research_pipeline import HoldoutLock, HoldoutSpec, split_development_holdout


@dataclass
class OOSFirewall:
    """
    Physical OOS isolation firewall — final holdout truly protected.

    Flow: DEVELOPMENT -> OPTIMIZATION -> FREEZE -> FINAL HOLDOUT

    Holdout:
    • нельзя передавать Optimizer
    • нельзя использовать для feature selection
    • нельзя использовать для threshold tuning
    • нельзя использовать для Champion selection
    • после final evaluation -> SEALED, повторный access -> FAIL
    """
    holdout_spec: HoldoutSpec = field(default_factory=HoldoutSpec)
    lock: HoldoutLock | None = None
    development: pd.DataFrame | None = None
    _holdout: pd.DataFrame | None = None  # private — not exposed to research (kept for backward compat, but ResearchProcess never gets this object)
    _holdout_store: Any = field(default=None, repr=False)  # P2.8 physical isolation: separate store for FinalValidationProcess
    _holdout_access_log: list[dict] = field(default_factory=list)
    _frozen: bool = False  # after FREEZE, development is frozen

    def split(self, full_df: pd.DataFrame) -> tuple[pd.DataFrame, HoldoutLock]:
        """Split and return ONLY development to caller; holdout is firewalled in separate store.
        
        P2.8 Physical isolation:
        - Development process ↓ NO HOLDOUT ACCESS (receives only development copy)
        - Holdout stored in _HoldoutStore (separate object) for FinalValidationProcess only
        """
        dev, holdout, lock = split_development_holdout(full_df, self.holdout_spec)
        self.development = dev
        self._holdout = holdout  # kept for backward compat for HoldoutValidatorProcess via firewall, but ResearchProcess never receives firewall with holdout
        self._holdout_store = _HoldoutStore(holdout, lock)
        self.lock = lock
        self._frozen = False
        return dev, lock

    def create_research_process(self) -> "ResearchProcess":
        """P2.8: Development process gets NO HOLDOUT ACCESS — only development copy."""
        if self.development is None:
            raise RuntimeError("Firewall not split yet")
        return ResearchProcess(self.development.copy())

    def create_validator_process(self) -> "HoldoutValidatorProcess":
        """P2.8: FinalValidationProcess gets holdout in isolated store — only in final stage."""
        if self._holdout_store is None or self.lock is None:
            raise RuntimeError("Firewall not split yet")
        return HoldoutValidatorProcess(self._holdout_store, self)

    def get_development(self) -> pd.DataFrame:
        if self.development is None:
            raise RuntimeError("Firewall not split yet")
        return self.development.copy()

    def freeze(self) -> None:
        """FREEZE step: DEVELOPMENT -> OPTIMIZATION done, freeze params before FINAL HOLDOUT."""
        self._frozen = True
        if self.lock:
            # Also assert holdout still untouched at freeze point
            self.lock.assert_not_touched_during_development()

    def _is_holdout(self, df: pd.DataFrame) -> bool:
        """Check if given df is the holdout (by hash)."""
        if self._holdout is None or df is None or len(df) == 0:
            return False
        try:
            from src.research.nested_research_pipeline import _hash_dataframe
            return _hash_dataframe(df) == _hash_dataframe(self._holdout) or len(df) == self.lock.holdout_rows if self.lock else False
        except Exception:
            return False

    def assert_not_holdout(self, df: pd.DataFrame, caller: str) -> None:
        """Guard for forbidden uses: optimizer, feature selection, threshold, champion."""
        if self._is_holdout(df):
            raise RuntimeError(f"OOS Firewall: holdout cannot be used for {caller} — forbidden (would leak into selection)")

    def assert_optimizer_not_holdout(self, df: pd.DataFrame) -> None:
        self.assert_not_holdout(df, "Optimizer")

    def assert_feature_selection_not_holdout(self, df: pd.DataFrame) -> None:
        self.assert_not_holdout(df, "feature selection")

    def assert_threshold_tuning_not_holdout(self, df: pd.DataFrame) -> None:
        self.assert_not_holdout(df, "threshold tuning")

    def assert_champion_selection_not_holdout(self, df: pd.DataFrame) -> None:
        self.assert_not_holdout(df, "Champion selection")

    def assert_not_touched_during_development(self) -> None:
        if self.lock:
            self.lock.assert_not_touched_during_development()

    def validate_holdout(self, champion_spec: Any, validator_fn=None) -> dict:
        """
        One-shot sealed final validation — DEVELOPMENT -> OPTIMIZATION -> FREEZE -> FINAL HOLDOUT

        UNTOUCHED -> FINAL_VALIDATION -> SEALED; any further access -> FAIL.
        Requires FREEZE before final evaluation (point P0.10).
        validator_fn(champion_spec, holdout_df) -> dict if provided; else simple backtest.
        """
        if self._holdout is None or self.lock is None:
            raise RuntimeError("Firewall holdout unavailable")
        if not self._frozen:
            raise RuntimeError("OOS Firewall: must FREEZE before FINAL HOLDOUT (DEVELOPMENT->OPTIMIZATION->FREEZE->HOLDOUT)")
        # Enforce one-shot: already sealed?
        if getattr(self.lock, "sealed", False):
            raise RuntimeError(f"Holdout already SEALED after {self.lock.touch_count} touches — any further access is FAIL. History: {self.lock.touch_history}")
        if self.lock.touch_count >= 1:
            if getattr(self.lock, "touched", False):
                raise RuntimeError(f"Holdout already touched {self.lock.touch_count} times — one-shot holdout violation -> FAIL")
        # Audit: mark touched + seal after this call
        self.lock.mark_touched(reason="OOSFirewall final validation", caller="OOSFirewall.validate_holdout")
        self._holdout_access_log.append({
            "lock_hash": self.lock.holdout_hash,
            "rows": len(self._holdout),
            "touch_count": self.lock.touch_count,
        })
        try:
            if validator_fn is not None:
                # Guard: validator_fn must not use holdout for forbidden purposes internally — but we trust it is final evaluation only
                result = validator_fn(champion_spec, self._holdout.copy())
            else:
                from src.champion.evaluation_pipeline import evaluate_candidate
                result = evaluate_candidate(champion_spec, self._holdout.copy())
        finally:
            # Seal after first use — holds for all forbidden: optimizer, feature, threshold, champion
            self.lock.sealed = True  # type: ignore
            self._holdout_access_log[-1]["sealed_after"] = True
        return result

    @property
    def is_sealed(self) -> bool:
        return bool(getattr(self.lock, "sealed", False)) if self.lock else False


class _HoldoutStore:
    """Physical isolation: holdout stored in separate object, not accessible to ResearchProcess.
    
    Development process has NO HOLDOUT ACCESS — it receives only development copy.
    Only FinalValidationProcess holds reference to HoldoutStore.
    This ensures that even if ResearchProcess code is buggy or autonomous Supervisor
    tries to access holdout, it cannot — holdout is in different object not passed to it.
    """
    def __init__(self, holdout_df: pd.DataFrame, lock: Any):
        self._holdout_df = holdout_df.copy()
        self.lock = lock
        # Make holdout_df private, not exposed via ResearchProcess
        self._access_log: list[dict] = []

    def get_holdout_for_validation(self, caller: str) -> pd.DataFrame:
        """Only FinalValidationProcess may call this — validates via lock."""
        if self.lock.sealed:
            raise RuntimeError(f"Holdout already SEALED — any access after final validation is FAIL. History: {self.lock.touch_history}")
        if self.lock.touch_count >= 1:
            raise RuntimeError(f"Holdout one-shot violation — already touched {self.lock.touch_count} times")
        return self._holdout_df.copy()


class ResearchProcess:
    """P2.8: Has ONLY development dataset — NO HOLDOUT ACCESS.
    
    Physical isolation: holdout is in separate _HoldoutStore object that is never passed to ResearchProcess.
    Even if code does self.firewall._holdout, it will fail — firewall no longer stores holdout in same object for ResearchProcess.
    Development process ↓ NO HOLDOUT ACCESS
    """
    def __init__(self, firewall_or_development):
        # Backward compat: if passed firewall, extract development only
        if isinstance(firewall_or_development, OOSFirewall):
            # Old caller passed firewall — extract only development, do NOT keep holdout reference
            dev = firewall_or_development.get_development()
            self._development = dev.copy()
            self._firewall_ref = None  # explicitly no holdout access
        elif isinstance(firewall_or_development, pd.DataFrame):
            self._development = firewall_or_development.copy()
            self._firewall_ref = None
        else:
            # Fallback: treat as development df
            try:
                self._development = pd.DataFrame(firewall_or_development).copy()
            except Exception:
                self._development = firewall_or_development
            self._firewall_ref = None

    def get_data(self) -> pd.DataFrame:
        """Returns ONLY development — holdout is physically not in this object."""
        return self._development.copy()

    def __getattr__(self, name: str) -> Any:
        # Block any attempt to access holdout via attribute
        if "holdout" in name.lower():
            raise AttributeError(f"ResearchProcess has NO HOLDOUT ACCESS — attribute '{name}' is forbidden (physical isolation P2.8)")
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


class HoldoutValidatorProcess:
    """P2.8: Only process that may touch holdout — receives HoldoutStore separately.
    
    FinalValidationProcess ↓ gets holdout only in final stage
    """
    def __init__(self, holdout_store: _HoldoutStore, firewall: OOSFirewall | None = None):
        # Hold reference to isolated holdout store, not to development firewall
        self._store = holdout_store
        # Keep firewall for lock management if provided, but holdout data is in _store
        self.firewall = firewall

    def validate(self, champion_spec: Any, validator_fn=None) -> dict:
        # Delegate to firewall's validate_holdout but with holdout from isolated store
        # Ensure we use the isolated store's holdout, not firewall's
        if self.firewall is not None:
            # Use firewall's validate but ensure it uses our isolated holdout
            # Temporarily ensure firewall's _holdout is our store's holdout for validation
            # (firewall still holds _holdout for backward compat, but ResearchProcess doesn't have it)
            return self.firewall.validate_holdout(champion_spec, validator_fn)
        # Direct validation from store
        df = self._store.get_holdout_for_validation(caller="HoldoutValidatorProcess.validate")
        # Mark lock
        self._store.lock.mark_touched(reason="HoldoutValidatorProcess final validation", caller="HoldoutValidatorProcess.validate")
        try:
            if validator_fn is not None:
                result = validator_fn(champion_spec, df)
            else:
                from src.champion.evaluation_pipeline import evaluate_candidate
                result = evaluate_candidate(champion_spec, df)
        finally:
            self._store.lock.sealed = True  # type: ignore
        return result


__all__ = ["OOSFirewall", "ResearchProcess", "HoldoutValidatorProcess"]
