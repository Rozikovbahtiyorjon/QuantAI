"""
Drift-Triggered Retraining (P-C1 drift -> new candidate)

Watches live Feature Store vs training reference.
When PSI > threshold on N features, triggers retraining
and pushes new candidate to Champion Bank.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from src.feature_store import FeatureStore


@dataclass
class DriftTriggerConfig:
    psi_threshold: float = 0.25
    min_drifted_features: int = 3
    cooldown_hours: int = 24          # don't retrigger too often
    reference_view: str = "btc_walk_forward_window_1"
    reference_version: int | str = 1
    live_view: str = "live_btc_15m"
    live_version: int | str = "latest"


class DriftRetrainTrigger:
    """
    Monitors drift; on trigger, calls retrain_fn and reports.
    retrain_fn: () -> dict with keys {candidate_id, metrics, model_path}
    """

    def __init__(
        self,
        config: DriftTriggerConfig | None = None,
        store: FeatureStore | None = None,
        retrain_fn: Optional[Callable[[], dict]] = None,
    ):
        self.config = config or DriftTriggerConfig()
        self.store = store or FeatureStore()
        self.retrain_fn = retrain_fn
        self._last_trigger_ts: float | None = None
        self._trigger_count: int = 0
        self._last_check: dict | None = None

    @property
    def trigger_count(self) -> int:
        return self._trigger_count

    def _in_cooldown(self) -> bool:
        if self._last_trigger_ts is None:
            return False
        hours = (time.time() - self._last_trigger_ts) / 3600
        return hours < self.config.cooldown_hours

    def check_drift(self) -> dict | None:
        """
        Compare live vs reference. Returns drift report or None.
        """
        try:
            ref_df = self.store.get_features(
                self.config.reference_view, self.config.reference_version
            )
            live_df = self.store.get_features(
                self.config.live_view, self.config.live_version
            )
        except FileNotFoundError:
            return None

        from src.feature_store.drift import detect_drift

        drift_df = detect_drift(
            ref_df,
            live_df,
            psi_threshold=self.config.psi_threshold,
        )
        if drift_df.empty:
            self._last_check = {"drifted": 0, "total": len(drift_df)}
            return None

        drifted = drift_df[drift_df["drifted"]]
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "drifted_count": len(drifted),
            "total_features": len(drift_df),
            "psi_max": float(drifted["psi"].max()) if not drifted.empty else 0.0,
            "drifted_features": drifted.to_dict(orient="records"),
            "threshold": self.config.psi_threshold,
            "triggered": len(drifted) >= self.config.min_drifted_features,
        }
        self._last_check = report

        if report["triggered"] and self._in_cooldown():
            report["triggered"] = False
            report["reason"] = f"cooldown {self.config.cooldown_hours}h active"

        return report if report["triggered"] else None

    def maybe_trigger_retrain(self) -> dict | None:
        """
        Check drift and, if triggered, run retrain_fn.
        Returns retrain result or None.
        """
        report = self.check_drift()
        if report is None or not report.get("triggered"):
            return None

        if self.retrain_fn is None:
            # No retrain function = just report trigger
            self._last_trigger_ts = time.time()
            self._trigger_count += 1
            return {"triggered": True, "drift_report": report, "retrain": None}

        try:
            result = self.retrain_fn()
            self._last_trigger_ts = time.time()
            self._trigger_count += 1
            try:
                from src.monitoring.metrics import feature_store_retrains_total

                feature_store_retrains_total.labels(
                    view=self.config.live_view, result="success"
                ).inc()
            except Exception:
                pass
            return {
                "triggered": True,
                "drift_report": report,
                "retrain": result,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            try:
                from src.monitoring.metrics import feature_store_retrains_total

                feature_store_retrains_total.labels(
                    view=self.config.live_view, result="error"
                ).inc()
            except Exception:
                pass
            return {
                "triggered": True,
                "drift_report": report,
                "error": f"{type(e).__name__}: {e}",
            }

    def force_trigger(self) -> dict | None:
        """Force retrain without drift check (for testing/manual)."""
        if self.retrain_fn is None:
            return None
        self._last_trigger_ts = time.time()
        self._trigger_count += 1
        return {"triggered": True, "retrain": self.retrain_fn()}
