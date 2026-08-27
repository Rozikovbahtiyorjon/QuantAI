"""
Live Feature Logger - auto-logging live features to Feature Store
with drift detection and non-blocking batch materialization.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from src.feature_store.store import FeatureStore
from src.feature_store.drift import detect_drift


@dataclass
class LiveLoggerConfig:
    view_name: str = "live_btc_15m"
    store_root: str = "data/feature_store"
    buffer_size: int = 100          # materialize every N features
    drift_check_every: int = 500    # check drift every N features
    psi_threshold: float = 0.2
    ks_p_threshold: float = 0.05
    reference_version: int | str = "latest"  # training reference for drift
    reference_view: Optional[str] = None     # if None, uses view_name's previous


@dataclass
class DriftAlert:
    timestamp: str
    drifted_features: list[dict]
    total_features: int
    drifted_count: int
    message: str


class LiveFeatureLogger:
    """
    Non-blocking live feature logger.

    Usage:
        logger = LiveFeatureLogger(view_name="live_btc_15m")
        features = build_features(df)  # dict
        logger.log(features)            # buffered, auto-flushes every 100

        # Or wrap FeatureEngine:
        engine = FeatureEngine(live_logger=logger)

    Thread-safe: log() is lock-protected, flush is atomic.
    """

    def __init__(
        self,
        config: LiveLoggerConfig | None = None,
        view_name: str | None = None,
        store: FeatureStore | None = None,
    ):
        if config is None and view_name is not None:
            config = LiveLoggerConfig(view_name=view_name)
        self.config = config or LiveLoggerConfig()
        self.store = store or FeatureStore(self.config.store_root)
        self._buffer: deque[dict] = deque()
        self._lock = threading.Lock()
        self._total_logged: int = 0
        self._last_materialized_version: int | None = None
        self._drift_alerts: list[DriftAlert] = []
        self._last_drift_check: int = 0

    @property
    def buffered_count(self) -> int:
        with self._lock:
            return len(self._buffer)

    @property
    def total_logged(self) -> int:
        return self._total_logged

    @property
    def drift_alerts(self) -> list[DriftAlert]:
        return list(self._drift_alerts)

    def log(self, features: dict, timestamp: str | None = None) -> bool:
        """
        Buffer one feature dict. Returns True if flush was triggered.
        Non-blocking, thread-safe.
        """
        if not isinstance(features, dict) or not features:
            return False

        entry = dict(features)
        entry["_logged_at"] = timestamp or datetime.now(timezone.utc).isoformat()

        with self._lock:
            self._buffer.append(entry)
            should_flush = len(self._buffer) >= self.config.buffer_size

        if should_flush:
            self.flush()
            return True
        return False

    def flush(self) -> dict | None:
        """
        Materialize buffered features to store as new version.
        Returns metadata or None if buffer empty.
        Thread-safe, non-blocking for callers (flush holds lock briefly).
        """
        with self._lock:
            if not self._buffer:
                return None
            batch = list(self._buffer)
            self._buffer.clear()

        df = pd.DataFrame(batch)
        # Drop internal column for drift comparison, keep for lineage
        df_features = df.drop(columns=["_logged_at"], errors="ignore")

        lineage = {
            "source": "live",
            "batch_size": len(df),
            "total_logged_before": self._total_logged,
        }

        try:
            meta = self.store.materialize(
                self.config.view_name, df_features, lineage=lineage
            )
            self._last_materialized_version = meta["version"]
            self._total_logged += len(df)

            # Periodic drift check vs reference
            if (
                self._total_logged - self._last_drift_check
                >= self.config.drift_check_every
            ):
                self._check_drift_async()
                self._last_drift_check = self._total_logged

            return meta
        except Exception as e:
            # Restore buffer on failure (don't lose data)
            with self._lock:
                self._buffer.extendleft(reversed(batch))
            print(f"[LiveLogger] materialize failed: {e}")
            return None

    def _check_drift_async(self) -> None:
        """Run drift check in background thread (non-blocking)."""
        def _run():
            try:
                self.check_drift()
            except Exception as e:
                print(f"[LiveLogger] drift check failed: {e}")

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

    def check_drift(self) -> DriftAlert | None:
        """
        Compare latest live version vs reference (training) version.
        Returns DriftAlert if drift detected, else None.
        """
        try:
            ref_view = self.config.reference_view or self.config.view_name
            ref_version = self.config.reference_version

            # Need at least 2 versions to compare; if reference_view == view_name,
            # compare latest vs previous
            if ref_view == self.config.view_name:
                versions = self.store.list_versions(self.config.view_name)
                if len(versions) < 2:
                    return None
                # Compare latest vs previous
                latest = self.store.get_features(self.config.view_name, "latest")
                prev_version = sorted(versions)[-2]
                ref_df = self.store.get_features(self.config.view_name, prev_version)
                curr_df = latest
            else:
                ref_df = self.store.get_features(ref_view, ref_version)
                curr_df = self.store.get_features(self.config.view_name, "latest")

            drift_df = detect_drift(
                ref_df,
                curr_df,
                psi_threshold=self.config.psi_threshold,
                ks_p_threshold=self.config.ks_p_threshold,
            )
            if drift_df.empty:
                return None

            drifted = drift_df[drift_df["drifted"]]
            if drifted.empty:
                return None

            alert = DriftAlert(
                timestamp=datetime.now(timezone.utc).isoformat(),
                drifted_features=drifted.to_dict(orient="records"),
                total_features=len(drift_df),
                drifted_count=len(drifted),
                message=(
                    f"[LiveLogger] DRIFT: {len(drifted)}/{len(drift_df)} features "
                    f"drifted (PSI>{self.config.psi_threshold})"
                ),
            )
            self._drift_alerts.append(alert)
            print(alert.message)
            for _, row in drifted.head(5).iterrows():
                print(f"  - {row['feature']}: PSI={row['psi']:.3f} KS_p={row['ks_pvalue']:.4f}")

            return alert

        except FileNotFoundError:
            return None
        except Exception as e:
            print(f"[LiveLogger] drift check error: {e}")
            return None

    def get_live_features(self, version: int | str = "latest") -> pd.DataFrame:
        return self.store.get_features(self.config.view_name, version)

    def force_drift_check(self) -> DriftAlert | None:
        """Synchronous drift check (for testing/monitoring)."""
        return self.check_drift()
