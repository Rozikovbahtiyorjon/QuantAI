"""
Centralized Feature Store with versioning + lineage + drift detection.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from src.feature_store.drift import detect_drift


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:8]
    except Exception:
        pass
    return None


def _hash_dataframe(df: pd.DataFrame) -> str:
    h = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()
    return h[:12]


@dataclass
class FeatureView:
    name: str
    features: list[str]
    description: str = ""
    version: int = 1


class FeatureStore:
    """
    File-based feature store: data/feature_store/<view>/v{n}/
      - features.parquet
      - metadata.json (version, hash, lineage, drift vs previous)
    """

    def __init__(self, root: str | Path = "data/feature_store"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._views: dict[str, FeatureView] = {}

    def register_view(self, view: FeatureView) -> None:
        self._views[view.name] = view

    def _view_dir(self, view_name: str) -> Path:
        return self.root / view_name

    def _next_version(self, view_name: str) -> int:
        vdir = self._view_dir(view_name)
        if not vdir.exists():
            return 1
        versions = [
            int(p.name[1:])
            for p in vdir.iterdir()
            if p.is_dir() and p.name.startswith("v") and p.name[1:].isdigit()
        ]
        return max(versions) + 1 if versions else 1

    def materialize(
        self,
        view_name: str,
        df: pd.DataFrame,
        lineage: dict | None = None,
    ) -> dict:
        """
        Persist feature dataframe as new version.
        Returns metadata dict.
        """
        if view_name not in self._views:
            # Auto-register minimal view
            self.register_view(FeatureView(name=view_name, features=list(df.columns)))

        view = self._views[view_name]
        version = self._next_version(view_name)
        view.version = version

        vpath = self._view_dir(view_name) / f"v{version}"
        vpath.mkdir(parents=True, exist_ok=True)

        # Write parquet
        df.to_parquet(vpath / "features.parquet", index=False)

        # Drift vs previous version
        drift_report = None
        if version > 1:
            prev_path = self._view_dir(view_name) / f"v{version-1}" / "features.parquet"
            if prev_path.exists():
                prev_df = pd.read_parquet(prev_path)
                drift_df = detect_drift(prev_df, df)
                drift_report = drift_df.to_dict(orient="records") if not drift_df.empty else []

        metadata = {
            "view": view_name,
            "version": version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "rows": len(df),
            "columns": list(df.columns),
            "hash": _hash_dataframe(df),
            "git_commit": _git_commit(),
            "lineage": lineage or {},
            "drift_vs_previous": drift_report,
        }

        (vpath / "metadata.json").write_text(
            json.dumps(metadata, indent=2, default=str), encoding="utf-8"
        )

        # Update latest pointer
        (self._view_dir(view_name) / "latest.json").write_text(
            json.dumps({"version": version, "path": str(vpath)}, indent=2),
            encoding="utf-8",
        )

        return metadata

    def get_features(
        self, view_name: str, version: int | str = "latest"
    ) -> pd.DataFrame:
        if version == "latest":
            latest_path = self._view_dir(view_name) / "latest.json"
            if not latest_path.exists():
                raise FileNotFoundError(f"No versions for view {view_name}")
            version = json.loads(latest_path.read_text(encoding="utf-8"))["version"]

        vpath = self._view_dir(view_name) / f"v{version}" / "features.parquet"
        if not vpath.exists():
            raise FileNotFoundError(f"Version v{version} not found for {view_name}")
        return pd.read_parquet(vpath)

    def get_metadata(self, view_name: str, version: int | str = "latest") -> dict:
        if version == "latest":
            # Resolve via latest.json
            latest_path = self._view_dir(view_name) / "latest.json"
            version = json.loads(latest_path.read_text(encoding="utf-8"))["version"]
        mpath = self._view_dir(view_name) / f"v{version}" / "metadata.json"
        return json.loads(mpath.read_text(encoding="utf-8"))

    def list_versions(self, view_name: str) -> list[int]:
        vdir = self._view_dir(view_name)
        if not vdir.exists():
            return []
        return sorted(
            int(p.name[1:])
            for p in vdir.iterdir()
            if p.is_dir() and p.name.startswith("v") and p.name[1:].isdigit()
        )

    def rollback(self, view_name: str, to_version: int) -> dict:
        """
        Rollback latest pointer to to_version (no data deletion).
        Returns new latest metadata.
        """
        versions = self.list_versions(view_name)
        if to_version not in versions:
            raise ValueError(f"Version v{to_version} not found, available: {versions}")
        vpath = self._view_dir(view_name) / f"v{to_version}"
        (self._view_dir(view_name) / "latest.json").write_text(
            json.dumps({"version": to_version, "path": str(vpath)}, indent=2),
            encoding="utf-8",
        )
        # Update view version
        if view_name in self._views:
            self._views[view_name].version = to_version
        return self.get_metadata(view_name, to_version)

    def check_drift(
        self,
        view_name: str,
        reference_version: int,
        current_df: pd.DataFrame,
        psi_threshold: float = 0.2,
    ) -> pd.DataFrame:
        ref_df = self.get_features(view_name, reference_version)
        return detect_drift(ref_df, current_df, psi_threshold=psi_threshold)
