"""
QuantAI Dataset Registry — Audit #32-33 + Task 8 Data Gates + Immutable Dataset

Each dataset has: dataset_id, symbol, timeframe, start, end, row_count, source, download_timestamp,
 schema_version, feature_version, label_version, hash (SHA256).

Canonical dataset identity is hash, not path. Experiments reference dataset_id.
Handles legacy *.bak.parquet / tmp_prepared.parquet by enforcing registry lookup.

Task 8 Enhancements:
- Automated data gates via src.data.data_gates.DataGates before hashing (fail-fast).
- Immutable dataset: file hash stored, parquet made read-only after registration,
  verification on load prevents in-place mutation / tampering.
- Hash verified on load(); tampered file raises ValueError.
- No in-place mutation: load() returns copy, file is read-only, hash is canonical.

Immutability guarantees:
1. hash: SHA256 of parquet bytes stored in record.
2. no in-place mutation: verify() recomputes hash and compares; mismatch raises.
3. read-only: parquet file chmod 0o444 after register (POSIX + Windows read-only flag).
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class DatasetRecord:
    # P2.7 exact 11 fields + legacy aliases
    dataset_id: str  # e.g., BTCUSDT_15M_v7
    exchange: str = "binance"  # P2.7
    symbol: str = ""
    timeframe: str = ""
    start: str = ""
    end: str = ""
    rows: int = 0  # P2.7 canonical
    row_count: int = 0  # alias for backward compat
    schema_hash: str = ""  # P2.7 hash of schema (columns+dtypes)
    raw_hash: str = ""  # P2.7 hash of raw source
    prepared_hash: str = ""  # P2.7 hash of prepared parquet (canonical)
    feature_version: str = "v1"  # P2.7
    label_version: str = "triple_barrier_v1"  # P2.7
    # Full reproducibility extensions + legacy
    source: str = "binance"
    download_timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    schema_version: str = "v1"
    hash: str = ""  # SHA256 of parquet bytes (alias prepared_hash)
    path: str = ""
    columns: List[str] = field(default_factory=list)
    dtypes: Dict[str, str] = field(default_factory=dict)
    # Legacy compatibility: keep hash as primary, prepared_hash mirrors it

    def __post_init__(self) -> None:
        # Sync aliases: rows <-> row_count, hash <-> prepared_hash, schema_hash fallback
        if self.rows and not self.row_count:
            self.row_count = self.rows
        if self.row_count and not self.rows:
            self.rows = self.row_count
        if self.prepared_hash and not self.hash:
            self.hash = self.prepared_hash
        if self.hash and not self.prepared_hash:
            self.prepared_hash = self.hash

    def to_dict(self) -> Dict:
        # Ensure sync before export
        if self.rows and not self.row_count:
            self.row_count = self.rows
        if self.row_count and not self.rows:
            self.rows = self.row_count
        return asdict(self)


class DatasetRegistry:
    def __init__(self, root: str = "data/registry"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._index: Dict[str, DatasetRecord] = {}
        self._load()

    def _load(self) -> None:
        for fp in self.root.glob("*.json"):
            try:
                d = json.loads(fp.read_text(encoding="utf-8"))
                rec = DatasetRecord(**{k: v for k, v in d.items() if k in DatasetRecord.__dataclass_fields__})
                self._index[rec.dataset_id] = rec
            except Exception:
                continue

    @staticmethod
    def hash_file(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    # ------------------------------------------------------------------
    # Immutability helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_readonly(path: Path) -> None:
        """Make parquet file read-only (immutable). POSIX 0o444 + Windows attribute."""
        try:
            # POSIX read-only for owner/group/others
            path.chmod(0o444)
        except Exception:
            try:
                os.chmod(path, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
            except Exception:
                pass
        # Windows read-only flag via file attribute
        if os.name == "nt":
            try:
                import ctypes  # type: ignore

                # FILE_ATTRIBUTE_READONLY = 0x01
                ctypes.windll.kernel32.SetFileAttributesW(str(path.resolve()), 1)
            except Exception:
                pass
        # Fallback: ensure write bit removed
        try:
            # Remove write permissions explicitly
            current = Path(path).stat().st_mode
            os.chmod(path, current & ~stat.S_IWRITE & ~stat.S_IWGRP & ~stat.S_IWOTH)
        except Exception:
            pass

    @staticmethod
    def _make_writable(path: Path) -> None:
        """Restore writable for testing / re-registration (removes read-only)."""
        try:
            path.chmod(0o644)
        except Exception:
            try:
                os.chmod(path, stat.S_IREAD | stat.S_IWRITE)
            except Exception:
                pass
        if os.name == "nt":
            try:
                import ctypes  # type: ignore

                # FILE_ATTRIBUTE_NORMAL = 0x80 (or 0 to clear read-only)
                ctypes.windll.kernel32.SetFileAttributesW(str(path.resolve()), 0x80)
                # Ensure writable
                os.chmod(path, stat.S_IREAD | stat.S_IWRITE)
            except Exception:
                pass

    @staticmethod
    def _is_readonly(path: Path) -> bool:
        """Check if file is read-only (no write bit)."""
        try:
            st = path.stat()
            # If any write bit is set, it's writable
            writable = bool(st.st_mode & (stat.S_IWRITE | stat.S_IWGRP | stat.S_IWOTH))
            return not writable
        except Exception:
            # Fallback via os.access
            return not os.access(path, os.W_OK)

    def is_readonly(self, dataset_id: str) -> bool:
        """Public check: is registered parquet file read-only?"""
        rec = self.get(dataset_id)
        if rec is None:
            raise KeyError(f"dataset_id '{dataset_id}' not found in registry")
        p = Path(rec.path)
        if not p.exists():
            raise FileNotFoundError(rec.path)
        return self._is_readonly(p)

    def verify(self, dataset_id: str) -> bool:
        """
        Verify dataset immutability: recompute hash and compare to stored hash.

        Returns True if intact, raises ValueError if tampered/mutated.

        Raises
        ------
        KeyError if dataset_id not found
        FileNotFoundError if parquet missing
        ValueError if hash mismatch (tampered)
        """
        rec = self.get(dataset_id)
        if rec is None:
            raise KeyError(f"dataset_id '{dataset_id}' not found in registry")
        p = Path(rec.path)
        if not p.exists():
            raise FileNotFoundError(f"Dataset file missing: {rec.path}")
        actual = self.hash_file(p)
        if actual != rec.hash:
            raise ValueError(
                f"Dataset {dataset_id} integrity check failed: hash mismatch (stored {rec.hash} != actual {actual}) — "
                f"file has been mutated in-place. Immutable dataset violation. Path: {rec.path}"
            )
        return True

    def verify_integrity(self, dataset_id: str) -> bool:
        """Alias for verify() for backward compatibility."""
        return self.verify(dataset_id)

    def load(self, dataset_id: str) -> pd.DataFrame:
        """
        Load dataset DataFrame with immutability verification.

        Steps:
        1. Verify hash matches stored hash (raises on tamper).
        2. Read parquet and return copy (no in-place mutation of stored file).
        3. Optionally re-validate gates on load (strict immutability).

        Raises ValueError on hash mismatch.
        """
        self.verify(dataset_id)
        rec = self.get(dataset_id)
        assert rec is not None  # verified above
        p = Path(rec.path)
        df = pd.read_parquet(p)
        # Return copy to prevent in-place mutation affecting cached reference
        # (File remains read-only, but DataFrame copy ensures no accidental mutation of shared object)
        return df.copy()

    def load_verified(self, dataset_id: str) -> pd.DataFrame:
        """Alias for load() — verified load."""
        return self.load(dataset_id)

    def load_dataframe(self, dataset_id: str) -> pd.DataFrame:
        """Alias for load() for API completeness."""
        return self.load(dataset_id)

    # ------------------------------------------------------------------
    # Registration with gates + immutability
    # ------------------------------------------------------------------

    def register(self, parquet_path: str, dataset_id: str, symbol: str, timeframe: str, label_version: str = "triple_barrier_v1", feature_version: str = "v2") -> DatasetRecord:
        p = Path(parquet_path)
        if not p.exists():
            raise FileNotFoundError(parquet_path)

        # Ensure writable for reading (if previous registration made it read-only, temporarily make writable for validation?)
        # We need to read the file even if read-only, so no need to make writable. Read works on read-only.

        df = pd.read_parquet(p)

        # --- Automated data gates (Task 8) ---
        # Import lazily to avoid circular import at module load time
        try:
            from src.data.data_gates import DataGates  # type: ignore
        except ImportError:
            # Fallback to legacy data_quality if data_gates not available (should not happen)
            from src.data_quality import validate_ohlcv as _legacy_validate  # type: ignore

            rep = _legacy_validate(df)
            if not rep.passed:
                raise ValueError(f"Data gates failed (legacy validator): {rep.summary()}")
            gates = None  # type: ignore
        else:
            gates = DataGates()
            # Validate DataFrame — raises DataGateError on failure
            gates.validate(df, timeframe=timeframe)

        # Assume timestamp column; handle both column and index cases
        if len(df):
            if "timestamp" in df.columns:
                start = str(df.iloc[0].get("timestamp", df.index[0]))
                end = str(df.iloc[-1].get("timestamp", df.index[-1]))
            else:
                # fallback to index
                start = str(df.index[0])
                end = str(df.index[-1])
        else:
            start = ""
            end = ""

        # Compute hash BEFORE making read-only (hash of bytes as stored)
        file_hash = self.hash_file(p)

        # P2.8 Immutability: if dataset_id exists and hash differs -> require new id
        existing = self._index.get(dataset_id)
        if existing is not None:
            if existing.prepared_hash and existing.prepared_hash != file_hash:
                raise ValueError(
                    f"DatasetRegistry immutability violation: dataset_id '{dataset_id}' already registered with hash {existing.prepared_hash[:12]} "
                    f"but new file hash is {file_hash[:12]}. Data changed -> new dataset_id required (P2.8)."
                )
            if existing.hash and existing.hash != file_hash:
                raise ValueError(
                    f"DatasetRegistry immutability violation: dataset_id '{dataset_id}' hash mismatch {existing.hash[:12]} != {file_hash[:12]} -> new dataset_id required"
                )

        # Full reproducibility metadata — P2.7 exact 11 fields
        cols = list(df.columns)
        dtypes_map = {str(k): str(v) for k, v in df.dtypes.items()}
        # P2.7 schema_hash: hash of schema (columns + dtypes) for schema drift detection
        schema_str = "|".join(f"{c}:{dtypes_map[c]}" for c in sorted(cols))
        schema_hash = hashlib.sha256(schema_str.encode()).hexdigest()[:16]
        # raw_hash: try to find raw source (e.g., data/btcusdt_15m.parquet without _prepared)
        raw_hash_val = ""
        try:
            raw_candidate = p.parent / p.name.replace("_prepared", "").replace("_fixed", "")
            if raw_candidate != p and raw_candidate.exists():
                raw_hash_val = self.hash_file(raw_candidate)
        except Exception:
            raw_hash_val = ""

        rec = DatasetRecord(
            dataset_id=dataset_id,
            exchange="binance",  # P2.7
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            rows=len(df),  # P2.7
            row_count=len(df),
            schema_hash=schema_hash,  # P2.7
            raw_hash=raw_hash_val,  # P2.7
            prepared_hash=file_hash,  # P2.7
            feature_version=feature_version,  # P2.7
            label_version=label_version,  # P2.7
            hash=file_hash,
            path=str(p),
            columns=cols,
            dtypes=dtypes_map,
        )
        self._index[dataset_id] = rec
        (self.root / f"{dataset_id}.json").write_text(json.dumps(rec.to_dict(), indent=2), encoding="utf-8")

        # --- Make immutable: set read-only flag after successful registration ---
        try:
            self._make_readonly(p)
        except Exception:
            # Non-critical: hash already stored, verification will still catch mutation
            pass

        return rec

    def get(self, dataset_id: str) -> Optional[DatasetRecord]:
        return self._index.get(dataset_id)

    def list_all(self) -> List[DatasetRecord]:
        return list(self._index.values())

    def canonical_path(self, dataset_id: str) -> Optional[str]:
        r = self.get(dataset_id)
        return r.path if r else None
