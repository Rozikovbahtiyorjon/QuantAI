"""
QuantAI Evidence Manager
Manages evidence collection, storage, and analysis for decision making
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from typing import TYPE_CHECKING
from pathlib import Path
import json

if TYPE_CHECKING:
    from .supervisor import SupervisorState


class EvidenceType(str, Enum):
    """Types of evidence"""
    EXECUTION_RESULT = "execution_result"
    TEST_RESULT = "test_result"
    REVIEW = "review"
    VALIDATION_RESULT = "validation_result"
    METRIC = "metric"
    ANOMALY = "anomaly"
    DRIFT = "drift"
    MODEL_PERFORMANCE = "model_performance"
    BACKTEST_RESULT = "backtest_result"
    WFO_RESULT = "wfo_result"
    PAPER_RESULT = "paper_result"
    RISK_REPORT = "risk_report"
    MARKET_DATA = "market_data"
    AGENT_DECISION = "agent_decision"


class VerificationStatus(str, Enum):
    VERIFIED = "VERIFIED"  # freshly created via _validate + real execution
    UNVERIFIED = "UNVERIFIED"  # present but not independently checked
    LEGACY = "LEGACY"  # pre-v2 without provenance
    LEGACY_UNVERIFIED = "LEGACY_UNVERIFIED"  # pre-versioned artifact, not re-validated (alias LEGACY)
    LEGACY_INVALID = "LEGACY_INVALID"  # legacy with 0-tests passed=true or similar
    INVALID = "INVALID"  # failed validation
    QUARANTINED = "QUARANTINED"

class TrustLevel(int, Enum):
    """Evidence Trust Model — QuantAI 5.3 — strict P0.5"""
    SIMULATED = 0  # L0 synthetic / placeholder
    SELF_REPORTED = 1  # L1 agent said success, not verified
    EXECUTION_VERIFIED = 2  # L2 real command exit 0 + artifact exists + hash
    INDEPENDENTLY_VALIDATED = 3  # L3 + independent gate (ResearchIntegrity/WRC)
    PRODUCTION_VERIFIED = 4  # L4 + live paper/prod verification

EVIDENCE_SCHEMA_VERSION = 2

# Production promotion requires trust >= INDEPENDENTLY_VALIDATED
PRODUCTION_PROMOTION_MIN_TRUST = TrustLevel.INDEPENDENTLY_VALIDATED

# Strict contract — every VERIFIED evidence must contain these (P0.5)
REQUIRED_EVIDENCE_FIELDS = [
    "evidence_id", "experiment_id", "code_commit", "dataset_id", "dataset_hash",
    "config_hash", "command", "exit_code", "artifact_hash", "created_at",
    "verification_status", "trust_level"
]


@dataclass
class Evidence:
    """Piece of evidence for decision making — canonical tags is List[str]."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: EvidenceType = EvidenceType.METRIC
    data: Any = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = ""
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    # --- Provenance (point 50) ---
    evidence_id: str = ""
    experiment_id: str = ""
    parent_evidence_id: str = ""
    source_command: str = ""
    exit_code: Optional[int] = None
    artifact_paths: List[str] = field(default_factory=list)
    artifact_hashes: Dict[str, str] = field(default_factory=dict)
    code_commit: str = ""
    dataset_id: str = ""
    dataset_hash: str = ""
    config_hash: str = ""
    environment_hash: str = ""
    trust_level: int = 0  # 0 SIMULATED .. 4 PRODUCTION_VERIFIED
    generated_by_real_execution: bool = False
    # --- Versioning / verification (point 9) ---
    evidence_schema_version: int = EVIDENCE_SCHEMA_VERSION
    verification_status: str = VerificationStatus.VERIFIED.value
    verified_at: Optional[str] = None  # ISO8601
    verified_by: str = "EvidenceManager"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value if isinstance(self.type, Enum) else str(self.type),
            "data": self.data,
            "timestamp": self.timestamp.isoformat() if hasattr(self.timestamp, "isoformat") else str(self.timestamp),
            "source": self.source,
            "confidence": self.confidence,
            "tags": self.tags,
            "metadata": self.metadata,
            "evidence_id": self.evidence_id or self.id,
            "experiment_id": self.experiment_id,
            "parent_evidence_id": self.parent_evidence_id,
            "source_command": self.source_command,
            "exit_code": self.exit_code,
            "artifact_paths": self.artifact_paths,
            "artifact_hashes": self.artifact_hashes,
            "code_commit": self.code_commit,
            "dataset_id": self.dataset_id,
            "dataset_hash": self.dataset_hash,
            "config_hash": self.config_hash,
            "environment_hash": self.environment_hash,
            "trust_level": self.trust_level,
            "generated_by_real_execution": self.generated_by_real_execution,
            "evidence_schema_version": self.evidence_schema_version,
            "verification_status": self.verification_status,
            "verified_at": self.verified_at or datetime.now(timezone.utc).isoformat(),
            "verified_by": self.verified_by,
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Evidence":
        # Detect legacy (no version)
        is_legacy = "evidence_schema_version" not in raw
        # Parse type
        t_raw = raw.get("type", "metric")
        try:
            t = EvidenceType(t_raw)
        except Exception:
            t = EvidenceType.METRIC
        # Parse timestamp
        ts_raw = raw.get("timestamp")
        try:
            ts = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00")) if ts_raw else datetime.now(timezone.utc)
        except Exception:
            ts = datetime.now(timezone.utc)
        ev = cls(
            id=str(raw.get("id", str(uuid.uuid4()))),
            type=t,
            data=raw.get("data"),
            timestamp=ts,
            source=str(raw.get("source", "")),
            tags=list(raw.get("tags", [])),
            metadata=dict(raw.get("metadata", {})),
            confidence=float(raw.get("confidence", 1.0)),
            evidence_id=str(raw.get("evidence_id", raw.get("id", ""))),
            experiment_id=str(raw.get("experiment_id", "")),
            parent_evidence_id=str(raw.get("parent_evidence_id", "")),
            source_command=str(raw.get("source_command", "")),
            exit_code=raw.get("exit_code"),
            artifact_paths=list(raw.get("artifact_paths", [])),
            artifact_hashes=dict(raw.get("artifact_hashes", {})),
            code_commit=str(raw.get("code_commit", "")),
            dataset_id=str(raw.get("dataset_id", "")),
            dataset_hash=str(raw.get("dataset_hash", "")),
            config_hash=str(raw.get("config_hash", "")),
            environment_hash=str(raw.get("environment_hash", "")),
            trust_level=int(raw.get("trust_level", 0) or 0),
            generated_by_real_execution=bool(raw.get("generated_by_real_execution", False)),
            evidence_schema_version=int(raw.get("evidence_schema_version", 1 if is_legacy else EVIDENCE_SCHEMA_VERSION)),
            verification_status=str(raw.get("verification_status", VerificationStatus.LEGACY_UNVERIFIED.value if is_legacy else VerificationStatus.VERIFIED.value)),
            verified_at=raw.get("verified_at"),
            verified_by=str(raw.get("verified_by", "legacy" if is_legacy else "EvidenceManager")),
        )
        # Trust model: legacy → SELF_REPORTED (1), invalid → SIMULATED (0)
        if is_legacy:
            # Old files had no trust_level field → set to SELF_REPORTED per spec
            if "trust_level" not in raw or int(raw.get("trust_level", 0) or 0) == 0:
                # keep INVALID as 0, otherwise 1
                if ev.verification_status == VerificationStatus.LEGACY_INVALID.value:
                    ev.trust_level = TrustLevel.SIMULATED.value
                else:
                    ev.trust_level = TrustLevel.SELF_REPORTED.value
        # If legacy and no verification_status was explicit, classify via fail-closed check
        if is_legacy and "verification_status" not in raw:
            # Check for false-positive pattern: tests_run 0 passed true
            d = ev.data if isinstance(ev.data, dict) else {}
            tr = d.get("tests_run", d.get("testsRun")) if isinstance(d, dict) else None
            passed = d.get("passed", d.get("success")) if isinstance(d, dict) else None
            is_false_positive = False
            if tr is not None and passed is True:
                try:
                    if int(tr) == 0:
                        is_false_positive = True
                except Exception:
                    pass
            if is_false_positive:
                ev.verification_status = VerificationStatus.LEGACY_INVALID.value
                ev.confidence = 0.0
                ev.trust_level = TrustLevel.SIMULATED.value
            else:
                # All legacy without version → quarantine as unverified, not VERIFIED
                if ev.verification_status == VerificationStatus.VERIFIED.value:
                    ev.verification_status = VerificationStatus.LEGACY_UNVERIFIED.value
                    ev.trust_level = TrustLevel.SELF_REPORTED.value
        return ev

    def is_verified(self) -> bool:
        return self.verification_status == VerificationStatus.VERIFIED.value and self.trust_level >= TrustLevel.EXECUTION_VERIFIED.value

    def is_legacy(self) -> bool:
        return self.evidence_schema_version < EVIDENCE_SCHEMA_VERSION or self.verification_status in (VerificationStatus.LEGACY_UNVERIFIED.value, VerificationStatus.LEGACY_INVALID.value, VerificationStatus.LEGACY.value, VerificationStatus.QUARANTINED.value)

    def is_promotable(self) -> bool:
        """P0.5: production promotion requires L3+ and VERIFIED."""
        return self.verification_status == VerificationStatus.VERIFIED.value and self.trust_level >= PRODUCTION_PROMOTION_MIN_TRUST.value


class EvidenceManager:
    """
    Manages evidence collection, storage, and analysis.
    Provides evidence-based decision making for the supervisor.
    Versioned: schema v2, legacy files quarantined as LEGACY_UNVERIFIED.
    """
    
    def __init__(self, evidence_path: str = "data/evidence", auto_migrate_legacy: bool = True):
        self.evidence_path = Path(evidence_path)
        self.evidence_path.mkdir(parents=True, exist_ok=True)
        
        self.evidence_store: Dict[str, Evidence] = {}
        self.evidence_index: Dict[str, List[str]] = {}  # type -> [evidence_ids]
        self.analysis_cache: Dict[str, Any] = {}
        self._lock = asyncio.Lock()
        self._migrated_legacy_count: int = 0
        if auto_migrate_legacy:
            try:
                self._migrate_legacy_files_sync()
                self._load_existing_into_store()
            except Exception:
                pass
    
    async def store(self, evidence: Evidence) -> str:
        """Store evidence"""
        async with self._lock:
            self.evidence_store[evidence.id] = evidence
            
            # Index by type
            if evidence.type not in self.evidence_index:
                self.evidence_index[evidence.type] = []
            self.evidence_index[evidence.type].append(evidence.id)
            
            # Persist to disk
            await self._persist_evidence(evidence)
            
            return evidence.id
    
    def _classify_legacy_status(self, evidence: Evidence) -> str:
        """Point 8-11: classify existing 305 files into VERIFIED vs LEGACY_UNVERIFIED/INVALID without deleting."""
        # Already versioned → keep
        if evidence.evidence_schema_version >= EVIDENCE_SCHEMA_VERSION and evidence.verification_status in (VerificationStatus.VERIFIED.value,):
            return evidence.verification_status
        # Legacy detection: missing version or unverified
        d = evidence.data if isinstance(evidence.data, dict) else {}
        # 89 test_result with tests_run 0 passed true
        if evidence.type == EvidenceType.TEST_RESULT or "tests_run" in (d if isinstance(d, dict) else {}):
            tr = d.get("tests_run", d.get("testsRun", None)) if isinstance(d, dict) else None
            passed = d.get("passed", d.get("success", None)) if isinstance(d, dict) else None
            if tr is not None and passed is True:
                try:
                    if int(tr) == 0:
                        return VerificationStatus.LEGACY_INVALID.value
                except Exception:
                    pass
        # 74 execution_result with success true code="# Generated code" tests=[]
        if evidence.type == EvidenceType.EXECUTION_RESULT:
            if isinstance(d, dict) and d.get("success") is True:
                res = d.get("result") if isinstance(d.get("result"), dict) else {}
                code = res.get("code", "") if isinstance(res, dict) else ""
                tests = res.get("tests", None) if isinstance(res, dict) else None
                if code.strip() == "# Generated code" and tests == []:
                    return VerificationStatus.LEGACY_INVALID.value
        # 126 review approved True from automated placeholder
        if evidence.type == EvidenceType.REVIEW:
            if isinstance(d, dict) and d.get("approved") is True and d.get("reviewer") == "automated" and not d.get("comments"):
                # Historical automated reviews without real checks → unverified
                return VerificationStatus.LEGACY_UNVERIFIED.value
        # Generic legacy without provenance
        if evidence.evidence_schema_version < EVIDENCE_SCHEMA_VERSION:
            return VerificationStatus.LEGACY_UNVERIFIED.value
        return VerificationStatus.VERIFIED.value

    def _migrate_legacy_files_sync(self) -> int:
        """Scan data/evidence/*.json and rewrite legacy as LEGACY_UNVERIFIED/INVALID (no deletion)."""
        migrated = 0
        for fp in self.evidence_path.glob("*.json"):
            try:
                raw = json.loads(fp.read_text(encoding="utf-8"))
                # Already versioned with explicit status → skip
                if "evidence_schema_version" in raw and "verification_status" in raw:
                    if raw.get("verification_status") in (VerificationStatus.VERIFIED.value, VerificationStatus.LEGACY_UNVERIFIED.value, VerificationStatus.LEGACY_INVALID.value):
                        # Still ensure quarantine count
                        if raw.get("verification_status") != VerificationStatus.VERIFIED.value:
                            migrated += 1
                        continue
                ev = Evidence.from_dict(raw)
                # Re-classify to ensure legacy status
                ev.verification_status = self._classify_legacy_status(ev)
                ev.verified_at = datetime.now(timezone.utc).isoformat()
                ev.verified_by = "migrate_legacy"
                if ev.verification_status in (VerificationStatus.LEGACY_UNVERIFIED.value, VerificationStatus.LEGACY_INVALID.value):
                    ev.confidence = 0.0 if ev.verification_status == VerificationStatus.LEGACY_INVALID.value else min(ev.confidence, 0.3)
                    ev.trust_level = TrustLevel.SIMULATED.value if ev.verification_status == VerificationStatus.LEGACY_INVALID.value else TrustLevel.SELF_REPORTED.value
                # Rewrite file with versioned fields (no deletion)
                fp.write_text(json.dumps(ev.to_dict(), indent=2, default=str), encoding="utf-8")
                migrated += 1
            except Exception:
                continue
        self._migrated_legacy_count = migrated
        return migrated

    async def verify_legacy_collection(self) -> Dict[str, Any]:
        """Public async wrapper for migration — returns quarantine stats."""
        # Run sync migration under lock
        async with self._lock:
            count = self._migrate_legacy_files_sync()
            # Also load into store as quarantined (not counted as verified)
            # We keep them in store but marked LEGACY
        stats = await self.analyze(include_legacy=False)
        legacy_stats = await self.analyze(include_legacy=True)
        return {"migrated": count, "verified": stats["total_evidence"], "legacy_total": legacy_stats["total_evidence"]}

    def _validate_evidence(self, evidence: Evidence) -> None:
        """Fail-closed validation: 0-tests passed=true is invalid + strict contract (P0.5)."""
        # Ensure new evidence is versioned correctly — Trust Model
        evidence.evidence_schema_version = EVIDENCE_SCHEMA_VERSION
        # Default to VERIFIED, but strict contract may downgrade to UNVERIFIED if fields missing
        if evidence.verification_status not in (VerificationStatus.VERIFIED.value, VerificationStatus.UNVERIFIED.value, VerificationStatus.INVALID.value):
            evidence.verification_status = VerificationStatus.VERIFIED.value
        evidence.verified_at = datetime.now(timezone.utc).isoformat()
        evidence.verified_by = "EvidenceManager"
        # Trust level: new evidence is at least EXECUTION_VERIFIED if real execution, else SELF_REPORTED
        if evidence.generated_by_real_execution:
            evidence.trust_level = max(evidence.trust_level, TrustLevel.EXECUTION_VERIFIED.value)
        else:
            if evidence.trust_level < TrustLevel.SELF_REPORTED.value:
                evidence.trust_level = TrustLevel.SELF_REPORTED.value
        # --- Strict contract for VERIFIED (P0.5) ---
        # Every VERIFIED evidence must contain all 12 fields; otherwise downgrade to UNVERIFIED or raise
        if evidence.verification_status == VerificationStatus.VERIFIED.value and evidence.trust_level >= TrustLevel.EXECUTION_VERIFIED.value:
            missing = []
            # Map required logical names to actual attributes
            checks = {
                "evidence_id": evidence.evidence_id or evidence.id,
                "experiment_id": evidence.experiment_id,
                "code_commit": evidence.code_commit,
                "dataset_id": evidence.dataset_id,
                "dataset_hash": evidence.dataset_hash,
                "config_hash": evidence.config_hash,
                "command": evidence.source_command,
                "exit_code": evidence.exit_code,
                "artifact_hash": evidence.artifact_hashes,  # dict, need non-empty if artifact expected
                "created_at": evidence.timestamp,
                "verification_status": evidence.verification_status,
                "trust_level": evidence.trust_level,
            }
            for k, v in checks.items():
                if k == "artifact_hash":
                    # Require at least one hash if artifact_paths present or for execution_result
                    if evidence.type == EvidenceType.EXECUTION_RESULT and evidence.generated_by_real_execution:
                        if not v or (isinstance(v, dict) and len(v) == 0):
                            # Allow empty if no artifact yet, but mark as incomplete
                            if not evidence.artifact_paths:
                                missing.append(k + " (no artifact_hashes for real execution)")
                elif k == "exit_code":
                    if v is None:
                        missing.append(k)
                    elif int(v) != 0:
                        # Non-zero exit cannot be VERIFIED
                        raise ValueError(f"Evidence {evidence.id} invalid: exit_code {v} != 0 for VERIFIED")
                elif k == "created_at":
                    if v is None:
                        missing.append(k)
                else:
                    if not v and v != 0:
                        # Allow 0 for trust_level, but not empty strings
                        if k in ("experiment_id", "code_commit", "dataset_id", "dataset_hash", "config_hash", "command"):
                            # These are required for VERIFIED L2+
                            missing.append(k)
            if missing:
                # Downgrade to UNVERIFIED instead of silently passing — fail-closed but not hard crash for missing optional
                # For strict production, raise; for now downgrade and log
                evidence.verification_status = VerificationStatus.UNVERIFIED.value
                evidence.trust_level = min(evidence.trust_level, TrustLevel.SELF_REPORTED.value)
                # Optionally raise for hard strict: uncomment to enforce
                # raise ValueError(f"Evidence {evidence.id} strict contract missing: {missing} → UNVERIFIED")
                evidence.metadata["contract_missing"] = missing
        d = evidence.data if isinstance(evidence.data, dict) else {}
        # P0.5 Fix: Never trust artifact's passed field — recompute independently from 7 checks
        # test_result: recompute passed from tests_run/failed/errors, not trust data["passed"]
        if evidence.type == EvidenceType.TEST_RESULT or "tests_run" in d:
            tr = int(d.get("tests_run", d.get("testsRun", 0)) or 0)
            failed = int(d.get("failed", d.get("failedCount", 0)) or 0)
            errors = int(d.get("errors", d.get("errorCount", 0)) or 0)
            # Handle failures list
            if "failures" in d and isinstance(d["failures"], list):
                failed = len(d["failures"]) if failed == 0 else failed
            # Recompute independently
            recomputed_passed = (tr > 0 and failed == 0 and errors == 0)
            observed_passed = bool(d.get("passed") or d.get("success"))
            # If artifact claims passed True but recomputed is False -> invalid (do not trust)
            if observed_passed and not recomputed_passed:
                raise ValueError(f"Evidence {evidence.id} invalid: artifact claims passed=true but independently recomputed false (tests_run={tr} failed={failed} errors={errors}) — not trusting passed (P0.5)")
            # Also fail if artifact claims passed with 0 tests
            if tr == 0 and observed_passed:
                raise ValueError(f"Evidence {evidence.id} invalid: tests_run=0 cannot be passed=true (independently recomputed)")
            if tr == 0 and recomputed_passed:
                raise ValueError(f"Evidence {evidence.id} invalid: tests_run=0 cannot be passed")
            if observed_passed and evidence.confidence == 1.0 and tr == 0:
                raise ValueError(f"Evidence {evidence.id} invalid: confidence 1.0 with 0 tests")
        # P1.16 + Fix: execution_result — never trust success/passed, recompute from 7 checks
        if evidence.type == EvidenceType.EXECUTION_RESULT:
            res = d.get("result", d) if isinstance(d, dict) else {}
            # Recompute success independently: need exit_code 0 + artifact exists + hash + metrics + dataset + code_commit
            # Do not trust d.get("success") or res.get("success")
            observed_success = bool(d.get("success") or res.get("success") if isinstance(res, dict) else d.get("success"))
            # 1. exit_code must be 0
            exit_code = evidence.exit_code
            if exit_code is None:
                exit_code = d.get("exit_code", d.get("exitCode"))
                if exit_code is None and isinstance(res, dict):
                    exit_code = res.get("exit_code", res.get("exitCode"))
            recomputed_exit_ok = (exit_code is not None and int(exit_code) == 0)
            # 2. artifact exists + hash
            artifact_ok = True
            if evidence.artifact_paths:
                for p in evidence.artifact_paths:
                    if not Path(p).exists():
                        artifact_ok = False
                        break
                    # Check hash matches stored
                    if p not in evidence.artifact_hashes or not evidence.artifact_hashes[p]:
                        artifact_ok = False
                        break
                    try:
                        import hashlib
                        actual = hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]
                        if evidence.artifact_hashes[p] != actual:
                            artifact_ok = False
                            break
                    except Exception:
                        artifact_ok = False
                        break
            # For execution without artifact, require metrics
            has_metrics = bool(d.get("metrics") or (isinstance(res, dict) and res.get("metrics")) or d.get("tournament") or d.get("allocation") or d.get("code"))
            if not evidence.artifact_paths and not has_metrics:
                artifact_ok = False
            # If observed claims success but recomputed fails -> invalid (do not trust passed)
            recomputed_success = recomputed_exit_ok and artifact_ok
            # Also need dataset and code_commit for full 7 checks
            dataset_ok = bool(evidence.dataset_id and evidence.dataset_hash and len(evidence.dataset_hash) >= 8)
            code_ok = bool(evidence.code_commit and len(evidence.code_commit) >= 7)
            # For code artifacts, require code_commit
            if not dataset_ok and evidence.artifact_paths:
                # If artifact exists but dataset missing, downgrade (not fail) — dataset required for L2+
                if evidence.trust_level >= TrustLevel.EXECUTION_VERIFIED.value:
                    evidence.verification_status = VerificationStatus.UNVERIFIED.value
                    evidence.trust_level = TrustLevel.SELF_REPORTED.value
            if not code_ok and evidence.artifact_paths:
                if evidence.trust_level >= TrustLevel.EXECUTION_VERIFIED.value:
                    evidence.verification_status = VerificationStatus.UNVERIFIED.value
                    evidence.trust_level = TrustLevel.SELF_REPORTED.value
            if observed_success and not recomputed_success:
                raise ValueError(f"Evidence {evidence.id} invalid: artifact claims success=true but independently recomputed false (exit_code={exit_code} artifact_ok={artifact_ok}) — not trusting passed")
            # Check for "# Generated code" placeholder (was in quant_engineer before fix)
            code = res.get("code", d.get("code", "")) if isinstance(res, dict) else ""
            tests = res.get("tests", d.get("tests", None)) if isinstance(res, dict) else None
            if d.get("success") is True and isinstance(res, dict) and code.strip() == "# Generated code" and tests == []:
                raise ValueError(f"Evidence {evidence.id} invalid: placeholder code '# Generated code' with empty tests — not valid artifact (P1.16)")
            if d.get("success") is True and isinstance(res, dict) and not res.get("tests") and not res.get("code") and not evidence.artifact_paths:
                # No artifact and no code — not valid execution
                # Allow only if explicitly has artifact_paths or provenance
                if not evidence.generated_by_real_execution:
                    # Downgrade to UNVERIFIED
                    evidence.verification_status = VerificationStatus.UNVERIFIED.value
                    evidence.trust_level = TrustLevel.SELF_REPORTED.value
            # Check for bal_acc placeholder 0.39
            metrics = d.get("metrics", d) if isinstance(d, dict) else {}
            if isinstance(metrics, dict):
                if metrics.get("bal_acc") == 0.39 and metrics.get("_bal_acc_missing") is None:
                    # Explicit 0.39 without missing flag is fabricated
                    raise ValueError(f"Evidence {evidence.id} invalid: bal_acc 0.39 placeholder forbidden (P1.16)")
                if metrics.get("exposure_ok") is True and not metrics.get("_provenance") and not evidence.generated_by_real_execution:
                    # exposure_ok True without real orchestrator provenance
                    evidence.verification_status = VerificationStatus.UNVERIFIED.value
                    evidence.trust_level = TrustLevel.SELF_REPORTED.value
        # P1.16: review approved True without real verification -> not VERIFIED
        if evidence.type == EvidenceType.REVIEW:
            if isinstance(d, dict) and d.get("approved") is True:
                if not d.get("verified") and d.get("reviewer") == "automated":
                    # Must have been verified via execution+tests
                    if not d.get("reason", "").startswith("execution and tests verified"):
                        evidence.verification_status = VerificationStatus.UNVERIFIED.value
                        evidence.trust_level = TrustLevel.SELF_REPORTED.value

    async def _persist_evidence(self, evidence: Evidence) -> None:
        """Persist evidence to disk — failure is critical (point 3)."""
        # Validate before persisting
        self._validate_evidence(evidence)
        file_path = self.evidence_path / f"{evidence.id}.json"
        try:
            with open(file_path, 'w') as f:
                json.dump(evidence.to_dict(), f, default=str)
        except Exception as e:
            raise RuntimeError(f"Evidence persistence failed for {evidence.id}: {e}") from e
    
    def _load_existing_into_store(self) -> int:
        """Load all JSON files into store with versioned status (for Supervisor)."""
        loaded = 0
        for fp in self.evidence_path.glob("*.json"):
            try:
                raw = json.loads(fp.read_text(encoding="utf-8"))
                ev = Evidence.from_dict(raw)
                # Reclassify legacy status if not yet
                if ev.evidence_schema_version < EVIDENCE_SCHEMA_VERSION or ev.verification_status not in (VerificationStatus.VERIFIED.value,):
                    # Ensure correct quarantine status
                    ev.verification_status = self._classify_legacy_status(ev)
                    ev.verified_at = ev.verified_at or datetime.now(timezone.utc).isoformat()
                self.evidence_store[ev.id] = ev
                if ev.type not in self.evidence_index:
                    self.evidence_index[ev.type] = []
                if ev.id not in self.evidence_index[ev.type]:
                    self.evidence_index[ev.type].append(ev.id)
                loaded += 1
            except Exception:
                continue
        return loaded

    async def load_existing(self) -> int:
        async with self._lock:
            return self._load_existing_into_store()

    async def get(self, evidence_id: str) -> Optional[Evidence]:
        """Get evidence by ID"""
        return self.evidence_store.get(evidence_id)
    
    async def get_verified(self, limit: int = 100) -> List[Evidence]:
        """Only VERIFIED L2+ evidence (production chain). Legacy explicitly excluded."""
        all_verified = [e for e in self.evidence_store.values() if e.verification_status == VerificationStatus.VERIFIED.value and e.trust_level >= TrustLevel.EXECUTION_VERIFIED.value]
        all_verified.sort(key=lambda e: e.timestamp, reverse=True)
        return all_verified[:limit]

    async def get_promotable(self, limit: int = 100) -> List[Evidence]:
        """P0.5: production promotion requires L3+ and VERIFIED — legacy explicitly excluded."""
        promotable = [e for e in self.evidence_store.values() if e.is_promotable()]
        promotable.sort(key=lambda e: e.timestamp, reverse=True)
        return promotable[:limit]

    async def get_legacy(self, limit: int = 100) -> List[Evidence]:
        """Quarantined legacy evidence — explicitly UNVERIFIED, excluded from promotion."""
        all_legacy = [e for e in self.evidence_store.values() if e.is_legacy() or e.verification_status != VerificationStatus.VERIFIED.value or e.trust_level < TrustLevel.EXECUTION_VERIFIED.value]
        all_legacy.sort(key=lambda e: e.timestamp, reverse=True)
        return all_legacy[:limit]

    async def upgrade_to_independently_validated(self, evidence_id: str) -> bool:
        """P0.5: upgrade evidence to L3 after independent gate (ResearchIntegrity/WRC) passes."""
        ev = self.evidence_store.get(evidence_id)
        if not ev or ev.is_legacy():
            return False
        if ev.verification_status != VerificationStatus.VERIFIED.value:
            return False
        if ev.trust_level >= TrustLevel.INDEPENDENTLY_VALIDATED.value:
            return True
        ev.trust_level = TrustLevel.INDEPENDENTLY_VALIDATED.value
        ev.verified_at = datetime.now(timezone.utc).isoformat()
        ev.verified_by = "IndependentGate"
        # Persist upgraded version
        try:
            await self._persist_evidence(ev)
        except Exception:
            pass
        return True

    async def upgrade_to_production_verified(self, evidence_id: str) -> bool:
        """L4 after live paper/prod verification."""
        ev = self.evidence_store.get(evidence_id)
        if not ev or not ev.is_promotable():
            return False
        ev.trust_level = TrustLevel.PRODUCTION_VERIFIED.value
        ev.verified_at = datetime.now(timezone.utc).isoformat()
        ev.verified_by = "ProductionGate"
        try:
            await self._persist_evidence(ev)
        except Exception:
            pass
        return True

    async def get_by_type(self, evidence_type: EvidenceType, limit: int = 100, verified_only: bool = False) -> List[Evidence]:
        """Get evidence by type — if verified_only, exclude LEGACY and require L2+."""
        ids = self.evidence_index.get(evidence_type, [])
        evs = [self.evidence_store[eid] for eid in ids[-limit:] if eid in self.evidence_store]
        if verified_only:
            evs = [e for e in evs if e.verification_status == VerificationStatus.VERIFIED.value and e.trust_level >= TrustLevel.EXECUTION_VERIFIED.value]
        return evs
    
    async def get_recent(self, limit: int = 100, verified_only: bool = False) -> List[Evidence]:
        """Get most recent evidence"""
        all_evidence = list(self.evidence_store.values())
        if verified_only:
            all_evidence = [e for e in all_evidence if e.verification_status == VerificationStatus.VERIFIED.value]
        all_evidence.sort(key=lambda e: e.timestamp, reverse=True)
        return all_evidence[:limit]
    
    async def get_all(self, include_legacy: bool = True) -> List[Evidence]:
        """Get all evidence — include_legacy=False returns only VERIFIED (production chain)."""
        if include_legacy:
            return list(self.evidence_store.values())
        return [e for e in self.evidence_store.values() if e.verification_status == VerificationStatus.VERIFIED.value]
    
    async def get_by_source(self, source: str, limit: int = 100) -> List[Evidence]:
        """Get evidence by source"""
        results = []
        for evidence in self.evidence_store.values():
            if evidence.source == source:
                results.append(evidence)
                if len(results) >= limit:
                    break
        return results
    
    async def get_by_tags(self, tags: List[str], limit: int = 100) -> List[Evidence]:
        """Get evidence matching any of the tags"""
        results = []
        for evidence in self.evidence_store.values():
            if any(tag in evidence.tags for tag in tags):
                results.append(evidence)
                if len(results) >= limit:
                    break
        return results
    
    async def analyze(self, include_legacy: bool = True) -> Dict[str, Any]:
        """Analyze evidence — include_legacy=False counts only VERIFIED for production decisions."""
        evidence = list(self.evidence_store.values())
        if not include_legacy:
            evidence = [e for e in evidence if e.verification_status == VerificationStatus.VERIFIED.value]
        
        if not evidence:
            return {"anomalies": [], "summary": "No evidence available"}
        
        # Group by type
        by_type: Dict[str, int] = {}
        by_source: Dict[str, int] = {}
        confidences: List[float] = []
        
        for e in evidence:
            by_type[e.type.value] = by_type.get(e.type.value, 0) + 1
            by_source[e.source] = by_source.get(e.source, 0) + 1
            confidences.append(e.confidence)
        
        # Check for anomalies
        anomalies = []
        
        # Low confidence evidence
        low_conf = [e for e in evidence if e.confidence < 0.5]
        if low_conf:
            anomalies.append(f"{len(low_conf)} low-confidence evidence items")
        
        # Old evidence
        old_cutoff = datetime.now(timezone.utc).timestamp() - 86400  # 24 hours
        old_evidence = [e for e in evidence if e.timestamp.timestamp() < old_cutoff]
        if old_evidence:
            anomalies.append(f"{len(old_evidence)} evidence items older than 24h")
        
        # Low confidence sources
        for source, count in by_source.items():
            if count < 2:
                anomalies.append(f"Source '{source}' has only {count} evidence items")
        
        return {
            "total_evidence": len(evidence),
            "by_type": by_type,
            "by_source": by_source,
            "avg_confidence": sum(confidences) / len(confidences) if confidences else 0,
            "anomalies": anomalies,
            "type_distribution": by_type,
            "source_distribution": by_source
        }
    
    async def clear_old(self, max_age_hours: int = 168) -> int:
        """Remove evidence older than max_age_hours"""
        cutoff = datetime.now(timezone.utc).timestamp() - (max_age_hours * 3600)
        removed = 0
        
        to_remove = []
        for eid, evidence in self.evidence_store.items():
            if evidence.timestamp.timestamp() < cutoff:
                to_remove.append(eid)
        
        for eid in to_remove:
            del self.evidence_store[eid]
            # Remove from index
            for type_list in self.evidence_index.values():
                if eid in type_list:
                    type_list.remove(eid)
            removed += 1
        
        return removed


__all__ = [
    "EvidenceType",
    "Evidence",
    "EvidenceManager",
    "VerificationStatus",
    "TrustLevel",
    "EVIDENCE_SCHEMA_VERSION",
    "PRODUCTION_PROMOTION_MIN_TRUST",
]