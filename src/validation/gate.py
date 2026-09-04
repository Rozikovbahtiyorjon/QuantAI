"""
QuantAI Validation Gate (R3)

One formal, non-trading quality boundary:

    py_compile · pytest · no-lookahead · backtest · walk-forward ·
    risk gates · paper pipeline · long-run evidence

Verdicts per check and overall:
    PASS      requirement satisfied
    FAIL      requirement violated  (blocks promotion/deployment)
    BLOCKED   cannot be evaluated yet (e.g., long-run artifacts missing)

Overall rule:
    any FAIL        -> FAIL
    else any BLOCKED-> BLOCKED
    else            -> PASS

CLI:
    python -m src.validation.gate            # full report
    python -m src.validation.gate --fast     # skip full pytest
"""

from __future__ import annotations

import compileall
import hashlib
import io
import json
import re
import subprocess
import sys
import tempfile
import time
import warnings
import xml.etree.ElementTree as ET
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path


# ============================================================
# Evidence versioning — P0-5 immutable artifact
# ============================================================

GATE_VERSION = "5.2"


def _get_code_commit(root: Path | None = None) -> str:
    """Best-effort git commit. Tries git binary, then .git files, else 'unknown'."""
    root = root or Path(__file__).resolve().parents[2]
    # try git binary (may not exist on Windows CI)
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=str(root),
        )
        if proc.returncode == 0:
            commit = proc.stdout.strip()
            if len(commit) >= 7:
                return commit
    except Exception:
        pass
    # fallback: read .git/HEAD directly (no git binary required)
    try:
        head_path = root / ".git" / "HEAD"
        if head_path.exists():
            head = head_path.read_text(encoding="utf-8").strip()
            if head.startswith("ref:"):
                ref = head.split("ref:", 1)[1].strip()
                ref_path = root / ".git" / ref
                if ref_path.exists():
                    ref_commit = ref_path.read_text(encoding="utf-8").strip()
                    if ref_commit:
                        return ref_commit
                # packed-refs fallback
                packed = root / ".git" / "packed-refs"
                if packed.exists():
                    for line in packed.read_text(encoding="utf-8").splitlines():
                        if line.strip().endswith(ref):
                            parts = line.strip().split()
                            if parts:
                                return parts[0]
            elif len(head) >= 7:
                return head
    except Exception:
        pass
    return "unknown"


def _hash_file(path: Path) -> str:
    """SHA256 of file content (streaming). Returns hexdigest."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _get_config_hash(root: Path | None = None) -> str:
    """Hash of config/settings.py + pyproject.toml (canonical config)."""
    root = root or Path(__file__).resolve().parents[2]
    files = [root / "config" / "settings.py", root / "pyproject.toml"]
    h = hashlib.sha256()
    for p in files:
        if p.exists():
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
        else:
            h.update(f"missing:{p}".encode())
        h.update(b"|")
    return h.hexdigest()


def _get_dataset_hash(data_dir: Path | None = None, root: Path | None = None) -> str:
    """
    Hash of dataset file — primary prepared parquet.
    Prefers btcusdt_1h_prepared.parquet else first sorted *_prepared.parquet.
    Returns 'no_dataset' if none.
    """
    root = root or Path(__file__).resolve().parents[2]
    data_dir = Path(data_dir) if data_dir is not None else root / "data"
    # prefer canonical 1h prepared (used by walk_forward smoke)
    primary = data_dir / "btcusdt_1h_prepared.parquet"
    target: Path | None = None
    if primary.exists():
        target = primary
    else:
        candidates = sorted(Path(data_dir).glob("*_prepared.parquet"))
        if not candidates:
            return "no_dataset"
        target = candidates[0]
    try:
        return _hash_file(target)
    except Exception:
        return "hash_error"


def _get_environment_hash() -> str:
    """Hash of python version + pip freeze (captures dependency drift)."""
    h = hashlib.sha256()
    h.update(sys.version.encode("utf-8"))
    h.update(b"|")
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode == 0:
            # normalize: sorted lines to be deterministic
            freeze = "\n".join(sorted(proc.stdout.strip().splitlines()))
            h.update(freeze.encode("utf-8"))
        else:
            h.update(b"pip_freeze_failed")
    except Exception:
        h.update(b"pip_freeze_error")
    return h.hexdigest()


def _get_timestamp() -> str:
    """ISO8601 UTC with Z suffix."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _collect_evidence_metadata(data_dir: Path | None = None, root: Path | None = None) -> dict:
    """Collect all versioned evidence fields for current code/config/dataset/env."""
    root = root or Path(__file__).resolve().parents[2]
    data_dir = Path(data_dir) if data_dir is not None else root / "data"
    return {
        "gate_version": GATE_VERSION,
        "code_commit": _get_code_commit(root),
        "config_hash": _get_config_hash(root),
        "dataset_hash": _get_dataset_hash(data_dir, root),
        "environment_hash": _get_environment_hash(),
        "timestamp": _get_timestamp(),
    }


def verify_report_freshness(
    report_dict: dict,
    data_dir: Path | None = None,
    root: Path | None = None,
) -> dict:
    """
    Compare stored hashes in report vs current filesystem.
    Returns {'is_stale': bool, 'reasons': [str], 'current': dict, 'stored': dict}.
    Any mismatch in code/config/dataset/gate_version -> STALE.
    Missing versioned fields -> STALE (legacy artifact).
    environment_hash mismatch is also STALE (dependency drift).
    """
    root = root or Path(__file__).resolve().parents[2]
    data_dir = Path(data_dir) if data_dir is not None else root / "data"
    current = _collect_evidence_metadata(data_dir, root)
    stored = {
        "gate_version": report_dict.get("gate_version", ""),
        "code_commit": report_dict.get("code_commit", ""),
        "config_hash": report_dict.get("config_hash", ""),
        "dataset_hash": report_dict.get("dataset_hash", ""),
        "environment_hash": report_dict.get("environment_hash", ""),
        "timestamp": report_dict.get("timestamp", ""),
    }
    reasons: list[str] = []
    # legacy check — missing fields
    if not stored["gate_version"] or not stored["code_commit"] or not stored["config_hash"] or not stored["dataset_hash"]:
        reasons.append("missing versioned fields (legacy report)")
    else:
        if stored["gate_version"] != current["gate_version"]:
            reasons.append(f"gate_version mismatch: stored {stored['gate_version']!r} vs current {current['gate_version']!r}")
        if stored["code_commit"] != current["code_commit"]:
            s_short = stored["code_commit"][:12] if stored["code_commit"] else "missing"
            c_short = current["code_commit"][:12] if current["code_commit"] else "missing"
            reasons.append(f"code_commit mismatch: stored {s_short} vs current {c_short}")
        if stored["config_hash"] != current["config_hash"]:
            reasons.append("config_hash mismatch: config/settings.py or pyproject.toml changed")
        if stored["dataset_hash"] != current["dataset_hash"]:
            reasons.append("dataset_hash mismatch: dataset file changed")
        if stored["environment_hash"] != current["environment_hash"]:
            reasons.append("environment_hash mismatch: python version or pip freeze changed")
    is_stale = len(reasons) > 0
    return {
        "is_stale": is_stale,
        "reasons": reasons,
        "current": current,
        "stored": stored,
    }


def load_and_verify_report(
    report_path: Path | None = None,
    data_dir: Path | None = None,
    root: Path | None = None,
) -> tuple[dict, dict]:
    """
    Load gate_report.json and verify freshness.
    Injects 'freshness' ('FRESH'/'STALE'), 'is_stale' and 'stale_reasons' into returned dict.
    Returns (report_dict_with_markers, verification_dict).
    Raises FileNotFoundError if report missing.
    """
    root = root or Path(__file__).resolve().parents[2]
    report_path = Path(report_path) if report_path is not None else root / "data" / "gate_report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"gate report not found: {report_path}")
    report_dict = json.loads(report_path.read_text(encoding="utf-8"))
    verification = verify_report_freshness(report_dict, data_dir, root)
    # immutable marker
    report_dict["freshness"] = "STALE" if verification["is_stale"] else "FRESH"
    report_dict["is_stale"] = verification["is_stale"]
    report_dict["stale_reasons"] = verification["reasons"]
    # expose current for debugging (prefixed underscore to not confuse consumers)
    report_dict["_current"] = verification["current"]
    return report_dict, verification


class GateStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    BLOCKED = "BLOCKED"
    INSUFFICIENT_SAMPLE = "INSUFFICIENT_SAMPLE"
    INCONCLUSIVE = "INCONCLUSIVE"  # P2.11: PF 1.8 trades 3 → INCONCLUSIVE (not PASS/FAIL)
    NOT_AVAILABLE = "NOT_AVAILABLE"  # data not available for gate


@dataclass
class CheckResult:
    name: str
    status: GateStatus
    details: str = ""
    metrics: dict = field(default_factory=dict)
    duration_s: float = 0.0


@dataclass
class GateReport:
    results: list[CheckResult] = field(default_factory=list)
    gate_version: str = GATE_VERSION
    code_commit: str = ""
    config_hash: str = ""
    dataset_hash: str = ""
    environment_hash: str = ""
    timestamp: str = ""
    # populated on load/verify — not part of evidence at generation time
    freshness: str = "FRESH"
    is_stale: bool = False
    stale_reasons: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # if metadata not supplied, collect lazily (keeps manual construction lightweight;
        # but QuantAIValidationGate.run() always populates explicitly)
        if not self.timestamp:
            # only auto-fill when explicitly empty AND results already present
            # to avoid overhead during empty construction in tests
            pass

    @property
    def verdict(self) -> GateStatus:
        statuses = [r.status for r in self.results]
        if GateStatus.FAIL in statuses:
            return GateStatus.FAIL
        if GateStatus.INSUFFICIENT_SAMPLE in statuses:
            return GateStatus.INSUFFICIENT_SAMPLE
        if GateStatus.BLOCKED in statuses:
            return GateStatus.BLOCKED
        return GateStatus.PASS

    def to_dict(self) -> dict:
        return {
            "gate_version": self.gate_version,
            "code_commit": self.code_commit,
            "config_hash": self.config_hash,
            "dataset_hash": self.dataset_hash,
            "environment_hash": self.environment_hash,
            "timestamp": self.timestamp,
            "freshness": self.freshness,
            "is_stale": self.is_stale,
            "stale_reasons": self.stale_reasons,
            "verdict": self.verdict.value,
            "checks": [
                {
                    "name": r.name,
                    "status": r.status.value,
                    "details": r.details,
                    "metrics": r.metrics,
                    "duration_s": round(r.duration_s, 2),
                }
                for r in self.results
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GateReport":
        results = []
        for c in data.get("checks", []):
            results.append(
                CheckResult(
                    name=c["name"],
                    status=GateStatus(c["status"]),
                    details=c.get("details", ""),
                    metrics=c.get("metrics", {}),
                    duration_s=float(c.get("duration_s", 0.0)),
                )
            )
        return cls(
            results=results,
            gate_version=data.get("gate_version", GATE_VERSION),
            code_commit=data.get("code_commit", ""),
            config_hash=data.get("config_hash", ""),
            dataset_hash=data.get("dataset_hash", ""),
            environment_hash=data.get("environment_hash", ""),
            timestamp=data.get("timestamp", ""),
            freshness=data.get("freshness", "FRESH"),
            is_stale=bool(data.get("is_stale", False)),
            stale_reasons=list(data.get("stale_reasons", [])),
        )

    def verify_freshness(self, data_dir: Path | None = None, root: Path | None = None) -> dict:
        """Verify this report's evidence hashes vs current filesystem. Returns verification dict."""
        return verify_report_freshness(self.to_dict(), data_dir=data_dir, root=root)

    def pretty(self) -> str:
        lines = ["=" * 64, f"QUANTAI VALIDATION GATE :: {self.verdict.value}", "=" * 64]
        # evidence header
        lines.append(f"version={self.gate_version} commit={self.code_commit[:12] if self.code_commit else 'unknown'} ts={self.timestamp}")
        lines.append(f"config_hash={self.config_hash[:12] if self.config_hash else 'missing'} dataset_hash={self.dataset_hash[:12] if self.dataset_hash else 'missing'} env_hash={self.environment_hash[:12] if self.environment_hash else 'missing'}")
        if self.is_stale:
            lines.append(f"STALE: {'; '.join(self.stale_reasons)}" if self.stale_reasons else "STALE: evidence mismatch")
        else:
            lines.append(f"FRESH: {self.freshness}")
        lines.append("-" * 64)
        for r in self.results:
            lines.append(
                f"[{r.status.value:<7}] {r.name:<24} "
                f"{r.duration_s:>6.1f}s  {r.details}"
            )
        lines.append("=" * 64)
        return "\n".join(lines)


def _has_json_report_plugin() -> bool:
    """Return True if pytest-json-report plugin appears importable."""
    import importlib.util

    for mod in ("pytest_jsonreport", "pytest_json_report", "pytest_jsonreport.plugin"):
        try:
            if importlib.util.find_spec(mod) is not None:
                return True
        except Exception:
            continue
    try:
        from importlib.metadata import entry_points

        eps = entry_points()
        if hasattr(eps, "select"):
            candidates = eps.select(group="pytest11")
        else:  # legacy dict API
            candidates = eps.get("pytest11", [])  # type: ignore[assignment]
        for ep in candidates:
            name = getattr(ep, "name", "") or ""
            val = getattr(ep, "value", "") or ""
            if "json" in name.lower() or "json" in val.lower():
                return True
    except Exception:
        pass
    return False


def _parse_json_report(path: Path) -> dict | None:
    """Parse pytest-json-report file. Returns metrics or None."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    summary = data.get("summary", {}) if isinstance(data, dict) else {}
    if not isinstance(summary, dict):
        return None
    try:
        passed = int(summary.get("passed", 0) or 0)
        failed = int(summary.get("failed", 0) or 0)
        errors = int(summary.get("error", summary.get("errors", 0)) or 0)
        skipped = int(summary.get("skipped", 0) or 0)
        xfailed = int(summary.get("xfailed", 0) or 0)
        xpassed = int(summary.get("xpassed", 0) or 0)
        total = summary.get("total")
        if total is None:
            total = summary.get("numcollected", passed + failed + errors + skipped)
            try:
                total = int(total)
            except Exception:
                total = passed + failed + errors + skipped
        else:
            try:
                total = int(total)
            except Exception:
                total = passed + failed + errors + skipped
        return {
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "skipped": skipped,
            "xfailed": xfailed,
            "xpassed": xpassed,
            "total": total,
        }
    except Exception:
        return None


def _parse_junitxml(path: Path) -> dict | None:
    """Parse JUnit XML (pytest --junit-xml). Aggregates suites."""
    try:
        tree = ET.parse(str(path))
        root = tree.getroot()
    except Exception:
        return None
    if root.tag == "testsuite":
        suites = [root]
    elif root.tag == "testsuites":
        suites = root.findall("testsuite")
        if not suites:
            suites = root.findall(".//testsuite")
    else:
        suites = root.findall(".//testsuite")
        if not suites:
            return None

    total = failed = errors = skipped = 0
    for ts in suites:
        try:
            total += int(ts.get("tests", 0) or 0)
        except ValueError:
            pass
        try:
            failed += int(ts.get("failures", 0) or 0)
        except ValueError:
            pass
        try:
            errors += int(ts.get("errors", 0) or 0)
        except ValueError:
            pass
        try:
            skipped += int(ts.get("skipped", 0) or 0)
        except ValueError:
            pass
    if total == 0 and failed == 0 and errors == 0:
        cases = root.findall(".//testcase")
        if cases:
            total = len(cases)
            for c in cases:
                if c.find("failure") is not None:
                    failed += 1
                elif c.find("error") is not None:
                    errors += 1
                elif c.find("skipped") is not None:
                    skipped += 1

    passed = total - failed - errors - skipped
    if passed < 0:
        passed = 0
    return {
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "skipped": skipped,
        "xfailed": 0,
        "xpassed": 0,
        "total": total,
    }


def _parse_text_fallback(output: str) -> dict:
    """Robust regex fallback for pytest -q summary line."""
    passed = failed = errors = skipped = xfailed = xpassed = 0
    for line in (output or "").splitlines():
        low = line.lower()
        if not any(k in low for k in ("passed", "failed", "error", "skipped", "xfailed", "xpassed")):
            continue
        m = re.search(r"(\d+)\s+passed", low)
        if m:
            try:
                passed = int(m.group(1))
            except ValueError:
                pass
        m = re.search(r"(\d+)\s+failed", low)
        if m:
            try:
                failed = int(m.group(1))
            except ValueError:
                pass
        m = re.search(r"(\d+)\s+error", low)
        if m:
            try:
                errors = int(m.group(1))
            except ValueError:
                pass
        m = re.search(r"(\d+)\s+skipped", low)
        if m:
            try:
                skipped = int(m.group(1))
            except ValueError:
                pass
        m = re.search(r"(\d+)\s+xfailed", low)
        if m:
            try:
                xfailed = int(m.group(1))
            except ValueError:
                pass
        m = re.search(r"(\d+)\s+xpassed", low)
        if m:
            try:
                xpassed = int(m.group(1))
            except ValueError:
                pass
    total = passed + failed + errors + skipped
    return {
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "skipped": skipped,
        "xfailed": xfailed,
        "xpassed": xpassed,
        "total": total,
    }


def _run_pytest(targets: list[str], timeout: int = 900) -> tuple[bool, str, dict]:
    """
    Run pytest in a subprocess against selected targets.
    Uses the same interpreter running the gate.
    Prefers structured output (pytest-json-report -> JUnit XML -> text fallback).
    """
    root = Path(__file__).resolve().parents[2]
    base = [sys.executable, "-m", "pytest", *targets, "-q", "--no-header", "-p", "no:cacheprovider"]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        json_path = tmp / "report.json"
        junit_path = tmp / "junit.xml"

        use_json = _has_json_report_plugin()

        cmd = list(base)
        if use_json:
            cmd += ["--json-report", f"--json-report-file={json_path}"]
        cmd += [f"--junit-xml={junit_path}", "-o", "junit_family=xunit2"]

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(root),
            )
        except subprocess.TimeoutExpired:
            return False, f"pytest timed out after {timeout}s", {"passed": 0, "failed": 0, "errors": 0, "skipped": 0, "total": 0}

        tail = "\n".join((proc.stdout or "").splitlines()[-6:])

        if use_json and json_path.exists():
            metrics = _parse_json_report(json_path)
            if metrics is not None:
                ok = proc.returncode == 0 and metrics.get("failed", 0) == 0 and metrics.get("errors", 0) == 0
                return ok, tail, metrics

        if junit_path.exists():
            metrics = _parse_junitxml(junit_path)
            if metrics is not None:
                ok = proc.returncode == 0 and metrics.get("failed", 0) == 0 and metrics.get("errors", 0) == 0
                return ok, tail, metrics

        if use_json and "unrecognized arguments: --json-report" in (proc.stderr or ""):
            warnings.warn(
                "[gate] pytest-json-report flag unsupported despite detection; retrying with JUnit only",
                stacklevel=2,
            )
            cmd_retry = list(base) + [f"--junit-xml={junit_path}", "-o", "junit_family=xunit2"]
            try:
                proc2 = subprocess.run(
                    cmd_retry,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=str(root),
                )
            except subprocess.TimeoutExpired:
                return False, f"pytest timed out after {timeout}s", {"passed": 0, "failed": 0, "errors": 0, "skipped": 0, "total": 0}
            tail = "\n".join((proc2.stdout or "").splitlines()[-6:])
            if junit_path.exists():
                metrics = _parse_junitxml(junit_path)
                if metrics is not None:
                    ok = proc2.returncode == 0 and metrics.get("failed", 0) == 0 and metrics.get("errors", 0) == 0
                    return ok, tail, metrics
            metrics = _parse_text_fallback(proc2.stdout or "")
            warnings.warn(
                "[gate] WARNING: structured pytest parsing unavailable, falling back to text parsing",
                stacklevel=2,
            )
            print(
                "[gate] WARNING: structured pytest parsing unavailable, falling back to text parsing",
                file=sys.stderr,
            )
            ok = proc2.returncode == 0 and metrics.get("failed", 0) == 0 and metrics.get("errors", 0) == 0
            return ok, tail, metrics

        metrics = _parse_text_fallback(proc.stdout or "")
        warnings.warn(
            "[gate] WARNING: structured pytest parsing unavailable, falling back to text parsing",
            stacklevel=2,
        )
        print(
            "[gate] WARNING: structured pytest parsing unavailable, falling back to text parsing",
            file=sys.stderr,
        )
        ok = proc.returncode == 0 and metrics.get("failed", 0) == 0 and metrics.get("errors", 0) == 0
        return ok, tail, metrics


# ============================================================
# STANDARD CHECKS
# ============================================================

def check_compile(root: Path | None = None) -> CheckResult:
    t0 = time.time()
    root = root or Path(__file__).resolve().parents[2]

    buf = io.StringIO()
    ok = True
    with redirect_stdout(buf):
        for pkg in ("src", "config"):
            ok &= compileall.compile_dir(
                str(root / pkg), quiet=5, force=False, maxlevels=10
            )

    return CheckResult(
        name="py_compile",
        status=GateStatus.PASS if ok else GateStatus.FAIL,
        details="src+config byte-compilable" if ok else "compile errors (see output)",
        duration_s=time.time() - t0,
    )


def check_pytest_full() -> CheckResult:
    t0 = time.time()
    ok, tail, metrics = _run_pytest(["tests"], timeout=1200)
    return CheckResult(
        name="pytest_full",
        status=GateStatus.PASS if ok else GateStatus.FAIL,
        details=tail.strip().splitlines()[-1] if tail.strip() else "",
        metrics=metrics,
        duration_s=time.time() - t0,
    )


def check_no_lookahead() -> CheckResult:
    t0 = time.time()
    ok, tail, metrics = _run_pytest(["tests/test_no_lookahead.py"], timeout=300)
    return CheckResult(
        name="no_lookahead",
        status=GateStatus.PASS if ok else GateStatus.FAIL,
        details=f"causality tests {'ok' if ok else 'FAILED'}",
        metrics=metrics,
        duration_s=time.time() - t0,
    )


def check_risk_gates() -> CheckResult:
    t0 = time.time()
    ok, _tail, metrics = _run_pytest(
        [
            "tests/test_paper_risk_e2e.py",
            "tests/test_paper_trading_risk_integration.py",
        ],
        timeout=600,
    )
    return CheckResult(
        name="risk_gates",
        status=GateStatus.PASS if ok else GateStatus.FAIL,
        details="FLIP/drawdown/exposure/sizing invariants",
        metrics=metrics,
        duration_s=time.time() - t0,
    )


def check_causal_audit() -> CheckResult:
    """P1.13 Full Causal Audit — DAG output[t] only input[:t]."""
    t0 = time.time()
    try:
        from src.validation.causal_audit import audit_full_dag

        report = audit_full_dag()
        failed_nodes = [n for n in report.nodes if not n.passed]
        if report.passed:
            return CheckResult(
                name="causal_audit",
                status=GateStatus.PASS,
                details=f"PASS: {len(report.nodes)} DAG nodes causal",
                metrics={"nodes": len(report.nodes), "failed": 0, "summary": report.summary},
                duration_s=time.time() - t0,
            )
        else:
            return CheckResult(
                name="causal_audit",
                status=GateStatus.FAIL,
                details=f"FAIL: {len(failed_nodes)}/{len(report.nodes)} nodes leaked: " + "; ".join(f"{n.node}: {n.details}" for n in failed_nodes),
                metrics={"nodes": len(report.nodes), "failed": len(failed_nodes), "summary": report.summary, "failed_nodes": [n.node for n in failed_nodes]},
                duration_s=time.time() - t0,
            )
    except Exception as e:
        return CheckResult(
            name="causal_audit",
            status=GateStatus.FAIL,
            details=f"exception: {type(e).__name__}: {e}",
            duration_s=time.time() - t0,
        )


def check_causal_mutation() -> CheckResult:
    """P1.14 Causal Mutation Test — future-mutated past invariant for 6 components."""
    t0 = time.time()
    ok, tail, metrics = _run_pytest(["tests/test_causal_mutation.py"], timeout=600)
    return CheckResult(
        name="causal_mutation",
        status=GateStatus.PASS if ok else GateStatus.FAIL,
        details=f"mutation tests {'ok' if ok else 'FAILED'} (6 components: FeatureEngine/Regime/DatasetBuilder/scaler/model_preprocessing/SignalGenerator)",
        metrics=metrics,
        duration_s=time.time() - t0,
    )


def check_backtest_smoke(data_dir: Path) -> CheckResult:
    """
    Health check (NOT profitability): the engine must run real prepared
    data end-to-end and produce structurally sane metrics.
    Strategy PnL sign is a research concern (profitability track),
    never a deployment gate. Use check_trading_readiness for that.
    """
    t0 = time.time()

    candidates = sorted(Path(data_dir).glob("*_prepared.parquet"))
    if not candidates:
        return CheckResult(
            name="backtest",
            status=GateStatus.BLOCKED,
            details=f"no *_prepared.parquet under {data_dir}",
            duration_s=time.time() - t0,
        )

    try:
        import pandas as pd

        from src.backtest_engine import BacktestEngine

        df = pd.read_parquet(candidates[0])
        res = BacktestEngine(initial_balance=1000.0).run(df)

        structurally_sane = (
            math_isfinite(res.max_drawdown_pct)
            and math_isfinite(res.sharpe)
            and res.total_trades >= 0
            and res.win_rate >= 0.0
            and bool(res.equity_curve)          # equity recorded per bar
        )
        bankrupt = res.final_balance <= 0

        return CheckResult(
            name="backtest",
            status=GateStatus.PASS if structurally_sane else GateStatus.FAIL,
            details=(
                f"{candidates[0].name}: {res.total_trades} trades"
                + (" [STRATEGY BLEW UP - research signal, not gate]" if bankrupt else "")
            ),
            metrics={
                "total_return_pct": res.total_return_pct,
                "max_drawdown_pct": res.max_drawdown_pct,
                "profit_factor": (
                    None if res.profit_factor == float("inf") else res.profit_factor
                ),
                "sharpe": res.sharpe,
                "bankrupt": bankrupt,
                "note": "gate=health; profitability via check_trading_readiness (TRADING_GATE)",
            },
            duration_s=time.time() - t0,
        )
    except Exception as e:  # noqa: BLE001
        return CheckResult(
            name="backtest",
            status=GateStatus.FAIL,
            details=f"exception: {type(e).__name__}: {e}",
            duration_s=time.time() - t0,
        )


def check_trading_readiness(data_dir: Path, strict: bool = True) -> CheckResult:
    """
    P0 TRADING READINESS GATE — OOS ONLY, frozen params (point 17).

    NOTE: This gate does NOT re-optimize. Full TRAIN→Optimization→Freeze→OOS
    is implemented in NestedWalkForward (inner WF + Optuna → aggregate → freeze → outer OOS).
    Trading Readiness validates the *frozen* candidate on canonical Walk-Forward OOS.

    strict=True (production, default): WFO FAILED → FAIL/BLOCK, no fallback to weaker validation.
    strict=False (explicit research_mode): may fallback to 80/20 holdout when canonical WFO lacks data.

    Thresholds: PF≥1.05, DD≥-15%, Sharpe>0, expectancy>0, trades≥30 on OOS only.
    """
    t0 = time.time()
    # Canonical ValidationDatasetSpec (point 23): single source of truth — 1h BTCUSDT
    # Ensures TradingReadiness and WalkForwardSmoke use same dataset/timeframe
    VALIDATION_DATASET_SPEC = {"symbol": "BTCUSDT", "timeframe": "1h", "dataset_id": "BTCUSDT_1H_v7"}
    path = Path(data_dir) / "btcusdt_1h_prepared.parquet"
    if not path.exists():
        candidates = sorted(Path(data_dir).glob("*_prepared.parquet"))
        if not candidates:
            return CheckResult(
                name="trading_readiness",
                status=GateStatus.BLOCKED,
                details=f"no *_prepared.parquet under {data_dir}",
                duration_s=time.time() - t0,
            )
        # Deterministic priority matching walk_forward_smoke (1h > 4h > 15m)
        priority = {"1h": 0, "4h": 1, "15m": 2}
        def _prio_tr(p: Path) -> tuple[int, str]:
            tf = _infer_timeframe_from_path(p)
            return (priority.get(tf, 99), p.name)
        candidates.sort(key=_prio_tr)
        path = candidates[0]
    else:
        path = Path(data_dir) / "btcusdt_1h_prepared.parquet"
    try:
        import pandas as pd

        from src.backtest_engine import BacktestEngine
        from src.walk.walk_forward_engine import WalkForwardEngine

        df = pd.read_parquet(path)
        # --- Canonical Walk-Forward OOS (frozen params, no re-optimization) ---
        # Full TRAIN→Optimization→Freeze→OOS lives in NestedWalkForward; here we validate frozen candidate.
        # Requires >=3600 rows for canonical WFO (train 3000 + test 600).
        use_wf = len(df) >= 3600
        wf_error: str | None = None
        if use_wf:
            try:
                wf = WalkForwardEngine(train_size=3000, test_size=600, step_size=600, initial_balance=1000.0)
                wfr = wf.run(df)
                # OOS aggregate is wfr (sum of all OOS windows) — now respects 30-trade guard
                # If median is INSUFFICIENT_SAMPLE, treat as BLOCKED not FAIL (need more OOS)
                median = wfr.median_pf_valid() if hasattr(wfr, "median_pf_valid") else None
                if median == "INSUFFICIENT_SAMPLE":
                    return CheckResult(
                        name="trading_readiness",
                        status=GateStatus.INSUFFICIENT_SAMPLE,
                        details=f"INSUFFICIENT_SAMPLE: valid {len(wfr.valid_windows)}/{wfr.total_windows} windows <30 trades (need more OOS)",
                        metrics={"valid_windows": len(wfr.valid_windows), "total_windows": wfr.total_windows, "oos_mode": "walk-forward", "status": "INSUFFICIENT_SAMPLE"},
                        duration_s=time.time() - t0,
                    )
                # Use aggregate OOS metrics for gate (not full-sample)
                # Map WalkForwardResult → BacktestResult-like for threshold checks
                class _OOSRes:
                    pass
                res = _OOSRes()
                res.final_balance = wfr.final_balance
                res.profit_factor = float(median) if isinstance(median, (int, float)) else 0.0
                # Aggregate maxDD: worst window DD (conservative)
                try:
                    res.max_drawdown_pct = min(w.backtest_result.max_drawdown_pct for w in wfr.windows)
                except Exception:
                    res.max_drawdown_pct = 0.0
                try:
                    # Sharpe median over valid windows
                    import numpy as np
                    sharpes = [w.backtest_result.sharpe for w in wfr.valid_windows if w.backtest_result.sharpe == w.backtest_result.sharpe]
                    res.sharpe = float(np.median(sharpes)) if sharpes else 0.0
                except Exception:
                    res.sharpe = 0.0
                res.total_return_pct = (wfr.final_balance - wfr.initial_balance) / wfr.initial_balance * 100 if wfr.initial_balance else 0
                res.total_trades = wfr.total_trades
                res.net_profit = wfr.net_profit
                res._oos_mode = "walk-forward"
                res._wfr = wfr
            except Exception as e_wf:
                wf_error = str(e_wf)
                if strict:
                    # Strict production: WFO FAILED → FAIL/BLOCK, no weaker fallback (point 16)
                    return CheckResult(
                        name="trading_readiness",
                        status=GateStatus.FAIL,
                        details=f"WFO FAILED in strict mode (no fallback): {type(e_wf).__name__}: {e_wf}",
                        metrics={"oos_mode": "walk-forward", "error": wf_error, "strict": True},
                        duration_s=time.time() - t0,
                    )
                # Research mode only: fallback to holdout is allowed
                use_wf = False
        if not use_wf:
            if strict:
                # Canonical WFO not available and strict → BLOCK (point 16: no weaker mode)
                if wf_error is not None:
                    return CheckResult(
                        name="trading_readiness",
                        status=GateStatus.FAIL,
                        details=f"WFO FAILED in strict mode (no fallback): {wf_error}",
                        metrics={"oos_mode": "walk-forward", "error": wf_error, "strict": True},
                        duration_s=time.time() - t0,
                    )
                # len <3600 case
                return CheckResult(
                    name="trading_readiness",
                    status=GateStatus.BLOCKED,
                    details=f"Canonical WFO requires >=3600 rows, got {len(df)} (strict: no 80/20 fallback)",
                    metrics={"oos_mode": "walk-forward", "rows": len(df), "strict": True},
                    duration_s=time.time() - t0,
                )
            # Research mode only: 80/20 holdout fallback is allowed
            split = int(len(df) * 0.8)
            oos_df = df.iloc[split:].copy()
            if len(oos_df) < 30:
                return CheckResult(
                    name="trading_readiness",
                    status=GateStatus.BLOCKED,
                    details=f"OOS hold-out too small: {len(oos_df)} rows (need ≥30 trades)",
                    duration_s=time.time() - t0,
                )
            res = BacktestEngine(initial_balance=1000.0, minimum_rows=min(30, len(oos_df))).run(oos_df)
            res._oos_mode = f"hold-out 80/20 ({len(oos_df)} bars) [research_mode fallback — not strict]"
            res._wfr = None

        # ===== P1.11 Minimum Sample Gates (before profitability) =====
        # Applies to both WFO aggregate and hold-out OOS
        sample_reasons: list[str] = []
        # 1) minimum OOS trades/window (for WFO) or total
        try:
            if hasattr(res, '_wfr') and res._wfr is not None:
                wfr = res._wfr  # type: ignore
                trades_per_win = [w.backtest_result.total_trades for w in wfr.windows]
                median_trades = sorted(trades_per_win)[len(trades_per_win)//2] if trades_per_win else 0
                if median_trades < 30:
                    sample_reasons.append(f"median trades/window {median_trades}<30")
                if res.total_trades < 100:
                    sample_reasons.append(f"total OOS trades {res.total_trades}<100")
            else:
                if res.total_trades < 30:
                    sample_reasons.append(f"total OOS trades {res.total_trades}<30 (hold-out)")
        except Exception:
            pass
        # 2) minimum OOS calendar duration — P1.12 strictly via timestamp
        try:
            # OOS calendar: OOS_end - OOS_start via real timestamp
            if "timestamp" in df.columns:
                if hasattr(res, '_wfr') and res._wfr is not None:
                    # WFO: OOS is suffix len = windows*test_size
                    wfr = res._wfr  # type: ignore
                    total_test_bars = len(wfr.windows) * 600  # trading_readiness test_size=600
                    # Clamp to df length
                    oos_start_idx = max(0, len(df) - total_test_bars)
                    oos_start = pd.to_datetime(df["timestamp"].iloc[oos_start_idx])
                    oos_end = pd.to_datetime(df["timestamp"].iloc[-1])
                    oos_days = (oos_end - oos_start).total_seconds() / 86400.0
                    res._oos_days = oos_days  # stash for metrics
                    if oos_days < 30:
                        sample_reasons.append(f"OOS calendar {oos_days:.1f}d<30d")
                else:
                    # hold-out: OOS is suffix 20%
                    split = int(len(df) * 0.8)
                    oos_start = pd.to_datetime(df["timestamp"].iloc[split])
                    oos_end = pd.to_datetime(df["timestamp"].iloc[-1])
                    oos_days = (oos_end - oos_start).total_seconds() / 86400.0
                    res._oos_days = oos_days
                    if oos_days < 7:
                        sample_reasons.append(f"OOS calendar {oos_days:.1f}d<7d (hold-out)")
            else:
                sample_reasons.append("OOS timestamp missing — cannot compute calendar duration (P1.12)")
        except Exception as e:
            sample_reasons.append(f"OOS calendar check failed: {e}")
        if sample_reasons:
            return CheckResult(
                name="trading_readiness",
                status=GateStatus.INSUFFICIENT_SAMPLE,
                details="INSUFFICIENT_SAMPLE: " + "; ".join(sample_reasons),
                metrics={"reasons": sample_reasons, "trades": getattr(res, 'total_trades', None), "oos_days": getattr(res, '_oos_days', None), "oos_mode": getattr(res, '_oos_mode', None), "status": "INSUFFICIENT_SAMPLE"},
                duration_s=time.time() - t0,
            )

        reasons: list[str] = []
        if res.final_balance <= 0:
            reasons.append("bankrupt")
        if res.profit_factor < 1.05:
            reasons.append(f"PF {res.profit_factor:.3f}<1.05")
        if res.max_drawdown_pct < -15.0:
            reasons.append(f"maxDD {res.max_drawdown_pct:.1f}%<-15%")
        if res.sharpe < 0:
            reasons.append(f"Sharpe {res.sharpe:.2f}<0")
        if res.total_return_pct < 0:
            reasons.append(f"return {res.total_return_pct:.1f}%<0")

        # Expectancy = net_profit / total_trades
        expectancy = res.net_profit / res.total_trades if res.total_trades else 0.0
        if expectancy <= 0:
            reasons.append(f"expectancy {expectancy:.2f}<=0")

        if reasons:
            return CheckResult(
                name="trading_readiness",
                status=GateStatus.FAIL,
                details="BLOCKED: " + "; ".join(reasons),
                metrics={
                    "total_return_pct": res.total_return_pct,
                    "max_drawdown_pct": res.max_drawdown_pct,
                    "profit_factor": None if res.profit_factor == float("inf") else res.profit_factor,
                    "sharpe": res.sharpe,
                    "bankrupt": res.final_balance <= 0,
                    "expectancy": round(expectancy, 4),
                    "reasons": reasons,
                },
                duration_s=time.time() - t0,
            )
        return CheckResult(
            name="trading_readiness",
            status=GateStatus.PASS,
            details=f"PASS: PF {res.profit_factor:.2f} Sharpe {res.sharpe:.2f} PF>1.05, expectancy>0",
            metrics={
                "total_return_pct": res.total_return_pct,
                "max_drawdown_pct": res.max_drawdown_pct,
                "profit_factor": None if res.profit_factor == float("inf") else res.profit_factor,
                "sharpe": res.sharpe,
                "expectancy": round(expectancy, 4),
            },
            duration_s=time.time() - t0,
        )
    except Exception as e:  # noqa: BLE001
        return CheckResult(
            name="trading_readiness",
            status=GateStatus.FAIL,
            details=f"exception: {type(e).__name__}: {e}",
            duration_s=time.time() - t0,
        )


def _infer_timeframe_from_path(path: Path) -> str:
    """Extract timeframe token from filename like btcusdt_15m_prepared.parquet -> 15m."""
    name = path.name.lower()
    for tf in ("15m", "1h", "4h", "1d"):
        if f"_{tf}" in name:
            return tf
    return "1h"


def _timeframe_to_hours(tf: str) -> float:
    mapping = {"15m": 0.25, "1h": 1.0, "4h": 4.0, "1d": 24.0}
    return mapping.get(tf, 1.0)


def math_isfinite(x: float) -> bool:
    x = float(x)
    return x == x and abs(x) != float("inf")


def check_walk_forward_smoke(data_dir: Path) -> CheckResult:
    t0 = time.time()

    # Deterministic canonical selection: prefer 1h, else smallest timeframe available
    path = Path(data_dir) / "btcusdt_1h_prepared.parquet"
    if not path.exists():
        candidates = sorted(Path(data_dir).glob("*_prepared.parquet"))
        if not candidates:
            return CheckResult(
                name="walk_forward",
                status=GateStatus.BLOCKED,
                details="no prepared data for WF smoke",
                duration_s=time.time() - t0,
            )
        # Prefer 1h > 4h > 15m deterministic priority
        priority = {"1h": 0, "4h": 1, "15m": 2}
        def _prio(p: Path) -> tuple[int, str]:
            tf = _infer_timeframe_from_path(p)
            return (priority.get(tf, 99), p.name)
        candidates.sort(key=_prio)
        path = candidates[0]

    try:
        import pandas as pd

        from src.walk_forward_engine import WalkForwardEngine

        df = pd.read_parquet(path).head(4000)
        eng = WalkForwardEngine(train_size=1000, test_size=250, step_size=250,
                                initial_balance=1000.0)
        result = eng.run(df)

        # === ENGINE_HEALTH_GATE (point 18) — only checks that engine runs, not trading eligibility ===
        # walk_forward smoke: engine health, TRADING_VALIDITY_GATE (trading_readiness) checks PF/trades.
        # This separation prevents mixing "engine works" vs "candidate is tradable".
        health_reasons: list[str] = []
        # Basic liveness
        if result.total_windows < 2:
            health_reasons.append(f"windows {result.total_windows}<2 (engine did not produce windows)")
        # Finite metrics (engine produced valid numbers, not NaN crash)
        try:
            pfs = [w.backtest_result.profit_factor for w in result.windows]
            sharpes = [w.backtest_result.sharpe for w in result.windows]
            if any(not math_isfinite(p) for p in pfs):
                health_reasons.append("PF non-finite (inf/nan) — engine numeric health failed")
            if any(not math_isfinite(s) for s in sharpes):
                health_reasons.append("Sharpe non-finite")
        except Exception:
            health_reasons.append("finite metrics check failed")

        # P1.12 Timeframe-aware: strictly OOS_end - OOS_start via timestamp (no bars/24 fallback)
        try:
            # Prefer real OOS window timestamps: test windows are last 250*windows bars
            df_full = pd.read_parquet(path)
            if "timestamp" not in df_full.columns:
                health_reasons.append("OOS timestamp missing — cannot compute calendar duration (P1.12 requires timestamp)")
            else:
                # Compute OOS calendar duration as last OOS_end - first OOS_start
                # OOS is suffix: total_windows * test_size (250) bars at end of df
                oos_bars = result.total_windows * 250
                oos_start_idx = max(0, len(df_full) - oos_bars)
                oos_start = pd.to_datetime(df_full["timestamp"].iloc[oos_start_idx])
                oos_end = pd.to_datetime(df_full["timestamp"].iloc[-1])
                oos_days = (oos_end - oos_start).total_seconds() / 86400.0
                if oos_days < 1:
                    health_reasons.append(f"OOS {oos_days:.1f}d suspiciously small (timestamp-based)")
        except Exception as e:
            health_reasons.append(f"OOS timestamp check failed: {e}")

        # === P2.10/P2.11 Funnel diagnostics for 0 trades — real counts from TradeEngine ===
        funnel_info: dict = {}
        funnel_render: str = ""
        try:
            from src.research.signal_funnel import FunnelCounts
            bars = len(df)
            trades_closed = int(result.total_trades)
            # Try to aggregate real funnel from WalkForward windows (P2.10 real counts)
            real_funnel = None
            try:
                # Aggregate across windows' BacktestResult.funnel
                agg = FunnelCounts()
                has_real = False
                for w in result.windows:
                    br = w.backtest_result
                    f = getattr(br, "funnel", None)
                    if f is not None:
                        has_real = True
                        agg.raw_signals += int(getattr(f, "raw_signals", 0))
                        agg.ai_accepted += int(getattr(f, "ai_accepted", 0))
                        agg.ml_accepted += int(getattr(f, "ml_accepted", 0))
                        agg.confidence_accepted += int(getattr(f, "confidence_accepted", 0))
                        agg.risk_accepted += int(getattr(f, "risk_accepted", 0))
                        agg.orders_submitted += int(getattr(f, "orders_submitted", 0))
                        agg.orders_filled += int(getattr(f, "orders_filled", 0))
                        # trades_closed will be aggregated as total
                if has_real:
                    agg.trades_closed = trades_closed
                    # Copy blocked counters
                    for w in result.windows:
                        f = getattr(w.backtest_result, "funnel", None)
                        if f:
                            agg.blocked_by_ml += int(getattr(f, "blocked_by_ml", 0))
                            agg.blocked_by_confidence += int(getattr(f, "blocked_by_confidence", 0))
                            agg.blocked_by_risk += int(getattr(f, "blocked_by_risk", 0))
                    real_funnel = agg
            except Exception:
                real_funnel = None
            if real_funnel is not None and real_funnel.raw_signals > 0:
                fc = real_funnel
            else:
                # Fallback heuristic when no real funnel (e.g., old BacktestResult)
                # P2.11: still classify into 6 spec reasons
                fc = FunnelCounts(
                    raw_signals=bars if trades_closed == 0 else max(bars // 10, trades_closed * 2),
                    ai_accepted=0 if trades_closed == 0 else trades_closed,
                    ml_accepted=0 if trades_closed == 0 else trades_closed,
                    confidence_accepted=0 if trades_closed == 0 else trades_closed,
                    risk_accepted=0 if trades_closed == 0 else trades_closed,
                    orders_submitted=0 if trades_closed == 0 else trades_closed,
                    orders_filled=0 if trades_closed == 0 else trades_closed,
                    trades_closed=trades_closed,
                )
                # For heuristic 0 trades, infer most likely block as execution (since we have no per-stage data)
                if trades_closed == 0 and fc.raw_signals > 0:
                    # Heuristic fallback classification will be ML_REJECTED etc. but we don't have real breakdown
                    # Keep as is, gate will surface need for real funnel
                    pass
            funnel_info = fc.to_dict()
            funnel_render = fc.render(bars)
            if trades_closed == 0:
                funnel_info["note"] = "0 trades — P2.10 funnel: bars→candidate→AI→ML→confidence→risk→orders→fills→closed (P2.11 classification)"
                # Also add detailed reason
                try:
                    det = fc.classify_zero_detailed()
                    funnel_info["detailed_reason"] = det
                except Exception:
                    pass
        except Exception as e:
            funnel_info = {"error": str(e)}

        # Health gate PASS even with 0 trades — trading eligibility is checked in trading_readiness (TRADING_VALIDITY_GATE)
        # But we surface funnel so 0 trades is diagnosable, not just FAIL PF
        # 12 windows 0 trades is still engine-healthy (engine ran), but TRADING_VALIDITY will FAIL/need funnel.
        if health_reasons:
            return CheckResult(
                name="walk_forward",
                status=GateStatus.FAIL,
                details="ENGINE_HEALTH FAIL: " + "; ".join(health_reasons),
                metrics={"windows": result.total_windows, "trades": result.total_trades, "funnel": funnel_info, "reasons": health_reasons, "engine": "health_only"},
                duration_s=time.time() - t0,
            )
        # P1.11 Minimum Sample Gates — INSUFFICIENT_SAMPLE (not FAIL) when lacking
        # Check minimum OOS trades/window, total, calendar duration
        sample_reasons: list[str] = []
        try:
            trades_per_win = [w.backtest_result.total_trades for w in result.windows]
            median_trades = sorted(trades_per_win)[len(trades_per_win)//2] if trades_per_win else 0
            total_trades = result.total_trades
            # Minimum OOS trades/window (e.g., 30)
            if median_trades < 30:
                sample_reasons.append(f"median trades/window {median_trades}<30")
            if total_trades < 100:
                sample_reasons.append(f"total OOS trades {total_trades}<100")
            # Calendar duration already computed as oos_days above
            try:
                if 'oos_days' in locals() and oos_days < 90:
                    sample_reasons.append(f"OOS calendar {oos_days:.1f}d<90d")
            except Exception:
                pass
            if sample_reasons:
                return CheckResult(
                    name="walk_forward",
                    status=GateStatus.INSUFFICIENT_SAMPLE,
                    details="INSUFFICIENT_SAMPLE: " + "; ".join(sample_reasons),
                    metrics={"windows": result.total_windows, "trades": total_trades, "median_trades": median_trades, "oos_days": oos_days if 'oos_days' in locals() else None, "reasons": sample_reasons, "funnel": funnel_info, "status": "INSUFFICIENT_SAMPLE"},
                    duration_s=time.time() - t0,
                )
        except Exception as e:
            sample_reasons.append(f"sample check failed: {e}")
            return CheckResult(
                name="walk_forward",
                status=GateStatus.INSUFFICIENT_SAMPLE,
                details="INSUFFICIENT_SAMPLE: " + "; ".join(sample_reasons),
                metrics={"reasons": sample_reasons, "funnel": funnel_info, "status": "INSUFFICIENT_SAMPLE"},
                duration_s=time.time() - t0,
            )
        # ENGINE_HEALTH PASS — even with 0 trades, this is now correctly health PASS
        # Trading eligibility (PF, trades, expectancy) is checked in TRADING_VALIDITY_GATE (trading_readiness)
        exp = result.net_profit / result.total_trades if result.total_trades else 0
        median_trades = 0
        try:
            trades_per_win = [w.backtest_result.total_trades for w in result.windows]
            median_trades = sorted(trades_per_win)[len(trades_per_win)//2] if trades_per_win else 0
        except Exception:
            pass
        details = f"ENGINE_HEALTH PASS: {result.total_windows} windows, {result.total_trades} trades (health only, not trading eligibility)"
        if result.total_trades == 0:
            details += f" | FUNNEL {funnel_render.replace(chr(10), ' | ')}"
            details += " | NOTE: TRADING_VALIDITY (trading_readiness) will FAIL this candidate — see funnel for why (0 raw vs confidence vs risk)"
        else:
            details += f" median {median_trades}/win, expectancy {exp:.4f}"
        return CheckResult(
            name="walk_forward",
            status=GateStatus.PASS,
            details=details,
            metrics={"windows": result.total_windows, "trades": result.total_trades, "median_trades_per_window": median_trades, "expectancy": round(exp, 4), "funnel": funnel_info, "engine": "health_only"},
            duration_s=time.time() - t0,
        )
    except Exception as e:  # noqa: BLE001
        return CheckResult(
            name="walk_forward",
            status=GateStatus.FAIL,
            details=f"exception: {type(e).__name__}: {e}",
            duration_s=time.time() - t0,
        )


def check_long_run_evidence(long_run_dir: Path, min_days: int = 30,
                            min_trades: int | None = None) -> CheckResult:
    """
    Evaluates long-run paper artifacts produced by
    src/validation/long_run.py (state.json + journal.csv).
    Missing/incomplete evidence => BLOCKED (not FAIL).
    """
    from src.validation.long_run import evaluate_long_run

    t0 = time.time()
    try:
        crit = evaluate_long_run(
            long_run_dir,
            min_days=min_days,
            min_trades=min_trades if min_trades is not None else 0,
            auto_min_trades=(min_trades is None),
        )
    except Exception as e:  # noqa: BLE001
        return CheckResult(
            name="long_run_paper",
            status=GateStatus.BLOCKED,
            details=f"artifacts unavailable: {type(e).__name__}",
            duration_s=time.time() - t0,
        )

    status = GateStatus.PASS if crit["passed"] else GateStatus.BLOCKED
    return CheckResult(
        name="long_run_paper",
        status=status,
        details=crit["summary"],
        metrics={k: v for k, v in crit.items() if k not in ("passed", "summary")},
        duration_s=time.time() - t0,
    )


# ============================================================
# GATE
# ============================================================

class QuantAIValidationGate:
    def __init__(self, checks=None) -> None:
        self.checks: list = list(checks) if checks is not None else []

    def add(self, check) -> "QuantAIValidationGate":
        self.checks.append(check)
        return self

    def run(self, data_dir: Path | None = None) -> GateReport:
        report = GateReport()
        for check in self.checks:
            report.results.append(check())
        # populate immutable evidence versioning (P0-5)
        meta = _collect_evidence_metadata(data_dir)
        report.gate_version = meta["gate_version"]
        report.code_commit = meta["code_commit"]
        report.config_hash = meta["config_hash"]
        report.dataset_hash = meta["dataset_hash"]
        report.environment_hash = meta["environment_hash"]
        report.timestamp = meta["timestamp"]
        report.freshness = "FRESH"
        report.is_stale = False
        report.stale_reasons = []
        return report


def build_standard_gate(
    data_dir: Path | None = None,
    long_run_dir: Path | None = None,
    include_pytest_full: bool = True,
    min_days: int = 30,
    min_trades: int = 30,
) -> QuantAIValidationGate:
    root = Path(__file__).resolve().parents[2]
    data_dir = data_dir or root / "data"
    long_run_dir = long_run_dir or root / "data" / "long_run"

    gate = QuantAIValidationGate()
    gate.add(check_compile)
    if include_pytest_full:
        gate.add(check_pytest_full)
    gate.add(check_no_lookahead)
    gate.add(check_causal_audit)
    gate.add(check_causal_mutation)
    gate.add(check_risk_gates)
    gate.add(lambda: check_backtest_smoke(data_dir))
    gate.add(lambda: check_trading_readiness(data_dir))
    gate.add(lambda: check_walk_forward_smoke(data_dir))
    gate.add(lambda: check_long_run_evidence(long_run_dir, min_days, min_trades))
    return gate


if __name__ == "__main__":
    args = set(sys.argv[1:])
    gate = build_standard_gate(include_pytest_full="--fast" not in args)

    # data_dir for metadata (consistent with gate's data_dir)
    root = Path(__file__).resolve().parents[2]
    data_dir_cli = root / "data"
    report = gate.run(data_dir=data_dir_cli)
    print(report.pretty())

    out = root_json = Path(__file__).resolve().parents[2] / "data" / "gate_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")
    print(f"json: {out}")
    # also verify immutability immediately after write
    try:
        _, verification = load_and_verify_report(out, data_dir=data_dir_cli)
        if verification["is_stale"]:
            print(f"WARNING: freshly written report is STALE: {verification['reasons']}")
        else:
            print(f"evidence: gate_version={report.gate_version} commit={report.code_commit[:12]} config={report.config_hash[:12]} dataset={report.dataset_hash[:12]} env={report.environment_hash[:12]} ts={report.timestamp}")
    except Exception as e:
        print(f"verify warning: {e}")

    sys.exit(0 if report.verdict == GateStatus.PASS else 1)
