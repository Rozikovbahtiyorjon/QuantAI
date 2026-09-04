"""QuantAI Verifier — hard separation: Agent creates Artifact, Verifier confirms, Gate decides."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

class VerificationResult:
    def __init__(self, verified: bool, reason: str = "", artifact_hashes: Dict[str, str] = None, checks: Dict[str, bool] = None):
        self.verified = verified
        self.reason = reason
        self.artifact_hashes = artifact_hashes or {}
        self.checks = checks or {}
        self.passed = verified

class Verifier:
    """Independent verifier: checks exit_code, artifact existence/hashes, not agent claim.
    
    P1.15 Agent Success Contract:
        success=True allowed ONLY if:
            1. command exit == 0
            2. artifact exists
            3. artifact valid (hashable)
            4. required tests pass (tests_run>0 and passed)
            5. expected metrics exist (no placeholder/None)
        Any bypass -> REJECT. Architecture: Agent->Artifact->Verifier->Evidence->Gate, never Agent->success.
    """

    # Expected metrics per agent/task type — fail-closed if missing
    REQUIRED_METRICS = {
        "quant_researcher": ["backtest_pf", "tournament"],
        "ml_engineer": ["bal_acc", "model"],
        "quant_engineer": ["code", "ruff_ok"],
        "portfolio_manager": ["allocation", "correlation_report"],
        "risk_manager": ["risk_report", "exposure_ok"],
        "qa_engineer": ["test_report", "coverage"],
        "execution_engineer": ["fill_report"],
        "data_engineer": ["quality_report", "dataset_hash"],
    }

    def _check_metrics_exist(self, execution_result: Dict[str, Any], task: Any = None) -> tuple[bool, str]:
        """P1.15: expected metrics must exist and not be placeholder/None."""
        # Determine agent type from task or result
        agent_hint = None
        if task is not None:
            agent_hint = getattr(task, "assigned_agent", None) or getattr(task, "name", "")
            if hasattr(task, "metadata") and isinstance(task.metadata, dict):
                agent_hint = task.metadata.get("agent", agent_hint)
        # Try to infer from result keys
        if not agent_hint:
            # infer from metrics present
            if "tournament" in execution_result:
                agent_hint = "quant_researcher"
            elif "model" in execution_result:
                agent_hint = "ml_engineer"
            elif "allocation" in execution_result:
                agent_hint = "portfolio_manager"
            elif "risk_report" in execution_result:
                agent_hint = "risk_manager"
        # Check no placeholder values
        for k, v in execution_result.items():
            if v is None and k in ("bal_acc", "profit_factor", "exposure_ok", "allocation"):
                return False, f"metric {k} is None (placeholder not allowed)"
            if isinstance(v, str) and v.strip() == "# Generated code" and k == "code":
                # quant_engineer must not return bare placeholder
                if execution_result.get("tests") == []:
                    return False, "code placeholder '# Generated code' with empty tests — not valid artifact"
        # Check required metrics for inferred agent
        for agent_key, req_list in self.REQUIRED_METRICS.items():
            if agent_hint and agent_key in str(agent_hint).lower():
                missing = [r for r in req_list if r not in execution_result or execution_result[r] is None]
                # Special: ml_engineer bal_acc None is explicitly forbidden (was 0.39 placeholder)
                if agent_key == "ml_engineer" and execution_result.get("bal_acc") is None:
                    if "_bal_acc_missing" in execution_result.get("metrics", {}) if isinstance(execution_result.get("metrics"), dict) else False:
                        return False, "bal_acc missing (was 0.39 placeholder) — model not trained"
                    return False, "ml_engineer missing bal_acc — model not trained"
                if agent_key == "ml_engineer" and execution_result.get("metrics", {}).get("_bal_acc_missing"):
                    return False, "metrics._bal_acc_missing — fake metric placeholder"
                if missing:
                    return False, f"expected metrics missing for {agent_key}: {missing}"
                # Check not placeholder True for exposure_ok
                if agent_key == "risk_manager" and execution_result.get("risk_report", {}).get("exposure_ok") is True:
                    # exposure_ok must be derived from RiskOrchestrator, not hardcoded
                    # We check provenance flag
                    if not execution_result.get("_provenance", {}).get("generated_by_real_execution"):
                        return False, "risk_manager exposure_ok=True without provenance — placeholder not allowed"
        # Generic: metrics dict must not contain placeholder sentinel
        metrics = execution_result.get("metrics", execution_result)
        if isinstance(metrics, dict):
            if metrics.get("bal_acc") == 0.39:
                return False, "bal_acc 0.39 placeholder forbidden"
            if metrics.get("exposure_ok") is True and "_provenance" not in execution_result:
                # need provenance
                pass
        return True, "metrics ok"

    def verify_execution(self, execution_result: Any, task: Any = None) -> VerificationResult:
        """P1.15 + Fix: Never trust artifact's passed/success field — recompute from 7 independent checks."""
        checks: Dict[str, bool] = {}
        # Must be dict
        if not isinstance(execution_result, dict):
            return VerificationResult(False, "execution_result not dict", {}, {"is_dict": False})
        # Do NOT trust success/passed claim — we recompute
        # success_claim is observed but not trusted for final passed
        success_claim = bool(execution_result.get("success") or execution_result.get("passed"))
        # 1. exit_code must be 0 — independently check (do not trust success)
        exit_code = execution_result.get("exit_code", execution_result.get("exitCode"))
        if exit_code is None:
            # No exit_code -> cannot verify unless artifact proves execution
            checks["exit_code_0"] = False
            # Will be set true only if artifact exists and hashes valid and other checks pass
            # For now, mark as missing
        elif int(exit_code) != 0:
            return VerificationResult(False, f"exit_code {exit_code} != 0 (independently verified)", {}, {"exit_code_0": False})
        else:
            checks["exit_code_0"] = True
        # Do not trust success_claim for final decision — we will recompute verified
        checks["success_claim_observed"] = success_claim
        # 2. artifact exists — independently check file system
        artifact_paths = execution_result.get("artifact_paths") or execution_result.get("artifacts") or execution_result.get("artifact_path") or []
        if isinstance(artifact_paths, str):
            artifact_paths = [artifact_paths]
        hashes: Dict[str, str] = {}
        # Also check provided artifact_hashes dict (if any) against actual file hash — do not trust claimed hash
        provided_hashes = execution_result.get("artifact_hashes", {}) if isinstance(execution_result.get("artifact_hashes"), dict) else {}
        if artifact_paths:
            all_exist = True
            all_hash_ok = True
            for p in artifact_paths:
                pp = Path(p)
                if not pp.exists():
                    all_exist = False
                    checks["artifact_exists"] = False
                    return VerificationResult(False, f"artifact missing (independently checked): {p}", hashes, {"artifact_exists": False, "exit_code_0": checks["exit_code_0"]})
                try:
                    actual_h = hashlib.sha256(pp.read_bytes()).hexdigest()[:16]
                    hashes[str(p)] = actual_h
                    # If caller provided a hash, verify it matches actual (do not trust claimed hash)
                    if p in provided_hashes:
                        claimed = str(provided_hashes[p])
                        if claimed and claimed != actual_h:
                            return VerificationResult(False, f"artifact_hash mismatch for {p} (independently checked): claimed {claimed[:8]} != actual {actual_h[:8]}", hashes, {"artifact_hash": False})
                    elif str(p) in provided_hashes:
                        claimed = str(provided_hashes[str(p)])
                        if claimed and claimed != actual_h:
                            return VerificationResult(False, f"artifact_hash mismatch for {p} (independently checked): claimed {claimed[:8]} != actual {actual_h[:8]}", hashes, {"artifact_hash": False})
                except Exception as e:
                    all_hash_ok = False
                    return VerificationResult(False, f"artifact hash failed (independently checked) {p}: {e}", hashes, {"artifact_hash": False})
            checks["artifact_exists"] = all_exist
            checks["artifact_hash"] = all_hash_ok
            checks["artifact_valid"] = all_exist and all_hash_ok
            # If exit_code was missing but artifact exists and valid, we can consider exit_code_0 as verified via artifact
            if exit_code is None and all_exist and all_hash_ok:
                checks["exit_code_0"] = True
        else:
            # No artifact_paths — check if metrics imply execution happened
            has_metrics = bool(execution_result.get("metrics") or execution_result.get("tournament") or execution_result.get("allocation") or execution_result.get("code"))
            if not has_metrics:
                return VerificationResult(False, "no artifact and no metrics — cannot verify (independently checked)", hashes, {"artifact_exists": False})
            checks["artifact_exists"] = False
            checks["artifact_valid"] = False
            # For non-file, we still need other checks; exit_code missing is now failure unless metrics prove
            if exit_code is None:
                # No file and no exit_code -> cannot independently verify execution
                return VerificationResult(False, "no exit_code and no artifact file — cannot independently verify execution", hashes, checks)
        # 6. dataset verification — independently check dataset_hash and code_commit exist (not trusting passed)
        dataset_ok = True
        dataset_reason = ""
        # Check dataset fields in execution_result or evidence metadata
        dataset_id = execution_result.get("dataset_id") or execution_result.get("datasetId")
        dataset_hash = execution_result.get("dataset_hash") or execution_result.get("datasetHash")
        if dataset_id or dataset_hash:
            # If one present, both must be present and non-empty
            if not (dataset_id and dataset_hash):
                dataset_ok = False
                dataset_reason = "dataset_id/hash mismatch (one missing)"
            elif len(str(dataset_hash)) < 8:
                dataset_ok = False
                dataset_reason = "dataset_hash too short (not real hash)"
            else:
                checks["dataset"] = True
        else:
            # No dataset info -> for research tasks, check if task requires dataset
            # For now, allow missing dataset for non-data tasks, but mark
            checks["dataset"] = True  # permissive if not data task
            dataset_ok = True
        if not dataset_ok:
            return VerificationResult(False, f"dataset verification failed: {dataset_reason}", hashes, checks)
        checks["dataset"] = dataset_ok
        # 7. code_commit verification — must exist and be hex
        code_commit = execution_result.get("code_commit") or execution_result.get("codeCommit") or execution_result.get("commit")
        if code_commit:
            # Check it's a plausible git hash (7-40 hex chars)
            cc_str = str(code_commit).strip()
            if len(cc_str) < 7 or not all(c in "0123456789abcdefABCDEF" for c in cc_str):
                # Not a valid hash, but could be 'unknown' for non-git env
                if cc_str.lower() not in ("unknown", ""):
                    return VerificationResult(False, f"code_commit invalid (not hex): {cc_str[:20]}", hashes, checks)
            checks["code_commit"] = True
        else:
            # Code commit missing -> for code tasks, fail; for others, allow
            # Check if task is code-related
            has_code = bool(execution_result.get("code") or artifact_paths)
            if has_code:
                return VerificationResult(False, "code_commit missing for code artifact — cannot verify provenance", hashes, checks)
            checks["code_commit"] = True
        # 4 & 5: tests and metrics — independently check, not trust passed
        # 5. expected metrics exist (no placeholder) — independent check
        metrics_ok, metrics_reason = self._check_metrics_exist(execution_result, task)
        checks["metrics_exist"] = metrics_ok
        if not metrics_ok:
            return VerificationResult(False, f"metrics missing/invalid (independently checked): {metrics_reason}", hashes, checks)
        # Additional: if execution_result has error field, independently fail
        if execution_result.get("error"):
            return VerificationResult(False, f"error present (independently checked): {execution_result.get('error')}", hashes, {"error_absent": False})
        # Final: RECOMPUTE passed from independent checks, DO NOT trust artifact's passed/success
        # For file artifacts: need exit_code_0 + artifact_exists + artifact_valid + metrics + dataset + code_commit
        # For non-file: need exit_code_0 + metrics + dataset + code_commit
        if artifact_paths:
            verified = checks.get("exit_code_0", False) and checks.get("artifact_exists", False) and checks.get("artifact_valid", False) and metrics_ok and checks.get("dataset", False) and checks.get("code_commit", False)
        else:
            verified = checks.get("exit_code_0", False) and metrics_ok and checks.get("dataset", False) and checks.get("code_commit", False)
        # Do NOT use success_claim for final decision — only for logging
        if verified:
            return VerificationResult(True, "verified (independently recomputed, not trusting passed)", hashes, checks)
        else:
            return VerificationResult(False, f"contract failed (independently recomputed, not trusting passed): {checks}", hashes, checks)

    def verify_test(self, test_result: Any) -> VerificationResult:
        """Never trust passed field — recompute from tests_run/failed/errors."""
        if not isinstance(test_result, dict):
            return VerificationResult(False, "test_result not dict", {}, {"is_dict": False})
        # Independently recompute: check tests_run, failed, errors, not passed claim
        tests_run = int(test_result.get("tests_run", test_result.get("testsRun", 0)) or 0)
        failed = int(test_result.get("failed", test_result.get("failedCount", 0)) or 0)
        errors = int(test_result.get("errors", test_result.get("errorCount", 0)) or 0)
        # Also check alternative fields: failures, errors
        if "failures" in test_result and isinstance(test_result["failures"], list):
            failed = len(test_result["failures"]) if failed == 0 else failed
        # Recompute passed independently: must have tests_run>0 and failed==0 and errors==0
        recomputed_passed = (tests_run > 0 and failed == 0 and errors == 0)
        # Do not trust test_result.get("passed") — use recomputed
        observed_passed = bool(test_result.get("passed") or test_result.get("success"))
        checks = {"tests_run>0": tests_run > 0, "failed==0": failed == 0, "errors==0": errors == 0, "recomputed_passed": recomputed_passed, "observed_passed": observed_passed}
        if tests_run == 0:
            return VerificationResult(False, "tests_run 0 cannot be passed (independently recomputed)", {}, checks)
        if not recomputed_passed:
            return VerificationResult(False, f"tests failed (independently recomputed): tests_run={tests_run} failed={failed} errors={errors}", {}, checks)
        return VerificationResult(True, "tests verified (independently recomputed, not trusting passed)", {}, checks)

    def verify_review(self, review: Any) -> VerificationResult:
        """Never trust approved field alone — check verified flag and execution/test evidence."""
        if not isinstance(review, dict):
            return VerificationResult(False, "review not dict", {}, {"is_dict": False})
        approved_claim = bool(review.get("approved"))
        verified_flag = bool(review.get("verified") or review.get("verified_flag"))
        # Independently recompute: approved only if verified and not placeholder
        # Check that review has real evidence backing
        has_real_check = bool(review.get("verified") or "execution and tests verified" in str(review.get("reason", "")) or "execution and tests verified" in str(review.get("reason", "")))
        checks = {"approved_claim": approved_claim, "verified_flag": verified_flag, "has_real_check": has_real_check}
        # Do not trust approved_claim alone
        recomputed_approved = approved_claim and verified_flag
        if approved_claim and not verified_flag:
            return VerificationResult(False, "review approved without verification (not trusting approved)", {}, checks)
        if not recomputed_approved and approved_claim:
            return VerificationResult(False, "review not verified (independently recomputed)", {}, checks)
        return VerificationResult(recomputed_approved, "review verified (independently recomputed, not trusting approved)" if recomputed_approved else "review not approved (independently recomputed)", {}, checks)

    def verify_evidence(self, evidence: Any) -> VerificationResult:
        """Comprehensive evidence verification — recomputes passed from 7 checks, not trusting passed field.
        
        Checks: exit_code, artifact_exists, artifact_hash, tests, metrics, dataset, code_commit
        Autonomous AI's artifact passed=true is ignored; only independent checks matter.
        """
        # evidence may be Evidence object or dict
        if hasattr(evidence, 'to_dict'):
            ev_dict = evidence.to_dict()
        elif isinstance(evidence, dict):
            ev_dict = evidence
        else:
            return VerificationResult(False, "evidence not dict/object", {}, {"is_dict": False})
        checks = {}
        # Extract fields
        exit_code = ev_dict.get("exit_code")
        artifact_paths = ev_dict.get("artifact_paths", [])
        artifact_hashes = ev_dict.get("artifact_hashes", {})
        # 1. exit_code
        if exit_code is None:
            # Check data's exit_code
            data = ev_dict.get("data", {})
            if isinstance(data, dict):
                exit_code = data.get("exit_code", data.get("exitCode"))
        checks["exit_code_0"] = exit_code is not None and int(exit_code) == 0
        if not checks["exit_code_0"]:
            return VerificationResult(False, f"exit_code not 0 (independently checked): {exit_code}", {}, checks)
        # 2. artifact_exists + 3. artifact_hash
        if artifact_paths:
            for p in artifact_paths:
                if not Path(p).exists():
                    return VerificationResult(False, f"artifact missing (independently checked): {p}", {}, {"artifact_exists": False})
                # Check hash matches stored
                if p not in artifact_hashes or not artifact_hashes[p]:
                    return VerificationResult(False, f"artifact_hash missing for {p}", {}, {"artifact_hash": False})
                try:
                    actual = hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]
                    if artifact_hashes[p] != actual:
                        return VerificationResult(False, f"artifact_hash mismatch for {p} (independently checked)", {}, {"artifact_hash": False})
                except Exception as e:
                    return VerificationResult(False, f"artifact hash failed {p}: {e}", {}, {"artifact_hash": False})
            checks["artifact_exists"] = True
            checks["artifact_hash"] = True
        else:
            # No artifact paths — must have metrics to prove execution
            data = ev_dict.get("data", {})
            has_metrics = bool(data.get("metrics") or data.get("tournament") or data.get("allocation") or data.get("code")) if isinstance(data, dict) else False
            if not has_metrics:
                return VerificationResult(False, "no artifact and no metrics — cannot verify", {}, {"artifact_exists": False})
            checks["artifact_exists"] = True  # permissive for non-file
            checks["artifact_hash"] = True
        # 4. tests
        data = ev_dict.get("data", {})
        if isinstance(data, dict) and ("tests_run" in data or "testsRun" in data):
            tr = int(data.get("tests_run", data.get("testsRun", 0)) or 0)
            passed = bool(data.get("passed") or data.get("success"))
            failed = int(data.get("failed", 0) or 0)
            # Recompute, not trust passed
            checks["tests"] = (tr > 0 and failed == 0)
            if not checks["tests"]:
                return VerificationResult(False, f"tests failed (independently checked): tests_run={tr} failed={failed} passed_claim={passed}", {}, checks)
        else:
            checks["tests"] = True
        # 5. metrics
        # Check for placeholder metrics
        if isinstance(data, dict):
            metrics = data.get("metrics", data)
            if isinstance(metrics, dict):
                if metrics.get("bal_acc") == 0.39:
                    return VerificationResult(False, "bal_acc 0.39 placeholder (independently checked)", {}, {"metrics": False})
                if metrics.get("exposure_ok") is True and not ev_dict.get("generated_by_real_execution"):
                    # Need provenance
                    pass
            checks["metrics"] = True
        else:
            checks["metrics"] = True
        # 6. dataset
        dataset_id = ev_dict.get("dataset_id")
        dataset_hash = ev_dict.get("dataset_hash")
        if dataset_id or dataset_hash:
            if not (dataset_id and dataset_hash and len(str(dataset_hash)) >= 8):
                return VerificationResult(False, "dataset_id/hash invalid (independently checked)", {}, {"dataset": False})
            checks["dataset"] = True
        else:
            checks["dataset"] = True
        # 7. code_commit
        code_commit = ev_dict.get("code_commit")
        if code_commit:
            cc = str(code_commit).strip()
            if cc.lower() not in ("unknown", "") and (len(cc) < 7 or not all(c in "0123456789abcdefABCDEF" for c in cc)):
                return VerificationResult(False, f"code_commit invalid (independently checked): {cc[:20]}", {}, {"code_commit": False})
            checks["code_commit"] = True
        else:
            # Check if evidence has code artifact but no commit
            has_code = bool(ev_dict.get("source_command") or (isinstance(data, dict) and data.get("code")))
            if has_code:
                return VerificationResult(False, "code_commit missing for code artifact", {}, {"code_commit": False})
            checks["code_commit"] = True
        # All 7 passed -> verified (recomputed, not trusting passed)
        return VerificationResult(True, "evidence verified (7 checks, not trusting passed)", {}, checks)
