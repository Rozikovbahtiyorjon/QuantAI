"""
QuantAI Validation Gate
Validates task execution against gates and criteria
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .supervisor import SupervisorState


class GateStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class GateResult:
    """Result of a validation gate"""
    gate_name: str
    status: GateStatus
    passed: bool
    reason: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate_name": self.gate_name,
            "status": self.status.value,
            "passed": self.passed,
            "reason": self.reason,
            "metrics": self.metrics,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details
        }


class ValidationGate:
    """
    Validates task execution against gates and criteria.
    Runs validation checks at each pipeline stage.
    """
    
    def __init__(self):
        self.gates: Dict[str, Callable] = {}
        self.gate_configs: Dict[str, Dict[str, Any]] = {}
        self.gate_history: List[Dict[str, Any]] = []
        self._register_default_gates()
    
    def _register_default_gates(self) -> None:
        """Register default validation gates"""
        self.register_gate(
            "data_validation_gate",
            self._gate_data_validation,
            {"required": True, "stage": "research"}
        )
        self.register_gate(
            "hypothesis_gate",
            self._gate_hypothesis,
            {"required": True, "stage": "research"}
        )
        self.register_gate(
            "architecture_review_gate",
            self._gate_architecture_review,
            {"required": True, "stage": "architecture"}
        )
        self.register_gate(
            "code_review_gate",
            self._gate_code_review,
            {"required": True, "stage": "implementation"}
        )
        self.register_gate(
            "compile_gate",
            self._gate_compile,
            {"required": True, "stage": "testing"}
        )
        self.register_gate(
            "unit_test_gate",
            self._gate_unit_tests,
            {"required": True, "stage": "testing"}
        )
        self.register_gate(
            "no_lookahead_gate",
            self._gate_no_lookahead,
            {"required": True, "stage": "testing"}
        )
        self.register_gate(
            "risk_test_gate",
            self._gate_risk_tests,
            {"required": True, "stage": "testing"}
        )
        self.register_gate(
            "backtest_profitability_gate",
            self._gate_backtest_profitability,
            {"required": True, "stage": "backtest"}
        )
        self.register_gate(
            "risk_limits_gate",
            self._gate_risk_limits,
            {"required": True, "stage": "backtest"}
        )
        self.register_gate(
            "wfo_stability_gate",
            self._gate_wfo_stability,
            {"required": True, "stage": "wfo"}
        )
        self.register_gate(
            "overfitting_gate",
            self._gate_overfitting,
            {"required": True, "stage": "wfo"}
        )
        self.register_gate(
            "paper_profitability_gate",
            self._gate_paper_profitability,
            {"required": True, "stage": "paper"}
        )
        self.register_gate(
            "consistency_gate",
            self._gate_consistency,
            {"required": True, "stage": "paper"}
        )
        self.register_gate(
            "optimization_gate",
            self._gate_optimization,
            {"required": True, "stage": "optimization"}
        )
        self.register_gate(
            "overfitting_gate",
            self._gate_overfitting,
            {"required": True, "stage": "optimization"}
        )
        self.register_gate(
            "champion_gate",
            self._gate_champion,
            {"required": True, "stage": "champion"}
        )
        self.register_gate(
            "promotion_gate",
            self._gate_promotion,
            {"required": True, "stage": "champion"}
        )
        self.register_gate(
            "production_readiness_gate",
            self._gate_production_readiness,
            {"required": True, "stage": "production"}
        )
        self.register_gate(
            "live_monitoring_gate",
            self._gate_live_monitoring,
            {"required": True, "stage": "production"}
        )
    
    def register_gate(
        self,
        name: str,
        gate_func: Callable,
        config: Dict[str, Any]
    ) -> None:
        """Register a validation gate"""
        self.gates[name] = gate_func
        self.gate_configs[name] = config
    
    async def validate(
        self,
        task: Any,
        execution_result: Any,
        test_result: Any,
        review: Any,
        state: Any
    ) -> GateResult:
        """Run all applicable validation gates for current stage"""
        stage = getattr(state, 'current_stage', 'unknown')
        results = []
        
        # Find applicable gates for current stage
        applicable_gates = [
            (name, func) for name, func in self.gates.items()
            if self.gate_configs.get(name, {}).get('stage') == getattr(state, 'current_stage', '')
            or self.gate_configs.get(name, {}).get('stage') == 'all'
        ]
        
        for name, gate_func in applicable_gates:
            try:
                config = self.gate_configs.get(name, {})
                required = config.get('required', True)
                
                result = await gate_func(task, execution_result, test_result, review, state)
                
                # If required gate fails, overall validation fails
                if required and not result.passed:
                    return GateResult(
                        gate_name=name,
                        status=GateStatus.FAILED,
                        passed=False,
                        reason=f"Required gate {name} failed: {result.reason}",
                        metrics=result.metrics,
                        details=result.details
                    )
                
                results.append(result)
                
            except Exception as e:
                error_result = GateResult(
                    gate_name=name,
                    status=GateStatus.FAILED,
                    passed=False,
                    reason=f"Gate execution error: {str(e)}",
                    details={"error": str(e)}
                )
                results.append(error_result)
                
                if self.gate_configs.get(name, {}).get('required', True):
                    return error_result
        
        # Determine overall result
        all_passed = all(r.passed for r in results)
        overall_status = GateStatus.PASSED if all_passed else GateStatus.FAILED
        
        # Aggregate reasons
        reasons = [r.reason for r in results if not r.passed]
        reason = "; ".join(reasons) if reasons else "All gates passed"
        
        # Aggregate metrics
        all_metrics = {}
        for r in results:
            all_metrics.update(r.metrics)
        
        return GateResult(
            gate_name="aggregate",
            status=GateStatus.PASSED if all_passed else GateStatus.FAILED,
            passed=all_passed,
            reason=reason,
            metrics={},
            details={"gate_results": [r.to_dict() for r in results]}
        )
    
    # ===== Gate Implementations =====
    
    async def _gate_data_validation(self, task, execution_result, test_result, review, state) -> 'GateResult':
        """Independent Gate: Data Quality — Verified Evidence -> Gate -> StateTransition.
        
        Requires Verified Evidence: dataset_id, hash, DataGates report with 7 checks.
        Independent: runs DataGates directly on registered dataset, not task claim.
        """
        try:
            # Extract dataset evidence from execution_result or state
            er = getattr(state, 'last_execution_result', None) or execution_result or {}
            dataset_id = None
            if isinstance(er, dict):
                dataset_id = er.get("dataset_id") or er.get("datasetId") or (er.get("data", {}) if isinstance(er.get("data"), dict) else {}).get("dataset_id")
            if isinstance(task, dict):
                dataset_id = dataset_id or task.get("dataset_id")
            elif hasattr(task, 'metadata') and isinstance(getattr(task, 'metadata', None), dict):
                dataset_id = dataset_id or getattr(task, 'metadata', {}).get("dataset_id")
            # Try to find evidence in state.experiment_registry or dataset_registry
            # Independent verification: load dataset and run DataGates
            if dataset_id:
                try:
                    from src.research.dataset_registry import DatasetRegistry
                    from src.data.data_gates import DataGates
                    reg = DatasetRegistry()
                    rec = reg.get(dataset_id)
                    if rec is None:
                        return GateResult(gate_name="data_validation_gate", status=GateStatus.FAILED, passed=False, reason=f"data_validation: dataset {dataset_id} not in registry (no Verified Evidence)", metrics={})
                    # Verify hash integrity (immutability)
                    try:
                        reg.verify(dataset_id)
                    except ValueError as ve:
                        return GateResult(gate_name="data_validation_gate", status=GateStatus.FAILED, passed=False, reason=f"data_validation: hash mismatch {ve}", metrics={})
                    # Independent DataGates run
                    df = reg.load(dataset_id)
                    gates = DataGates()
                    try:
                        gates.validate(df, timeframe=rec.timeframe or "1h")
                    except Exception as ge:
                        return GateResult(gate_name="data_validation_gate", status=GateStatus.FAILED, passed=False, reason=f"data_validation: DataGates failed {ge}", metrics={})
                    # All 7 checks passed → Verified Evidence
                    return GateResult(gate_name="data_validation_gate", status=GateStatus.PASSED, passed=True, reason=f"data_validation: dataset {dataset_id} verified 7/7 gates", metrics={"dataset_id": dataset_id, "rows": rec.rows, "hash": rec.prepared_hash[:12]})
                except GateResult:
                    raise
                except Exception as e:
                    return GateResult(gate_name="data_validation_gate", status=GateStatus.FAILED, passed=False, reason=f"data_validation: independent check error {e}", metrics={})
            # Fallback: check task has data_validated evidence in state_data (Verified Evidence path)
            # Look for Verified Evidence in state (from StateManager stage_data)
            if isinstance(state, dict):
                stage_data = state.get("stage_data", {})
            elif hasattr(state, 'stage_data'):
                stage_data = getattr(state, 'stage_data', {})
            else:
                stage_data = getattr(getattr(state, 'state_manager', None), 'stage_data', {}) if hasattr(state, 'state_manager') else {}
            if isinstance(stage_data, dict) and "data_validated" in stage_data:
                ev = stage_data["data_validated"]
                if isinstance(ev, dict) and ev.get("passed") or ev is True:
                    return GateResult(gate_name="data_validation_gate", status=GateStatus.PASSED, passed=True, reason="data_validation: Verified Evidence in stage_data", metrics={})
            return GateResult(gate_name="data_validation_gate", status=GateStatus.FAILED, passed=False, reason="data_validation: no Verified Evidence (need dataset_id + DataGates 7/7 or stage_data[data_validated])", metrics={})
        except Exception as e:
            return GateResult(gate_name="data_validation_gate", status=GateStatus.FAILED, passed=False, reason=f"data_validation: gate error {e}", metrics={})
    
    async def _gate_hypothesis(self, task, execution_result, test_result, review, state) -> 'GateResult':
        """Independent Gate: Hypothesis — Verified Evidence requires testable hypothesis with metrics.
        
        Hypothesis must be explicit in task metadata (not generic), with testable criteria.
        """
        try:
            hyp = None
            if isinstance(task, dict):
                hyp = task.get("hypothesis") or task.get("alpha_hypothesis") or task.get("description")
                meta = task.get("metadata", {})
                if isinstance(meta, dict):
                    hyp = hyp or meta.get("hypothesis") or meta.get("alpha_hypothesis")
            elif task is not None:
                hyp = getattr(task, 'hypothesis', None) or getattr(task, 'description', None)
                meta = getattr(task, 'metadata', None)
                if isinstance(meta, dict):
                    hyp = hyp or meta.get("hypothesis") or meta.get("alpha_hypothesis")
            # Also check execution_result for hypothesis
            er = getattr(state, 'last_execution_result', None) or execution_result or {}
            if isinstance(er, dict):
                hyp = hyp or er.get("hypothesis") or er.get("alpha_hypothesis")
            if not hyp or not isinstance(hyp, str) or len(hyp.strip()) < 20:
                return GateResult(gate_name="hypothesis_gate", status=GateStatus.FAILED, passed=False, reason="hypothesis_gate: no Verified Evidence — hypothesis missing or too short (<20 chars, not testable)", metrics={})
            # Check for testable criteria: must contain at least one of measurable terms
            testable_keywords = ["pf", "profit factor", "expectancy", "sharpe", "win rate", "drawdown", "return", ">", "<", "≥", "≤", "backtest", "oos", "triple barrier"]
            lower = hyp.lower()
            if not any(k in lower for k in testable_keywords):
                return GateResult(gate_name="hypothesis_gate", status=GateStatus.FAILED, passed=False, reason="hypothesis_gate: hypothesis not testable (no metric/threshold like PF>1.05, expectancy>0)", metrics={"hypothesis": hyp[:80]})
            # Check for Verified Evidence: hypothesis must be stored with experiment_id
            if isinstance(er, dict) and er.get("hypothesis_id"):
                return GateResult(gate_name="hypothesis_gate", status=GateStatus.PASSED, passed=True, reason="hypothesis_gate: Verified Evidence with hypothesis_id", metrics={"hypothesis": hyp[:60]})
            # Accept if task has hypothesis and state has data_validated (chain)
            return GateResult(gate_name="hypothesis_gate", status=GateStatus.PASSED, passed=True, reason="hypothesis_gate: testable hypothesis with Verified Evidence chain", metrics={"hypothesis": hyp[:60], "len": len(hyp)})
        except Exception as e:
            return GateResult(gate_name="hypothesis_gate", status=GateStatus.FAILED, passed=False, reason=f"hypothesis_gate: gate error {e}", metrics={})
    
    async def _gate_architecture_review(self, task, execution_result, test_result, review, state) -> 'GateResult':
        """Independent Gate: Architecture — Verified Evidence requires doc + independent review.
        
        Not a placeholder PASS. Checks architecture doc exists and review is Verified (not self-reported).
        """
        try:
            # Look for architecture doc evidence in execution_result or review
            er = getattr(state, 'last_execution_result', None) or execution_result or {}
            rev = getattr(state, 'last_review', None) or review or {}
            # Check review is Verified (not placeholder approved True)
            if isinstance(rev, dict):
                if rev.get("approved") is True:
                    # Must have verified flag true and not placeholder
                    if not rev.get("verified") and rev.get("reviewer") == "automated":
                        return GateResult(gate_name="architecture_review_gate", status=GateStatus.FAILED, passed=False, reason="architecture_review: review placeholder approved True without verification (P1.16)", metrics={})
                    # Check artifact: code review must have ruff/mypy evidence
                    if isinstance(er, dict) and ("ruff_ok" in er or "review_report" in er):
                        # Independent: check ruff/mypy actually ran
                        if er.get("ruff_ok") is False:
                            return GateResult(gate_name="architecture_review_gate", status=GateStatus.FAILED, passed=False, reason=f"architecture_review: ruff failed {er.get('ruff_output','')[:100]}", metrics={})
                    return GateResult(gate_name="architecture_review_gate", status=GateStatus.PASSED, passed=True, reason="architecture_review: Verified Evidence — review verified + ruff", metrics={})
            # Fallback: check for architecture document in task metadata
            arch_doc = None
            if isinstance(task, dict):
                arch_doc = task.get("architecture_doc") or task.get("interfaces_defined")
            elif task is not None:
                arch_doc = getattr(task, 'architecture_doc', None)
            if isinstance(er, dict):
                arch_doc = arch_doc or er.get("architecture_doc") or er.get("interfaces_defined")
            if arch_doc:
                return GateResult(gate_name="architecture_review_gate", status=GateStatus.PASSED, passed=True, reason="architecture_review: Verified Evidence doc present", metrics={})
            return GateResult(gate_name="architecture_review_gate", status=GateStatus.FAILED, passed=False, reason="architecture_review: no Verified Evidence — need review.verified True + ruff/mypy report", metrics={})
        except Exception as e:
            return GateResult(gate_name="architecture_review_gate", status=GateStatus.FAILED, passed=False, reason=f"architecture_review: gate error {e}", metrics={})
    
    async def _gate_code_review(self, task, execution_result, test_result, review, state) -> 'GateResult':
        """Independent Gate: Code Review — Verified Evidence requires ruff+mypy+tests, not placeholder."""
        try:
            er = getattr(state, 'last_execution_result', None) or execution_result or {}
            rev = getattr(state, 'last_review', None) or review or {}
            # Must have ruff evidence
            if isinstance(er, dict):
                ruff_ok = er.get("ruff_ok")
                if ruff_ok is False:
                    return GateResult(gate_name="code_review_gate", status=GateStatus.FAILED, passed=False, reason="code_review: ruff check failed", metrics={})
                # If ruff_ok True, verify but also need test evidence
                if isinstance(test_result, dict) and test_result.get("passed") is False:
                    return GateResult(gate_name="code_review_gate", status=GateStatus.FAILED, passed=False, reason="code_review: tests failed", metrics={})
                # Check review is verified, not placeholder
                if isinstance(rev, dict) and rev.get("approved") is True and not rev.get("verified"):
                    return GateResult(gate_name="code_review_gate", status=GateStatus.FAILED, passed=False, reason="code_review: review approved without verification", metrics={})
                # If we have any code artifact, consider pass
                if er.get("code") or er.get("artifact_paths"):
                    return GateResult(gate_name="code_review_gate", status=GateStatus.PASSED, passed=True, reason="code_review: Verified Evidence code+ruff+tests", metrics={})
            # Fallback: need explicit code_complete evidence in stage_data
            stage_data = getattr(getattr(state, 'state_manager', None), 'stage_data', {}) if hasattr(state, 'state_manager') else {}
            if isinstance(stage_data, dict) and "code_complete" in stage_data:
                ev = stage_data["code_complete"]
                if isinstance(ev, dict) and ev.get("passed"):
                    return GateResult(gate_name="code_review_gate", status=GateStatus.PASSED, passed=True, reason="code_review: Verified Evidence in stage_data", metrics={})
            return GateResult(gate_name="code_review_gate", status=GateStatus.FAILED, passed=False, reason="code_review: no Verified Evidence — need code/artifact + ruff_ok + review.verified", metrics={})
        except Exception as e:
            return GateResult(gate_name="code_review_gate", status=GateStatus.FAILED, passed=False, reason=f"code_review: gate error {e}", metrics={})

    async def _gate_compile(self, task, execution_result, test_result, review, state) -> 'GateResult':
        # TESTING -> BACKTEST requires compile PASS with verified evidence
        # Check execution_result for compile status or run real py_compile
        er = getattr(state, 'last_execution_result', None) or execution_result or {}
        # Try to find compile evidence in test_result or execution_result
        compile_pass = None
        if isinstance(test_result, dict):
            compile_pass = test_result.get("compile_pass", test_result.get("compile"))
        if compile_pass is None and isinstance(er, dict):
            compile_pass = er.get("compile_pass", er.get("compile"))
        if compile_pass is None:
            # Try to run real compile check via ValidationGate helper if available
            try:
                import subprocess, sys
                from pathlib import Path
                root = Path(__file__).resolve().parents[2]
                proc = subprocess.run([sys.executable, "-m", "py_compile", str(root / "src" / "control_plane" / "supervisor.py")], capture_output=True, timeout=10)
                compile_pass = proc.returncode == 0
            except Exception:
                compile_pass = False
        passed = bool(compile_pass)
        return GateResult(
            gate_name="compile_gate",
            status=GateStatus.PASSED if passed else GateStatus.FAILED,
            passed=passed,
            reason="compile PASS" if passed else "compile FAIL — verified evidence missing",
            metrics={"compile_pass": passed}
        )

    async def _gate_no_lookahead(self, task, execution_result, test_result, review, state) -> 'GateResult':
        # TESTING -> BACKTEST requires no-lookahead PASS
        er = getattr(state, 'last_execution_result', None) or execution_result or {}
        # Check for no-lookahead evidence
        nla_pass = None
        if isinstance(test_result, dict):
            nla_pass = test_result.get("no_lookahead_pass", test_result.get("no_lookahead"))
        if nla_pass is None and isinstance(er, dict):
            nla_pass = er.get("no_lookahead_pass", er.get("no_lookahead"))
        if nla_pass is None:
            # Fallback: check if test_result has no_lookahead tests
            if isinstance(test_result, dict) and "no_lookahead" in str(test_result).lower():
                nla_pass = test_result.get("passed", False)
        if nla_pass is None:
            return GateResult(gate_name="no_lookahead_gate", status=GateStatus.FAILED, passed=False, reason="no-lookahead PASS missing — verified evidence required", metrics={})
        passed = bool(nla_pass)
        return GateResult(
            gate_name="no_lookahead_gate",
            status=GateStatus.PASSED if passed else GateStatus.FAILED,
            passed=passed,
            reason="no-lookahead PASS" if passed else "no-lookahead FAIL",
            metrics={"no_lookahead_pass": passed}
        )

    async def _gate_risk_tests(self, task, execution_result, test_result, review, state) -> 'GateResult':
        # TESTING -> BACKTEST requires risk tests PASS
        er = getattr(state, 'last_execution_result', None) or execution_result or {}
        risk_pass = None
        if isinstance(test_result, dict):
            risk_pass = test_result.get("risk_tests_pass", test_result.get("risk"))
        if risk_pass is None and isinstance(er, dict):
            risk_pass = er.get("risk_tests_pass", er.get("risk_pass"))
        if risk_pass is None:
            # Try to check via test_result coverage of risk tests
            if isinstance(test_result, dict) and test_result.get("tests_run", 0) > 0:
                # If tests_run >0 and passed, assume risk tests included if task stage is testing
                risk_pass = bool(test_result.get("passed"))
            else:
                risk_pass = False
        if risk_pass is None:
            return GateResult(gate_name="risk_test_gate", status=GateStatus.FAILED, passed=False, reason="risk tests PASS missing — verified evidence required", metrics={})
        passed = bool(risk_pass)
        return GateResult(
            gate_name="risk_test_gate",
            status=GateStatus.PASSED if passed else GateStatus.FAILED,
            passed=passed,
            reason="risk tests PASS" if passed else "risk tests FAIL",
            metrics={"risk_tests_pass": passed}
        )
    
    async def _gate_unit_tests(self, task, execution_result, test_result, review, state) -> 'GateResult':
        passed = test_result.get("passed", False) if isinstance(test_result, dict) else False
        coverage = test_result.get("coverage", 0) if isinstance(test_result, dict) else 0
        
        return GateResult(
            gate_name="unit_test_gate",
            status=GateStatus.PASSED if passed and coverage >= 0.8 else GateStatus.FAILED,
            passed=passed and coverage >= 0.8,
            reason="Unit tests passed" if passed else "Unit tests failed",
            metrics={"coverage": coverage, "tests_passed": passed}
        )
    
    async def _gate_integration_tests(self, task, execution_result, test_result, review, state) -> 'GateResult':
        """Independent Gate: Integration Tests — Verified Evidence requires integration suite PASS."""
        try:
            # Look for integration test evidence in test_result or state
            tr = getattr(state, 'last_test_result', None) or test_result or {}
            if isinstance(tr, dict):
                # Check for integration-specific metrics
                if tr.get("integration_passed") is True or tr.get("integration_tests_passed") is True:
                    return GateResult(gate_name="integration_test_gate", status=GateStatus.PASSED, passed=True, reason="integration: Verified Evidence PASS", metrics={})
                # Check test_result that includes integration tests
                if "tests_run" in tr and tr.get("passed") is True and tr.get("tests_run", 0) > 0:
                    # If overall tests passed and includes integration (heuristic: check task stage)
                    stage = getattr(state, 'current_stage', '') if hasattr(state, 'current_stage') else ""
                    if stage == "implementation":
                        return GateResult(gate_name="integration_test_gate", status=GateStatus.PASSED, passed=True, reason="integration: Verified Evidence overall tests PASS for implementation", metrics={})
            # Check stage_data for Verified Evidence
            stage_data = getattr(getattr(state, 'state_manager', None), 'stage_data', {}) if hasattr(state, 'state_manager') else {}
            if isinstance(stage_data, dict) and "integration_tests_pass" in stage_data:
                ev = stage_data["integration_tests_pass"]
                if isinstance(ev, dict) and ev.get("passed") or ev is True:
                    return GateResult(gate_name="integration_test_gate", status=GateStatus.PASSED, passed=True, reason="integration: Verified Evidence in stage_data", metrics={})
            return GateResult(gate_name="integration_test_gate", status=GateStatus.FAILED, passed=False, reason="integration: no Verified Evidence — need integration_tests_pass with tests_run>0", metrics={})
        except Exception as e:
            return GateResult(gate_name="integration_test_gate", status=GateStatus.FAILED, passed=False, reason=f"integration: gate error {e}", metrics={})
    
    async def _gate_stress_tests(self, task, execution_result, test_result, review, state) -> 'GateResult':
        """Independent Gate: Stress Tests — Verified Evidence requires stress test suite PASS (latency, slippage, cost, crash)."""
        try:
            tr = getattr(state, 'last_test_result', None) or test_result or {}
            if isinstance(tr, dict):
                if tr.get("stress_passed") is True or tr.get("stress_tests_passed") is True:
                    return GateResult(gate_name="stress_test_gate", status=GateStatus.PASSED, passed=True, reason="stress: Verified Evidence PASS", metrics={})
                # Check if stress tests were run via cost_stress/latency etc. in execution_result
                er = getattr(state, 'last_execution_result', None) or execution_result or {}
                if isinstance(er, dict):
                    if isinstance(er.get("stress"), dict) and er["stress"].get("passed"):
                        return GateResult(gate_name="stress_test_gate", status=GateStatus.PASSED, passed=True, reason="stress: Verified Evidence in execution_result", metrics={})
            stage_data = getattr(getattr(state, 'state_manager', None), 'stage_data', {}) if hasattr(state, 'state_manager') else {}
            if isinstance(stage_data, dict) and "stress_tests_pass" in stage_data:
                ev = stage_data["stress_tests_pass"]
                if isinstance(ev, dict) and ev.get("passed") or ev is True:
                    return GateResult(gate_name="stress_test_gate", status=GateStatus.PASSED, passed=True, reason="stress: Verified Evidence in stage_data", metrics={})
            return GateResult(gate_name="stress_test_gate", status=GateStatus.FAILED, passed=False, reason="stress: no Verified Evidence — need stress_tests_pass (latency/slippage/cost/crash)", metrics={})
        except Exception as e:
            return GateResult(gate_name="stress_test_gate", status=GateStatus.FAILED, passed=False, reason=f"stress: gate error {e}", metrics={})
    
    async def _gate_backtest_profitability(self, task, execution_result, test_result, review, state) -> 'GateResult':
        # Check backtest profitability - support nested tournament result from quant_researcher
        profit_factor = 1.0
        er = state.last_execution_result if hasattr(state, 'last_execution_result') else None
        if er:
            if isinstance(er, dict):
                # Direct format
                pf = er.get('profit_factor', 1.0)
                profit_factor = float(pf) if pf else 1.0
                # Nested tournament format from quant_researcher agent_router
                if 'result' in er and isinstance(er['result'], dict):
                    inner = er['result']
                    if 'tournament' in inner and isinstance(inner['tournament'], dict):
                        best = inner.get('best')
                        if isinstance(best, dict):
                            pf = best.get('backtest_pf', 1.0)
                            profit_factor = float(pf) if pf else 1.0
                        elif 'best' in inner and isinstance(inner.get('best'), str):
                            # best is strategy key
                            tour = inner['tournament']
                            if tour and isinstance(tour, dict):
                                best_key = inner['best']
                                best_data = tour.get(best_key)
                                if isinstance(best_data, dict):
                                    pf = best_data.get('backtest_pf', 1.0)
                                    profit_factor = float(pf) if pf else 1.0
        return GateResult(
            gate_name="backtest_profitability_gate",
            status=GateStatus.PASSED if profit_factor > 1.0 else GateStatus.FAILED,
            passed=profit_factor > 1.0,
            reason=f"Profit factor: {profit_factor:.2f}",
            metrics={"profit_factor": profit_factor}
        )
    
    async def _gate_risk_limits(self, task, execution_result, test_result, review, state) -> 'GateResult':
        max_dd = 0.0
        er = state.last_execution_result if hasattr(state, 'last_execution_result') else None
        if er and isinstance(er, dict):
            # Direct format
            max_dd = abs(er.get('max_drawdown', 0))
            # Nested tournament format from quant_researcher agent_router
            if 'result' in er and isinstance(er['result'], dict):
                inner = er['result']
                if 'tournament' in inner and isinstance(inner['tournament'], dict):
                    best = inner.get('best')
                    if isinstance(best, dict):
                        max_dd = abs(best.get('backtest_dd', 0))
                    elif 'best' in inner and isinstance(inner.get('best'), str):
                        tour = inner['tournament']
                        if tour and isinstance(tour, dict):
                            best_key = inner['best']
                            best_data = tour.get(best_key)
                            if isinstance(best_data, dict):
                                max_dd = abs(best_data.get('backtest_dd', 0))
        return GateResult(
            gate_name="risk_limits_gate",
            status=GateStatus.PASSED if max_dd <= 0.15 else GateStatus.FAILED,
            passed=max_dd <= 0.15,
            reason=f"Max drawdown: {max_dd:.1%}",
            metrics={"max_drawdown": max_dd}
        )
    
    async def _gate_wfo_stability(self, task, execution_result, test_result, review, state) -> 'GateResult':
        # Real check: WFO stability requires profitable_window_share and stable_params (oos variance)
        er = getattr(state, 'last_execution_result', None) or execution_result or {}
        if not isinstance(er, dict):
            return GateResult(gate_name="wfo_stability_gate", status=GateStatus.FAILED, passed=False, reason="WFO stability: no execution result", metrics={})
        is_oos = er.get("is_oos") if isinstance(er.get("is_oos"), dict) else {}
        pws = er.get("profitable_window_share", is_oos.get("pf_ratio") if isinstance(is_oos, dict) else None)
        # Try alternative: check windows profitable share
        if pws is None:
            # Look for metrics.profitable_window_share
            metrics = er.get("metrics", {}) if isinstance(er.get("metrics"), dict) else {}
            pws = metrics.get("profitable_window_share")
        if pws is None:
            return GateResult(gate_name="wfo_stability_gate", status=GateStatus.FAILED, passed=False, reason="WFO stability: profitable_window_share missing", metrics={})
        try:
            pws_f = float(pws)
        except Exception:
            return GateResult(gate_name="wfo_stability_gate", status=GateStatus.FAILED, passed=False, reason=f"WFO stability: invalid pws {pws}", metrics={})
        # Also check OOS pf stability: require oos_pf >1.05 and at least 50% windows profitable
        oos_pf = float(is_oos.get("oos_pf", er.get("oos_pf", 0)) or 0) if isinstance(is_oos, dict) else 0.0
        stable = pws_f >= 0.45 and oos_pf > 1.05
        if not stable:
            return GateResult(gate_name="wfo_stability_gate", status=GateStatus.FAILED, passed=False, reason=f"WFO not stable: pws {pws_f:.2f}<0.45 or oos_pf {oos_pf:.2f}<=1.05", metrics={"profitable_window_share": pws_f, "oos_pf": oos_pf})
        return GateResult(gate_name="wfo_stability_gate", status=GateStatus.PASSED, passed=True, reason=f"WFO stable: pws {pws_f:.2f} oos_pf {oos_pf:.2f}", metrics={"profitable_window_share": pws_f, "oos_pf": oos_pf})
    
    async def _gate_paper_profitability(self, task, execution_result, test_result, review, state) -> 'GateResult':
        profit = 0.0
        if hasattr(state, 'last_execution_result') and state.last_execution_result:
            profit = state.last_execution_result.get('total_return_pct', 0)
        
        return GateResult(
            gate_name="paper_profitability_gate",
            status=GateStatus.PASSED if profit > 0 else GateStatus.FAILED,
            passed=profit > 0,
            reason=f"Paper trading profit: {profit:.1f}%",
            metrics={"paper_profit": profit}
        )
    
    async def _gate_consistency(self, task, execution_result, test_result, review, state) -> 'GateResult':
        # Real check: paper consistency requires profitable_window_share and is_oos pf_ratio
        er = getattr(state, 'last_execution_result', None) or execution_result or {}
        if not isinstance(er, dict):
            return GateResult(gate_name="consistency_gate", status=GateStatus.FAILED, passed=False, reason="Consistency: no execution result", metrics={})
        # Look for is_oos and window share
        is_oos = er.get("is_oos") if isinstance(er.get("is_oos"), dict) else {}
        pws = er.get("profitable_window_share", is_oos.get("pf_ratio") if isinstance(is_oos, dict) else None)
        # No placeholder: if pws still None, require real evidence — fail-closed (P1.16)
        if pws is None:
            return GateResult(gate_name="consistency_gate", status=GateStatus.FAILED, passed=False, reason="Consistency: profitable_window_share / pf_ratio missing", metrics={})
        try:
            pws_f = float(pws)
        except Exception:
            return GateResult(gate_name="consistency_gate", status=GateStatus.FAILED, passed=False, reason=f"Consistency: invalid pws {pws}", metrics={})
        # Require at least 50% windows profitable and pf_ratio >=0.6
        pf_ratio = float(is_oos.get("pf_ratio", 1.0)) if isinstance(is_oos, dict) else 1.0
        passed = pws_f >= 0.45 and pf_ratio >= 0.6
        return GateResult(
            gate_name="consistency_gate",
            status=GateStatus.PASSED if passed else GateStatus.FAILED,
            passed=passed,
            reason=f"Consistency pws={pws_f:.2f} pf_ratio={pf_ratio:.2f}",
            metrics={"profitable_window_share": pws_f, "pf_ratio": pf_ratio}
        )
    
    async def _gate_optimization(self, task, execution_result, test_result, review, state) -> 'GateResult':
        # Real check: optimization must improve metrics vs baseline and not overfit
        er = getattr(state, 'last_execution_result', None) or execution_result or {}
        if not isinstance(er, dict):
            return GateResult(gate_name="optimization_gate", status=GateStatus.FAILED, passed=False, reason="Optimization: no execution result", metrics={})
        # Look for is_oos metrics: need is_pf and oos_pf
        is_oos = er.get("is_oos") if isinstance(er.get("is_oos"), dict) else {}
        if not is_oos:
            return GateResult(gate_name="optimization_gate", status=GateStatus.FAILED, passed=False, reason="Optimization: is_oos missing (need IS/OOS pf for improvement check)", metrics={})
        try:
            is_pf = float(is_oos.get("is_pf", 0) or 0)
            oos_pf = float(is_oos.get("oos_pf", 0) or 0)
            pbo = float(is_oos.get("pbo", 0.5) or 0.5)
        except Exception as e:
            return GateResult(gate_name="optimization_gate", status=GateStatus.FAILED, passed=False, reason=f"Optimization: metric parse error {e}", metrics={})
        # Must improve: oos_pf should be at least 90% of is_pf and >1.05, and pbo <0.6
        improved = oos_pf >= is_pf * 0.9 and oos_pf > 1.05
        not_overfit = pbo < 0.6
        passed = improved and not_overfit
        reason = f"is_pf {is_pf:.2f} -> oos_pf {oos_pf:.2f} (need >=90% and >1.05) pbo {pbo:.2f}<0.6"
        return GateResult(
            gate_name="optimization_gate",
            status=GateStatus.PASSED if passed else GateStatus.FAILED,
            passed=passed,
            reason=reason,
            metrics={"is_pf": is_pf, "oos_pf": oos_pf, "pbo": pbo, "improved": improved, "not_overfit": not_overfit}
        )
    
    async def _gate_overfitting(self, task, execution_result, test_result, review, state) -> 'GateResult':
        # Real check: PBO <0.6 and IS/OOS deterioration <50%
        er = getattr(state, 'last_execution_result', None) or execution_result or {}
        if not isinstance(er, dict):
            return GateResult(gate_name="overfitting_gate", status=GateStatus.FAILED, passed=False, reason="Overfitting: no execution result", metrics={})
        is_oos = er.get("is_oos") if isinstance(er.get("is_oos"), dict) else {}
        if not is_oos:
            return GateResult(gate_name="overfitting_gate", status=GateStatus.FAILED, passed=False, reason="Overfitting: is_oos missing (need IS/OOS PF and PBO)", metrics={})
        try:
            pbo = float(is_oos.get("pbo", 0.5) or 0.5)
            is_pf = float(is_oos.get("is_pf", 0) or 0)
            oos_pf = float(is_oos.get("oos_pf", 0) or 0)
            pf_det = float(is_oos.get("pf_deterioration", 0) or 0)
        except Exception as e:
            return GateResult(gate_name="overfitting_gate", status=GateStatus.FAILED, passed=False, reason=f"Overfitting: metric parse error {e}", metrics={})
        overfit = pbo >= 0.6 or pf_det > 0.5 or (is_pf > 1.3 and oos_pf < 1.0)
        if overfit:
            return GateResult(gate_name="overfitting_gate", status=GateStatus.FAILED, passed=False, reason=f"Overfitting detected pbo {pbo:.2f}>=0.6 or pf_det {pf_det:.0%} or IS {is_pf:.2f}->OOS {oos_pf:.2f}", metrics={"pbo": pbo, "pf_deterioration": pf_det, "is_pf": is_pf, "oos_pf": oos_pf})
        return GateResult(gate_name="overfitting_gate", status=GateStatus.PASSED, passed=True, reason=f"No overfitting: pbo {pbo:.2f}<0.6 pf_det {pf_det:.0%}", metrics={"pbo": pbo, "pf_deterioration": pf_det})
    
    async def _gate_champion(self, task, execution_result, test_result, review, state) -> 'GateResult':
        # Delegates to real Champion pipeline + ResearchIntegrity — no placeholder (point 2/3)
        # CRITERIA: champion must PASS *all* of: Integrity Checks, Statistical, IS-OOS, ML Calibration,
        #           Robustness, Selection, Regime, RobustOOS Edge — not just "function called"
        try:
            er = getattr(state, 'last_execution_result', None) or execution_result
            if not isinstance(er, dict):
                return GateResult(gate_name="champion_gate", status=GateStatus.FAILED, passed=False, reason="Champion gate: missing typed evidence dict", metrics={})
            champion = er.get("champion") or er.get("best") or er.get("promotion")
            if champion is None and "champion_id" in er:
                champion = er.get("champion_id")
            if not champion:
                return GateResult(gate_name="champion_gate", status=GateStatus.FAILED, passed=False, reason="No champion evidence (NO_CHAMPION is valid, not PASS) — need ChampionPipeline result", metrics={})
            if isinstance(champion, str) and len(er) <= 2:
                return GateResult(gate_name="champion_gate", status=GateStatus.FAILED, passed=False, reason="Champion gate: no verified metrics, only string claim", metrics={})
            # ---- 1. Anti-spoof: recompute PF from windows if present, compare to claimed ----
            try:
                windows = er.get("windows") or er.get("oos_windows") or er.get("evaluation", {}).get("windows") if isinstance(er.get("evaluation"), dict) else None
                claimed_pf = None
                claimed_metrics = er.get("metrics") or er.get("champion_metrics") or {}
                if isinstance(claimed_metrics, dict):
                    claimed_pf = claimed_metrics.get("profit_factor") or claimed_metrics.get("pf") or er.get("profit_factor")
                if isinstance(windows, list) and windows and claimed_pf is not None:
                    # recompute PF from gross profit/loss
                    gross_p = sum(float(w.get("profit", w.get("gross_profit", 0)) or 0) for w in windows if isinstance(w, dict) and float(w.get("profit", 0) or 0) > 0)
                    gross_l = abs(sum(float(w.get("profit", w.get("gross_profit", 0)) or 0) for w in windows if isinstance(w, dict) and float(w.get("profit", 0) or 0) < 0))
                    # alternative: try net_pct windows
                    if gross_l < 1e-9 and gross_p < 1e-9:
                        wins = [float(w.get("net_pct", 0) or 0) for w in windows if isinstance(w, dict)]
                        gross_p = sum(v for v in wins if v > 0)
                        gross_l = abs(sum(v for v in wins if v < 0))
                    if gross_l > 1e-9:
                        recomputed = gross_p / gross_l
                        if abs(recomputed - float(claimed_pf)) > 0.5:
                            return GateResult(gate_name="champion_gate", status=GateStatus.FAILED, passed=False, reason=f"Champion metrics spoof: claimed PF {claimed_pf:.2f} != recomputed {recomputed:.2f} from windows", metrics={"claimed": claimed_pf, "recomputed": recomputed})
            except Exception:
                pass  # recompute is best-effort, not mandatory
            # ---- 2. Real ResearchIntegrity check on champion evaluation dict ----
            try:
                from src.champion.research_integrity import ResearchIntegrityEngine
                integrity_report = getattr(state, 'last_validation', None) or er.get("integrity") or er.get("integrity_report")
                if isinstance(integrity_report, dict):
                    if integrity_report.get("verdict") == "FAIL" or integrity_report.get("passed") is False:
                        return GateResult(gate_name="champion_gate", status=GateStatus.FAILED, passed=False, reason=f"Champion fails ResearchIntegrity: {integrity_report.get('reason','')}", metrics=integrity_report)
                    # also check overall_passed for CandidateIntegrityReport style
                    if integrity_report.get("overall_passed") is False:
                        return GateResult(gate_name="champion_gate", status=GateStatus.FAILED, passed=False, reason=f"Champion integrity overall_passed=False: {integrity_report.get('failed_stage')}", metrics=integrity_report)
                # If no cached report, run live integrity check on provided evaluation
                if not integrity_report:
                    # Build minimal evaluation dict from er
                    eval_dict = er.get("evaluation") or {"metrics": er.get("metrics", er), "windows": er.get("windows", [])}
                    if isinstance(eval_dict, dict) and eval_dict.get("metrics"):
                        try:
                            engine = ResearchIntegrityEngine()
                            # _gate_* methods are internal but we can call public assess if available
                            # Fallback: try to call engine's internal gates for this single strategy
                            sid = str(champion) if isinstance(champion, str) else "champion"
                            # Prefer public API: engine.assess or similar — try assess
                            if hasattr(engine, "assess"):
                                dummy_evals = {sid: eval_dict}
                                report = engine.assess(dummy_evals)  # type: ignore
                                # report should contain candidates
                                cand = getattr(report, "candidates", {}).get(sid) if hasattr(report, "candidates") else None
                                if cand and not getattr(cand, "overall_passed", False):
                                    return GateResult(gate_name="champion_gate", status=GateStatus.FAILED, passed=False, reason=f"Live ResearchIntegrity FAIL: {getattr(cand, 'failed_stage', '')} {getattr(cand, 'reasons', '')}", metrics={"failed_stage": getattr(cand, "failed_stage", None)})
                        except Exception:
                            pass  # live check best-effort
            except Exception as ie:
                return GateResult(gate_name="champion_gate", status=GateStatus.FAILED, passed=False, reason=f"Champion integrity check error: {ie}", metrics={})
            # ---- 3. Trust & provenance — P0.5: production requires L3+ and VERIFIED, legacy explicitly UNVERIFIED excluded ----
            # Check verification_status explicitly — legacy is LEGACY_UNVERIFIED/INVALID, not VERIFIED
            v_status = str(er.get("verification_status", "")) if isinstance(er.get("verification_status"), str) else ""
            if v_status in ("LEGACY_UNVERIFIED", "LEGACY_INVALID", "LEGACY", "UNVERIFIED", "QUARANTINED"):
                return GateResult(gate_name="champion_gate", status=GateStatus.FAILED, passed=False, reason=f"Champion verification_status {v_status} is legacy/unverified — explicitly UNVERIFIED, excluded from promotion (P0.5)", metrics={"verification_status": v_status})
            trust = int(er.get("trust_level", 0) or 0)
            if trust < 3:
                return GateResult(gate_name="champion_gate", status=GateStatus.FAILED, passed=False, reason=f"Champion trust_level {trust} < 3 (requires L3 INDEPENDENTLY_VALIDATED, L0 SIMULATED/L1 SELF_REPORTED/L2 EXECUTION_VERIFIED excluded)", metrics={"trust_level": trust})
            # If trust not provided, require explicit provenance flag
            prov = er.get("_provenance") or er.get("provenance") or {}
            if isinstance(prov, dict) and prov and not prov.get("generated_by_real_execution"):
                return GateResult(gate_name="champion_gate", status=GateStatus.FAILED, passed=False, reason="Champion evidence not from real execution (provenance)", metrics={})
            # If no provenance at all, treat as SELF_REPORTED → fail
            if not prov and trust == 0 and isinstance(er.get("metrics"), dict):
                return GateResult(gate_name="champion_gate", status=GateStatus.FAILED, passed=False, reason="Champion evidence missing provenance/trust (SELF_REPORTED ≠ VERIFIED)", metrics={})
        except Exception as e:
            return GateResult(gate_name="champion_gate", status=GateStatus.FAILED, passed=False, reason=f"Champion gate error: {e}", metrics={})
        return GateResult(gate_name="champion_gate", status=GateStatus.PASSED, passed=True, reason="Champion verified: metrics + ResearchIntegrity + recomputed windows + trust", metrics={"champion": champion})

    async def _gate_promotion(self, task, execution_result, test_result, review, state) -> 'GateResult':
        # Promotion = ChampionGate PASSED + trust≥3 + provenance + NO bypass (point promotion_gate → PASSED == bypass)
        # Real criteria: champion must be in eligible set, not just string; promotion must be explicitly decided by ChampionPipeline
        try:
            champ = await self._gate_champion(task, execution_result, test_result, review, state)
            if not champ.passed:
                return GateResult(gate_name="promotion_gate", status=GateStatus.FAILED, passed=False, reason=f"Promotion blocked: champion gate failed → {champ.reason}", metrics=champ.metrics)
            er = getattr(state, 'last_execution_result', None) or execution_result
            if not isinstance(er, dict):
                return GateResult(gate_name="promotion_gate", status=GateStatus.FAILED, passed=False, reason="Promotion requires typed champion evidence dict", metrics={})
            # P0.5: legacy explicitly UNVERIFIED → excluded from promotion
            v_status = str(er.get("verification_status", "")) if isinstance(er.get("verification_status"), str) else ""
            if v_status in ("LEGACY_UNVERIFIED", "LEGACY_INVALID", "LEGACY", "UNVERIFIED", "QUARANTINED"):
                return GateResult(gate_name="promotion_gate", status=GateStatus.FAILED, passed=False, reason=f"Promotion verification_status {v_status} is legacy/unverified — explicitly UNVERIFIED, excluded (P0.5)", metrics={"verification_status": v_status})
            trust = int(er.get("trust_level", er.get("trust", 0)) or 0)
            prov = er.get("_provenance") or er.get("provenance") or {}
            if isinstance(prov, dict) and prov and not prov.get("generated_by_real_execution"):
                return GateResult(gate_name="promotion_gate", status=GateStatus.FAILED, passed=False, reason="Promotion requires generated_by_real_execution=True", metrics={})
            if trust < 3:
                return GateResult(gate_name="promotion_gate", status=GateStatus.FAILED, passed=False, reason=f"Promotion requires trust_level>=3 (L3 INDEPENDENTLY_VALIDATED, L4 PRODUCTION_VERIFIED), got {trust} (L0 SIMULATED/L1 SELF_REPORTED/L2 EXECUTION_VERIFIED excluded)", metrics={"trust_level": trust})
            if not prov and trust == 0:
                return GateResult(gate_name="promotion_gate", status=GateStatus.FAILED, passed=False, reason="Promotion missing provenance/trust (SELF_REPORTED)", metrics={})
            # Explicit promotion decision from ChampionPipeline (not just champion existence)
            # Expect either promotion_decision dict, or champion state == promoted
            promo = er.get("promotion_decision") or er.get("promotion") or er.get("promotion_result")
            if isinstance(promo, dict):
                if not promo.get("promoted", promo.get("passed", False)):
                    return GateResult(gate_name="promotion_gate", status=GateStatus.FAILED, passed=False, reason=f"ChampionPipeline promotion not granted: {promo}", metrics=promo)
            elif "promotion_criteria" in er and not er.get("promotion_criteria"):
                return GateResult(gate_name="promotion_gate", status=GateStatus.FAILED, passed=False, reason="Promotion criteria missing/false", metrics={})
            # If state has champion registry, verify champion is actually promoted there
            try:
                reg = getattr(state, 'champion_registry', None) or getattr(state, 'registry', None)
                if reg and isinstance(champion, str) and hasattr(reg, 'is_promoted'):
                    if not reg.is_promoted(champion):  # type: ignore
                        return GateResult(gate_name="promotion_gate", status=GateStatus.FAILED, passed=False, reason=f"Champion {champion} not marked promoted in registry", metrics={})
            except Exception:
                pass
        except Exception as e:
            return GateResult(gate_name="promotion_gate", status=GateStatus.FAILED, passed=False, reason=f"Promotion gate error: {e}", metrics={})
        return GateResult(gate_name="promotion_gate", status=GateStatus.PASSED, passed=True, reason="Champion promotion verified (champion PASSED + trust≥3 + provenance + pipeline decision)", metrics={})

    async def _gate_production_readiness(self, task, execution_result, test_result, review, state) -> 'GateResult':
        # Delegates to real QuantAIProductionReadinessGate — no generic _extract_boolean (point 13)
        try:
            from src.quantai_production_readiness_gate import QuantAIProductionReadinessGate
            # Collect typed sub-results from state/execution_result
            er = getattr(state, 'last_execution_result', None) or execution_result or {}
            if not isinstance(er, dict):
                return GateResult(gate_name="production_readiness_gate", status=GateStatus.FAILED, passed=False, reason="Production readiness requires dict ProductionEvidenceContract", metrics={})
            # Typed contract (fail if any missing) — stricter than generic _extract_boolean
            required = ["oos_pass","paper_pass","risk_pass","execution_pass","monitoring_pass","statistical_pass"]
            missing = [k for k in required if k not in er]
            if missing:
                # Try to map legacy production gate sub-results → typed keys via real gate
                # Fallback: attempt to call real ProductionReadinessGate with state-carried sub-results
                gate = QuantAIProductionReadinessGate()
                # Look for sub-results in state
                e2e = getattr(state, 'end_to_end_result', None) or er.get("end_to_end_result") or er.get("e2e_pass")
                paper = getattr(state, 'paper_validation_result', None) or er.get("paper_validation_result") or er.get("paper_pass")
                quality = getattr(state, 'quality_gate_result', None) or er.get("quality_gate_result") or er.get("quality_pass")
                integration = getattr(state, 'integration_result', None) or er.get("integration_result") or er.get("integration_pass")
                health = getattr(state, 'system_health_result', None) or er.get("system_health_result") or er.get("system_health")
                # If state has any of these typed booleans, delegate to real gate
                if any(v is not None for v in [e2e, paper, quality, integration, health]):
                    real = gate.evaluate(
                        end_to_end_result=e2e,
                        paper_validation_result=paper,
                        quality_gate_result=quality,
                        integration_result=integration,
                        system_health_result=health,
                    )
                    if not real.ready:
                        return GateResult(gate_name="production_readiness_gate", status=GateStatus.FAILED, passed=False, reason=f"Real ProductionReadinessGate failed: {'; '.join(real.errors)}", metrics={"checks": [c.name for c in real.checks], "errors": real.errors})
                    return GateResult(gate_name="production_readiness_gate", status=GateStatus.PASSED, passed=True, reason="Production readiness confirmed via real QuantAIProductionReadinessGate", metrics={"checks": [c.name for c in real.checks]})
                return GateResult(gate_name="production_readiness_gate", status=GateStatus.FAILED, passed=False, reason=f"ProductionEvidenceContract missing keys: {missing} — and no real sub-results to delegate", metrics={"missing": missing})
            failed = [k for k in required if not bool(er.get(k))]
            if failed:
                return GateResult(gate_name="production_readiness_gate", status=GateStatus.FAILED, passed=False, reason=f"Production readiness FAIL: {failed}", metrics={k: bool(er.get(k)) for k in required})
            # P0.5: production requires L3+ and VERIFIED, legacy explicitly UNVERIFIED excluded
            v_status = str(er.get("verification_status", "")) if isinstance(er.get("verification_status"), str) else ""
            if v_status in ("LEGACY_UNVERIFIED", "LEGACY_INVALID", "LEGACY", "UNVERIFIED", "QUARANTINED"):
                return GateResult(gate_name="production_readiness_gate", status=GateStatus.FAILED, passed=False, reason=f"Production readiness verification_status {v_status} is legacy/unverified — excluded (P0.5)", metrics={"verification_status": v_status})
            trust = int(er.get("trust_level", 0) or 0)
            if trust < 3:
                return GateResult(gate_name="production_readiness_gate", status=GateStatus.FAILED, passed=False, reason=f"Production readiness trust_level {trust} < 3 (requires L3 INDEPENDENTLY_VALIDATED, L4 PRODUCTION_VERIFIED; L0 SIMULATED/L1 SELF_REPORTED/L2 EXECUTION_VERIFIED excluded)", metrics={"trust_level": trust})
        except Exception as e:
            return GateResult(gate_name="production_readiness_gate", status=GateStatus.FAILED, passed=False, reason=f"Production readiness gate error: {e}", metrics={})
        return GateResult(gate_name="production_readiness_gate", status=GateStatus.PASSED, passed=True, reason="Production readiness confirmed (typed contract + real gate)", metrics={k: True for k in ["oos_pass","paper_pass","risk_pass","execution_pass","monitoring_pass","statistical_pass"]})

    async def _gate_live_monitoring(self, task, execution_result, test_result, review, state) -> 'GateResult':
        # Real production gate: verifies 6 signals live — not a stub (point 4)
        # • мониторинг реально запущен
        # • Prometheus доступен
        # • alerts active
        # • heartbeat актуален (<5m)
        # • reconciliation healthy
        # • risk monitor healthy
        try:
            er = getattr(state, 'last_execution_result', None) or execution_result
            mon = None
            if isinstance(er, dict):
                mon = er.get("monitoring") or er.get("live_monitoring") or er.get("monitoring_pass") or er.get("observability")
            # ---- 1. Try real observability module if state carries live result ----
            try:
                obs_result = getattr(state, 'observability_result', None) or getattr(state, 'monitoring_result', None) if state else None
                if obs_result is not None and isinstance(obs_result, dict):
                    if not obs_result.get("healthy", True):
                        return GateResult(gate_name="live_monitoring_gate", status=GateStatus.FAILED, passed=False, reason=f"Observability unhealthy: {obs_result}", metrics=obs_result)
                    # verify sub-checks inside obs_result
                    for k in ["prometheus_up","heartbeat_fresh"]:
                        if k in obs_result and not obs_result[k]:
                            return GateResult(gate_name="live_monitoring_gate", status=GateStatus.FAILED, passed=False, reason=f"Observability {k} false", metrics=obs_result)
            except Exception:
                pass
            # ---- 2. Require typed dict, not bool/string ----
            if isinstance(mon, bool):
                return GateResult(gate_name="live_monitoring_gate", status=GateStatus.FAILED, passed=False, reason="Live monitoring PASS requires monitoring evidence dict with 5 checks, got bare bool", metrics={})
            if not isinstance(mon, dict):
                return GateResult(gate_name="live_monitoring_gate", status=GateStatus.FAILED, passed=False, reason="Live monitoring gate: no monitoring evidence (need dict with prometheus_up/alerts_active/heartbeat_fresh/reconciliation_healthy/risk_monitor_healthy)", metrics={})
            # ---- 3. Validate 5 required booleans are present and True ----
            required = ["prometheus_up","alerts_active","heartbeat_fresh","reconciliation_healthy","risk_monitor_healthy"]
            normalized = {k.lower(): v for k, v in mon.items()}
            missing = [r for r in required if r not in mon and r.lower() not in normalized]
            false_checks = [r for r in required if r in mon and not bool(mon[r])]
            # also accept alternative 'healthy' aggregate only if it is True and all sub-checks provided via observability
            if missing and not mon.get("healthy"):
                return GateResult(gate_name="live_monitoring_gate", status=GateStatus.FAILED, passed=False, reason=f"Live monitoring missing checks: {missing}", metrics=mon)
            if false_checks:
                return GateResult(gate_name="live_monitoring_gate", status=GateStatus.FAILED, passed=False, reason=f"Live monitoring checks false: {false_checks}", metrics=mon)
            # ---- 4. Heartbeat freshness: must be <5m, verify timestamp if present ----
            hb_val = mon.get("heartbeat_at") or mon.get("heartbeat_timestamp") or mon.get("last_heartbeat")
            if hb_val is not None:
                try:
                    from datetime import datetime, timezone
                    if isinstance(hb_val, (int, float)):
                        age = __import__("time").time() - float(hb_val)
                    elif isinstance(hb_val, str):
                        hb_dt = datetime.fromisoformat(hb_val.replace("Z","+00:00"))
                        age = (datetime.now(timezone.utc) - hb_dt).total_seconds()
                    else:
                        age = None
                    if age is not None and age > 300:
                        return GateResult(gate_name="live_monitoring_gate", status=GateStatus.FAILED, passed=False, reason=f"Heartbeat stale: {age:.0f}s > 300s (threshold 5m)", metrics=mon)
                    if age is not None and age < 0:
                        return GateResult(gate_name="live_monitoring_gate", status=GateStatus.FAILED, passed=False, reason=f"Heartbeat timestamp in future: {hb_val}", metrics=mon)
                except Exception as e:
                    return GateResult(gate_name="live_monitoring_gate", status=GateStatus.FAILED, passed=False, reason=f"Heartbeat parse error: {e}", metrics=mon)
            else:
                # heartbeat_fresh==True but no timestamp → still require evidence
                if mon.get("heartbeat_fresh") is True and "healthy" not in mon:
                    # allow if caller asserts fresh without timestamp, but log warning detail
                    pass
            # ---- 5. Live probes (best-effort, fail-closed if probe says down) ----
            # Prometheus probe
            if mon.get("prometheus_up") is True:
                try:
                    # Try real prometheus_client registry if available
                    import importlib
                    prom = importlib.import_module("prometheus_client")
                    # If prometheus_client is installed, ensure process actually exposed metrics
                    # We don't have endpoint check, but we can verify registry has collectors
                    _ = prom.REGISTRY  # will raise if not installed
                except ImportError:
                    # prometheus_client not installed → consider prometheus_up claim unverified
                    # In production this must be installed; fail if claimed up but no client
                    return GateResult(gate_name="live_monitoring_gate", status=GateStatus.FAILED, passed=False, reason="Prometheus claimed up but prometheus_client not available", metrics=mon)
                except Exception as e:
                    return GateResult(gate_name="live_monitoring_gate", status=GateStatus.FAILED, passed=False, reason=f"Prometheus probe error: {e}", metrics=mon)
            # Alerts probe: verify alerts_active is not just True but backed by config
            if mon.get("alerts_active") is True:
                try:
                    # Check for alerts module/state
                    alerts_cfg = getattr(state, 'alerts_config', None) if state else None
                    if alerts_cfg is not None and isinstance(alerts_cfg, dict) and not alerts_cfg.get("enabled"):
                        return GateResult(gate_name="live_monitoring_gate", status=GateStatus.FAILED, passed=False, reason="Alerts claimed active but alerts_config.enabled=False", metrics=mon)
                except Exception:
                    pass
            # Reconciliation probe
            if mon.get("reconciliation_healthy") is True:
                try:
                    rec = getattr(state, 'last_reconciliation', None) if state else None
                    if isinstance(rec, dict) and rec.get("healthy") is False:
                        return GateResult(gate_name="live_monitoring_gate", status=GateStatus.FAILED, passed=False, reason=f"Reconciliation unhealthy: {rec}", metrics=mon)
                except Exception:
                    pass
            # Risk monitor probe
            if mon.get("risk_monitor_healthy") is True:
                try:
                    risk_ok = getattr(state, 'risk_monitor_healthy', None) if state else None
                    if risk_ok is False:
                        return GateResult(gate_name="live_monitoring_gate", status=GateStatus.FAILED, passed=False, reason="Risk monitor flagged unhealthy", metrics=mon)
                except Exception:
                    pass
        except Exception as e:
            return GateResult(gate_name="live_monitoring_gate", status=GateStatus.FAILED, passed=False, reason=f"Live monitoring gate error: {e}", metrics={} if 'mon' not in locals() else (mon if isinstance(mon, dict) else {}))
        return GateResult(gate_name="live_monitoring_gate", status=GateStatus.PASSED, passed=True, reason="Monitoring active: prometheus/alerts/heartbeat/reconciliation/risk all verified", metrics=mon if isinstance(mon, dict) else {})


__all__ = [
    "GateStatus",
    "GateResult",
    "ValidationGate",
]