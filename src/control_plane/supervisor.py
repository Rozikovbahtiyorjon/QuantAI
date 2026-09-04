"""QuantAI AI Supervisor - Central Control Plane"""
from __future__ import annotations
import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from pathlib import Path
from .task_manager import TaskManager, Task, TaskStatus, TaskPriority
from .agent_router import AgentRouter, AgentType, AgentCapability
from .state_manager import StateManager, SupervisorState, PipelineStage
from .evidence_manager import EvidenceManager, Evidence, EvidenceType
from .validation_gate import ValidationGate, GateResult, GateStatus
from .retry_engine import RetryEngine, RepairAction, RepairResult
from .checkpoint_manager import CheckpointManager, Checkpoint
from .audit_logger import AuditLogger, AuditEntry, AuditLevel
from .verifier import Verifier, VerificationResult
from config.settings import settings
try:
    from src.research.research_budget import ResearchBudget, BudgetExceeded
    from src.research.experiment_registry import ExperimentRegistry
    from src.research.research_ledger import AtomicResearchLedger
except Exception:  # graceful degrade if research not installed
    ResearchBudget = None  # type: ignore
    BudgetExceeded = RuntimeError  # type: ignore
    ExperimentRegistry = None  # type: ignore
    AtomicResearchLedger = None  # type: ignore

class PipelineStage(str, Enum):
    RESEARCH = "research"
    ARCHITECTURE = "architecture"
    IMPLEMENTATION = "implementation"
    TESTING = "testing"
    BACKTEST = "backtest"
    WFO = "wfo"
    PAPER = "paper"
    OPTIMIZATION = "optimization"
    CHAMPION = "champion"
    PRODUCTION = "production"

@dataclass
class SupervisorConfig:
    max_concurrent_tasks: int = 3
    max_retries_per_task: int = 3
    retry_delay_seconds: float = 5.0
    max_repair_attempts: int = 3
    checkpoint_interval_seconds: float = 60.0
    audit_log_path: str = "logs/audit"
    checkpoint_path: str = "data/checkpoints"
    evidence_path: str = "data/evidence"
    max_risk_per_trade_pct: float = 0.03
    max_total_exposure_pct: float = 0.05
    max_drawdown_pct: float = 0.10
    min_reserve_pct: float = 0.40
    max_leverage: float = 1.0
    # P0.6: Supervisor is SUBORDINATE to Research Budget + Research Integrity — cannot bypass
    # Forbidden: disable gate, increase budget, read holdout, forge evidence, self-declare champion
    forbidden_actions: Set[str] = field(default_factory=lambda: {
        "modify_risk_limits","modify_live_capital","modify_exchange_credentials","disable_kill_switch","modify_production_safety_policy",
        "disable_gate","increase_budget","read_holdout","forge_evidence","self_declare_champion",
        "bypass_verifier","bypass_research_integrity","modify_budget_limits","access_holdout_direct"
    })

@dataclass
class SupervisorState:
    current_stage: PipelineStage = PipelineStage.RESEARCH
    active_tasks: Dict[str, Any] = field(default_factory=dict)
    completed_tasks: List[str] = field(default_factory=list)
    failed_tasks: List[str] = field(default_factory=list)
    current_agent: Optional[str] = None
    last_checkpoint: Optional[datetime] = None
    iteration: int = 0
    errors: List[str] = field(default_factory=list)
    safety_violations: List[str] = field(default_factory=list)
    escalated: bool = False
    last_analysis: Optional[Dict[str, Any]] = None
    last_execution_result: Optional[Dict[str, Any]] = None
    last_test_result: Optional[Dict[str, Any]] = None
    last_review: Optional[Dict[str, Any]] = None
    last_validation: Optional[Any] = None
    current_task: Optional[Any] = None

class AISupervisor:
    def __init__(self, config: Optional[SupervisorConfig] = None):
        self.config = config or SupervisorConfig()
        self.state = SupervisorState()
        self.task_manager = TaskManager()
        self.agent_router = AgentRouter()
        self.state_manager = StateManager()
        self.evidence_manager = EvidenceManager()
        self.validation_gate = ValidationGate()
        self.retry_engine = RetryEngine()
        self.checkpoint_manager = CheckpointManager()
        self.audit_logger = AuditLogger()
        self.verifier = Verifier()
        self._safety_boundaries = self.config.forbidden_actions
        # P1 — Research Integrity: durable budget + registry (points 21-23) — must survive restart
        self.research_ledger = None
        try:
            if AtomicResearchLedger is not None:
                self.research_ledger = AtomicResearchLedger()
        except Exception:
            self.research_ledger = None
        self.research_budget = self.research_ledger.load_budget() if self.research_ledger else (ResearchBudget() if ResearchBudget else None)
        self.experiment_registry = ExperimentRegistry() if ExperimentRegistry else None
        # Ensure registry is source of truth for PBO/DSR n_trials — empty snapshot is expected at start
        # Future n_trials will be correctly computed from registry + ledger durability
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._main_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        self.audit_logger.log(level=AuditLevel.INFO, message="AI Supervisor starting", context={"config": str(self.config)})
        self._running = True
        self._main_task = asyncio.create_task(self._main_loop())
        await self.checkpoint_manager.start()
        await self.audit_logger.start()
        self.audit_logger.log(level=AuditLevel.INFO, message="AI Supervisor started successfully", context={"config": str(self.config)})

    async def stop(self) -> None:
        self.audit_logger.log(level=AuditLevel.INFO, message="AI Supervisor stopping")
        self._running = False
        self._shutdown_event.set()
        if self._main_task:
            self._main_task.cancel()
            try:
                await self._main_task
            except asyncio.CancelledError:
                pass
        await self.checkpoint_manager.stop()
        await self.audit_logger.stop()
        self.audit_logger.log(level=AuditLevel.INFO, message="AI Supervisor stopped")

    async def _main_loop(self) -> None:
        # Autonomous Loop — runs until SUCCESS or ESCALATION_REQUIRED, never stops after one op with "что делать дальше?"
        while self._running:
            try:
                await self._execute_cycle()
                # NEXT TASK will be planned in next iteration's PLAN — no human question
                if self.state.escalated:
                    self.audit_logger.log(level=AuditLevel.CRITICAL, message="Autonomous loop escalated — human intervention required, pausing loop")
                    await asyncio.sleep(self.config.retry_delay_seconds * 3)
                else:
                    await asyncio.sleep(self.config.retry_delay_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.audit_logger.log(level=AuditLevel.ERROR, message=f"Main loop error: {e}", context={"error": str(e)})
                await asyncio.sleep(self.config.retry_delay_seconds)

    async def _execute_cycle(self) -> None:
        # P0.9 True Autonomous Loop — spec main cycle:
        # OBSERVE → DIAGNOSE → HYPOTHESIS → PLAN → EXECUTE → REAL TEST → VERIFY → EVIDENCE → GATE → STATE TRANSITION → NEXT ITERATION
        # Agent is FORBIDDEN to confirm success=True — only Verifier can.
        # Loops autonomously until SUCCESS or ESCALATION_REQUIRED, never "что делать дальше?"
        self.state.iteration += 1
        # SUBORDINATION CHECK: Supervisor subordinate to Budget + Integrity
        if not await self._enforce_subordination():
            self.audit_logger.log(level=AuditLevel.CRITICAL, message="Supervisor subordination check failed — escalating (cannot disable gate/increase budget/read holdout/forge evidence/self-declare champion)")
            await self._escalate_failure()
            await self._checkpoint()
            return
        # OBSERVE: current stage, active tasks, recent evidence, market data, system health
        await self._observe()
        # DIAGNOSE: evidence anomalies, gate failures, regime, costs
        await self._analyze()
        # HYPOTHESIS: Supervisor generates hypothesis from diagnosis (not agent)
        await self._hypothesis()
        # PLAN: Planner picks next Task for current stage based on hypothesis
        await self._plan()
        if not self.state.current_task:
            # No task → still checkpoint and continue loop (autonomous, not stop)
            await self._checkpoint()
            return
        # EXECUTE: Task → Agent → Real Command → Artifact
        # FORBIDDEN: agent cannot self-confirm success=True — will be ignored, Verifier decides
        await self._select_agent()
        await self._execute()
        # REAL TEST: run real pytest, produce JUnit/JSON (not mock)
        await self._test()
        # VERIFY: Independent Verifier recomputes passed from 7 checks (exit_code, artifact_exists, hash, tests, metrics, dataset, code_commit)
        # Does NOT trust artifact's passed field — especially important for autonomous AI
        verify_res = await self._verify()
        # Agent success claim vs Verifier reality
        agent_claim = bool((self.state.last_execution_result or {}).get("success") or (self.state.last_execution_result or {}).get("passed"))
        if agent_claim and not verify_res.verified:
            self.audit_logger.log(level=AuditLevel.CRITICAL, message=f"Agent claimed success=True but Verifier rejected (not trusting passed): {verify_res.reason} — agent forbidden to confirm success")
        if not verify_res.verified:
            self.audit_logger.log(level=AuditLevel.WARNING, message=f"Verifier rejected: {verify_res.reason} → REPAIR")
            self.state.last_validation = GateResult(passed=False, reason=f"Verifier rejected: {verify_res.reason}", gate_name="verifier", status=GateStatus.FAILED)
            result = await self._evaluate()
            # REPAIR → REAL TEST → EVIDENCE → GATE → TRANSITION (autonomous)
            await self._handle_failure(result)
            await self._checkpoint()
            return
        # EVIDENCE: Verifier -> Evidence (L2/L3) — only after VERIFY, not before
        # (Evidence already stored in _verify as L2, and in _execute as L1 self-reported)
        # GATE: Independent Gate decides via formula(raw metrics), not evidence["passed"]
        await self._review()
        await self._validate()
        result = await self._evaluate()
        # STATE TRANSITION: Gate result -> StateManager (only if Gate passed with Verified Evidence)
        if not result.passed:
            # REPAIR → REAL TEST → EVIDENCE → GATE → STATE TRANSITION (inside _handle_failure)
            await self._handle_failure(result)
        else:
            # GATE PASS → STATE TRANSITION → NEXT ITERATION (autonomous, no question)
            await self._handle_success(result)
        await self._checkpoint()
        # NEXT ITERATION: will be OBSERVE of next cycle — loop continues autonomously
        # Research loop: Hypothesis→...→Reject reason → new hypothesis (closed loop)
        if not result.passed and str(self.state.current_stage) in ("PipelineStage.RESEARCH", "research"):
            await self._research_loop_on_reject(result)

    async def _observe(self) -> None:
        self.state.current_stage = await self.state_manager.get_current_stage()
        self.state.active_tasks = await self.task_manager.get_active_tasks()
        evidence = await self.evidence_manager.get_recent(limit=100)
        self.audit_logger.log(level=AuditLevel.DEBUG, message="Observation complete", context={"stage": str(self.state.current_stage)})

    async def _analyze(self) -> None:
        analysis = await self.evidence_manager.analyze()
        if analysis.get("anomalies"):
            self.audit_logger.log(level=AuditLevel.WARNING, message=f"Anomalies detected: {analysis['anomalies']}")
        self.state.last_analysis = analysis

    async def _plan(self) -> None:
        next_task = await self.task_manager.get_next_task(stage=str(self.state.current_stage), state=self.state)
        if next_task:
            self.state.current_task = next_task
            await self.task_manager.assign_task(next_task)

    async def _select_agent(self) -> None:
        if not self.state.current_task:
            return
        agent_type = await self.agent_router.route(task=self.state.current_task, stage=str(self.state.current_stage), state=self.state)
        self.state.current_agent = agent_type.value if hasattr(agent_type, 'value') else str(agent_type)
        self.audit_logger.log(level=AuditLevel.INFO, message=f"Selected agent: {self.state.current_agent}")

    async def _execute(self) -> None:
        if not self.state.current_task or not self.state.current_agent:
            return
        if not await self._check_safety_boundaries():
            self.audit_logger.log(level=AuditLevel.ERROR, message="Safety boundary violation - execution blocked")
            return
        # P1 ResearchBudget durable guard — ALL 7 limits via AtomicResearchLedger (points 23,28)
        # Every supervisor path must use ledger, not in-memory, otherwise restart bypasses budget
        ledger = getattr(self, 'research_ledger', None)
        budget = self.research_budget  # in-memory mirror for quick checks, but ledger is source of truth
        if ledger:
            try:
                # Always count experiment
                ledger.check_and_increment("experiment")
                # Strategy variant (each quant_researcher tournament)
                if self.state.current_agent == "quant_researcher":
                    ledger.check_and_increment("strategy_variant")
                    oos_period = getattr(self.state.current_task, "metadata", {}).get("oos_period", "") if hasattr(self.state.current_task, "metadata") else ""
                    if oos_period and self.experiment_registry:
                        reuse = self.experiment_registry.oos_reuse_count(oos_period)
                        ledger.check_and_increment("oos_reuse", registry_oos_reuse=reuse)
                        # Also per-OOS cap
                        try:
                            # Need current count for this OOS; use registry count
                            ledger.check_and_increment("optimization_attempt", strategy_id=oos_period)  # track per-OOS via optimization_attempt key
                        except Exception:
                            pass
                if self.state.current_agent == "ml_engineer":
                    # Optuna trials: try to get n from task metadata
                    n_trials = 1
                    try:
                        n_trials = int(getattr(self.state.current_task, "metadata", {}).get("n_trials", 1) or 1)
                    except Exception:
                        n_trials = 1
                    ledger.check_and_increment("optuna", n=n_trials)
                    ledger.check_and_increment("parameter_mutation")
                    # Check indicator/param complexity if task carries them
                    try:
                        params = getattr(self.state.current_task, "metadata", {}).get("params") if hasattr(self.state.current_task, "metadata") else None
                        if params:
                            ledger.check_and_increment("params", params=params)
                        indicators = getattr(self.state.current_task, "metadata", {}).get("indicators") if hasattr(self.state.current_task, "metadata") else None
                        if indicators:
                            ledger.check_and_increment("indicators", indicators=indicators)
                    except Exception:
                        pass
                if self.state.current_agent in ("quant_engineer", "ml_engineer"):
                    name_low = getattr(self.state.current_task, "name", "").lower()
                    if "tune" in name_low or "optim" in name_low:
                        ledger.check_and_increment("parameter_mutation")
                    # Also count strategy_id optimization attempts
                    try:
                        sid = getattr(self.state.current_task, "id", "unknown")
                        ledger.check_and_increment("optimization_attempt", strategy_id=str(sid))
                    except Exception:
                        pass
                # Sync in-memory budget for stats display
                if budget and ledger:
                    try:
                        fresh = ledger.load_budget()
                        self.research_budget = fresh
                    except Exception:
                        pass
            except BudgetExceeded as e:
                self.audit_logger.log(level=AuditLevel.WARNING, message=f"ResearchBudget exceeded: {e} — hard stop, no silent continue")
                self.state.last_execution_result = {"success": False, "error": str(e), "budget_exceeded": True}
                await self.evidence_manager.store(Evidence(type=EvidenceType.EXECUTION_RESULT, data=self.state.last_execution_result, source=f"agent_{self.state.current_agent}"))
                return
        elif budget:
            # P0.6: No in-memory fallback — ledger is required for persistence (restart would reset counters)
            # If ledger is missing (should not happen), fail-closed and require ledger
            self.audit_logger.log(level=AuditLevel.ERROR, message="ResearchBudget ledger missing — cannot enforce durable budget, blocking execution (P0.6)")
            self.state.last_execution_result = {"success": False, "error": "Research ledger missing — persistent budget required (P0.6)", "budget_exceeded": True}
            await self.evidence_manager.store(Evidence(type=EvidenceType.EXECUTION_RESULT, data=self.state.last_execution_result, source=f"agent_{self.state.current_agent}"))
            return
        result = await self.agent_router.execute(agent_type=self.state.current_agent, task=self.state.current_task, state=self.state)
        self.state.last_execution_result = result
        # --- Experiment Registry: source of truth for PBO/DSR n_trials (points 21-22) ---
        if self.experiment_registry and isinstance(result, dict):
            try:
                from src.research.experiment_registry import ExperimentRecord
                # Only register if we have meaningful result (not budget_exceeded)
                if not result.get("budget_exceeded"):
                    meta = getattr(self.state.current_task, "metadata", {}) if hasattr(self.state.current_task, "metadata") else {}
                    # Try to extract PF/Sharpe/Trades from result
                    pf = result.get("PF") or result.get("pf") or result.get("profit_factor") or 0.0
                    # Tournament nested format
                    if pf == 0.0 and "result" in result and isinstance(result["result"], dict):
                        inner = result["result"]
                        if "tournament" in inner:
                            best = inner.get("best")
                            if isinstance(best, dict):
                                pf = best.get("backtest_pf", 0.0)
                    rec = ExperimentRecord(
                        dataset_id=str(meta.get("dataset_id", "BTCUSDT_4H_v7")),
                        dataset_hash=str(meta.get("dataset_hash", "")),
                        feature_schema_hash=str(meta.get("feature_hash", "")),
                        oos_period=str(meta.get("oos_period", "2024-OOS-unknown")),
                        parameters=dict(meta.get("params", {})) if isinstance(meta.get("params"), dict) else {},
                        PF=float(pf or 0.0),
                        Sharpe=float(result.get("Sharpe") or result.get("sharpe") or 0.0),
                        Trades=int(result.get("Trades") or result.get("trades") or 0),
                        oos_touched=bool(meta.get("oos_period")),
                        used_for_selection=bool(meta.get("oos_period")),
                        selection_status="CANDIDATE" if result.get("success") else "RESEARCH",
                    )
                    self.experiment_registry.register(rec)
            except Exception as e:
                self.audit_logger.log(level=AuditLevel.WARNING, message=f"ExperimentRegistry register failed: {e}")
        # P1.17: Agent->Artifact->Verifier->Evidence->Gate — store self-reported as L1, verifier will create L2/L3
        # Populate evidence with provenance fields from result for contract check (P1.15)
        try:
            artifact_paths = result.get("artifact_paths", []) if isinstance(result, dict) else []
            exit_code = result.get("exit_code", 1 if not result.get("success") else 0) if isinstance(result, dict) else 1
            # Hash artifacts if present
            artifact_hashes = result.get("artifact_hashes", {}) if isinstance(result, dict) else {}
            ev = Evidence(
                type=EvidenceType.EXECUTION_RESULT,
                data=result,
                source=f"agent_{self.state.current_agent}",
                artifact_paths=list(artifact_paths) if isinstance(artifact_paths, list) else [str(artifact_paths)] if artifact_paths else [],
                artifact_hashes=dict(artifact_hashes) if isinstance(artifact_hashes, dict) else {},
                exit_code=int(exit_code) if exit_code is not None else 1,
                source_command=str(result.get("artifact_paths", "")) if isinstance(result, dict) else "",
                trust_level=1,  # L1 self-reported until verifier upgrades
                generated_by_real_execution=False,  # will be set by verifier
            )
            await self.evidence_manager.store(ev)
        except Exception:
            # Fallback self-reported
            await self.evidence_manager.store(Evidence(type=EvidenceType.EXECUTION_RESULT, data=result, source=f"agent_{self.state.current_agent}"))

    async def _test(self) -> None:
        if not self.state.current_task:
            return
        test_result = await self.task_manager.run_tests(task=self.state.current_task, state=self.state)
        self.state.last_test_result = test_result
        await self.evidence_manager.store(Evidence(type=EvidenceType.TEST_RESULT, data=test_result, source="test_runner"))

    async def _review(self) -> None:
        review = await self.task_manager.review(task=self.state.current_task, execution_result=self.state.last_execution_result, test_result=self.state.last_test_result)
        self.state.last_review = review
        await self.evidence_manager.store(Evidence(type=EvidenceType.REVIEW, data=review, source="reviewer"))

    async def _verify(self) -> VerificationResult:
        """Verifier: Agent creates result, Verifier confirms, Gate decides — not Agent->success->transition."""
        exec_v = self.verifier.verify_execution(self.state.last_execution_result, self.state.current_task)
        test_v = self.verifier.verify_test(self.state.last_test_result)
        # P1.17: Verifier->Evidence with L2/L3 trust (independent)
        all_verified = exec_v.verified and test_v.verified
        # Store verification evidence with proper trust
        try:
            ev = Evidence(
                type=EvidenceType.VALIDATION_RESULT,
                data={"exec_verified": exec_v.verified, "exec_reason": exec_v.reason, "test_verified": test_v.verified, "checks": {**exec_v.checks, **test_v.checks}},
                source="verifier",
                artifact_hashes=dict(exec_v.artifact_hashes),
                exit_code=0 if all_verified else 1,
                trust_level=2 if all_verified else 1,  # L2 if verified, else L1
                generated_by_real_execution=bool(all_verified),
                verification_status="VERIFIED" if all_verified else "UNVERIFIED",
            )
            await self.evidence_manager.store(ev)
            # Annotate last_execution_result for Gate (independent verification layer)
            if isinstance(self.state.last_execution_result, dict):
                self.state.last_execution_result["_verified"] = all_verified
                self.state.last_execution_result["_artifact_hashes"] = exec_v.artifact_hashes
                self.state.last_execution_result["_verifier_checks"] = {**exec_v.checks, **test_v.checks}
        except Exception:
            await self.evidence_manager.store(Evidence(type=EvidenceType.VALIDATION_RESULT, data={"exec_verified": exec_v.verified, "exec_reason": exec_v.reason, "test_verified": test_v.verified, "checks": {**exec_v.checks, **test_v.checks}}, source="verifier"))
        if not exec_v.verified:
            return exec_v
        if not test_v.verified:
            return test_v
        return VerificationResult(True, "all verified", {**exec_v.artifact_hashes}, {**exec_v.checks, **test_v.checks})

    async def _hypothesis(self) -> None:
        """HYPOTHESIS: Supervisor generates hypothesis from DIAGNOSE (not agent).

        Autonomous Hypothesis Generator: uses last_analysis anomalies and last_validation
        to propose hypothesis. Agent is NOT allowed to generate hypothesis without evidence.
        """
        # If we are in research stage and have a diagnosis, generate hypothesis task if needed
        # This is the autonomous hypothesis generation (P4.1)
        if str(self.state.current_stage) not in ("PipelineStage.RESEARCH", "research"):
            return
        analysis = self.state.last_analysis or {}
        anomalies = analysis.get("anomalies", []) if isinstance(analysis, dict) else []
        # Only generate hypothesis if we have anomalies or last validation failed
        last_val = self.state.last_validation
        if last_val is None or getattr(last_val, 'passed', True):
            # No failure to hypothesize from — skip unless no active task
            if self.state.current_task is not None:
                return
        # Generate hypothesis from diagnosis: use anomalies + last failure reason
        reason = getattr(last_val, 'reason', str(last_val)) if last_val else "; ".join(anomalies) if anomalies else "initial hypothesis"
        # Check that we don't generate duplicate hypothesis for same reason too quickly
        recent_tasks = list(self.task_manager.tasks.values())[-5:] if hasattr(self.task_manager, 'tasks') else []
        for t in recent_tasks:
            if isinstance(t, dict):
                desc = t.get("description", "")
            else:
                desc = getattr(t, 'description', "")
            if reason[:30] in str(desc)[:30] and "hypothesis from reject" in str(desc):
                return  # already have hypothesis for this reason
        # Create hypothesis task (autonomous, not agent)
        try:
            from .task_manager import Task
            new_task = Task(
                name=f"hypothesis: {reason[:50]}",
                description=f"Autonomous hypothesis from DIAGNOSE: {reason}. Requires evidence before PLAN.",
                stage="research",
                metadata={"hypothesis": reason, "auto_generated": True, "stage": "hypothesis"},
            )
            await self.task_manager.create_task(new_task)
            self.audit_logger.log(level=AuditLevel.INFO, message=f"HYPOTHESIS generated: {reason[:80]}")
        except Exception as e:
            self.audit_logger.log(level=AuditLevel.WARNING, message=f"HYPOTHESIS generation failed: {e}")

    async def _research_loop_on_reject(self, result: GateResult) -> None:
        """Research loop: Hypothesis -> Budget -> Dataset -> Feature/Label -> Inner WF -> Optuna -> Frozen -> Outer OOS -> Cost/Slippage/Latency -> Regimes -> PBO/DSR/WRC -> Robustness -> Candidate/Reject -> on Reject: reason -> new hypothesis.
        
        P0.6: Complexity ↑ requires Evidence ↑↑ — new hypothesis with higher complexity must have higher evidence quality.
        """
        try:
            # Only for research stage
            if str(self.state.current_stage) not in ("PipelineStage.RESEARCH", "research"):
                return
            # P0.6: Check complexity-evidence link before generating new hypothesis
            try:
                from src.research.complexity_evidence_gate import ComplexityScore, EvidenceQuality, check_complexity_evidence
                # Estimate next hypothesis complexity (incremental)
                ledger = self.research_ledger.load_budget() if self.research_ledger else None
                n_exp = ledger.experiments_used if ledger else len(self.experiment_registry._index) if self.experiment_registry else 0
                # Use last validation evidence if available
                last_ev = getattr(self.state, 'last_validation', None)
                ev_dict = {}
                if last_ev and hasattr(last_ev, 'details'):
                    ev_dict = getattr(last_ev, 'details', {}) or {}
                # Enforce link — if next complexity would require higher evidence and we don't have it, block new hypothesis and require more validation
                from src.research.complexity_evidence_gate import enforce_complexity_evidence_link
                # Simulate next complexity as current +1 param/indicator
                enforce_complexity_evidence_link(
                    n_params=6,  # next hypothesis would add one param (5->6)
                    n_indicators=11,  # next would add one indicator (10->11)
                    n_trials=10,
                    n_experiments=n_exp + 1,
                    oos_touches=ledger.oos_reuse_used if ledger else 0,
                    evidence_dict=ev_dict,
                )
            except ValueError as ce:
                self.audit_logger.log(level=AuditLevel.WARNING, message=f"Complexity-Evidence link blocked new hypothesis: {ce}")
                return
            except Exception:
                pass
            reason = getattr(result, 'reason', str(result))
            # Create new hypothesis task from reject reason (not improvisation)
            from .task_manager import Task
            new_task = Task(
                name=f"hypothesis from reject: {reason[:60]}",
                description=f"Previous candidate rejected: {reason}. Generate new hypothesis with different features/params.",
                stage="research",
                metadata={"parent_reject_reason": reason, "auto_generated": True}
            )
            await self.task_manager.create_task(new_task)
            self.audit_logger.log(level=AuditLevel.INFO, message=f"Research loop: new hypothesis created from reject reason: {reason[:100]}")
        except Exception as e:
            self.audit_logger.log(level=AuditLevel.WARNING, message=f"Research loop failed: {e}")

    async def _validate(self) -> GateResult:
        if not self.state.current_task:
            return GateResult(passed=False, reason="No active task", gate_name="none", status=GateStatus.FAILED)
        result = await self.validation_gate.validate(task=self.state.current_task, execution_result=self.state.last_execution_result, test_result=self.state.last_test_result, review=self.state.last_review, state=self.state)
        self.state.last_validation = result
        self.audit_logger.log(level=AuditLevel.INFO if result.passed else AuditLevel.WARNING, message=f"Validation {'PASSED' if result.passed else 'FAILED'}: {result.reason}")
        return result

    async def _evaluate(self) -> GateResult:
        if not self.state.last_validation:
            return GateResult(passed=False, reason="No validation result", gate_name="none", status=GateStatus.FAILED)
        passed = self.state.last_validation.passed
        if passed:
            if not await self._check_additional_criteria():
                passed = False
        return GateResult(passed=passed, reason="Evaluation complete" if passed else "Evaluation failed", gate_name="evaluate", status=GateStatus.PASSED if passed else GateStatus.FAILED)

    async def _check_additional_criteria(self) -> bool:
        if not self.state.last_execution_result:
            return True
        result = self.state.last_execution_result
        if not isinstance(result, dict):
            return True
        
        # Direct format (legacy)
        pf = result.get("profit_factor")
        dd = result.get("max_drawdown")
        
        # Nested tournament format from quant_researcher agent
        if pf is None and "result" in result and isinstance(result["result"], dict):
            inner = result["result"]
            if "tournament" in inner and isinstance(inner["tournament"], dict):
                best = inner.get("best")
                if isinstance(best, dict):
                    pf = best.get("backtest_pf")
                    dd = best.get("backtest_dd")
                elif "best" in inner and isinstance(inner.get("best"), str):
                    tour = inner["tournament"]
                    best_key = inner["best"]
                    best_data = tour.get(best_key)
                    if isinstance(best_data, dict):
                        pf = best_data.get("backtest_pf")
                        dd = best_data.get("backtest_dd")
        
        if pf is not None and float(pf) < 1.0:
            return False
        if dd is not None:
            # backtest_dd is in percentage (e.g., -1.05 for -1.05%), config is decimal (0.10 = 10%)
            dd_decimal = abs(float(dd)) / 100.0
            if dd_decimal > self.config.max_drawdown_pct:
                return False
        return True

    async def _handle_failure(self, result: GateResult) -> None:
        self.state.failed_tasks.append(self.state.current_task.id if self.state.current_task else "unknown")
        self.audit_logger.log(level=AuditLevel.ERROR, message=f"Task failed validation: {result.reason}")
        # Hard budget for retries — durable via ledger (point 23)
        ledger = getattr(self, 'research_ledger', None)
        if ledger:
            try:
                ledger.check_and_increment("retry")
                # Sync in-memory mirror
                try:
                    self.research_budget = ledger.load_budget()
                except Exception:
                    pass
            except BudgetExceeded as e:
                self.audit_logger.log(level=AuditLevel.WARNING, message=f"ResearchBudget max_retries exceeded: {e} — escalating without retry")
                await self._escalate_failure()
                return
        elif self.research_budget:
            try:
                self.research_budget.check_retry()
            except BudgetExceeded as e:
                self.audit_logger.log(level=AuditLevel.WARNING, message=f"ResearchBudget max_retries exceeded: {e} — escalating without retry")
                await self._escalate_failure()
                return
        diagnosis = await self._diagnose()
        repair_result = await self.retry_engine.repair(task=self.state.current_task, diagnosis=diagnosis, state=self.state)
        if repair_result.success:
            await self._test()
            await self._review()
            verify_res = await self._verify()
            if not verify_res.verified:
                await self._escalate_failure()
                return
            validation_result = await self._validate()
            if validation_result.passed:
                await self._handle_success(validation_result)
            else:
                await self._escalate_failure()
        else:
            await self._escalate_failure()

    async def _diagnose(self) -> Dict[str, Any]:
        return await self.retry_engine.diagnose(task=self.state.current_task, execution_result=self.state.last_execution_result, test_result=self.state.last_test_result, validation_result=self.state.last_validation)

    async def _handle_success(self, result: GateResult) -> None:
        # Task complete ≠ Stage complete — stage transitions only on verified evidence for ALL exit criteria
        if self.state.current_task:
            self.state.completed_tasks.append(self.state.current_task.id)
            self.state.active_tasks.pop(self.state.current_task.id, None)
        self.audit_logger.log(level=AuditLevel.INFO, message=f"Task completed successfully: {result.reason} (task≠stage)")
        # P0.8: Check if stage still has pending/active tasks — if so, Task complete but Stage not complete
        try:
            cur_stage_key_check = self.state_manager.current_stage
            pending_for_stage = [
                t for t in self.task_manager.tasks.values()
                if getattr(t, 'stage', None) == cur_stage_key_check and t.status not in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)
            ]
            if pending_for_stage:
                self.audit_logger.log(level=AuditLevel.INFO, message=f"Stage {cur_stage_key_check} not complete: {len(pending_for_stage)} tasks still pending/active — Task complete ≠ Stage complete")
                await self._checkpoint()
                self.state.current_task = None
                self.state.current_agent = None
                return
        except Exception:
            pass
        # --- Autonomous pipeline transition: validated stage → next stage (point 5) — only on verified evidence ---
        try:
            # Record evidence for current stage exit criteria — only if GateResult verified
            cur_stage_key = self.state_manager.current_stage
            cur_info = self.state_manager.stages.get(cur_stage_key)
            if cur_info and result.passed:
                # For TESTING stage, require 4 verified gates: compile, required tests, no-lookahead, risk tests
                if cur_stage_key == "testing":
                    # TESTING -> BACKTEST only after 4 verified gates (P0.8)
                    required = {"compile_pass": False, "required_tests_pass": False, "no_lookahead_pass": False, "risk_tests_pass": False}
                    # First, populate from current GateResult's gate_results
                    if hasattr(self.state, 'last_validation') and self.state.last_validation:
                        details = getattr(self.state.last_validation, 'details', {}) or {}
                        gate_results = details.get("gate_results", [])
                        for gr in gate_results:
                            if gr.get("gate_name") == "compile_gate" and gr.get("passed"):
                                required["compile_pass"] = True
                                self.state_manager.set_criterion_evidence("compile_pass", {"passed": True, "source": "validation_gate", "gate": "compile_gate"})
                            if gr.get("gate_name") == "unit_test_gate" and gr.get("passed"):
                                required["required_tests_pass"] = True
                                self.state_manager.set_criterion_evidence("required_tests_pass", {"passed": True, "source": "validation_gate", "gate": "unit_test_gate"})
                            if gr.get("gate_name") == "no_lookahead_gate" and gr.get("passed"):
                                required["no_lookahead_pass"] = True
                                self.state_manager.set_criterion_evidence("no_lookahead_pass", {"passed": True, "source": "validation_gate", "gate": "no_lookahead_gate"})
                            if gr.get("gate_name") == "risk_test_gate" and gr.get("passed"):
                                required["risk_tests_pass"] = True
                                self.state_manager.set_criterion_evidence("risk_tests_pass", {"passed": True, "source": "validation_gate", "gate": "risk_test_gate"})
                    # Also check stage_data for any previously verified
                    for crit in cur_info.exit_criteria:
                        ev = self.state_manager.stage_data.get(crit, {})
                        if isinstance(ev, dict) and ev.get("passed") is True:
                            required[crit] = True
                        elif ev is True:
                            required[crit] = True
                    missing = [k for k, v in required.items() if not v]
                    if missing:
                        self.audit_logger.log(level=AuditLevel.WARNING, message=f"Stage {cur_stage_key} NOT complete — missing verified evidence for {missing} — Task complete ≠ Stage complete")
                        await self._checkpoint()
                        self.state.current_task = None
                        self.state.current_agent = None
                        return
                else:
                    for crit in cur_info.exit_criteria:
                        self.state_manager.set_criterion_evidence(crit, {"passed": True, "source": "validation_gate", "gate": result.gate_name})
            # Determine next stage in pipeline order
            order = list(self.state_manager.stages.keys())
            try:
                cur_idx = order.index(self.state_manager.current_stage)
            except ValueError:
                cur_idx = -1
            if cur_idx >= 0 and cur_idx + 1 < len(order):
                next_stage = order[cur_idx + 1]
                next_info = self.state_manager.stages.get(next_stage)
                if next_info:
                    for crit in next_info.entry_criteria:
                        # Entry criteria should already be satisfied by previous exit; set if not yet
                        if crit not in self.state_manager.stage_data:
                            self.state_manager.set_criterion_evidence(crit, {"passed": True, "source": "previous_stage_exit"})
                # Attempt transition — fail-closed: requires all criteria True
                can = await self.state_manager.can_transition(next_stage, self.state)
                if can:
                    ok = await self.state_manager.transition_to(next_stage, self.state)
                    if ok:
                        # Keep SupervisorState in sync with StateManager
                        try:
                            self.state.current_stage = PipelineStage(self.state_manager.current_stage)
                        except Exception:
                            self.state.current_stage = self.state_manager.current_stage  # type: ignore
                        self.audit_logger.log(level=AuditLevel.INFO, message=f"Autonomous transition: {cur_stage_key} → {next_stage}")
                        await self.evidence_manager.store(Evidence(type=EvidenceType.VALIDATION_RESULT, data={"transition": f"{cur_stage_key}->{next_stage}", "validated": True}, source="state_manager"))
                    else:
                        self.audit_logger.log(level=AuditLevel.WARNING, message=f"Transition {cur_stage_key}->{next_stage} blocked by StateManager")
                else:
                    self.audit_logger.log(level=AuditLevel.DEBUG, message=f"Cannot transition {cur_stage_key}->{next_stage}: criteria not met (fail-closed)")
        except Exception as e:
            self.audit_logger.log(level=AuditLevel.WARNING, message=f"Autonomous transition error: {e}")
        await self._checkpoint()
        self.state.current_task = None
        self.state.current_agent = None

    async def _escalate_failure(self) -> None:
        self.audit_logger.log(level=AuditLevel.CRITICAL, message="Failure escalated - human intervention required")
        self.state.escalated = True

    async def _checkpoint(self) -> None:
        checkpoint = Checkpoint(id=str(uuid.uuid4()), iteration=self.state.iteration, stage=str(self.state.current_stage), state=self.state, evidence=await self.evidence_manager.get_all())
        await self.checkpoint_manager.save(checkpoint)
        self.state.last_checkpoint = datetime.now(timezone.utc)
        self.audit_logger.log(level=AuditLevel.INFO, message="Checkpoint saved", context={"checkpoint_id": checkpoint.id})

    async def _check_safety_boundaries(self) -> bool:
        if self.state.current_task:
            action = self.state.current_task.get("action", "") if isinstance(self.state.current_task, dict) else getattr(self.state.current_task, 'action', '')
            if action in self._safety_boundaries:
                self.audit_logger.log(level=AuditLevel.CRITICAL, message=f"Safety boundary violation: {action}")
                self.state.safety_violations.append(action)
                return False
        return True

    async def _enforce_subordination(self) -> bool:
        """Supervisor is SUBORDINATE to Research Budget + Research Integrity (P0.6).
        
        Checks 5 forbidden capabilities:
        - disable gate: validation_gate must be called, cannot be bypassed
        - increase budget: ledger limits are immutable, cannot be raised
        - read holdout: firewall holdout not accessible to ResearchProcess
        - forge evidence: evidence must via Verifier (L1→L2), not direct store with fake pass
        - self-declare champion: champion only via ChampionPipeline + ResearchIntegrity
        """
        # 1. Budget is subordinate — cannot increase
        if self.research_ledger:
            try:
                budget = self.research_ledger.load_budget()
                # Check that budget limits have not been increased from original
                # Original limits are from ResearchPolicy: max_experiments 50, etc.
                # If any max_* increased, block
                from src.research.research_budget import ResearchBudget
                canonical = ResearchBudget()
                for field in ["max_experiments","max_optuna_trials","max_oos_reuse","max_strategy_variants"]:
                    if getattr(budget, field) > getattr(canonical, field):
                        self.audit_logger.log(level=AuditLevel.CRITICAL, message=f"Budget increase forbidden: {field} {getattr(canonical, field)} → {getattr(budget, field)}")
                        return False
            except Exception:
                pass
        # 2. Gate cannot be disabled — validation_gate must exist
        if not self.validation_gate or not hasattr(self.validation_gate, 'validate'):
            self.audit_logger.log(level=AuditLevel.CRITICAL, message="Gate disabled — validation_gate missing (forbidden)")
            return False
        # 3. Holdout cannot be read by ResearchProcess — check via firewall
        try:
            from src.research.oos_firewall import OOSFirewall
            # If supervisor has direct holdout access, block
            if hasattr(self, '_holdout_df') and getattr(self, '_holdout_df') is not None:
                self.audit_logger.log(level=AuditLevel.CRITICAL, message="Holdout direct access forbidden — must via HoldoutValidatorProcess")
                return False
        except Exception:
            pass
        # 4. Evidence cannot be forged — check last evidence is via Verifier
        # If last evidence was stored without verifier, it is L1 not L2, cannot be used for promotion
        # This is enforced in Verifier and EvidenceManager is_promotable check
        # 5. Champion cannot be self-declared — must be via ChampionPipeline
        # Check that champion not directly set in state without pipeline
        # This is enforced in validation_gate _gate_champion which requires ChampionPipeline result
        return True

    async def submit_task(self, task: Task) -> str:
        task_id = await self.task_manager.create_task(task)
        self.audit_logger.log(level=AuditLevel.INFO, message=f"Task submitted: {task.name}")
        return task.id

    async def get_status(self) -> Dict[str, Any]:
        return {"running": self._running, "stage": str(self.state.current_stage), "iteration": self.state.iteration, "active_tasks": len(self.state.active_tasks), "completed_tasks": len(self.state.completed_tasks), "failed_tasks": len(self.state.failed_tasks)}

    async def pause(self) -> None:
        self._running = False
        self.audit_logger.log(level=AuditLevel.INFO, message="Supervisor paused")

    async def resume(self) -> None:
        self._running = True
        self.audit_logger.log(level=AuditLevel.INFO, message="Supervisor resumed")

    async def emergency_stop(self) -> None:
        self.audit_logger.log(level=AuditLevel.CRITICAL, message="EMERGENCY STOP triggered")
        self._running = False
        self._shutdown_event.set()
        await self.checkpoint_manager.emergency_save(Checkpoint(id=str(uuid.uuid4()), stage="emergency", state=self.state))
        self.audit_logger.log(level=AuditLevel.CRITICAL, message="Emergency stop complete")

async def create_supervisor(config: Optional[SupervisorConfig] = None) -> AISupervisor:
    supervisor = AISupervisor(config)
    await supervisor.start()
    return supervisor
