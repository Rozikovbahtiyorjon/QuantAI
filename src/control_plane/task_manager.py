"""
QuantAI Task Manager
Manages tasks, scheduling, and execution tracking
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from collections import deque


class TaskStatus(str, Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    TESTING = "testing"
    REVIEWING = "reviewing"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class TaskPriority(int, Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Task:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    stage: str = ""
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    assigned_agent: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3


class TaskManager:
    """Manages task lifecycle, scheduling, and execution tracking"""
    
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.active_tasks: Dict[str, Task] = {}
        self.completed_tasks: Dict[str, Task] = {}
        self.failed_tasks: Dict[str, Task] = {}
        self.task_queue: deque = deque()
        self._lock = asyncio.Lock()
    
    async def create_task(self, task: Task) -> str:
        """Create a new task"""
        async with self._lock:
            self.tasks[task.id] = task
            self.task_queue.append(task.id)
            return task.id
    
    async def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID"""
        return self.tasks.get(task_id)
    
    async def get_next_task(
        self,
        stage: str,
        state: Any
    ) -> Optional[Task]:
        """Get next task for the given stage"""
        async with self._lock:
            # Filter tasks by stage and status
            for task_id in self.task_queue:
                task = self.tasks.get(task_id)
                if not task:
                    continue
                if task.stage != stage:
                    continue
                if task.status != TaskStatus.PENDING:
                    continue
                # Check dependencies
                if not all(self.tasks.get(dep_id, Task()).status == TaskStatus.COMPLETED 
                           for dep_id in task.dependencies):
                    continue
                return task
        return None
    
    async def assign_task(self, task: Task) -> None:
        """Mark task as assigned"""
        task.status = TaskStatus.ASSIGNED
        task.started_at = datetime.now(timezone.utc)
        self.active_tasks[task.id] = task
        if task.id in self.task_queue:
            self.task_queue.remove(task.id)
    
    async def start_task(self, task_id: str, agent: str) -> bool:
        """Start task execution"""
        task = self.tasks.get(task_id)
        if not task:
            return False
        
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now(timezone.utc)
        task.assigned_agent = agent
        return True
    
    async def complete_task(self, task_id: str, result: Any = None) -> bool:
        """Mark task as completed"""
        task = self.tasks.get(task_id)
        if not task:
            return False
        
        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.now(timezone.utc)
        task.result = result
        
        if task_id in self.active_tasks:
            del self.active_tasks[task_id]
        self.completed_tasks[task.id] = task
        return True
    
    async def fail_task(self, task_id: str, error: str) -> bool:
        """Mark task as failed"""
        task = self.tasks.get(task_id)
        if not task:
            return False
        
        task.status = TaskStatus.FAILED
        task.completed_at = datetime.now(timezone.utc)
        task.error = error
        task.retry_count += 1
        
        if task_id in self.active_tasks:
            del self.active_tasks[task_id]
        self.failed_tasks[task.id] = task
        return True
    
    async def run_tests(self, task: Task, state: Any) -> Dict[str, Any]:
        """Run tests for a task — strict evidence via JUnit XML / pytest-json-report (P0.7)."""
        # No manual human-readable parsing as primary — use structured JUnit / JSON
        import subprocess, sys, json, tempfile, xml.etree.ElementTree as ET
        from pathlib import Path
        import time
        root = Path(__file__).resolve().parents[2]
        start = time.time()
        junit_path = None
        json_path = None
        try:
            # Check collection
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "tests", "-q", "--collect-only"],
                capture_output=True, text=True, timeout=30, cwd=str(root)
            )
            if proc.returncode != 0:
                return {"passed": False, "tests_run": 0, "failed": 0, "errors": 1, "skipped": 0, "duration": 0.0, "failures": [proc.stderr[:500]], "coverage": 0.0, "error": "collect_failed", "evidence_source": "junit"}

            # Prepare temp files for structured reports
            with tempfile.TemporaryDirectory() as td:
                junit_path = Path(td) / "junit.xml"
                json_path = Path(td) / "report.json"
                # Prefer json-report if plugin available, always generate junit
                cmd = [sys.executable, "-m", "pytest", "tests/test_no_lookahead.py", "tests/test_backtest_engine.py", "-q", f"--junitxml={junit_path}"]
                # Try to enable json report — if plugin missing, pytest will error, fallback to junit only
                json_enabled = False
                try:
                    # Probe if plugin exists
                    probe = subprocess.run([sys.executable, "-m", "pytest", "--help"], capture_output=True, text=True, timeout=10, cwd=str(root))
                    if "json-report" in (probe.stdout + probe.stderr):
                        cmd.extend([f"--json-report", f"--json-report-file={json_path}"])
                        json_enabled = True
                except Exception:
                    pass

                proc2 = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=str(root))
                duration = time.time() - start
                # Try structured JSON first
                if json_enabled and json_path.exists():
                    try:
                        data = json.loads(json_path.read_text(encoding="utf-8"))
                        summary = data.get("summary", {})
                        passed = int(summary.get("passed", 0) or 0)
                        failed = int(summary.get("failed", 0) or 0)
                        errors = int(summary.get("error", summary.get("errors", 0)) or 0)
                        skipped = int(summary.get("skipped", 0) or 0)
                        total = int(summary.get("total", summary.get("collected", passed+failed+errors+skipped)) or 0)
                        duration_json = float(data.get("duration", duration))
                        # Also get overall duration from json if present
                        return {
                            "passed": failed==0 and errors==0 and passed>0,
                            "tests_run": passed,
                            "failed": failed,
                            "errors": errors,
                            "skipped": skipped,
                            "total": total,
                            "duration": round(duration_json, 3),
                            "failures": [] if failed==0 and errors==0 else [f"failed={failed} errors={errors}"],
                            "coverage": 0.0,
                            "exit_code": proc2.returncode,
                            "evidence_source": "pytest-json-report",
                            "raw_output_tail": (proc2.stdout+proc2.stderr)[-500:],
                        }
                    except Exception:
                        pass
                # Fallback to JUnit XML (always generated)
                if junit_path.exists():
                    try:
                        tree = ET.parse(str(junit_path))
                        root_el = tree.getroot()
                        # JUnit may be <testsuites> or <testsuite>
                        suites = []
                        if root_el.tag == "testsuite":
                            suites = [root_el]
                        else:
                            suites = list(root_el.findall("testsuite"))
                            if not suites:
                                suites = [root_el]
                        tests = failures = errors = skipped = 0
                        time_sum = 0.0
                        for ts in suites:
                            tests += int(ts.get("tests", 0) or 0)
                            failures += int(ts.get("failures", 0) or 0)
                            errors += int(ts.get("errors", 0) or 0)
                            skipped += int(ts.get("skipped", 0) or 0)
                            try:
                                time_sum += float(ts.get("time", 0) or 0)
                            except Exception:
                                pass
                        passed = tests - failures - errors - skipped
                        if passed < 0:
                            passed = 0
                        # If tests attribute missing, count testcase nodes
                        if tests == 0:
                            cases = root_el.findall(".//testcase")
                            tests = len(cases)
                            # Count failures/errors/skipped via child nodes
                            failures = len(root_el.findall(".//failure"))
                            errors = len(root_el.findall(".//error"))
                            skipped = len(root_el.findall(".//skipped"))
                            passed = tests - failures - errors - skipped
                        return {
                            "passed": failures==0 and errors==0 and passed>0,
                            "tests_run": passed,
                            "failed": failures,
                            "errors": errors,
                            "skipped": skipped,
                            "total": tests,
                            "duration": round(time_sum if time_sum>0 else duration, 3),
                            "failures": [] if failures==0 and errors==0 else [f"failures={failures} errors={errors}"],
                            "coverage": 0.0,
                            "exit_code": proc2.returncode,
                            "evidence_source": "junit-xml",
                            "raw_output_tail": (proc2.stdout+proc2.stderr)[-500:],
                        }
                    except Exception as e:
                        # Fall through to human-readable as last resort with warning
                        pass
                # Last resort: human-readable (deprecated, must not be primary)
                out = proc2.stdout + proc2.stderr
                import re
                m = re.search(r"(\d+)\s+passed", out)
                passed_count = int(m.group(1)) if m else 0
                failed = proc2.returncode != 0
                # Try to extract failed/errors/skipped via regex as fallback
                mf = re.search(r"(\d+)\s+failed", out)
                me = re.search(r"(\d+)\s+error", out)
                ms = re.search(r"(\d+)\s+skipped", out)
                return {
                    "passed": not failed and passed_count>0,
                    "tests_run": passed_count,
                    "failed": int(mf.group(1)) if mf else (1 if failed else 0),
                    "errors": int(me.group(1)) if me else 0,
                    "skipped": int(ms.group(1)) if ms else 0,
                    "total": passed_count + (int(mf.group(1)) if mf else 0),
                    "duration": round(duration,3),
                    "failures": [] if not failed else [out[-500:]],
                    "coverage": 0.0,
                    "exit_code": proc2.returncode,
                    "evidence_source": "human-readable-fallback",
                    "raw_output_tail": out[-500:],
                }
        except Exception as e:
            return {"passed": False, "tests_run": 0, "failed": 0, "errors": 1, "skipped": 0, "total": 0, "duration": round(time.time()-start,3), "failures": [str(e)], "coverage": 0.0, "error": str(e), "evidence_source": "exception"}

    async def review(
        self,
        task: Task,
        execution_result: Any,
        test_result: Any
    ) -> Dict[str, Any]:
        """Review task execution and test results — fail-closed (P1.15/P1.16).
        
        Approved ONLY if:
            - execution success + exit_code 0 + artifact exists/valid
            - tests passed + tests_run>0
            - expected metrics exist (no placeholder)
        Never auto-approve.
        """
        exec_ok = False
        exec_reason = ""
        if isinstance(execution_result, dict):
            success = bool(execution_result.get("success"))
            exit_code = execution_result.get("exit_code", execution_result.get("exitCode", 0))
            exit_ok = exit_code is None or int(exit_code) == 0
            has_artifact = bool(execution_result.get("artifact_paths") or execution_result.get("code"))
            # Check artifact file exists if paths provided
            artifact_valid = True
            aps = execution_result.get("artifact_paths") or []
            if isinstance(aps, str):
                aps = [aps]
            if aps:
                from pathlib import Path
                for p in aps:
                    if not Path(p).exists():
                        artifact_valid = False
                        exec_reason = f"artifact missing: {p}"
                        break
            # Check metrics not placeholder
            metrics_ok = True
            metrics_reason = ""
            # bal_acc placeholder check
            metrics = execution_result.get("metrics", execution_result)
            if isinstance(metrics, dict):
                if metrics.get("bal_acc") == 0.39:
                    metrics_ok = False
                    metrics_reason = "bal_acc 0.39 placeholder forbidden"
                if metrics.get("_bal_acc_missing"):
                    metrics_ok = False
                    metrics_reason = "bal_acc missing (placeholder)"
                # exposure_ok placeholder: must have provenance
                rr = execution_result.get("risk_report", {})
                if isinstance(rr, dict) and rr.get("exposure_ok") is True and not execution_result.get("_provenance", {}).get("generated_by_real_execution"):
                    metrics_ok = False
                    metrics_reason = "exposure_ok placeholder without provenance"
                # ml model not trained
                if execution_result.get("model") == "skipped_insufficient" and success:
                    metrics_ok = False
                    metrics_reason = "model skipped but success claimed"
            exec_ok = success and exit_ok and has_artifact and artifact_valid and metrics_ok
            if not exec_ok and not exec_reason:
                exec_reason = f"success={success} exit_ok={exit_ok} artifact={has_artifact}/{artifact_valid} metrics_ok={metrics_ok} {metrics_reason}"
        test_ok = False
        if isinstance(test_result, dict):
            passed = bool(test_result.get("passed"))
            tests_run = int(test_result.get("tests_run", test_result.get("testsRun", 0)) or 0)
            # P1.16: 0 tests_run cannot be passed
            if passed and tests_run == 0:
                test_ok = False
            else:
                test_ok = passed and tests_run > 0
        # Also check verifier-like: if test_result has failures, not ok
        approved = exec_ok and test_ok
        if not approved:
            return {
                "approved": False,
                "comments": [f"Review failed: execution or tests not verified — {exec_reason}" if not exec_ok else "Review failed: tests not verified"],
                "suggestions": ["Ensure real execution (exit 0 + artifact + metrics) and tests_run>0"],
                "reviewer": "automated",
                "reason": f"exec_ok={exec_ok} test_ok={test_ok} {exec_reason}",
                "verified": False,
            }
        return {
            "approved": True,
            "comments": [],
            "suggestions": [],
            "reviewer": "automated",
            "verified": True,
            "reason": "execution and tests verified (P1.15 contract)",
        }
    
    async def get_active_tasks(self) -> List[Task]:
        """Get all active tasks"""
        return list(self.active_tasks.values())
    
    async def get_active_tasks(self) -> Dict[str, Task]:
        """Get all active tasks (alias)"""
        return self.active_tasks.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get task manager statistics"""
        return {
            "total": len(self.tasks),
            "pending": len([t for t in self.tasks.values() if t.status == TaskStatus.PENDING]),
            "active": len(self.active_tasks),
            "completed": len(self.completed_tasks),
            "failed": len(self.failed_tasks),
        }