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
import io
import json
import subprocess
import sys
import time
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class GateStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    BLOCKED = "BLOCKED"


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

    @property
    def verdict(self) -> GateStatus:
        statuses = [r.status for r in self.results]
        if GateStatus.FAIL in statuses:
            return GateStatus.FAIL
        if GateStatus.BLOCKED in statuses:
            return GateStatus.BLOCKED
        return GateStatus.PASS

    def to_dict(self) -> dict:
        return {
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

    def pretty(self) -> str:
        lines = ["=" * 64, f"QUANTAI VALIDATION GATE :: {self.verdict.value}", "=" * 64]
        for r in self.results:
            lines.append(
                f"[{r.status.value:<7}] {r.name:<24} "
                f"{r.duration_s:>6.1f}s  {r.details}"
            )
        lines.append("=" * 64)
        return "\n".join(lines)


def _run_pytest(targets: list[str], timeout: int = 900) -> tuple[bool, str, dict]:
    """
    Run pytest in a subprocess against selected targets.
    Uses the same interpreter running the gate.
    """
    cmd = [sys.executable, "-m", "pytest", *targets, "-q", "--no-header", "-p", "no:cacheprovider"]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(Path(__file__).resolve().parents[2]),
        )
    except subprocess.TimeoutExpired:
        return False, f"pytest timed out after {timeout}s", {}

    tail = "\n".join((proc.stdout or "").splitlines()[-6:])

    # parse "N passed, M failed" summary line
    passed = failed = errors = 0
    for line in (proc.stdout or "").splitlines():
        if "passed" in line or "failed" in line or "error" in line:
            for part in line.replace(",", " ").split():
                if part.endswith("passed"):
                    try:
                        passed = int(part.split("passed")[0] or 0)
                    except ValueError:
                        pass
                elif part.endswith("failed"):
                    try:
                        failed = int(part.split("failed")[0] or 0)
                    except ValueError:
                        pass
                elif part.endswith("error"):
                    try:
                        errors = int(part.split("error")[0] or 0)
                    except ValueError:
                        pass

    ok = proc.returncode == 0
    return ok, tail, {"passed": passed, "failed": failed, "errors": errors}


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


def check_backtest_smoke(data_dir: Path) -> CheckResult:
    """
    Health check (NOT profitability): the engine must run real prepared
    data end-to-end and produce structurally sane metrics.
    Strategy PnL sign is a research concern (profitability track),
    never a deployment gate.
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
                "note": "gate=health; profitability tracked separately",
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


def math_isfinite(x: float) -> bool:
    x = float(x)
    return x == x and abs(x) != float("inf")


def check_walk_forward_smoke(data_dir: Path) -> CheckResult:
    t0 = time.time()

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
        path = candidates[0]

    try:
        import pandas as pd

        from src.walk_forward_engine import WalkForwardEngine

        df = pd.read_parquet(path).head(4000)
        eng = WalkForwardEngine(train_size=1000, test_size=250, step_size=250,
                                initial_balance=1000.0)
        result = eng.run(df)

        ok = result.total_windows > 0
        return CheckResult(
            name="walk_forward",
            status=GateStatus.PASS if ok else GateStatus.FAIL,
            details=f"{result.total_windows} windows",
            metrics={"windows": result.total_windows,
                     "trades": result.total_trades},
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

    def run(self) -> GateReport:
        report = GateReport()
        for check in self.checks:
            report.results.append(check())
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
    gate.add(check_risk_gates)
    gate.add(lambda: check_backtest_smoke(data_dir))
    gate.add(lambda: check_walk_forward_smoke(data_dir))
    gate.add(lambda: check_long_run_evidence(long_run_dir, min_days, min_trades))
    return gate


if __name__ == "__main__":
    args = set(sys.argv[1:])
    gate = build_standard_gate(include_pytest_full="--fast" not in args)

    report = gate.run()
    print(report.pretty())

    out = root_json = Path(__file__).resolve().parents[2] / "data" / "gate_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")
    print(f"json: {out}")

    sys.exit(0 if report.verdict == GateStatus.PASS else 1)
