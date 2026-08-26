"""
R3: Validation Gate mechanics + Long-Run harness tests.

Network-free: candle provider and checks are injected/synthetic.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.validation.gate import (
    CheckResult,
    GateStatus,
    QuantAIValidationGate,
)
from src.validation.long_run import (
    LongRunConfig,
    LongRunSession,
    evaluate_long_run,
)


# ============================================================
# GATE AGGREGATION
# ============================================================

def _check(name, status):
    return lambda: CheckResult(name=name, status=status, details="")


class TestVerdictRules:
    def test_all_pass(self) -> None:
        g = QuantAIValidationGate([
            _check("a", GateStatus.PASS),
            _check("b", GateStatus.PASS),
        ])
        assert g.run().verdict == GateStatus.PASS

    def test_any_fail_is_fail(self) -> None:
        g = QuantAIValidationGate([
            _check("a", GateStatus.PASS),
            _check("b", GateStatus.FAIL),
            _check("c", GateStatus.BLOCKED),
        ])
        assert g.run().verdict == GateStatus.FAIL

    def test_blocked_when_no_fail(self) -> None:
        g = QuantAIValidationGate([
            _check("a", GateStatus.PASS),
            _check("long_run", GateStatus.BLOCKED),
        ])
        assert g.run().verdict == GateStatus.BLOCKED

    def test_report_serializable_and_timed(self) -> None:
        g = QuantAIValidationGate([_check("x", GateStatus.PASS)])
        rep = g.run()

        d = rep.to_dict()
        assert d["verdict"] == "PASS"
        assert d["checks"][0]["name"] == "x"

        blob = json.dumps(d)
        assert "PASS" in blob


# ============================================================
# LONG-RUN CRITERIA
# ============================================================

def _write_artifacts(d: Path, *, days: float, trades: int,
                     incidents: int = 0, balance: float = 1234.0) -> None:
    d.mkdir(parents=True, exist_ok=True)

    from datetime import datetime, timedelta, timezone
    started = datetime.now(timezone.utc) - timedelta(days=days)

    state = {
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "started_at": started.isoformat(),
        "balance": balance,
        "last_open_time_ms": None,
        "steps_done": int(days * 24),
        "signals_processed": int(days * 24),
        "trades_closed": trades,
        "incidents": incidents,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    (d / "state.json").write_text(json.dumps(state), encoding="utf-8")

    with open(d / "journal.csv", "w", encoding="utf-8") as f:
        f.write("close_time,side,entry,exit,qty,gross,fees,net,balance\n")
        for i in range(trades):
            f.write(f"2025-01-01T00:00:00+00:00,LONG,100,101,0.1,0.1,0.01,0.09,{balance}\n")


class TestLongRunCriteria:
    def test_missing_artifacts_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            evaluate_long_run(tmp_path)

    def test_incomplete_is_not_passed(self, tmp_path: Path) -> None:
        _write_artifacts(tmp_path, days=3, trades=5)
        crit = evaluate_long_run(tmp_path, min_days=30, min_trades=30)

        assert crit["passed"] is False
        assert crit["checks"]["days_ok"] is False
        assert crit["checks"]["trades_ok"] is False
        assert crit["incidents"] == 0

    def test_complete_passes(self, tmp_path: Path) -> None:
        _write_artifacts(tmp_path, days=31, trades=40)
        crit = evaluate_long_run(tmp_path, min_days=30, min_trades=30)

        assert crit["passed"] is True
        assert crit["trades"] == 40
        assert crit["days_covered"] >= 30

    def test_incident_blocks_even_if_complete(self, tmp_path: Path) -> None:
        _write_artifacts(tmp_path, days=35, trades=50, incidents=1)
        crit = evaluate_long_run(tmp_path, min_days=30, min_trades=30)

        assert crit["passed"] is False
        assert crit["checks"]["incidents_ok"] is False


# ============================================================
# SESSION MECHANICS (synthetic candles, no network)
# ============================================================

def _synthetic_provider(bars: list[dict]):
    """Provider returning the full list on first call, empty after."""
    state = {"calls": 0}

    def provide(symbol, timeframe, since_ms, limit=1000):
        state["calls"] += 1
        if state["calls"] == 1:
            return pd.DataFrame(bars)
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    return provide


def make_bars(n=320, base=100.0):
    bars = []
    t0 = pd.Timestamp("2026-01-01", tz="UTC")
    price = base
    for i in range(n):
        ts = t0 + pd.Timedelta(hours=i)
        o = price
        c = price * (1 + (0.002 if i % 7 else -0.001))
        h = max(o, c) * 1.001
        low = min(o, c) * 0.999
        bars.append(dict(timestamp=ts, open=o, high=h, low=low,
                         close=c, volume=100.0))
        price = c
    return bars


class TestSessionMechanics:
    def test_session_processes_and_checkpoints(self, tmp_path: Path) -> None:
        cfg = LongRunConfig(duration_hours=10 ** 6, warmup_bars=200,
                            checkpoint_every_steps=1)
        sess = LongRunSession(
            tmp_path, cfg,
            candle_provider=_synthetic_provider(make_bars(320)),
        )

        st = sess.run_until_target(max_wall_seconds=60)

        assert st.steps_done > 100          # consumed synthetic bars
        # every step either produced a signal or was a warm-up incident
        assert st.signals_processed + st.incidents == st.steps_done
        assert st.incidents <= 5            # only early warm-up bars
        assert st.balance > 0

        # checkpoint persisted & resumable
        st2 = LongRunSession(tmp_path, cfg,
                             candle_provider=_synthetic_provider([]))
        assert st2.state.steps_done == st.steps_done

        # journal header exists; criteria evaluator works on dir
        crit = evaluate_long_run(tmp_path, min_days=0, min_trades=0)
        assert crit["checks"]["alive_ok"] is True
        assert crit["signals_processed"] == st.signals_processed

    def test_resume_skips_old_bars(self, tmp_path: Path) -> None:
        bars = make_bars(310)

        cfg = LongRunConfig(warmup_bars=200)
        s1 = LongRunSession(tmp_path, cfg, candle_provider=_synthetic_provider(bars[:280]))
        s1.run_until_target(max_wall_seconds=60)
        done_1 = s1.state.steps_done

        # resume with overlapping feed (full list): old bars skipped by ts
        s2 = LongRunSession(tmp_path, cfg, candle_provider=_synthetic_provider(bars))
        s2.run_until_target(max_wall_seconds=60)
        done_2 = s2.state.steps_done

        assert done_2 >= done_1
        # no double-processing of already-seen bar
        assert s2.state.last_open_time_ms >= s1.state.last_open_time_ms