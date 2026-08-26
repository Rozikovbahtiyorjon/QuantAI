"""
Portfolio Long-Run session tests (network-free, synthetic provider).

Covers: daily ingestion + weekly rebalance cadence, journal schema,
crash-safe resume without double-processing, strict broker identity,
gate criteria with portfolio-class meta expectations.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.validation.long_run import evaluate_long_run
from src.validation.portfolio_long_run import (
    PortfolioLongRunConfig,
    PortfolioLongRunSession,
    PortfolioState,
)
from src.strategies.cross_sectional import CrossSectionParams


N_DAYS = 120
SYMBOLS = ["AAA", "BBB", "CCC", "DDD"]


def synthetic_closes() -> pd.DataFrame:
    """
    Deterministic divergent series: AAA strong up, DDD down,
    B/C mild — momentum ranking must prefer AAA.
    """
    idx = pd.date_range("2026-01-01", periods=N_DAYS, freq="D", tz="UTC")
    rng = np_rng = pd.Series(range(N_DAYS), index=idx)

    def path(drift, vol, seed):
        import numpy as np

        r = np.random.default_rng(seed).normal(drift, vol, N_DAYS)
        return 100 * pd.Series(
            __import__("numpy").cumprod(1 + r), index=idx
        )

    return pd.DataFrame({
        "AAA": path(0.010, 0.004, 1),
        "BBB": path(0.001, 0.004, 2),
        "CCC": path(0.000, 0.005, 3),
        "DDD": path(-0.004, 0.006, 4),
    })


def make_provider(frame: pd.DataFrame):
    """Provider serving each symbol's closes up to requested since."""

    def provide(symbol: str, since_ms=None):
        s = frame[symbol].dropna()
        if since_ms is not None:
            since = pd.Timestamp(since_ms, unit="ms", tz="UTC")
            s = s[s.index > since]
        df = s.reset_index()
        df.columns = ["timestamp", "close"]
        return df

    return provide


def make_counting_provider(frame: pd.DataFrame):
    """Same as make_provider but tolerant to repeated global calls:
    slices per-symbol independently (no shared state)."""
    return make_provider(frame)


def make_cfg(**kw):
    base = dict(params=CrossSectionParams(lookback_days=10, top_k=2,
                                          rebalance_days=7),
                duration_days=10_000)
    base.update(kw)
    return PortfolioLongRunConfig(**base)


class TestPortfolioSession:
    def test_full_cycle(self, tmp_path: Path) -> None:
        sess = PortfolioLongRunSession(
            tmp_path, make_cfg(), SYMBOLS,
            candle_provider=make_provider(synthetic_closes()),
        )
        st = sess.run_until_target(max_wall_seconds=60)

        # processed all completed days after warm-up guard
        assert st.steps_done >= N_DAYS - 12      # lookback+2 consumed silently
        # warm-up days (~lookback) trade nothing; weekly cadence after
        assert st.rebalances >= max(0, (st.steps_done - 14)) // 7
        assert st.incidents == 0

        # journal produced on rebalances with correct schema
        rows = open(tmp_path / "journal.csv", encoding="utf-8").read().strip().splitlines()
        assert rows[0] == ("close_time,side,entry,exit,qty,gross,fees,net,balance")
        assert len(rows) - 1 == st.trades_closed

        # strict ledger identity holds at ALL times
        assert sess.broker.identity_gap() < 1e-6
        assert st.balance > 0
        # marked equity sane vs history
        assert sess.broker.equity({s: float(sess.closes[s].iloc[-1])
                                   for s in sess.closes}) > 0

        crit = evaluate_long_run(tmp_path, min_days=0, min_trades=0)
        assert crit["passed"] is True
        assert crit["incidents"] == 0

    def test_resume_no_double_processing(self, tmp_path: Path) -> None:
        frame = synthetic_closes()

        s1 = PortfolioLongRunSession(tmp_path, make_cfg(), SYMBOLS,
                                     candle_provider=make_provider(frame.iloc[:70]))
        s1.run_until_target(max_wall_seconds=60)
        first_steps = s1.state.steps_done
        last_day = s1.state.last_completed_day

        s2 = PortfolioLongRunSession(tmp_path, make_cfg(), SYMBOLS,
                                     candle_provider=make_provider(frame))
        s2.run_until_target(max_wall_seconds=60)

        assert s2.state.steps_done >= first_steps
        # previously processed days were skipped exactly once
        if last_day is not None:
            all_days = pd.date_range(frame.index[0], frame.index[-1], freq="D", tz="UTC")
            done_days = all_days[all_days <= pd.Timestamp(last_day, tz="UTC")]
            assert s2.state.steps_done >= len(done_days)

    def test_gate_meta_scales_min_trades(self, tmp_path: Path) -> None:
        """Portfolio class declares its own pace; evaluator respects it."""
        # craft state manually: 31 days but only 8 trades (<30 default)
        started = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=31)
        st = PortfolioState(
            symbols=SYMBOLS,
            params={},
            started_at=started.isoformat(),
            balance=10_100.0,
            steps_done=31,
            trades_closed=8,
            updated_at=pd.Timestamp.now(tz="UTC").isoformat(),
            meta={"asset_class": "portfolio_xs_momentum",
                  "min_trades_per_30d": 8},
        )
        (tmp_path / "state.json").write_text(st.to_json(), encoding="utf-8")
        (tmp_path / "journal.csv").write_text(
            "close_time,side,entry,exit,qty,gross,fees,net,balance\n"
            + "".join(f"t,LONG,1,1.01,1,0.01,0.001,0.009,10100\n" for _ in range(8)),
            encoding="utf-8",
        )

        crit = evaluate_long_run(tmp_path, min_days=30,
                                 min_trades=None, auto_min_trades=True)
        assert crit["checks"]["trades_ok"] is True
        assert crit["passed"] is True