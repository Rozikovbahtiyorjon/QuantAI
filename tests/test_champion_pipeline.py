"""
R4 Champion Pipeline tests.

End-to-end cycle on deterministic stub strategies (network-free):
    submit -> WF evaluate -> rules -> tournament rank ->
    promotion vs champion -> rollback on degradation -> persistence

Plus feedback loop: journal telemetry + deterministic mutations.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.champion.evaluation_pipeline import (
    CandidateSpec,
    PromotionRules,
)
from src.champion.feedback import (
    MutationBounds,
    feedback_from_long_run,
    params_differ,
    suggest_mutations,
)
from src.champion.pipeline import ChampionPipeline
from src.strategy.signal_generator import SignalResult
from src.strategy_bank import StrategyRegistry


# ============================================================
# FIXTURES: synthetic prepared data + stub candidates
# ============================================================

def make_prepared(rows=900, seed=7, drift=0.0012, vol=0.003):
    import numpy as np
    from src.indicators import add_indicators

    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, rows)
    close = 100 * np.cumprod(1 + rets)

    o = np.empty(rows); o[0] = 100.0; o[1:] = close[:-1]
    spread = np.abs(rng.normal(0, 0.0008, rows)) * close
    ts = pd.date_range("2026-01-01", periods=rows, freq="15min")

    return add_indicators(pd.DataFrame({
        "timestamp": ts, "open": o,
        "high": np.maximum(o, close) + spread,
        "low": np.minimum(o, close) - spread,
        "close": close, "volume": rng.uniform(50, 200, rows),
    }))


class StubStrategy:
    """
    Deterministic momentum-ish stub:
    BUY when fast>trend and close above; SELL mirrored.
    SL/TP geometry fixed; exit policy (trail-only) drives outcomes.
    """

    def __init__(self, aggressiveness: float = 0.0, sl_mult: float = 3.0):
        self.agg = aggressiveness
        self.sl = sl_mult

    def generate(self, df: pd.DataFrame) -> SignalResult:
        row = df.iloc[-1]
        r = SignalResult()
        r.entry = float(row["close"])
        atr = float(row.get("atr", 0.0)) or 1e-9
        r.stop_loss = r.entry - self.sl * atr
        r.take_profit = r.entry + 2 * self.sl * atr
        r.confidence = 70.0

        long_cond = (row["ema_fast"] > row["ema_trend"]
                     and row["close"] > row["ema_trend"] * (1 + self.agg))
        short_cond = (row["ema_fast"] < row["ema_trend"]
                      and row["close"] < row["ema_trend"] * (1 - self.agg))

        if long_cond:
            r.signal = "BUY"
        elif short_cond:
            r.signal = "SELL"
        else:
            r.signal = "HOLD"
        return r


def make_genome(sid: str, params: dict | None = None):
    from src.strategy_genome import StrategyGenome

    return StrategyGenome(
        strategy_id=sid,
        version="1.0",
        market="crypto",
        timeframes=("1h",),
        features=("core4",),
        indicators=("ema", "atr"),
        ml_model="none",
        regime_filters=("none",),
        entry_logic={"type": "stub"},
        exit_logic={"trail_atr": 3.0},
        risk_profile="conservative",
        position_sizing={"risk_percent": 1.0},
        portfolio_constraints={},
        parameters=params or {},
    )


@pytest.fixture(scope="module")
def wf_df() -> pd.DataFrame:
    return make_prepared()


EVAL_KW = dict(train_size=400, test_size=100, step_size=100)


class TestEvaluationPipeline:
    def test_evaluation_produces_vector(self, wf_df) -> None:
        from src.champion.evaluation_pipeline import evaluate_candidate

        spec = CandidateSpec("stub0", lambda: StubStrategy())
        res = evaluate_candidate(spec, wf_df, costs={"commission": 0.0, "slippage": 0.0}, **EVAL_KW)

        m = res["metrics"]
        assert m["windows"] > 0
        assert m["windows"] == len(res["windows"])
        assert 0.0 <= m["profitable_window_share"] <= 1.0
        assert m["maxdd_median_pct"] <= 0.0

    def test_rules_gate(self, wf_df) -> None:
        from src.champion.evaluation_pipeline import evaluate_candidate

        strict = PromotionRules(min_pf_median=99.0)     # impossible
        loose = PromotionRules(
            min_pf_median=0.0,
            min_profitable_window_share=0.0,
            max_drawdown_median_pct=-10_000.0,
            min_net_median_pct=-10_000.0,
            min_trades_total=0,
            max_net_std_pct=10_000.0,
        )

        spec = CandidateSpec("stub", lambda: StubStrategy())

        res = evaluate_candidate(spec, wf_df, costs={"commission": 0.0, "slippage": 0.0}, **EVAL_KW)

        assert not all(strict.evaluate_flags(res["metrics"]).values())
        assert all(loose.evaluate_flags(res["metrics"]).values())


class TestPromotionCycle:
    def _pipeline(self, tmp_path):
        rules = PromotionRules(
            min_pf_median=0.0, min_profitable_window_share=0.0,
            max_drawdown_median_pct=-100.0, min_net_median_pct=-100.0,
            min_trades_total=0, max_net_std_pct=10_000.0,
        )
        return ChampionPipeline(
            registry=StrategyRegistry(),
            rules=rules,
            store_path=tmp_path / "champions.json",
        )

    def test_promotion_and_history(self, tmp_path, wf_df) -> None:
        pipe = self._pipeline(tmp_path)

        pipe.submit_candidate(
            CandidateSpec("alpha", lambda: StubStrategy()),
            make_genome("alpha"),
        )
        evals = pipe.evaluate_all(wf_df, costs={"commission": 0.0, "slippage": 0.0}, **EVAL_KW)
        dec = pipe.decide_promotion(evals)

        assert dec["promoted"] is True
        assert dec["to"] == "alpha"
        assert pipe.current_champion_id() == "alpha"
        assert any(h.event == "promotion" for h in pipe.history)

    def test_beats_champion_required(self, tmp_path, wf_df) -> None:
        pipe = self._pipeline(tmp_path)

        # alpha becomes champion first
        pipe.submit_candidate(CandidateSpec("alpha", lambda: StubStrategy()),
                              make_genome("alpha"))
        evals = pipe.evaluate_all(wf_df, costs={"commission": 0.0, "slippage": 0.0}, **EVAL_KW)
        assert pipe.decide_promotion(evals)["promoted"]

        # beta identical to champion -> cannot beat it (improvement=0)
        pipe.submit_candidate(
            CandidateSpec("beta", lambda: StubStrategy(sl_mult=3.0)),
            make_genome("beta"),
        )
        evals2 = pipe.evaluate_all(wf_df, costs={"commission": 0.0, "slippage": 0.0}, **EVAL_KW)
        dec = pipe.decide_promotion(evals2)

        if dec.get("strategy_id") == "beta":
            assert dec["promoted"] is False
            assert "champion" in dec["reason"]

    def test_rollback_on_degradation(self, tmp_path, wf_df) -> None:
        pipe = self._pipeline(tmp_path)
        pipe.submit_candidate(CandidateSpec("alpha", lambda: StubStrategy()),
                              make_genome("alpha"))
        evals = pipe.evaluate_all(wf_df, costs={"commission": 0.0, "slippage": 0.0}, **EVAL_KW)
        assert pipe.decide_promotion(evals)["promoted"]

        bad_eval = {"rules_passed": False}
        rb = pipe.rollback_if_degraded(bad_eval)

        # no previous champion in stack -> stays but reports honestly
        assert rb["rolled_back"] is False
        assert "no previous" in rb["reason"]

    def test_persistence_roundtrip(self, tmp_path, wf_df) -> None:
        store = tmp_path / "champions.json"
        pipe = self._pipeline(tmp_path)
        pipe.submit_candidate(CandidateSpec("alpha", lambda: StubStrategy(), {"x": 1}),
                              make_genome("alpha", {"x": 1}))
        evals = pipe.evaluate_all(wf_df, costs={"commission": 0.0, "slippage": 0.0}, **EVAL_KW)
        pipe.decide_promotion(evals)
        pipe.save()

        assert store.exists()

        pipe2 = ChampionPipeline(
            registry=StrategyRegistry(),
            rules=pipe.rules,
            store_path=store,
        )
        pipe2._load()

        assert pipe2.current_champion_id() == "alpha"
        assert any(h.event == "promotion" for h in pipe2.history)
        assert pipe2.loaded_params.get("alpha", {}).get("x") == 1


# ============================================================
# FEEDBACK LOOP
# ============================================================

class TestFeedback:
    def test_feedback_from_journal(self, tmp_path: Path) -> None:
        journal = tmp_path / "journal.csv"
        journal.write_text(
            "close_time,side,entry,exit,qty,gross,fees,net,balance\n"
            "t,LONG,100,101,1,1,0.1,0.9,1000\n"
            "t,SHORT,101,100,1,1,0.1,0.9,1001\n"
            "t,LONG,100,99,1,-1,0.1,-1.1,1000\n",
            encoding="utf-8",
        )

        fb = feedback_from_long_run(tmp_path)

        assert fb.trades == 3
        assert fb.wins == 2
        assert fb.win_rate == pytest.approx(66.66, abs=0.1)
        assert fb.net_pnl == pytest.approx(0.9 + 0.9 - 1.1)
        assert fb.balance == pytest.approx(1000.0)

    def test_mutations_deterministic_and_bounded(self) -> None:
        bounds = MutationBounds(bounds={
            "trail_atr_mult": (1.5, 5.0, 0.5),
            "min_adx": (15.0, 35.0, 5.0),
        })
        base = {"trail_atr_mult": 3.0, "min_adx": 20.0}

        v1 = suggest_mutations(base, bounds, max_variants=10)
        v2 = suggest_mutations(base, bounds, max_variants=10)

        assert v1 == v2                       # deterministic
        assert len(v1) == 4                   # 2 params x +-1 step
        for variant in v1:
            assert params_differ(variant, base)
            for k, (lo, hi, _) in bounds.bounds.items():
                assert lo <= variant[k] <= hi

    def test_mutation_cap_respected(self) -> None:
        bounds = MutationBounds(bounds={
            f"p{i}": (0.0, 10.0, 1.0) for i in range(5)
        })
        variants = suggest_mutations({"p0": 5, "p1": 5}, bounds, max_variants=3)
        assert len(variants) == 3

class TestGovernanceR4A:
    def _pipe(self, tmp_path):
        rules = PromotionRules(
            min_pf_median=0.0, min_profitable_window_share=0.0,
            max_drawdown_median_pct=-10_000.0, min_net_median_pct=-10_000.0,
            min_trades_total=0, max_net_std_pct=10_000.0,
        )
        return ChampionPipeline(registry=StrategyRegistry(), rules=rules,
                                store_path=tmp_path / "c.json")

    @staticmethod
    def _genome(sid):
        from src.strategy_genome import StrategyGenome

        return StrategyGenome(
            strategy_id=sid, version="1", market="crypto",
            timeframes=("1D",), features=("f",), indicators=("i",),
            ml_model="none", regime_filters=("none",),
            entry_logic={}, exit_logic={}, risk_profile="r",
            position_sizing={}, portfolio_constraints={},
        )

    def test_resubmit_is_idempotent(self, tmp_path) -> None:
        from src.champion.evaluation_pipeline import CandidateSpec

        pipe = self._pipe(tmp_path)
        spec = CandidateSpec("a", lambda: None)

        pipe.submit_candidate(spec, self._genome("a"))
        n_first = pipe.registry.count()
        pipe.submit_candidate(spec, self._genome("a"))

        assert pipe.registry.count() == n_first
        assert any(h.event == "resubmit" for h in pipe.history)

    def test_review_flags_failing_champion_without_demotion(self, tmp_path) -> None:
        pipe = self._pipe(tmp_path)
        pipe.submit_candidate(CandidateSpec("a", lambda: None), self._genome("a"))
        pipe.registry.update_status("a", "champion")
        pipe.registry.set_champion("a")

        evals = {"a": {"rules_passed": False, "rules_flags": {"pf_ok": False}}}
        res = pipe.review_champion(evals)

        assert res["flagged"] is True
        # status unchanged - system stays deployable
        assert pipe.registry.get("a").status == "champion"
        assert any(h.event == "champion_under_review" for h in pipe.history)

    def test_review_recovers_on_pass(self, tmp_path) -> None:
        pipe = self._pipe(tmp_path)
        pipe.submit_candidate(CandidateSpec("a", lambda: None), self._genome("a"))
        pipe.registry.update_status("a", "champion")
        pipe.registry.set_champion("a")

        fail = {"a": {"rules_passed": False, "rules_flags": {"pf_ok": False}}}
        pipe.review_champion(fail)

        ok = {"a": {"rules_passed": True, "rules_flags": {}}}
        res = pipe.review_champion(ok)

        assert res.get("recovered") is True
        assert any(h.event == "champion_recovered" for h in pipe.history)

    def test_flag_survives_persistence(self, tmp_path) -> None:
        pipe = self._pipe(tmp_path)
        pipe.submit_candidate(CandidateSpec("a", lambda: None), self._genome("a"))
        pipe.registry.update_status("a", "champion")
        pipe.registry.set_champion("a")

        fail = {"a": {"rules_passed": False, "rules_flags": {}}}
        pipe.review_champion(fail)
        pipe.save()

        pipe2 = ChampionPipeline(registry=StrategyRegistry(),
                                 rules=pipe.rules,
                                 store_path=tmp_path / "c.json")
        pipe2._load()

        assert getattr(pipe2, "flag:a", False) is True
