"""
P-C1 Cross-Sectional Momentum tests.

    - planted momentum -> correct picks
    - causal: future shock cannot change past holdings/equity
    - costs reduce returns monotonically
    - portfolio adapter plugs into the R4 promotion contract
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.champion.evaluation_pipeline import PromotionRules
from src.champion.pipeline import ChampionPipeline
from src.champion.portfolio_adapter import evaluate_portfolio_candidate
from src.strategies.cross_sectional import CrossSectionParams, backtest
from src.strategy_bank import StrategyRegistry


def make_wide(rows=420, seed=11):
    """
    Planted structure: A strong up-drift, C down-drift,
    B flat, D/E noisy — momentum ranking must prefer A.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=rows, freq="D")

    def path(drift, vol):
        return 100 * np.cumprod(1 + rng.normal(drift, vol, rows))

    return pd.DataFrame({
        "A": path(0.004, 0.004),
        "B": path(0.000, 0.003),
        "C": path(-0.002, 0.005),
        "D": path(0.0005, 0.010),
        "E": path(-0.0005, 0.008),
    }, index=idx)


class TestBacktest:
    def test_picks_the_trending_asset(self) -> None:
        wide = make_wide()
        res = backtest(wide, CrossSectionParams(lookback_days=20, top_k=1,
                                                rebalance_days=7))
        picked_counts: dict[str, int] = {}
        for per in res["periods"]:
            for name in per["picked"]:
                picked_counts[name] = picked_counts.get(name, 0) + 1

        assert picked_counts.get("A", 0) > 0
        assert picked_counts["A"] == max(picked_counts.values())
        # downtrending asset never selected as the single top pick
        assert "C" not in picked_counts

    def test_causality_future_shock_invariant(self) -> None:
        wide = make_wide(rows=420)

        cut = 360
        params = CrossSectionParams(lookback_days=14, top_k=1, rebalance_days=7)

        base = backtest(wide.iloc[:cut], params)

        shocked = wide.copy()
        shocked.iloc[cut:] *= 2.0                      # violent future
        shocked = backtest(shocked, params)

        # periods whose FULL holding window lies before the cut
        safe_end = wide.index[cut - params.rebalance_days]
        b_periods = [p for p in base["periods"] if p["end"] <= safe_end]
        s_periods = [p for p in shocked["periods"] if p["end"] <= safe_end]

        assert len(b_periods) > 10
        assert [(p["picked"], round(p["net"], 8)) for p in b_periods] == \
               [(p["picked"], round(p["net"], 8)) for p in s_periods]

    def test_costs_reduce_returns(self) -> None:
        wide = make_wide()
        p_free = CrossSectionParams(lookback_days=14, top_k=2, fee_per_side=0.0)
        p_cost = CrossSectionParams(lookback_days=14, top_k=2, fee_per_side=0.002)

        r_free = backtest(wide, p_free)["stats"]["total_ret_pct"]
        r_cost = backtest(wide, p_cost)["stats"]["total_ret_pct"]

        assert r_cost <= r_free

    def test_insufficient_history_raises(self) -> None:
        with pytest.raises(ValueError):
            backtest(make_wide(30), CrossSectionParams(lookback_days=60))


# ============================================================
# ADAPTER + PROMOTION CONTRACT
# ============================================================

class TestAdapterAndPromotion:
    def _pipeline(self, tmp_path):
        rules = PromotionRules(
            min_pf_median=0.0,
            min_profitable_window_share=0.0,
            max_drawdown_median_pct=-10_000.0,
            min_net_median_pct=-10_000.0,
            min_trades_total=0,
            max_net_std_pct=10_000.0,
        )
        return ChampionPipeline(registry=StrategyRegistry(), rules=rules,
                                store_path=tmp_path / "champ.json")

    @staticmethod
    def _genome(sid):
        from src.strategy_genome import StrategyGenome

        return StrategyGenome(
            strategy_id=sid, version="1", market="crypto",
            timeframes=("1D",), features=("mom",), indicators=("ret",),
            ml_model="none", regime_filters=("none",),
            entry_logic={}, exit_logic={}, risk_profile="portfolio",
            position_sizing={}, portfolio_constraints={},
        )

    def test_adapter_metrics_schema(self) -> None:
        wide = make_wide()
        spec = __import__("src.champion.evaluation_pipeline",
                          fromlist=["CandidateSpec"]).CandidateSpec(
            "xs", lambda: CrossSectionParams(lookback_days=14, top_k=1)
        )
        res = evaluate_portfolio_candidate(spec, wide, test_days=120)

        m = res["metrics"]
        for key in ("net_median_pct", "pf_median", "profitable_window_share",
                    "maxdd_median_pct", "trades", "win_rate", "windows"):
            assert key in m
        assert m["windows"] >= 2

    def test_end_to_end_promotion(self, tmp_path) -> None:
        from src.champion.evaluation_pipeline import CandidateSpec

        wide = make_wide()
        pipe = self._pipeline(tmp_path)

        for sid, k in (("xs_k1", 1), ("xs_k2", 2)):
            pipe.submit_candidate(
                CandidateSpec(sid, lambda k=k: CrossSectionParams(
                    lookback_days=14, top_k=k)),
                self._genome(sid),
            )

        evals = pipe.evaluate_all(
            wide,
            evaluate_fn=evaluate_portfolio_candidate,
            test_days=120,
        )
        dec = pipe.decide_promotion(evals)

        assert dec["promoted"] is True
        assert pipe.current_champion_id() in {"xs_k1", "xs_k2"}
        pipe.save()
        assert (tmp_path / "champ.json").exists()