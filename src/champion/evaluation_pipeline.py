"""
QuantAI Candidate Evaluation Pipeline (R4)

Turns a CandidateSpec into an objective MetricsVector using the SAME
walk-forward harness validated in research phases:

    CandidateSpec --WF windows--> TradeEngine runs --> vector + rule gates

No profitability assumptions here: the pipeline MEASURES, rules DECIDE.
"""

from __future__ import annotations

import statistics as stats
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

import src.trade_engine as te_mod
from src.backtest_engine import BacktestEngine
from src.trade_engine import ExitPolicy, TradeEngine
from src.walk_forward_engine import WalkForwardEngine


# =====================================================
# SPEC
# =====================================================

@dataclass
class CandidateSpec:
    """
    Executable candidate description.

    factory: zero-arg callable returning a generator object with
             .generate(df) -> SignalResult (SignalGenerator-compatible).
    params:  free-form parameters snapshot (persisted into genome).
    """
    name: str
    factory: Callable[[], Any]
    params: dict = field(default_factory=dict)


# =====================================================
# RULES
# =====================================================

@dataclass
class PromotionRules:
    """Objective promotion thresholds (all must hold)."""

    min_pf_median: float = 1.05
    min_profitable_window_share: float = 0.45
    max_drawdown_median_pct: float = -15.0     # median must be >= this
    min_net_median_pct: float = 0.0
    min_trades_total: int = 30
    max_net_std_pct: float = 10.0              # stability cap

    def evaluate_flags(self, m: dict) -> dict[str, bool]:
        return {
            "pf_ok": m["pf_median"] >= self.min_pf_median,
            "window_share_ok": (
                m["profitable_window_share"] >= self.min_profitable_window_share
            ),
            "drawdown_ok": m["maxdd_median_pct"] >= self.max_drawdown_median_pct,
            "net_ok": m["net_median_pct"] >= self.min_net_median_pct,
            "trades_ok": m["trades"] >= self.min_trades_total,
            "stability_ok": m["net_std_pct"] <= self.max_net_std_pct,
        }

    @staticmethod
    def all_pass(flags: dict[str, bool]) -> bool:
        return all(flags.values())


DEFAULT_POLICY = ExitPolicy(use_take_profit=False, break_even_atr=None,
                            trail_atr_mult=3.0)


# =====================================================
# EVALUATION
# =====================================================

def _cap_pf(pf: float) -> float:
    return 99.0 if pf == float("inf") else pf


def aggregate_windows(window_stats: list[dict]) -> dict:
    nets = [w["net_pct"] for w in window_stats]
    pfs = [w["pf"] for w in window_stats]
    dds = [w["maxdd_pct"] for w in window_stats]
    sharpes = [w["sharpe"] for w in window_stats]

    trades = sum(w["trades"] for w in window_stats)
    wins = sum(w["wins"] for w in window_stats)

    return {
        "net_median_pct": stats.median(nets) if nets else 0.0,
        "net_mean_pct": stats.mean(nets) if nets else 0.0,
        "net_std_pct": stats.pstdev(nets) if len(nets) > 1 else 0.0,
        "pf_median": stats.median(pfs) if pfs else 0.0,
        "maxdd_median_pct": stats.median(dds) if dds else 0.0,
        "sharpe_median": stats.median(sharpes) if sharpes else 0.0,
        "profitable_window_share": (
            sum(1 for n in nets if n > 0) / len(nets) if nets else 0.0
        ),
        "trades": trades,
        "win_rate": 100.0 * wins / trades if trades else 0.0,
        "windows": len(nets),
    }


def evaluate_candidate(
    spec: CandidateSpec,
    df: pd.DataFrame,
    *,
    train_size: int = 2000,
    test_size: int = 500,
    step_size: int = 500,
    initial_balance: float = 1000.0,
    exit_policy: ExitPolicy | None = None,
    costs: dict | None = None,
    warmup_bars: int = 0,
    history_window: int = 300,
) -> dict:
    """
    Walk-forward evaluation of one candidate.

    Returns {"metrics": aggregated vector, "windows": [...],
             "rules": PromotionRules-evaluated later}.
    """
    policy = exit_policy or DEFAULT_POLICY
    costs = costs or {"commission": 0.0004, "slippage": 0.0002}

    eng = WalkForwardEngine(
        train_size=train_size, test_size=test_size,
        step_size=step_size, initial_balance=initial_balance,
    )

    saved_c, saved_s = te_mod.COMMISSION, te_mod.SLIPPAGE
    te_mod.COMMISSION = costs["commission"]
    te_mod.SLIPPAGE = costs["slippage"]

    window_stats: list[dict] = []
    try:
        for _wnum, _tr, test_df in eng.generate_windows(df):
            te = TradeEngine(exit_policy=policy)
            te.initial_balance = te.balance = te.equity = float(initial_balance)

            te.run(
                test_df,
                history_window=history_window,
                warmup_bars=warmup_bars,
                signal_generator=spec.factory(),
            )

            m = BacktestEngine._compute_risk_metrics(te, initial_balance)
            final = te.balance
            trades_n = len(te.closed_positions)
            wins_n = sum(1 for p in te.closed_positions if p.net_profit > 0)

            window_stats.append({
                "net_pct": (final - initial_balance) / initial_balance * 100.0,
                "pf": _cap_pf(m["profit_factor"]),
                "maxdd_pct": m["max_drawdown_pct"],
                "sharpe": m["sharpe"],
                "trades": trades_n,
                "wins": wins_n,
            })
    finally:
        te_mod.COMMISSION, te_mod.SLIPPAGE = saved_c, saved_s

    return {
        "metrics": aggregate_windows(window_stats),
        "windows": window_stats,
    }


__all__ = [
    "CandidateSpec",
    "PromotionRules",
    "DEFAULT_POLICY",
    "aggregate_windows",
    "evaluate_candidate",
]
