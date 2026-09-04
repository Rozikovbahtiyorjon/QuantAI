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
    with_is_metrics: bool = True,
) -> dict:
    """
    Walk-forward evaluation of one candidate.

    Returns {
        "metrics": OOS aggregated vector,
        "windows": OOS per-window stats,
        "is_metrics": IS aggregated (if with_is_metrics),
        "is_windows": IS per-window,
        "is_sharpes": list IS Sharpe per window,
        "oos_sharpes": list OOS Sharpe per window,
        "is_oos": {is_pf, oos_pf, deterioration, is_sharpe, oos_sharpe, pbo}
    }
    IS = in-sample (train) performance per window, OOS = out-of-sample (test).
    IS vs OOS divergence is the core alpha validation: IS != OOS => no real alpha.
    Breakout as research hypothesis is expected to show IS good PF but OOS
    deterioration until proven robust.
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
    is_window_stats: list[dict] = []
    is_sharpes: list[float] = []
    oos_sharpes: list[float] = []
    try:
        for _wnum, train_df, test_df in eng.generate_windows(df):
            # --- OOS (test) — with sufficient history from train tail ---
            # WalkForward test windows are short (100 bars) but history_window is 300.
            # Running only on test_df gives artificial 0-trade OOS (IS-OOS artifact).
            # Provide combined history: train tail + test, warmup on history.
            te = TradeEngine(exit_policy=policy)
            te.initial_balance = te.balance = te.equity = float(initial_balance)
            if with_is_metrics and len(train_df) >= min(history_window, 50):
                tail_len = min(history_window, len(train_df))
                combined = pd.concat([train_df.tail(tail_len), test_df], ignore_index=True)
                # Warmup = tail_len so trading starts at test start (no trades in history tail)
                te.run(
                    combined,
                    history_window=history_window,
                    warmup_bars=tail_len,
                    signal_generator=spec.factory(),
                )
                m = BacktestEngine._compute_risk_metrics(te, initial_balance)
                final = te.balance
                trades_n = len(te.closed_positions)
                wins_n = sum(1 for p in te.closed_positions if p.net_profit > 0)
            else:
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
            oos_sharpes.append(float(m["sharpe"]))

            # --- IS (train) — same spec, same costs, for divergence check ---
            if with_is_metrics:
                te_is = TradeEngine(exit_policy=policy)
                te_is.initial_balance = te_is.balance = te_is.equity = float(initial_balance)
                te_is.run(
                    train_df,
                    history_window=history_window,
                    warmup_bars=warmup_bars,
                    signal_generator=spec.factory(),
                )
                m_is = BacktestEngine._compute_risk_metrics(te_is, initial_balance)
                final_is = te_is.balance
                trades_is = len(te_is.closed_positions)
                wins_is = sum(1 for p in te_is.closed_positions if p.net_profit > 0)
                is_window_stats.append({
                    "net_pct": (final_is - initial_balance) / initial_balance * 100.0,
                    "pf": _cap_pf(m_is["profit_factor"]),
                    "maxdd_pct": m_is["max_drawdown_pct"],
                    "sharpe": m_is["sharpe"],
                    "trades": trades_is,
                    "wins": wins_is,
                })
                is_sharpes.append(float(m_is["sharpe"]))
    finally:
        te_mod.COMMISSION, te_mod.SLIPPAGE = saved_c, saved_s

    oos_metrics = aggregate_windows(window_stats)
    result: dict[str, Any] = {
        "metrics": oos_metrics,
        "windows": window_stats,
        "is_metrics": aggregate_windows(is_window_stats) if with_is_metrics else {},
        "is_windows": is_window_stats,
        "is_sharpes": is_sharpes,
        "oos_sharpes": oos_sharpes,
    }
    # IS-OOS summary for Research Integrity
    if with_is_metrics and is_window_stats:
        is_pf = result["is_metrics"].get("pf_median", 0.0)
        oos_pf = oos_metrics.get("pf_median", 0.0)
        is_sh = result["is_metrics"].get("sharpe_median", 0.0)
        oos_sh = oos_metrics.get("sharpe_median", 0.0)
        # Deterioration ratios
        pf_deterioration = (is_pf - oos_pf) / max(is_pf, 1e-9) if is_pf > 0 else (0.0 if oos_pf == 0 else 1.0)
        sharpe_deterioration = (is_sh - oos_sh) / max(abs(is_sh), 1e-9) if is_sh != 0 else 0.0
        # PBO proxy
        try:
            from src.validation.bootstrap import pbo_combinatorial
            pbo = pbo_combinatorial(is_sharpes, oos_sharpes)
        except Exception:
            pbo = 0.5
        result["is_oos"] = {
            "is_pf": is_pf,
            "oos_pf": oos_pf,
            "pf_ratio": oos_pf / max(is_pf, 1e-9) if is_pf > 0 else 0.0,
            "pf_deterioration": pf_deterioration,
            "is_sharpe": is_sh,
            "oos_sharpe": oos_sh,
            "sharpe_deterioration": sharpe_deterioration,
            "pbo": pbo,
            "is_net": result["is_metrics"].get("net_median_pct", 0.0),
            "oos_net": oos_metrics.get("net_median_pct", 0.0),
        }
        # Surface for integrity gates
        result["metrics"]["is_pf_median"] = is_pf
        result["metrics"]["pbo"] = pbo
        # Keep legacy pbo key for gate
    return result


__all__ = [
    "CandidateSpec",
    "PromotionRules",
    "DEFAULT_POLICY",
    "aggregate_windows",
    "evaluate_candidate",
]
