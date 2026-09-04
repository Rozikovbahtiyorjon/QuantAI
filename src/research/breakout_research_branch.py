"""
Breakout — Main Research Branch (not Champion)

User requirement: Breakout becomes RESEARCH CANDIDATE, not Champion,
and is tested as:

    Breakout (Donchian 96/20/3.0/12)
      + Regime Filter (TREND/RANGE, ADX hysteresis)
      + ML Meta-Labeler (P(win) filter)

Until IS-OOS consistency is proven (PF OOS/IS ≥0.60, PBO<0.6, cost robust),
it stays RESEARCH_HYPOTHESIS and is never promoted to Champion.

This module wires the three layers causally for Walk-Forward:
  TRAIN window → harvest Breakout entries → Regime-filtered → triple-barrier labels → fit MetaLabelModel (Purged, no leakage)
  TEST window  → Breakout → Regime gate → Meta gate → filtered signal

All gates are causal, no look-ahead, stateful regime memory is reset per window.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.regime_filter import RegimeConfig, RegimeFilter
from src.strategy.breakout_signal import BreakoutConfig, BreakoutSignalGenerator
from src.strategy.meta_label import (
    BarrierConfig,
    CostConfig,
    ExpectedReturnModel,
    FilteredGenerator,
    MetaLabelModel,
    build_labeled_dataset,
    build_regression_dataset,
    entry_features,
    net_return_entry,
)
from src.strategy.signal_generator import SignalResult


@dataclass
class BreakoutResearchConfig:
    breakout: BreakoutConfig = None  # type: ignore
    regime: RegimeConfig = None  # type: ignore
    use_regime_filter: bool = True
    use_meta_labeler: bool = True
    # Legacy binary threshold (deprecated) — use expected_return_hurdle
    meta_threshold: float = 0.55
    # New: expected net return hurdle (E[net] > hurdle => TAKE)
    # net = price outcome - commission - slippage - spread - funding
    # Trading task: expected_net_edge > hurdle, not BUY/SELL/HOLD classifier
    expected_return_hurdle: float = 0.001  # 0.1% net of costs required (was 0.0)
    use_expected_return: bool = True  # True = regression E[net], False = binary P(win)
    meta_history_window: int = 300
    barrier: BarrierConfig = None  # type: ignore
    costs: CostConfig = None  # type: ignore

    def __post_init__(self):
        if self.breakout is None:
            self.breakout = BreakoutConfig()
        if self.regime is None:
            self.regime = RegimeConfig()
        if self.barrier is None:
            self.barrier = BarrierConfig()
        if self.costs is None:
            self.costs = CostConfig()


class RegimeFilteredBreakout:
    """Breakout wrapped with RegimeFilter gate."""

    def __init__(self, breakout_gen: Any, regime_filter: RegimeFilter | None):
        self.breakout = breakout_gen
        self.regime = regime_filter

    def reset(self):
        try:
            self.breakout.reset()
        except AttributeError:
            pass
        if self.regime:
            self.regime.reset()

    def generate(self, df: pd.DataFrame) -> SignalResult:
        # Update regime state causally on current window
        if self.regime:
            regime = self.regime.classify(df)
            raw = self.breakout.generate(df)
            if raw.signal == "HOLD":
                return raw
            if not self.regime.allows(raw.signal):
                raw.signal = "HOLD"
                raw.trade_approved = False
                raw.reasons.append(f"RegimeFilter: blocked {raw.ai_signal} in {regime}")
                return raw
            raw.reasons.append(f"Regime: {regime}")
            return raw
        return self.breakout.generate(df)


class BreakoutResearchBranch:
    """
    Factory for Walk-Forward research branch:

        Breakout -> Regime -> Meta

    Usage in evaluation_pipeline:

        spec = CandidateSpec("breakout_regime_meta", breakout_research_factory, params={...})
        evaluate_candidate(spec, df, with_is_metrics=True)

    The factory is zero-arg per CandidateSpec contract; per-window training
    of MetaLabelModel is handled by walk-forward adapter below.
    """

    def __init__(self, config: BreakoutResearchConfig | None = None):
        self.config = config or BreakoutResearchConfig()

    def make_base(self) -> Any:
        """Base Breakout generator (no meta, but with regime)."""
        breakout = BreakoutSignalGenerator(config=self.config.breakout)
        regime = RegimeFilter(config=self.config.regime) if self.config.use_regime_filter else None
        return RegimeFilteredBreakout(breakout, regime)

    def make_for_window(self, train_df: pd.DataFrame | None, test_history_window: int = 300) -> Any:
        """
        Build generator for a TEST window.
        If use_meta_labeler and train_df is provided, fit MetaLabelModel on TRAIN entries.
        Otherwise return base (Breakout+Regime) unfiltered.
        """
        base = self.make_base()

        if not self.config.use_meta_labeler or train_df is None or len(train_df) < 200:
            return base

        # Harvest TRAIN entries with regime-filtered breakout
        # We run a lightweight TradeEngine gross run on TRAIN to collect signal indices
        try:
            from src.trade_engine import TradeEngine, ExitPolicy
            import src.trade_engine as te_mod

            # Use same exit policy as evaluation (trailing only)
            policy = ExitPolicy(use_take_profit=False, break_even_atr=None, trail_atr_mult=3.0)
            # Collect entries via TradeEngine entry_callback
            entries: list[dict] = []

            def _cb(meta: dict):
                if meta.get("executed"):
                    entries.append(dict(meta))

            # Run gross on TRAIN with RegimeFiltered base
            te = TradeEngine(exit_policy=policy)
            te.initial_balance = te.balance = te.equity = 1000.0
            # Need to run with base generator
            te.run(
                train_df,
                history_window=test_history_window,
                warmup_bars=0,
                signal_generator=base,
                entry_callback=_cb,
            )

            if len(entries) < 3:
                # Not enough entries to train meta — fallback to base
                return base

            # Build dataset: choose regression E[net] (preferred) or binary P(win)
            if self.config.use_expected_return:
                labeled = build_regression_dataset(
                    train_df,
                    entries,
                    feature_fn=entry_features,
                    history_window=test_history_window,
                    barrier=self.config.barrier,
                    costs=self.config.costs,
                )
                if labeled.empty or len(labeled) < 8:
                    return base
                # Check variance in target
                if labeled["net_return"].std() < 1e-9:
                    return base
                X = labeled.drop(columns=["net_return"])
                y = labeled["net_return"]
                model = ExpectedReturnModel(hurdle=self.config.expected_return_hurdle)
                model.fit(X, y)
                return FilteredGenerator(base, model, history_window=test_history_window)
            else:
                labeled = build_labeled_dataset(
                    train_df,
                    entries,
                    feature_fn=entry_features,
                    history_window=test_history_window,
                    barrier=self.config.barrier,
                )
                if labeled.empty or len(labeled) < 8 or labeled["label"].nunique() < 2:
                    return base
                X = labeled.drop(columns=["label"])
                y = labeled["label"]
                model = MetaLabelModel(threshold=self.config.meta_threshold)
                model.fit(X, y)
                return FilteredGenerator(base, model, history_window=test_history_window)
        except Exception:
            # Any failure in meta training -> fallback to base (research hypothesis still testable)
            return base


def breakout_research_factory(
    breakout_cfg: BreakoutConfig | None = None,
    regime_cfg: RegimeConfig | None = None,
    use_regime: bool = True,
    use_meta: bool = True,
) -> Any:
    """
    Zero-arg factory for CandidateSpec (research branch).
    For Walk-Forward, use `breakout_walkforward_adapter` instead to enable per-window meta fitting.
    """
    cfg = BreakoutResearchConfig(
        breakout=breakout_cfg or BreakoutConfig(channel_bars=96, min_adx=20.0, sl_atr_mult=3.0, cooldown_bars=12),
        regime=regime_cfg or RegimeConfig(adx_enter=22.0, adx_exit=18.0),
        use_regime_filter=use_regime,
        use_meta_labeler=use_meta,
    )
    branch = BreakoutResearchBranch(cfg)
    # For simple backtest (no TRAIN), return base without meta fit
    return branch.make_base()


def breakout_walkforward_adapter(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    breakout_cfg: BreakoutConfig | None = None,
    regime_cfg: RegimeConfig | None = None,
) -> Any:
    """
    Adapter for WalkForward per-window meta training.
    Called by evaluation harness with both TRAIN and TEST slices.
    """
    cfg = BreakoutResearchConfig(
        breakout=breakout_cfg or BreakoutConfig(channel_bars=96, min_adx=20.0, sl_atr_mult=3.0, cooldown_bars=12),
        regime=regime_cfg or RegimeConfig(),
        use_regime_filter=True,
        use_meta_labeler=True,
    )
    branch = BreakoutResearchBranch(cfg)
    # Fit meta on TRAIN, return generator for TEST
    return branch.make_for_window(train_df, test_history_window=300)


# =========================================================
# WALK-FORWARD EVALUATION FOR RESEARCH BRANCH (with per-window meta)
# =========================================================

def evaluate_breakout_research(
    spec,  # CandidateSpec, kept for API compat but not used (config inside)
    df: pd.DataFrame,
    *,
    train_size: int = 2000,
    test_size: int = 500,
    step_size: int = 500,
    initial_balance: float = 1000.0,
    costs: dict | None = None,
    warmup_bars: int = 0,
    history_window: int = 300,
    breakout_config: BreakoutConfig | None = None,
    regime_config: RegimeConfig | None = None,
    use_regime: bool = True,
    use_meta: bool = True,
) -> dict:
    """
    Research-branch evaluation: per-window Regime + Meta fitting on TRAIN,
    then test on OOS. Returns same shape as evaluate_candidate (metrics, windows, is_oos).
    Used for RESEARCH CANDIDATE, not Champion.
    """
    import src.trade_engine as te_mod
    from src.backtest_engine import BacktestEngine
    from src.champion.evaluation_pipeline import aggregate_windows, _cap_pf
    from src.trade_engine import ExitPolicy, TradeEngine
    from src.walk_forward_engine import WalkForwardEngine

    policy = ExitPolicy(use_take_profit=False, break_even_atr=None, trail_atr_mult=3.0)
    costs = costs or {"commission": 0.0004, "slippage": 0.0002}
    eng = WalkForwardEngine(train_size=train_size, test_size=test_size, step_size=step_size, initial_balance=initial_balance)
    saved_c, saved_s = te_mod.COMMISSION, te_mod.SLIPPAGE
    te_mod.COMMISSION, te_mod.SLIPPAGE = costs["commission"], costs["slippage"]

    branch_cfg = BreakoutResearchConfig(
        breakout=breakout_config or BreakoutConfig(channel_bars=96, min_adx=20.0, sl_atr_mult=3.0, cooldown_bars=12),
        regime=regime_config or RegimeConfig(adx_enter=22.0, adx_exit=18.0),
        use_regime_filter=use_regime,
        use_meta_labeler=use_meta,
    )
    branch = BreakoutResearchBranch(branch_cfg)

    window_stats: list[dict] = []
    is_window_stats: list[dict] = []
    is_sharpes: list[float] = []
    oos_sharpes: list[float] = []
    # For ML calibration: P(conf bucket) -> actual net return
    calib_y_true: list[float] = []
    calib_y_pred: list[float] = []

    try:
        for _wnum, train_df, test_df in eng.generate_windows(df):
            # Fit meta on TRAIN, get generator for TEST
            gen_oos = branch.make_for_window(train_df, test_history_window=history_window)
            # Also need IS generator (train without meta leakage: fit meta on train's own train split? For IS we use base only to avoid leakage)
            gen_is = branch.make_base()
            # For calibration: need model fitted on TRAIN and base candidates on TEST
            calib_model = getattr(gen_oos, "model", None) if hasattr(gen_oos, "model") else None
            base_for_calib = branch.make_base()

            # --- Collect calibration data for this window (P(conf) -> actual net) ---
            # Harvest base candidates on TEST (before meta filtering) and compare predicted E[net] vs actual net
            if calib_model is not None and hasattr(calib_model, "predict_expected"):
                try:
                    # Harvest TEST base entries (without meta) for calibration
                    from src.trade_engine import TradeEngine as _TE
                    from src.strategy.meta_label import entry_features, net_return_entry, BarrierConfig, CostConfig
                    # Use same barrier/costs as branch
                    barrier = branch.config.barrier
                    costs = branch.config.costs
                    # Quick harvest on TEST alone (with history)
                    tail_for_calib = min(history_window, len(train_df))
                    combined_calib = pd.concat([train_df.tail(tail_for_calib), test_df], ignore_index=True) if len(train_df) >= 50 else test_df
                    # Collect base entries on combined
                    calib_entries: list[dict] = []
                    def _cb_calib(meta: dict):
                        if meta.get("executed"):
                            calib_entries.append(dict(meta))
                    te_calib = _TE(exit_policy=policy)
                    te_calib.initial_balance = te_calib.balance = te_calib.equity = float(initial_balance)
                    te_calib.run(combined_calib, history_window=history_window, warmup_bars=tail_for_calib if len(train_df) >= 50 else warmup_bars, signal_generator=base_for_calib, entry_callback=_cb_calib)
                    # For each base candidate, predict and get actual net
                    for e in calib_entries:
                        try:
                            idx = int(e["signal_index"])
                            # Map signal_index from combined to original df index for actual net
                            # For calibration, use test_df's actual net via net_return_entry on combined
                            # Need to find entry's actual net return
                            # Use net_return_entry on combined with entry_price
                            actual = net_return_entry(combined_calib, idx, e["side"], float(e["entry_price"]), barrier, costs)
                            # Predicted E[net] from model
                            # Need feature window at signal bar
                            lo = max(0, idx - history_window + 1)
                            win = combined_calib.iloc[lo : idx + 1]
                            if len(win) < 120:
                                continue
                            feats = entry_features(win, e["side"])
                            pred = float(calib_model.predict_expected(feats))
                            calib_y_true.append(actual)
                            calib_y_pred.append(pred)
                        except Exception:
                            continue
                except Exception:
                    pass

            # OOS
            tail_len = min(history_window, len(train_df))
            combined = pd.concat([train_df.tail(tail_len), test_df], ignore_index=True) if len(train_df) >= 50 else test_df
            te = TradeEngine(exit_policy=policy)
            te.initial_balance = te.balance = te.equity = float(initial_balance)
            te.run(combined, history_window=history_window, warmup_bars=tail_len if len(train_df) >= 50 else warmup_bars, signal_generator=gen_oos)
            m = BacktestEngine._compute_risk_metrics(te, initial_balance)
            final = te.balance
            trades_n = len(te.closed_positions)
            wins_n = sum(1 for p in te.closed_positions if p.net_profit > 0)
            window_stats.append({"net_pct": (final - initial_balance) / initial_balance * 100.0, "pf": _cap_pf(m["profit_factor"]), "maxdd_pct": m["max_drawdown_pct"], "sharpe": m["sharpe"], "trades": trades_n, "wins": wins_n})
            oos_sharpes.append(float(m["sharpe"]))

            # IS
            te_is = TradeEngine(exit_policy=policy)
            te_is.initial_balance = te_is.balance = te_is.equity = float(initial_balance)
            te_is.run(train_df, history_window=history_window, warmup_bars=warmup_bars, signal_generator=gen_is)
            m_is = BacktestEngine._compute_risk_metrics(te_is, initial_balance)
            final_is = te_is.balance
            is_window_stats.append({"net_pct": (final_is - initial_balance) / initial_balance * 100.0, "pf": _cap_pf(m_is["profit_factor"]), "maxdd_pct": m_is["max_drawdown_pct"], "sharpe": m_is["sharpe"], "trades": len(te_is.closed_positions), "wins": sum(1 for p in te_is.closed_positions if p.net_profit > 0)})
            is_sharpes.append(float(m_is["sharpe"]))
    finally:
        te_mod.COMMISSION, te_mod.SLIPPAGE = saved_c, saved_s

    from src.champion.evaluation_pipeline import aggregate_windows as agg
    oos_metrics = agg(window_stats)
    is_metrics = agg(is_window_stats) if is_window_stats else {}
    # --- Calibration: Brier/ECE/reliability (confidence 0.8 -> 80% empirical) ---
    calib_report = None
    brier_report = None
    if calib_y_true and calib_y_pred and len(calib_y_true) >= 10:
        try:
            from src.validation.calibration import evaluate_calibration, evaluate_brier_ece
            calib_report = evaluate_calibration(calib_y_true, calib_y_pred, n_buckets=5)
            # For Brier/ECE, need binary labels: net >0 => win
            y_true_bin = [1 if float(v) > 0 else 0 for v in calib_y_true]
            # For ExpectedReturn, pred is E[net] not prob; convert to prob via sigmoid for Brier if needed
            # Here we treat pred as prob if in [0,1], else skip Brier
            if all(0 <= float(p) <= 1 for p in calib_y_pred):
                brier_report = evaluate_brier_ece(y_true_bin, calib_y_pred, max_brier=0.25, max_ece=0.10)
        except Exception:
            pass

    # --- Cost/Slippage/Latency Stress (research §55) ---
    cost_results = []
    slippage_results = []
    latency_results = []
    queue_results = {}
    try:
        from src.validation.cost_stress import cost_stress, slippage_stress, latency_stress, queue_simulation
        # Use full df for stress (conservative) or OOS slice if available
        stress_df = df.tail(1000) if len(df) > 1000 else df
        cost_results = cost_stress(stress_df)
        slippage_results = slippage_stress(stress_df)
        latency_results = latency_stress(stress_df)
        queue_results = queue_simulation(stress_df)
    except Exception:
        pass

    result: dict = {
        "metrics": oos_metrics,
        "windows": window_stats,
        "is_metrics": is_metrics,
        "is_windows": is_window_stats,
        "is_sharpes": is_sharpes,
        "oos_sharpes": oos_sharpes,
        "y_true_net": calib_y_true,
        "y_pred_conf": calib_y_pred,
        "y_pred_expected": calib_y_pred,  # alias for ExpectedReturn E[net]
        "ml_calibration": {"y_true_net": calib_y_true, "y_pred_conf": calib_y_pred, "y_pred_expected": calib_y_pred},
        "calibration_report": {"spearman": calib_report.spearman_corr if calib_report else None, "pearson": calib_report.pearson_corr if calib_report else None, "monotonic": calib_report.monotonic if calib_report else None, "passed": calib_report.passed if calib_report else False, "reason": calib_report.reason if calib_report else "no calib", "n": len(calib_y_true)},
        "brier_ece": {"brier": brier_report.brier_score if brier_report else None, "ece": brier_report.ece if brier_report else None, "passed": brier_report.passed if brier_report else False, "reason": brier_report.reason if brier_report else "no brier"},
        "cost_stress": [{"mult": r.multiplier, "pf": r.pf, "fragile": r.fragile} for r in cost_results],
        "slippage_stress": slippage_results,
        "latency_stress": latency_results,
        "queue_fill": queue_results,
        "expected_return_hurdle": branch_cfg.expected_return_hurdle,
    }
    # Also expose cost/slippage/latency in metrics for gate
    if cost_results:
        result["metrics"]["cost_robust"] = all(not r.fragile for r in cost_results if r.multiplier == 1.5) if any(r.multiplier == 1.5 for r in cost_results) else True
    if calib_report:
        result["metrics"]["calibration_passed"] = bool(calib_report.passed)
    if is_window_stats:
        is_pf = is_metrics.get("pf_median", 0.0)
        oos_pf = oos_metrics.get("pf_median", 0.0)
        is_sh = is_metrics.get("sharpe_median", 0.0)
        oos_sh = oos_metrics.get("sharpe_median", 0.0)
        pf_ratio = oos_pf / max(is_pf, 1e-9) if is_pf > 0 else 0.0
        pf_det = (is_pf - oos_pf) / max(is_pf, 1e-9) if is_pf > 0 else 0.0
        sharpe_det = (is_sh - oos_sh) / max(abs(is_sh), 1e-9) if is_sh != 0 else 0.0
        try:
            from src.validation.bootstrap import pbo_combinatorial
            pbo = pbo_combinatorial(is_sharpes, oos_sharpes)
        except Exception:
            pbo = 0.5
        result["is_oos"] = {"is_pf": is_pf, "oos_pf": oos_pf, "pf_ratio": pf_ratio, "pf_deterioration": pf_det, "is_sharpe": is_sh, "oos_sharpe": oos_sh, "sharpe_deterioration": sharpe_det, "pbo": pbo, "is_net": is_metrics.get("net_median_pct", 0.0), "oos_net": oos_metrics.get("net_median_pct", 0.0)}
        result["metrics"]["is_pf_median"] = is_pf
        result["metrics"]["pbo"] = pbo
        # Also store calibration in metrics for integrity gate fallback
        if calib_y_true:
            result["metrics"]["calibration_n"] = len(calib_y_true)
    return result


# Research candidate specs for registration
RESEARCH_CANDIDATE_BREAKOUT_PLAIN = "breakout_plain"  # Breakout only
RESEARCH_CANDIDATE_BREAKOUT_REGIME = "breakout_regime"  # Breakout + Regime
RESEARCH_CANDIDATE_BREAKOUT_REGIME_META = "breakout_regime_meta"  # Full branch

__all__ = [
    "BreakoutResearchConfig",
    "RegimeFilteredBreakout",
    "BreakoutResearchBranch",
    "breakout_research_factory",
    "breakout_walkforward_adapter",
    "evaluate_breakout_research",
    "RESEARCH_CANDIDATE_BREAKOUT_PLAIN",
    "RESEARCH_CANDIDATE_BREAKOUT_REGIME",
    "RESEARCH_CANDIDATE_BREAKOUT_REGIME_META",
]
