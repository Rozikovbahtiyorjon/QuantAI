"""
PHASE 16 — RESEARCH VALIDATION
ENTRY-69 — Walk-Forward (Inner WF → Freeze → Outer OOS)
ENTRY-70 — Parameter Optimization (Optuna only in INNER WF)
ENTRY-71 — Setup-specific Robustness (neighbor params)
ENTRY-72 — Entry Robustness (±0.1/0.2/0.3 ATR)
ENTRY-73 — Execution Robustness (best/normal/degraded fill)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod


class ValidationStage(str, Enum):
    DEVELOPMENT = "DEVELOPMENT"
    OPTIMIZATION = "OPTIMIZATION"  # Inner WF only
    FREEZE = "FREEZE"
    OUTER_OOS = "OUTER_OOS"
    SEALED = "SEALED"


@dataclass
class RobustnessResult:
    """Result of robustness testing."""
    param_name: str
    base_value: float
    neighbor_results: dict[float, dict]  # param_value -> metrics
    all_reasonable: bool  # All neighbors have reasonable performance
    degradation_pct: float  # Max degradation from base


@dataclass
class EntryRobustnessResult:
    """ENTRY-72: Entry robustness at different slippage levels."""
    ideal_entry: float
    atr: float
    results: dict[str, dict]  # "+0.1_ATR", "-0.1_ATR", etc -> metrics
    is_fragile: bool  # Strategy breaks with small entry degradation


@dataclass
class ExecutionRobustnessResult:
    """ENTRY-73: Execution robustness."""
    best_fill: dict  # metrics at best fill
    normal_fill: dict  # metrics at normal fill
    degraded_fill: dict  # metrics at degraded fill
    fill_degradation_pct: float


class WalkForwardValidator:
    """ENTRY-69: True Nested Walk-Forward (Inner WF → Freeze → Outer OOS)."""

    def __init__(
        self,
        inner_windows: int = 5,
        outer_windows: int = 3,
        embargo_bars: int = 100,
        min_trades_per_window: int = 30,
    ):
        self.inner_windows = inner_windows
        self.outer_windows = outer_windows
        self.embargo_bars = embargo_bars
        self.min_trades = min_trades_per_window

    def validate(
        self,
        df: pd.DataFrame,
        strategy_factory: Callable,
        param_search_fn: Callable,  # param_search_fn(inner_windows) -> frozen_params
    ) -> dict:
        """
        True nested WF:
        1. Outer split: train/test windows
        2. For each outer train: Inner WF on train portion only
        3. param_search_fn selects params from INNER only
        4. Freeze params
        5. Test on outer OOS (never seen during optimization)
        6. Aggregate outer OOS results
        """
        # This delegates to nested_walk_forward.py which has the full implementation
        from src.validation.nested_walk_forward import NestedWalkForward

        nwf = NestedWalkForward(
            n_splits=self.outer_windows,
            embargo=self.embargo_bars,
            min_trades_per_split=self.min_trades,
        )

        def outer_objective(inner_windows, inner_result, aggregate):
            # This is where param_search_fn runs
            return param_search_fn(inner_windows, inner_result, aggregate)

        results = nwf.run(df, strategy_factory, outer_objective)
        return {
            "outer_oos_results": results,
            "frozen_params": nwf.frozen_params,
            "stage": ValidationStage.SEALED.value,
        }


class ParameterOptimizer:
    """ENTRY-70: Optuna optimization ONLY in INNER WF."""

    def __init__(self, n_trials: int = 100, timeout: int = 3600):
        self.n_trials = n_trials
        self.timeout = timeout

    def optimize_inner(
        self,
        inner_windows: list,
        inner_result: Any,
        aggregate: Callable,
        param_space: dict,
    ) -> dict:
        """
        Optimize parameters using ONLY inner WF data.
        NEVER touches outer OOS.
        Returns frozen parameters.
        """
        import optuna

        def objective(trial):
            params = {}
            for name, (low, high) in param_space.items():
                if isinstance(low, int):
                    params[name] = trial.suggest_int(name, low, high)
                else:
                    params[name] = trial.suggest_float(name, low, high)

            # Evaluate on inner windows using aggregate function
            # This is a simplified interface - real impl uses inner_windows data
            score = aggregate(params, inner_windows, inner_result)
            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        return study.best_params


class SetupRobustnessTester:
    """ENTRY-71: Setup-specific robustness on neighbor parameters."""

    NEIGHBOR_RANGES = {
        "channel_bars": 0.15,   # ±15%
        "adx_threshold": 0.20,  # ±20%
        "atr_mult": 0.25,       # ±25%
        "cooldown_bars": 0.30,  # ±30%
    }

    def test_setup_params(
        self,
        base_params: dict,
        backtest_fn: Callable[[dict], dict],  # params -> metrics
    ) -> dict[str, RobustnessResult]:
        """
        For each param, test neighbors.
        Example Breakout:
        - channel_bars: 88-104 (base 96)
        - adx: 20-30 (base 25)
        - atr_stop: 1.5-2.5 (base 2.0)
        - cooldown: 7-13 (base 10)
        All should be reasonable, not just one perfect value.
        """
        results = {}

        for param_name, base_value in base_params.items():
            if param_name not in self.NEIGHBOR_RANGES:
                continue

            range_pct = self.NEIGHBOR_RANGES[param_name]
            neighbor_values = self._generate_neighbors(base_value, range_pct)

            neighbor_results = {}
            for val in neighbor_values:
                test_params = {**base_params, param_name: val}
                metrics = backtest_fn(test_params)
                neighbor_results[val] = metrics

            # Check if ALL neighbors are reasonable
            base_metrics = backtest_fn(base_params)
            base_pf = base_metrics.get("profit_factor", 0)
            all_reasonable = all(
                m.get("profit_factor", 0) > 1.0 and m.get("expectancy", 0) > 0
                for m in neighbor_results.values()
            )

            max_degradation = max(
                (base_pf - m.get("profit_factor", 0)) / max(base_pf, 0.001)
                for m in neighbor_results.values()
            ) if base_pf > 0 else 0

            results[param_name] = RobustnessResult(
                param_name=param_name,
                base_value=base_value,
                neighbor_results=neighbor_results,
                all_reasonable=all_reasonable,
                degradation_pct=max_degradation,
            )

        return results

    def _generate_neighbors(self, base: float, range_pct: float) -> list[float]:
        if isinstance(base, int):
            step = max(1, int(base * range_pct / 3))
            return [base - 2*step, base - step, base, base + step, base + 2*step]
        else:
            step = base * range_pct / 3
            return [base - 2*step, base - step, base, base + step, base + 2*step]


class EntryRobustnessTester:
    """ENTRY-72: Entry robustness at ±0.1/0.2/0.3 ATR."""

    ATR_OFFSETS = [0.1, 0.2, 0.3]

    def test_entry_robustness(
        self,
        ideal_entry: float,
        atr: float,
        direction: str,
        backtest_fn: Callable[[float], dict],  # entry_price -> metrics
    ) -> EntryRobustnessResult:
        """
        Test entry at:
        - ideal
        - ideal ± 0.1 ATR
        - ideal ± 0.2 ATR
        - ideal ± 0.3 ATR
        If strategy collapses at small degradation → FRAGILE.
        """
        results = {}
        base_metrics = backtest_fn(ideal_entry)
        base_expectancy = base_metrics.get("expectancy", 0)

        for offset in self.ATR_OFFSETS:
            for sign in [-1, 1]:
                test_entry = ideal_entry + sign * offset * atr
                if direction == "SHORT":
                    test_entry = ideal_entry - sign * offset * atr  # inverse for shorts
                metrics = backtest_fn(test_entry)
                results[f"{sign:+.1f}_ATR"] = metrics

        # Check fragility: any offset causes expectancy to drop below 0 or PF < 1
        is_fragile = any(
            m.get("expectancy", 0) <= 0 or m.get("profit_factor", 0) < 1.0
            for m in results.values()
        )

        return EntryRobustnessResult(
            ideal_entry=ideal_entry,
            atr=atr,
            results=results,
            is_fragile=is_fragile,
        )


class ExecutionRobustnessTester:
    """ENTRY-73: Execution robustness (best/normal/degraded fill)."""

    def test_execution_robustness(
        self,
        backtest_fn: Callable[[dict], dict],  # fill_model_params -> metrics
    ) -> ExecutionRobustnessResult:
        """
        Test three fill scenarios:
        - best_fill:   queue_position=0.1, latency=10ms, spread=0.5bp
        - normal_fill: queue_position=0.5, latency=100ms, spread=1bp
        - degraded:    queue_position=0.9, latency=500ms, spread=3bp
        """
        best_params = {"queue_position": 0.1, "latency_ms": 10, "spread_bps": 0.5}
        normal_params = {"queue_position": 0.5, "latency_ms": 100, "spread_bps": 1.0}
        degraded_params = {"queue_position": 0.9, "latency_ms": 500, "spread_bps": 3.0}

        best = backtest_fn(best_params)
        normal = backtest_fn(normal_params)
        degraded = backtest_fn(degraded_params)

        base_pf = best.get("profit_factor", 0)
        deg_pf = degraded.get("profit_factor", 0)
        degradation = (base_pf - deg_pf) / max(base_pf, 0.001) if base_pf > 0 else 0

        return ExecutionRobustnessResult(
            best_fill=best,
            normal_fill=normal,
            degraded_fill=degraded,
            fill_degradation_pct=degradation,
        )