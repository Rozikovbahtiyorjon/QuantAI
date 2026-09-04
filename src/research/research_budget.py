"""
Research Budget — Audit #22 + Task 13 LESS OPTIMIZATION + MORE VALIDATION

Supervisor must have max experiments / trials / OOS reuse limits
to avoid researcher overfitting (machine for overfitting).

Task 13 principle: LESS OPTIMIZATION + MORE VALIDATION
- Don't try to increase win rate/PF by Optuna x10 + 1000 new params + 20 new indicators.
- This worsens overfitting. Enforce hard caps:
    max_optuna_trials = 50 per budget (was 200) — total budget
    max_optuna_trials_per_run = 50 — single optimization can't do 500 trials
    max_params_per_strategy = 5 — no 20-param overfit
    max_indicators = 10 — Occam's razor
    max_experiments_per_oos = 10 — fresh OOS required after 10 touches
    max_optimizations_per_strategy = 3 — no 10x retry without validation
  If strategy failed validation, require PBO/DSR/WRC before next optimization.

All limits raise BudgetExceeded with clear LESS OPTIMIZATION message.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


class BudgetExceeded(RuntimeError):
    pass


_LESS_OPT_MSG = (
    " [LESS OPTIMIZATION principle: don't increase win rate/PF by "
    "Optuna x10 + 1000 new params + 20 indicators — worsens overfitting. "
    "MORE VALIDATION required (PBO/DSR/WRC) instead.]"
)


@dataclass
class ResearchBudget:
    # --- Core limits (Task 13: tightened for LESS OPTIMIZATION) ---
    max_experiments: int = 50  # was 100 — less is more
    max_optuna_trials: int = 50  # was 200 — total Optuna budget hard cap 50
    max_optuna_trials_per_run: int = 50  # NEW: single run can't exceed 50
    max_parameter_mutations: int = 20  # was 50 — fewer mutations
    max_oos_reuse: int = 10  # strict OOS reuse (statistical power)
    max_experiments_per_oos: int = 10  # NEW: per-OOS cap (fresh OOS after 10)
    max_strategy_variants: int = 10  # was 20 — fewer variants
    max_retries: int = 3
    # --- Task 13 NEW: complexity caps ---
    max_params_per_strategy: int = 5  # NEW: max tunable params per strategy
    max_indicators: int = 10  # NEW: max indicators per strategy
    max_optimizations_per_strategy: int = 3  # NEW: no endless retry
    # P1.2 additional durable budgets
    max_model_retrains: int = 20
    max_feature_additions: int = 10
    max_oos_accesses: int = 50
    max_dataset_reuses: int = 20

    experiments_used: int = 0
    optuna_trials_used: int = 0
    oos_reuse_used: int = 0
    strategy_variants_used: int = 0
    parameter_mutations_used: int = 0
    retries_used: int = 0
    model_retrains_used: int = 0
    feature_additions_used: int = 0
    oos_accesses_used: int = 0
    dataset_reuses_used: int = 0
    # --- Tracking for new caps (durable via ledger) ---
    _per_oos_counts: Dict[str, int] = field(default_factory=dict, repr=False)
    _per_strategy_opt_counts: Dict[str, int] = field(default_factory=dict, repr=False)

    def check_experiment(self) -> None:
        if self.experiments_used >= self.max_experiments:
            raise BudgetExceeded(
                f"OPTIMIZATION BLOCKED: max_experiments {self.max_experiments} exceeded "
                f"(used {self.experiments_used}).{ _LESS_OPT_MSG}"
            )
        self.experiments_used += 1

    def check_optuna(self, n: int = 1) -> None:
        # Per-run guard first (clear message if single request is too large)
        if n > self.max_optuna_trials_per_run:
            raise BudgetExceeded(
                f"OPTIMIZATION BLOCKED: requested {n} Optuna trials > max per run "
                f"{self.max_optuna_trials_per_run}.{ _LESS_OPT_MSG} "
                f"Reduce trials (e.g., 50 max) and add validation (PBO/DSR/WRC) instead of more trials."
            )
        if self.optuna_trials_used + n > self.max_optuna_trials:
            raise BudgetExceeded(
                f"OPTIMIZATION BLOCKED: max_optuna_trials {self.max_optuna_trials} exceeded "
                f"(used {self.optuna_trials_used}, requested {n}, would be {self.optuna_trials_used + n})."
                f"{ _LESS_OPT_MSG} Total Optuna budget exhausted — need validation, not more trials."
            )
        self.optuna_trials_used += n

    def check_optuna_per_run(self, n: int) -> None:
        """Enforce per-run cap separately (useful for guard before budget accounting)."""
        if n > self.max_optuna_trials_per_run:
            raise BudgetExceeded(
                f"OPTIMIZATION BLOCKED: requested {n} Optuna trials > max per run "
                f"{self.max_optuna_trials_per_run}.{ _LESS_OPT_MSG}"
            )

    def check_oos_reuse(self, registry_oos_reuse: int) -> None:
        if registry_oos_reuse >= self.max_oos_reuse:
            raise BudgetExceeded(
                f"OPTIMIZATION BLOCKED: max_oos_reuse {self.max_oos_reuse} for OOS period exceeded "
                f"(current reuse {registry_oos_reuse}).{ _LESS_OPT_MSG} OOS has been reused too many times "
                f"— PF loses statistical power. Use fresh OOS period or add WRC/DSR validation."
            )

    def check_experiments_per_oos(self, oos_period: str, current_count: int | None = None) -> None:
        """
        Task 13: max experiments per single OOS period.
        Provide current_count explicitly or rely on internal per-OOS tracker.
        """
        count = current_count if current_count is not None else self._per_oos_counts.get(oos_period, 0)
        if count >= self.max_experiments_per_oos:
            raise BudgetExceeded(
                f"OPTIMIZATION BLOCKED: max_experiments_per_oos {self.max_experiments_per_oos} "
                f"for OOS '{oos_period}' exceeded (used {count}).{ _LESS_OPT_MSG} "
                f"Fresh OOS required — don't overfit the same period."
            )
        # Increment tracker
        self._per_oos_counts[oos_period] = count + 1
        # Also count as experiment if caller hasn't already
        # Don't double-increment experiments_used here — caller decides

    def check_parameter_mutation(self, n: int = 1) -> None:
        """Hard budget for parameter mutations (Audit: previously not enforced)."""
        if self.parameter_mutations_used + n > self.max_parameter_mutations:
            raise BudgetExceeded(
                f"OPTIMIZATION BLOCKED: max_parameter_mutations {self.max_parameter_mutations} exceeded "
                f"(used {self.parameter_mutations_used}, requested {n}).{ _LESS_OPT_MSG}"
            )
        self.parameter_mutations_used += n

    def check_strategy_variant(self, n: int = 1) -> None:
        """Hard budget for strategy variants (Audit: previously not enforced)."""
        if self.strategy_variants_used + n > self.max_strategy_variants:
            raise BudgetExceeded(
                f"OPTIMIZATION BLOCKED: max_strategy_variants {self.max_strategy_variants} exceeded "
                f"(used {self.strategy_variants_used}, requested {n}).{ _LESS_OPT_MSG}"
            )
        self.strategy_variants_used += n

    def check_retry(self, n: int = 1) -> None:
        """Hard budget for retries (Audit: previously not enforced)."""
        if self.retries_used + n > self.max_retries:
            raise BudgetExceeded(
                f"OPTIMIZATION BLOCKED: max_retries {self.max_retries} exceeded "
                f"(used {self.retries_used}).{ _LESS_OPT_MSG}"
            )
        self.retries_used += n

    # --- Task 13 NEW: complexity gates ---
    def check_params(self, params: Dict[str, Any] | int | None) -> None:
        """
        Enforce max_params_per_strategy.
        Accepts dict (counts keys), int (explicit count), or None.
        """
        if params is None:
            return
        n = len(params) if isinstance(params, dict) else int(params)
        if n > self.max_params_per_strategy:
            raise BudgetExceeded(
                f"OPTIMIZATION BLOCKED: strategy has {n} params > max {self.max_params_per_strategy} per strategy."
                f"{ _LESS_OPT_MSG} Reduce to <=5 tuned params (Occam's razor) and validate instead."
            )

    def check_indicators(self, indicators: Any | None) -> None:
        """
        Enforce max_indicators.
        Accepts list/tuple/set/dict/int or None.
        """
        if indicators is None:
            return
        if isinstance(indicators, int):
            n = indicators
        elif isinstance(indicators, dict):
            n = len(indicators)
        elif isinstance(indicators, (list, tuple, set)):
            n = len(indicators)
        else:
            try:
                n = len(indicators)  # type: ignore[arg-type]
            except Exception:
                return
        if n > self.max_indicators:
            raise BudgetExceeded(
                f"OPTIMIZATION BLOCKED: strategy uses {n} indicators > max {self.max_indicators}."
                f"{ _LESS_OPT_MSG} Use <=10 indicators and validate (WRC/PBO/DSR)."
            )

    def check_optimization_attempt(self, strategy_id: str) -> None:
        """
        Task 13: per-strategy optimization count gate.
        Prevents 10x Optuna on same failing strategy without validation.
        Call before each optimization; raises if limit exceeded.
        Validation-gated retry is handled by OptimizationGuard (resets on passed validation).
        Here we enforce hard budget.
        """
        count = self._per_strategy_opt_counts.get(strategy_id, 0)
        if count >= self.max_optimizations_per_strategy:
            raise BudgetExceeded(
                f"OPTIMIZATION BLOCKED: strategy '{strategy_id}' has {count} optimizations "
                f">= max per strategy {self.max_optimizations_per_strategy}.{ _LESS_OPT_MSG} "
                f"Strategy already optimized {count} times — provide MORE VALIDATION (PBO<0.6, DSR>0.95, WRC p<0.05) "
                f"before next optimization. Don't add more params/trials."
            )
        self._per_strategy_opt_counts[strategy_id] = count + 1

    def record_strategy_validation_pass(self, strategy_id: str) -> None:
        """
        Reset per-strategy optimization counter when strategy PASSES validation.
        More validation unlocks careful next optimization; failed validation does NOT reset.
        """
        # On pass, reset to 0 to allow fresh cycle, but keep history
        self._per_strategy_opt_counts[strategy_id] = 0

    def check_model_retrain(self, n: int = 1) -> None:
        if self.model_retrains_used + n > self.max_model_retrains:
            raise BudgetExceeded(f"OPTIMIZATION BLOCKED: max_model_retrains {self.max_model_retrains} exceeded (used {self.model_retrains_used}){ _LESS_OPT_MSG}")
        self.model_retrains_used += n

    def check_feature_addition(self, n: int = 1) -> None:
        if self.feature_additions_used + n > self.max_feature_additions:
            raise BudgetExceeded(f"OPTIMIZATION BLOCKED: max_feature_additions {self.max_feature_additions} exceeded (used {self.feature_additions_used}){ _LESS_OPT_MSG}")
        self.feature_additions_used += n

    def check_oos_access(self, n: int = 1) -> None:
        if self.oos_accesses_used + n > self.max_oos_accesses:
            raise BudgetExceeded(f"OPTIMIZATION BLOCKED: max_oos_accesses {self.max_oos_accesses} exceeded (used {self.oos_accesses_used}){ _LESS_OPT_MSG}")
        self.oos_accesses_used += n

    def check_dataset_reuse(self, n: int = 1) -> None:
        if self.dataset_reuses_used + n > self.max_dataset_reuses:
            raise BudgetExceeded(f"OPTIMIZATION BLOCKED: max_dataset_reuses {self.max_dataset_reuses} exceeded (used {self.dataset_reuses_used}){ _LESS_OPT_MSG}")
        self.dataset_reuses_used += n

    def to_dict(self) -> dict:
        return {
            "max_experiments": self.max_experiments,
            "experiments_used": self.experiments_used,
            "max_optuna_trials": self.max_optuna_trials,
            "optuna_trials_used": self.optuna_trials_used,
            "max_optuna_trials_per_run": self.max_optuna_trials_per_run,
            "max_oos_reuse": self.max_oos_reuse,
            "oos_reuse_used": self.oos_reuse_used,
            "max_experiments_per_oos": self.max_experiments_per_oos,
            "max_parameter_mutations": self.max_parameter_mutations,
            "parameter_mutations_used": self.parameter_mutations_used,
            "max_strategy_variants": self.max_strategy_variants,
            "strategy_variants_used": self.strategy_variants_used,
            "max_retries": self.max_retries,
            "retries_used": self.retries_used,
            "max_params_per_strategy": self.max_params_per_strategy,
            "max_indicators": self.max_indicators,
            "max_optimizations_per_strategy": self.max_optimizations_per_strategy,
            "max_model_retrains": self.max_model_retrains,
            "model_retrains_used": self.model_retrains_used,
            "max_feature_additions": self.max_feature_additions,
            "feature_additions_used": self.feature_additions_used,
            "max_oos_accesses": self.max_oos_accesses,
            "oos_accesses_used": self.oos_accesses_used,
            "max_dataset_reuses": self.max_dataset_reuses,
            "dataset_reuses_used": self.dataset_reuses_used,
            "remaining_experiments": self.max_experiments - self.experiments_used,
            "remaining_optuna_trials": self.max_optuna_trials - self.optuna_trials_used,
            "remaining_variants": self.max_strategy_variants - self.strategy_variants_used,
            "remaining_mutations": self.max_parameter_mutations - self.parameter_mutations_used,
            "per_oos_counts": dict(self._per_oos_counts),
            "per_strategy_opt_counts": dict(self._per_strategy_opt_counts),
        }
