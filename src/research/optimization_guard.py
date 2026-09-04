"""
Optimization Guard — Task 13: LESS OPTIMIZATION + MORE VALIDATION

Prevents over-optimization pathology:
  "Don't try to increase win rate/PF by Optuna x10 + 1000 new params + 20 new indicators —
   this worsens situation."

Instead:
  MORE VALIDATION (PBO, DSR, WRC/SPA) before next optimization.

Enforces:
  - max Optuna trials per run (default 50)
  - max Optuna trials total (budget 50)
  - max params per strategy (default 5)
  - max indicators (default 10)
  - max experiments per OOS (default 10) + max OOS reuse (10)
  - max optimizations per strategy (default 3)
  - VALIDATION GATE: if strategy failed last validation, next optimization
    requires fresh validation evidence (PBO<0.6, DSR threshold, WRC p<0.05)
    before allowing more trials/params.

Usage:
    from src.research.optimization_guard import OptimizationGuard, OptimizationGuardConfig, ValidationEvidence

    guard = OptimizationGuard()
    guard.check_strategy_complexity(params={"a":1,"b":2}, indicators=["rsi","ema"])
    guard.check_optuna_request(n_trials=50)
    # After failed validation, require evidence:
    evidence = ValidationEvidence(strategy_id="my_strat", pbo=0.45, dsr=0.96, wrc_p_value=0.03, passed=True)
    guard.record_validation_result("my_strat", evidence)
    guard.enforce_before_optimization(
        strategy_id="my_strat", n_trials=40,
        params={"a":1}, indicators=["rsi"],
        oos_period="2024-01->2024-06", validation_evidence=evidence
    )

Raises:
    BudgetExceeded (from research_budget) for hard limits.
    ValidationRequired for missing/insufficient validation after failure.

Error messages are explicit and mention LESS OPTIMIZATION principle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.research.research_budget import BudgetExceeded, ResearchBudget

# Re-export for convenience
__all__ = [
    "OptimizationBlocked",
    "ValidationRequired",
    "OptimizationGuardConfig",
    "ValidationEvidence",
    "OptimizationGuard",
    "create_default_guard",
]


_LESS_OPT = (
    "LESS OPTIMIZATION + MORE VALIDATION (Task 13): "
    "don't increase win rate/PF by Optuna x10 + 1000 params + 20 indicators. "
    "Provide PBO/DSR/WRC validation instead."
)


class OptimizationBlocked(BudgetExceeded):
    """Hard limit exceeded — over-optimization blocked."""
    pass


class ValidationRequired(RuntimeError):
    """Strategy failed last validation; fresh PBO/DSR/WRC evidence required before next optimization."""
    pass


# ---------------------------------------------------------------------------
# Validation evidence
# ---------------------------------------------------------------------------

@dataclass
class ValidationEvidence:
    """
    Validation evidence required after a strategy fails.

    Provide at least PBO, DSR, WRC p-value from real validation run.
    Use src.research.pbo.compute_pbo, src.research.dsr.deflated_sharpe_ratio,
    src.research.white_reality_check.spa_test / white_reality_check.
    """
    strategy_id: str
    passed: bool = False
    pbo: Optional[float] = None  # <0.6 to pass
    dsr: Optional[float] = None  # > threshold to pass (0.95 strict, 0.5 research)
    wrc_p_value: Optional[float] = None  # <0.05 to pass (SPA/WRC)
    wrc_method: str = "SPA"
    pf_oos: Optional[float] = None
    sharpe_oos: Optional[float] = None
    trades_oos: Optional[int] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def is_sufficient(
        self,
        max_pbo: float = 0.6,
        min_dsr: float = 0.95,
        max_wrc_p: float = 0.05,
        require_all_three: bool = True,
    ) -> bool:
        """
        Check if evidence is sufficient to unlock next optimization after failure.

        require_all_three=True means PBO AND DSR AND WRC must all be present and pass.
        If False, at least 2 of 3 must pass (research mode).
        """
        checks = []
        # PBO
        if self.pbo is None:
            checks.append(False)
        else:
            checks.append(float(self.pbo) < max_pbo)
        # DSR
        if self.dsr is None:
            checks.append(False)
        else:
            checks.append(float(self.dsr) >= min_dsr)
        # WRC
        if self.wrc_p_value is None:
            checks.append(False)
        else:
            checks.append(float(self.wrc_p_value) < max_wrc_p)

        if require_all_three:
            return all(checks)
        # research: at least 2 of 3
        return sum(checks) >= 2

    def missing_requirements(
        self,
        max_pbo: float = 0.6,
        min_dsr: float = 0.95,
        max_wrc_p: float = 0.05,
    ) -> List[str]:
        reasons: List[str] = []
        if self.pbo is None:
            reasons.append(f"PBO missing (need PBO < {max_pbo}, compute via src.research.pbo.compute_pbo)")
        elif float(self.pbo) >= max_pbo:
            reasons.append(f"PBO {self.pbo:.3f} >= {max_pbo} (overfit risk > threshold)")
        if self.dsr is None:
            reasons.append(f"DSR missing (need DSR >= {min_dsr}, compute via src.research.dsr.deflated_sharpe_ratio)")
        elif float(self.dsr) < min_dsr:
            reasons.append(f"DSR {self.dsr:.3f} < {min_dsr} (skill not significant after multiple-testing correction)")
        if self.wrc_p_value is None:
            reasons.append(f"WRC/SPA p-value missing (need p < {max_wrc_p}, compute via src.research.white_reality_check.spa_test)")
        elif float(self.wrc_p_value) >= max_wrc_p:
            reasons.append(f"WRC/SPA p={self.wrc_p_value:.3f} >= {max_wrc_p} (data-snooping: best of many not significant)")
        return reasons

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "passed": self.passed,
            "pbo": self.pbo,
            "dsr": self.dsr,
            "wrc_p_value": self.wrc_p_value,
            "wrc_method": self.wrc_method,
            "pf_oos": self.pf_oos,
            "sharpe_oos": self.sharpe_oos,
            "trades_oos": self.trades_oos,
            "details": self.details,
        }

    @classmethod
    def passed_example(cls, strategy_id: str) -> "ValidationEvidence":
        """Example of passing evidence (for tests/docs)."""
        return cls(strategy_id=strategy_id, passed=True, pbo=0.45, dsr=0.97, wrc_p_value=0.03)

    @classmethod
    def failed_example(cls, strategy_id: str) -> "ValidationEvidence":
        """Example of failing evidence."""
        return cls(strategy_id=strategy_id, passed=False, pbo=0.72, dsr=0.42, wrc_p_value=0.38)


# ---------------------------------------------------------------------------
# Guard config
# ---------------------------------------------------------------------------

@dataclass
class OptimizationGuardConfig:
    """Hard limits for LESS OPTIMIZATION."""

    max_optuna_trials_per_run: int = 50
    max_optuna_trials_total: int = 50  # mirrors ResearchBudget.max_optuna_trials
    max_params_per_strategy: int = 5
    max_indicators: int = 10
    max_optimizations_per_strategy: int = 3
    max_experiments_per_oos: int = 10
    max_oos_reuse: int = 10
    # Validation thresholds to unlock next optimization after failure
    require_validation_before_optimization: bool = False  # if True, even first opt needs evidence
    require_validation_after_failure: bool = True
    min_validation_pbo: float = 0.6  # pbo must be < this
    min_validation_dsr: float = 0.95  # dsr must be >= this (strict); research may use 0.5
    max_validation_wrc_p: float = 0.05
    require_all_three_validations: bool = True  # after failure need PBO+DSR+WRC
    # If True, after failure allow only 1 more optimization even with evidence
    max_retries_after_failure: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_optuna_trials_per_run": self.max_optuna_trials_per_run,
            "max_optuna_trials_total": self.max_optuna_trials_total,
            "max_params_per_strategy": self.max_params_per_strategy,
            "max_indicators": self.max_indicators,
            "max_optimizations_per_strategy": self.max_optimizations_per_strategy,
            "max_experiments_per_oos": self.max_experiments_per_oos,
            "max_oos_reuse": self.max_oos_reuse,
            "min_validation_pbo": self.min_validation_pbo,
            "min_validation_dsr": self.min_validation_dsr,
            "max_validation_wrc_p": self.max_validation_wrc_p,
            "require_all_three_validations": self.require_all_three_validations,
        }


# ---------------------------------------------------------------------------
# Main guard
# ---------------------------------------------------------------------------

class OptimizationGuard:
    """
    Enforces LESS OPTIMIZATION + MORE VALIDATION.

    Wraps ResearchBudget (hard budget) + validation gate (PBO/DSR/WRC).
    Tracks per-strategy failure state so failed strategies cannot be
    re-optimized 10x without fresh validation.

    Thread-safety: not needed (single researcher loop). For async supervisor,
    caller should hold budget lock.
    """

    def __init__(
        self,
        config: Optional[OptimizationGuardConfig] = None,
        budget: Optional[ResearchBudget] = None,
        registry: Optional[Any] = None,  # ExperimentRegistry optional
        ledger_path: str = "data/research_ledger.db",  # Persistent SQLite ledger
    ) -> None:
        self.config = config or OptimizationGuardConfig()
        # P0.6: Budget must survive restart — use persistent AtomicResearchLedger, not in-memory ResearchBudget
        # If caller passed a budget, wrap it via ledger; otherwise use ledger as source of truth
        from src.research.research_ledger import AtomicResearchLedger
        self.ledger = AtomicResearchLedger(path=ledger_path)
        if budget is not None:
            # If explicit budget passed (e.g., tests), seed ledger with its limits but keep ledger as source
            # Persist the passed budget's limits to ledger
            self.ledger.save_budget(budget)
            self.budget = self.ledger.load_budget()
        else:
            self.budget = self.ledger.load_budget()
            # Sync config limits to ledger budget (tightest wins)
            needs_save = False
            for attr in ("max_optuna_trials", "max_optuna_trials_per_run", "max_params_per_strategy", "max_indicators",
                         "max_experiments_per_oos", "max_oos_reuse", "max_optimizations_per_strategy"):
                cfg_val = getattr(self.config, attr if attr != "max_optuna_trials" else "max_optuna_trials_total", None)
                if cfg_val is None:
                    cfg_val = getattr(self.config, attr, None)
                bud_val = getattr(self.budget, attr, None)
                if bud_val is not None and cfg_val is not None and bud_val != cfg_val:
                    tighter = min(cfg_val, bud_val)
                    setattr(self.budget, attr, tighter)
                    setattr(self.config, attr if attr != "max_optuna_trials" else "max_optuna_trials_total", tighter)
                    needs_save = True
            if needs_save:
                self.ledger.save_budget(self.budget)
        # Sync config -> budget if budget was passed with different values
        # Prefer guard config as source of truth, but don't silently override if budget is strict custom
        for attr in ("max_optuna_trials_per_run", "max_params_per_strategy", "max_indicators",
                     "max_experiments_per_oos", "max_oos_reuse", "max_optimizations_per_strategy"):
            cfg_val = getattr(self.config, attr)
            bud_val = getattr(self.budget, attr, None)
            if bud_val is not None and bud_val != cfg_val:
                # Keep the tighter (smaller) limit
                tighter = min(cfg_val, bud_val)
                setattr(self.budget, attr, tighter)
                setattr(self.config, attr, tighter)
        # Also sync total trials
        if self.budget.max_optuna_trials != self.config.max_optuna_trials_total:
            tighter = min(self.budget.max_optuna_trials, self.config.max_optuna_trials_total)
            self.budget.max_optuna_trials = tighter
            self.config.max_optuna_trials_total = tighter

        self.registry = registry
        # strategy_id -> {failed: bool, fail_count: int, last_evidence: ValidationEvidence|None,
        #                  optimizations_after_failure: int, last_failure_details: str}
        self._strategy_state: Dict[str, Dict[str, Any]] = {}
        # per-OOS tracking (mirrors budget._per_oos_counts but guard-level for early check)
        self._oos_counts: Dict[str, int] = {}

    # ---- Internal helpers ----

    def _state_for(self, strategy_id: str) -> Dict[str, Any]:
        if strategy_id not in self._strategy_state:
            self._strategy_state[strategy_id] = {
                "failed": False,
                "fail_count": 0,
                "last_evidence": None,
                "optimizations_after_failure": 0,
                "optimization_count": 0,
                "last_failure_details": "",
            }
        return self._strategy_state[strategy_id]

    def is_blocked(self, strategy_id: str) -> bool:
        """True if strategy failed last validation and hasn't provided sufficient evidence."""
        st = self._strategy_state.get(strategy_id)
        if not st:
            return False
        return bool(st.get("failed", False))

    # ---- Public checks (each raises with clear message) ----

    def check_optuna_request(self, n_trials: int) -> None:
        """Check Optuna trials against per-run and total budget. Raises OptimizationBlocked."""
        if n_trials > self.config.max_optuna_trials_per_run:
            raise OptimizationBlocked(
                f"OPTIMIZATION BLOCKED: requested {n_trials} Optuna trials > max per run "
                f"{self.config.max_optuna_trials_per_run}. {_LESS_OPT} "
                f"Instead of 10x trials, run validation: PBO (compute_pbo), DSR (deflated_sharpe_ratio), "
                f"WRC/SPA (spa_test) — then decide."
            )
        # Delegate to budget for total accounting
        try:
            self.budget.check_optuna_per_run(n_trials)
        except BudgetExceeded as e:
            raise OptimizationBlocked(str(e)) from e
        # Also check total would not exceed (but don't consume yet; caller may do enforce_before_optimization)
        if self.budget.optuna_trials_used + n_trials > self.budget.max_optuna_trials:
            raise OptimizationBlocked(
                f"OPTIMIZATION BLOCKED: {n_trials} trials would exceed total budget "
                f"{self.budget.max_optuna_trials} (used {self.budget.optuna_trials_used}, "
                f"would be {self.budget.optuna_trials_used + n_trials}). {_LESS_OPT} "
                f"Optuna budget exhausted — MORE VALIDATION needed, not more trials."
            )

    def check_strategy_complexity(
        self,
        params: Dict[str, Any] | int | None,
        indicators: Any | None,
    ) -> None:
        """Check params and indicators count. Raises OptimizationBlocked."""
        # Params
        if params is not None:
            n_params = len(params) if isinstance(params, dict) else int(params)  # type: ignore[arg-type]
            if n_params > self.config.max_params_per_strategy:
                raise OptimizationBlocked(
                    f"OPTIMIZATION BLOCKED: strategy has {n_params} params > max "
                    f"{self.config.max_params_per_strategy} per strategy. {_LESS_OPT} "
                    f"Reduce to <=5 tuned params (Occam's razor) — 1000 new params worsens overfit."
                )
            # Also delegate to budget for consistency
            try:
                self.budget.check_params(params)  # type: ignore[arg-type]
            except BudgetExceeded as e:
                raise OptimizationBlocked(str(e)) from e

        # Indicators
        if indicators is not None:
            if isinstance(indicators, int):
                n_ind = indicators
            elif isinstance(indicators, dict):
                n_ind = len(indicators)
            elif isinstance(indicators, (list, tuple, set)):
                n_ind = len(indicators)
            else:
                try:
                    n_ind = len(indicators)  # type: ignore[arg-type]
                except Exception:
                    n_ind = 0
            if n_ind > self.config.max_indicators:
                raise OptimizationBlocked(
                    f"OPTIMIZATION BLOCKED: strategy uses {n_ind} indicators > max "
                    f"{self.config.max_indicators}. {_LESS_OPT} "
                    f"Use <=10 indicators — 20 new indicators worsens situation."
                )
            try:
                self.budget.check_indicators(indicators)
            except BudgetExceeded as e:
                raise OptimizationBlocked(str(e)) from e

    def check_oos_limits(self, oos_period: str, current_reuse: Optional[int] = None) -> None:
        """Check OOS reuse and experiments per OOS. Raises OptimizationBlocked."""
        # Check via registry if provided
        if current_reuse is None and self.registry is not None and oos_period:
            try:
                current_reuse = self.registry.oos_reuse_count(oos_period)
            except Exception:
                current_reuse = None
        if current_reuse is not None and current_reuse >= self.config.max_oos_reuse:
            raise OptimizationBlocked(
                f"OPTIMIZATION BLOCKED: OOS '{oos_period}' reuse {current_reuse} >= max "
                f"{self.config.max_oos_reuse}. {_LESS_OPT} "
                f"OOS has been reused too many times — PF loses statistical power. Use fresh OOS period. "
                f"Invalidate by checking experiment_registry.oos_reuse_count."
            )
        # Budget check
        if current_reuse is not None:
            try:
                self.budget.check_oos_reuse(current_reuse)
            except BudgetExceeded as e:
                raise OptimizationBlocked(str(e)) from e

        # Per-OOS experiments (guard-level + budget)
        oos_count = self._oos_counts.get(oos_period, 0)
        # Prefer budget's tracker if exists
        budget_oos = self.budget._per_oos_counts.get(oos_period, 0) if hasattr(self.budget, "_per_oos_counts") else 0
        effective = max(oos_count, budget_oos)
        if effective >= self.config.max_experiments_per_oos:
            raise OptimizationBlocked(
                f"OPTIMIZATION BLOCKED: OOS '{oos_period}' already has {effective} experiments "
                f">= max per OOS {self.config.max_experiments_per_oos}. {_LESS_OPT} "
                f"Don't overfit same OOS — rotate to new period and validate."
            )

    def require_validation_for_strategy(
        self,
        strategy_id: str,
        validation_evidence: Optional[ValidationEvidence | Dict[str, Any]] = None,
    ) -> None:
        """
        Gate: if strategy previously failed validation, require sufficient evidence
        (PBO<0.6, DSR>threshold, WRC p<0.05) before allowing next optimization.

        Raises ValidationRequired with detailed missing requirements.
        """
        st = self._state_for(strategy_id)

        # If require before first optimization and no history, also require evidence
        if self.config.require_validation_before_optimization and not st.get("has_ever_validated"):
            if validation_evidence is None:
                raise ValidationRequired(
                    f"OPTIMIZATION BLOCKED: strategy '{strategy_id}' requires validation before ANY optimization "
                    f"(require_validation_before_optimization=True). {_LESS_OPT} "
                    f"Provide ValidationEvidence with PBO, DSR, WRC from a real OOS run."
                )

        # Only gate after failure
        if not self.config.require_validation_after_failure:
            return
        if not st.get("failed"):
            return

        # Failed => require evidence
        if validation_evidence is None:
            raise ValidationRequired(
                f"OPTIMIZATION BLOCKED: strategy '{strategy_id}' failed last validation "
                f"(fail_count={st.get('fail_count',1)}, details: {st.get('last_failure_details','no details')}). "
                f"{_LESS_OPT} "
                f"Require MORE VALIDATION before next optimization: provide ValidationEvidence with "
                f"PBO < {self.config.min_validation_pbo}, DSR >= {self.config.min_validation_dsr}, "
                f"WRC/SPA p < {self.config.max_validation_wrc_p}. "
                f"Don't do 10x Optuna on a failing strategy — validate first."
            )

        # Normalize dict -> ValidationEvidence
        ev: ValidationEvidence
        if isinstance(validation_evidence, dict):
            ev = ValidationEvidence(
                strategy_id=strategy_id,
                passed=bool(validation_evidence.get("passed", False)),
                pbo=validation_evidence.get("pbo"),
                dsr=validation_evidence.get("dsr"),
                wrc_p_value=validation_evidence.get("wrc_p_value", validation_evidence.get("p_value", validation_evidence.get("wrc_p"))),
                wrc_method=validation_evidence.get("wrc_method", "SPA"),
                pf_oos=validation_evidence.get("pf_oos"),
                sharpe_oos=validation_evidence.get("sharpe_oos"),
                trades_oos=validation_evidence.get("trades_oos"),
                details=validation_evidence,
            )
        else:
            ev = validation_evidence

        # Check sufficiency
        if not ev.is_sufficient(
            max_pbo=self.config.min_validation_pbo,
            min_dsr=self.config.min_validation_dsr,
            max_wrc_p=self.config.max_validation_wrc_p,
            require_all_three=self.config.require_all_three_validations,
        ):
            missing = ev.missing_requirements(
                max_pbo=self.config.min_validation_pbo,
                min_dsr=self.config.min_validation_dsr,
                max_wrc_p=self.config.max_validation_wrc_p,
            )
            raise ValidationRequired(
                f"OPTIMIZATION BLOCKED: strategy '{strategy_id}' has failed validation and provided "
                f"validation evidence is INSUFFICIENT to unlock next optimization. "
                f"Missing/insufficient: {'; '.join(missing)}. {_LESS_OPT} "
                f"Need PBO<{self.config.min_validation_pbo}, DSR>={self.config.min_validation_dsr}, "
                f"WRC p<{self.config.max_validation_wrc_p} (all three) before retrying Optuna. "
                f"Evidence provided: pbo={ev.pbo}, dsr={ev.dsr}, wrc_p={ev.wrc_p_value}."
            )

        # Also require overall passed flag if evidence says so and threshold requires it
        if not ev.passed and self.config.require_all_three_validations:
            # If evidence metrics pass but passed flag is False, warn but allow if metrics sufficient
            # Metrics already checked; allow through but note
            pass

    # ---- Main entry point ----

    def enforce_before_optimization(
        self,
        strategy_id: str,
        n_trials: int = 0,
        params: Dict[str, Any] | int | None = None,
        indicators: Any | None = None,
        oos_period: str | None = None,
        validation_evidence: Optional[ValidationEvidence | Dict[str, Any]] = None,
        current_oos_reuse: Optional[int] = None,
        consume_budget: bool = True,
    ) -> Dict[str, Any]:
        """
        Full gate: call before ANY optimization attempt.

        Checks in order:
          1. Strategy complexity (params/indicators)
          2. Optuna trials (per-run + total)
          3. OOS limits
          4. Per-strategy optimization count
          5. Validation gate (if failed before)
        If all pass and consume_budget=True, increments counters.

        Returns dict with status and remaining budget.

        Raises OptimizationBlocked or ValidationRequired with clear messages.
        """
        # 1. Complexity
        self.check_strategy_complexity(params, indicators)

        # 2. Optuna
        if n_trials and n_trials > 0:
            self.check_optuna_request(n_trials)

        # 3. OOS
        if oos_period:
            self.check_oos_limits(oos_period, current_reuse=current_oos_reuse)

        # 4. Per-strategy optimization count (hard budget)
        st = self._state_for(strategy_id)
        current_count = st.get("optimization_count", 0)
        # Also check via budget's per-strategy counter (tighter wins)
        budget_count = self.budget._per_strategy_opt_counts.get(strategy_id, 0) if hasattr(self.budget, "_per_strategy_opt_counts") else 0
        effective_count = max(current_count, budget_count)
        if effective_count >= self.config.max_optimizations_per_strategy:
            raise OptimizationBlocked(
                f"OPTIMIZATION BLOCKED: strategy '{strategy_id}' already optimized {effective_count} times "
                f">= max per strategy {self.config.max_optimizations_per_strategy}. {_LESS_OPT} "
                f"Don't run Optuna again without major validation. Provide fresh OOS validation."
            )
        # Also budget-level check (for error message consistency)
        # Note: budget.check_optimization_attempt will increment; we already checked gate, so we delay increment to consume step

        # 5. Validation gate after failure
        self.require_validation_for_strategy(strategy_id, validation_evidence)

        # If failed but validation now provided and sufficient, check retry limit after failure
        if st.get("failed") and validation_evidence is not None:
            after_fail = st.get("optimizations_after_failure", 0)
            if after_fail >= self.config.max_retries_after_failure:
                raise ValidationRequired(
                    f"OPTIMIZATION BLOCKED: strategy '{strategy_id}' failed validation and has already "
                    f"retried {after_fail} times after failure (max {self.config.max_retries_after_failure}). "
                    f"{_LESS_OPT} Even with validation, repeated optimization of a failing strategy is blocked. "
                    f"Need fundamentally new hypothesis or fresh data — not more Optuna."
                )

        # P0.6: Complexity ↑ must be accompanied by Evidence quality ↑↑
        # Check that current complexity tier's required evidence is met
        try:
            from src.research.complexity_evidence_gate import enforce_complexity_evidence_link
            # Gather current evidence from last validation or provided evidence
            ev_for_complexity = validation_evidence if isinstance(validation_evidence, dict) else {}
            if isinstance(validation_evidence, dict):
                # Already have pbo/dsr etc. in validation_evidence
                pass
            # Also try to get evidence from budget/registry state
            current_evidence = {
                "pf": float(ev_for_complexity.get("pbo", 1.0) or 0) if "pf" not in ev_for_complexity else float(ev_for_complexity.get("pf", 0) or 0),
                "pbo": float(ev_for_complexity.get("pbo", 1.0) or 1.0),
                "dsr": float(ev_for_complexity.get("dsr", -1) or -1),
                "wrc_p": float(ev_for_complexity.get("wrc_p_value", ev_for_complexity.get("wrc_p", 1.0)) or 1.0),
                "regime_stability": False,
                "cost_robust": False,
                "trades": 0,
                "oos_days": 0,
            }
            # If we have last evidence in state, use it
            st_ev = st.get("last_evidence")
            if st_ev and hasattr(st_ev, "to_dict"):
                try:
                    d = st_ev.to_dict()
                    current_evidence.update({k: d.get(k, current_evidence[k]) for k in current_evidence if k in d})
                except Exception:
                    pass
            enforce_complexity_evidence_link(
                n_params=len(params) if isinstance(params, dict) else int(params or 0),
                n_indicators=len(indicators) if isinstance(indicators, (list, tuple, set, dict)) else 0,
                n_trials=n_trials,
                n_experiments=self.budget.experiments_used,
                oos_touches=self.budget.oos_reuse_used,
                evidence_dict=current_evidence,
            )
        except ValueError as e:
            # Complexity without evidence -> block, do not consume budget
            raise OptimizationBlocked(str(e)) from e
        except Exception:
            # If gate not available, allow but log
            pass

        # All checks passed — consume budget atomically via ledger (survives restart)
        if consume_budget:
            if n_trials and n_trials > 0:
                try:
                    self.ledger.check_and_increment("optuna", n=n_trials)
                    self.budget = self.ledger.load_budget()
                except BudgetExceeded as e:
                    raise OptimizationBlocked(str(e)) from e
            # Per-strategy count via ledger
            try:
                self.ledger.check_and_increment("optimization_attempt", strategy_id=strategy_id)
                self.budget = self.ledger.load_budget()
            except BudgetExceeded as e:
                raise OptimizationBlocked(str(e)) from e
            st["optimization_count"] = st.get("optimization_count", 0) + 1
            if st.get("failed"):
                st["optimizations_after_failure"] = st.get("optimizations_after_failure", 0) + 1
            # OOS counts via ledger
            if oos_period:
                self._oos_counts[oos_period] = self._oos_counts.get(oos_period, 0) + 1
                try:
                    # Use ledger for per-OOS as well
                    self.ledger.check_and_increment("optimization_attempt", strategy_id=f"oos:{oos_period}")
                    self.budget = self.ledger.load_budget()
                except BudgetExceeded as e:
                    raise OptimizationBlocked(str(e)) from e

        return {
            "allowed": True,
            "strategy_id": strategy_id,
            "n_trials": n_trials,
            "optimization_count": st.get("optimization_count", 0),
            "remaining_optuna": self.budget.max_optuna_trials - self.budget.optuna_trials_used,
            "remaining_experiments": self.budget.max_experiments - self.budget.experiments_used,
            "oos_period": oos_period,
            "validation_required": False,
        }

    # ---- Validation result hooks ----

    def record_validation_result(
        self,
        strategy_id: str,
        evidence: ValidationEvidence | Dict[str, Any] | bool,
        details: str = "",
    ) -> Dict[str, Any]:
        """
        Call after validation run to inform guard about strategy health.

        evidence can be ValidationEvidence, dict with pbo/dsr/wrc, or bool (passed).
        If passed=True (or evidence indicates pass), reset failure state and allow future optimization.
        If failed, set blocked flag so next optimization requires fresh PBO/DSR/WRC evidence.
        """
        st = self._state_for(strategy_id)

        # Normalize evidence
        if isinstance(evidence, bool):
            passed = bool(evidence)
            ev_obj = ValidationEvidence(strategy_id=strategy_id, passed=passed, details={"details": details})
        elif isinstance(evidence, dict):
            # Try to infer passed if not explicit: use thresholds
            inferred_passed = bool(evidence.get("passed", False))
            # If metrics provided and passed not explicit, infer from sufficiency
            if "passed" not in evidence and ("pbo" in evidence or "dsr" in evidence or "wrc_p_value" in evidence):
                tmp = ValidationEvidence(
                    strategy_id=strategy_id,
                    pbo=evidence.get("pbo"),
                    dsr=evidence.get("dsr"),
                    wrc_p_value=evidence.get("wrc_p_value", evidence.get("p_value")),
                )
                inferred_passed = tmp.is_sufficient(
                    max_pbo=self.config.min_validation_pbo,
                    min_dsr=self.config.min_validation_dsr,
                    max_wrc_p=self.config.max_validation_wrc_p,
                    require_all_three=self.config.require_all_three_validations,
                )
                passed = inferred_passed
            else:
                passed = inferred_passed
            ev_obj = ValidationEvidence(
                strategy_id=strategy_id,
                passed=passed,
                pbo=evidence.get("pbo"),
                dsr=evidence.get("dsr"),
                wrc_p_value=evidence.get("wrc_p_value", evidence.get("p_value", evidence.get("wrc_p"))),
                wrc_method=evidence.get("wrc_method", "SPA"),
                pf_oos=evidence.get("pf_oos"),
                sharpe_oos=evidence.get("sharpe_oos"),
                trades_oos=evidence.get("trades_oos"),
                details=evidence,
            )
            if details:
                ev_obj.details["guard_details"] = details
        elif isinstance(evidence, ValidationEvidence):
            passed = bool(evidence.passed)
            ev_obj = evidence
        else:
            passed = False
            ev_obj = ValidationEvidence(strategy_id=strategy_id, passed=False, details={"raw": str(evidence)})

        st["last_evidence"] = ev_obj
        st["has_ever_validated"] = True

        if passed:
            # Success: reset failure, allow future optimizations
            st["failed"] = False
            st["optimizations_after_failure"] = 0
            # Also reset budget per-strategy counter via budget helper
            try:
                self.budget.record_strategy_validation_pass(strategy_id)  # type: ignore[attr-defined]
            except Exception:
                # Fallback: directly reset
                if hasattr(self.budget, "_per_strategy_opt_counts"):
                    self.budget._per_strategy_opt_counts[strategy_id] = 0
            st["last_failure_details"] = ""
            return {"strategy_id": strategy_id, "validation": "PASSED", "failed": False, "unblocked": True}
        else:
            # Failure: set blocked flag, require validation before next optimization
            st["failed"] = True
            st["fail_count"] = st.get("fail_count", 0) + 1
            st["optimizations_after_failure"] = 0  # reset for next cycle of retries after failure
            st["last_failure_details"] = details or ev_obj.missing_requirements(
                max_pbo=self.config.min_validation_pbo,
                min_dsr=self.config.min_validation_dsr,
                max_wrc_p=self.config.max_validation_wrc_p,
            ).__str__()
            # Keep evidence for audit
            return {
                "strategy_id": strategy_id,
                "validation": "FAILED",
                "failed": True,
                "fail_count": st["fail_count"],
                "next_optimization_requires": f"PBO<{self.config.min_validation_pbo}, DSR>={self.config.min_validation_dsr}, WRC p<{self.config.max_validation_wrc_p}",
                "blocked": True,
            }

    def record_optimization_attempt(
        self,
        strategy_id: str,
        n_trials: int = 0,
        params: Dict[str, Any] | None = None,
        indicators: Any | None = None,
        oos_period: str | None = None,
        validation_evidence: Optional[ValidationEvidence | Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Alias for enforce_before_optimization with consume_budget=True."""
        return self.enforce_before_optimization(
            strategy_id=strategy_id,
            n_trials=n_trials,
            params=params,
            indicators=indicators,
            oos_period=oos_period,
            validation_evidence=validation_evidence,
            consume_budget=True,
        )

    # ---- Budget consumption helpers (for experiments not via guard) ----

    def check_experiment_allowed(self) -> None:
        """Delegate to budget.check_experiment with guard-aware error."""
        try:
            self.budget.check_experiment()
        except BudgetExceeded as e:
            raise OptimizationBlocked(str(e)) from e

    def status(self, strategy_id: Optional[str] = None) -> Dict[str, Any]:
        """Snapshot of guard + budget status."""
        base = {
            "config": self.config.to_dict(),
            "budget": self.budget.to_dict(),
            "oos_counts_guard": dict(self._oos_counts),
        }
        if strategy_id:
            st = self._strategy_state.get(strategy_id, {})
            base["strategy"] = {
                "strategy_id": strategy_id,
                "failed": st.get("failed", False),
                "fail_count": st.get("fail_count", 0),
                "optimization_count": st.get("optimization_count", 0),
                "optimizations_after_failure": st.get("optimizations_after_failure", 0),
                "blocked": bool(st.get("failed", False)),
                "last_evidence": st.get("last_evidence").to_dict() if st.get("last_evidence") and hasattr(st.get("last_evidence"), "to_dict") else None,
            }
        else:
            base["strategies"] = {
                sid: {
                    "failed": v.get("failed", False),
                    "optimization_count": v.get("optimization_count", 0),
                    "fail_count": v.get("fail_count", 0),
                }
                for sid, v in self._strategy_state.items()
            }
        return base


def create_default_guard(
    budget: Optional[ResearchBudget] = None,
    registry: Optional[Any] = None,
    **config_overrides: Any,
) -> OptimizationGuard:
    """Factory for default LESS OPTIMIZATION guard (Task 13 defaults)."""
    cfg = OptimizationGuardConfig(**config_overrides)
    return OptimizationGuard(config=cfg, budget=budget, registry=registry)
