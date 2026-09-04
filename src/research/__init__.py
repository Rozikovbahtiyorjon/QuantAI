"""QuantAI Research Integrity Engine v1 — Audit 5.2 + Task 13 LESS OPTIMIZATION + Task 14 MAX ROBUST OOS EDGE

Modules:
- experiment_registry: tracks every experiment with dataset/model/commit hash, OOS touches
- dataset_registry: canonical dataset registry with SHA256
- research_budget: enforces max experiments / trials / OOS reuse to prevent researcher overfitting
- optimization_guard: Task 13 LESS OPTIMIZATION + MORE VALIDATION guard
- robust_oos_edge: Task 14 MAX ROBUST OOS EDGE (8-component weighted KPI + gate)
"""

from .experiment_registry import ExperimentRegistry, ExperimentRecord
from .dataset_registry import DatasetRegistry, DatasetRecord
from .research_budget import ResearchBudget, BudgetExceeded

try:
    from .optimization_guard import (
        OptimizationBlocked,
        OptimizationGuard,
        OptimizationGuardConfig,
        ValidationEvidence,
        ValidationRequired,
        create_default_guard,
    )
except Exception:  # graceful degrade if guard not available
    OptimizationBlocked = BudgetExceeded  # type: ignore
    OptimizationGuard = None  # type: ignore
    OptimizationGuardConfig = None  # type: ignore
    ValidationEvidence = None  # type: ignore
    ValidationRequired = RuntimeError  # type: ignore
    create_default_guard = None  # type: ignore

try:
    from .robust_oos_edge import (
        RobustOOSComponents,
        RobustOOSConfig,
        RobustOOSResult,
        ComponentResult,
        compute_robust_oos_edge,
        is_robust_edge,
        evaluate_metrics_for_robust_edge,
    )
except Exception:
    RobustOOSComponents = None  # type: ignore
    RobustOOSConfig = None  # type: ignore
    RobustOOSResult = None  # type: ignore
    ComponentResult = None  # type: ignore
    compute_robust_oos_edge = None  # type: ignore
    is_robust_edge = None  # type: ignore
    evaluate_metrics_for_robust_edge = None  # type: ignore

__all__ = [
    "ExperimentRegistry",
    "ExperimentRecord",
    "DatasetRegistry",
    "DatasetRecord",
    "ResearchBudget",
    "BudgetExceeded",
    "OptimizationBlocked",
    "OptimizationGuard",
    "OptimizationGuardConfig",
    "ValidationEvidence",
    "ValidationRequired",
    "create_default_guard",
    "RobustOOSComponents",
    "RobustOOSConfig",
    "RobustOOSResult",
    "ComponentResult",
    "compute_robust_oos_edge",
    "is_robust_edge",
    "evaluate_metrics_for_robust_edge",
]
