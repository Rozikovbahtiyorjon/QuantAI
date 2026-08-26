"""Allow archived tests to resolve modules that remained in src/.

Some frozen tests cover BOTH moved and conflict-pinned modules.
For pinned modules (still living in src/) we alias
experimental.src.<mod> -> src.<mod> so archived tests import fine.
"""
import importlib
import sys

PINNED_IN_SRC = [
    "champion_admission_controller",
    "champion_governance_engine",
    "champion_history",
    "champion_performance_feedback",
    "champion_promotion_engine",
    "champion_registry",
    "champion_replacement_guard",
    "champion_rollback_guard",
    "champion_stability_monitor",
    "champion_transition_decision",
    "champion_transition_executor",
    "order_flow_strategy_integration",
    "order_flow_unified_integration",
    "paper_trading_market_session",
    "paper_trading_monitor",
    "paper_trading_presenter",
    "paper_trading_quality_gate",
    "paper_trading_report",
    "quantai_production_model_runtime_lifecycle_recovery_coordination",
    "quantai_production_model_runtime_monitoring_integration",
    "quantai_production_observability_analytics",
    "quantai_production_readiness_integration",
    "quantai_production_runtime_control",
    "quantai_production_runtime_supervisor_integration",
    "quantai_production_safe_startup_controller",
    "strategy_champion",
    "strategy_tournament",
    "trading_activity_policy",
    "walk_forward_analyzer",
    "monte_carlo_engine",
]

for _m in PINNED_IN_SRC:
    try:
        importlib.import_module(f"experimental.src.{_m}")
    except ModuleNotFoundError:
        try:
            sys.modules[f"experimental.src.{_m}"] = importlib.import_module(f"src.{_m}")
        except Exception:
            pass
