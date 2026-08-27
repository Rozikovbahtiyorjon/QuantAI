# QuantAI Module Status Registry

**Generated**: 2026-08-25  
**Total Modules**: 90+  
**Status**: Baseline audit

---

## Legend

| Status | Criteria |
|--------|----------|
| **WRITTEN** | Code exists, `python -m py_compile` clean |
| **UNIT_TESTED** | Dedicated test file, ≥80% coverage, all pass |
| **INTEGRATED** | Imported/used by ≥1 consumer in production path |
| **E2E_VALIDATED** | Passes Validation Gates 1-3 (Backtest, Performance, Walk-Forward) |
| **LONG_RUN_VALIDATED** | Passes Gates 4-5 (7-day paper, 30-day paper) |
| **PRODUCTION_APPROVED** | All gates + security audit + ops runbook + rollback tested |

---

## Level 1: Core Trading (22 modules)

| Module | Written | Unit | Integrated | E2E | Long-Run | Prod | Notes |
|--------|---------|------|------------|-----|----------|------|-------|
| data_loader | ✓ | ✓ | ✓ | ✓ | ? | — | CCXT Binance only |
| exchange_market_data | ✓ | ✓ | ✓ | ✓ | ? | — | |
| indicators | ✓ | ✓ | ✓ | ✓ | ? | — | `add_indicators(core_only=True)` |
| feature_engine | ✓ | ✓ | ✓ | ✓ | ? | — | Microstructure stubs return 0 |
| model_manager | ✓ | ✓ | ✓ | ✓ | ? | — | pickle load/save |
| confidence_engine | ✓ | ✓ | ✓ | ✓ | ? | ✗ | Weights hardcoded |
| strategy | ✓ | ✓ | ✓ | ✓ | ? | ✗ | ML model injected |
| risk_orchestrator | ✓ | ✓ | ✓ | ✓ | ? | ✗ | |
| drawdown_guard | ✓ | ✓ | ✓ | ✓ | ? | ✗ | |
| exposure_manager | ✓ | ✓ | ✓ | ? | ? | ✗ | |
| position_sizer | ✓ | ✓ | ✓ | ✓ | ? | ✗ | Leverage validation |
| risk_manager | ✓ | ✓ | ✓ | ✓ | ? | ✗ | SL/TP calc |
| trade_engine | ✓ | ✓ | ✓ | ✓ | ? | ✗ | SL/TP/BE/trailing |
| trade_engine (execution) | ✓ | ✓ | ? | ? | ? | ✗ | Paper only |
| backtest_engine | ✓ | ✓ | ✓ | ✓ | ? | ✗ | Fresh engine per run |
| walk_forward_engine | ✓ | ✓ | ✓ | ✓ | ? | ✗ | Rolling windows |
| paper_trading_runner | ✓ | ✓ | ✓ | ✓ | ? | ✗ | ML gate |
| paper_trading_session | ✓ | ✓ | ✓ | ? | ? | ✗ | |
| paper_trading_engine | ✓ | ✓ | ✓ | ✓ | ? | ✗ | |
| execution_engine | ✓ | ✓ | ? | ? | ? | ✗ | DRY_RUN/LIVE untested |
| order_manager | ✓ | ✓ | ✓ | ? | ? | ✗ | |
| binance_adapter | ✓ | ✓ | ✓ | ? | ? | ✗ | REST + WS |
| reconciliation_engine | ✓ | ✓ | ? | ? | ? | ✗ | 30s interval |

---

## Level 2: Validation (8 modules)

| Module | Written | Unit | Integrated | E2E | Long-Run | Prod | Notes |
|--------|---------|------|------------|-----|----------|------|-------|
| performance_analyzer | ✓ | ✓ | ✓ | ? | ? | ✗ | |
| walk_forward_analyzer | ✓ | ✓ | ✓ | ? | ? | ✗ | |
| walk_forward_report | ✓ | ✓ | ✓ | ? | ? | ✗ | |
| walk_forward_runner | ✓ | ✓ | ✓ | ? | ? | ✗ | |
| walk_forward_validator | ✓ | ✓ | ✓ | ? | ? | ✗ | v2 exists |
| purged_kfold | ✓ | ✓ | ✓ | ✓ | ? | — | In validation/ |
| quantai_walk_forward_validation | ✓ | ? | ? | ? | ? | ✗ | Integration |
| quantai_ml_walk_forward_integration | ✓ | ? | ? | ? | ? | ✗ | |

---

## Level 3: Execution Boundary (5 modules)

| Module | Written | Unit | Integrated | E2E | Long-Run | Prod | Notes |
|--------|---------|------|------------|-----|----------|------|-------|
| orders | ✓ | ✓ | ✓ | ? | ? | ✗ | |
| order_manager | ✓ | ✓ | ✓ | ? | ? | ✗ | |
| binance_adapter | ✓ | ✓ | ✓ | ? | ? | ✗ | REST + WS |
| execution_engine | ✓ | ✓ | ? | ? | ? | ✗ | 3 modes |
| reconciliation_engine | ✓ | ✓ | ? | ? | ? | ✗ | |

---

## Level 4: Governance & Meta (40+ modules)

| Module | Written | Unit | Integrated | E2E | Long-Run | Prod | Notes |
|--------|---------|------|------------|-----|----------|------|-------|
| champion_registry | ✓ | ? | ? | ? | ? | ✗ | |
| champion_lifecycle | ✓ | ? | ? | ? | ? | ✗ | |
| champion_evaluator | ✓ | ? | ? | ? | ? | ✗ | |
| champion_governance_engine | ✓ | ? | ? | ? | ? | ✗ | |
| champion_admission_controller | ✓ | ? | ? | ? | ? | ✗ | |
| champion_promotion_engine | ✓ | ? | ? | ? | ? | ✗ | |
| champion_replacement_guard | ✓ | ? | ? | ? | ? | ✗ | |
| champion_rollback_guard | ✓ | ? | ? | ? | ? | ✗ | |
| champion_stability_monitor | ✓ | ? | ? | ? | ? | ✗ | |
| champion_improvement_cycle | ✓ | ? | ? | ? | ? | ✗ | |
| champion_performance_feedback | ✓ | ? | ? | ? | ? | ✗ | |
| champion_history | ✓ | ? | ? | ? | ? | ✗ | |
| champion_transition_decision | ✓ | ? | ? | ? | ? | ✗ | |
| champion_transition_executor | ✓ | ? | ? | ? | ? | ✗ | |
| quantai_production_runtime | ✓ | ? | ? | ? | ? | ✗ | |
| quantai_production_runtime_control | ✓ | ? | ? | ? | ? | ✗ | |
| quantai_production_runtime_supervisor | ✓ | ? | ? | ? | ? | ✗ | |
| quantai_production_runtime_lifecycle | ✓ | ? | ? | ? | ? | ✗ | |
| quantai_production_safe_startup_controller | ✓ | ? | ? | ? | ? | ✗ | |
| quantai_production_deployment_preparation | ✓ | ? | ? | ? | ? | ✗ | |
| quantai_production_readiness_gate | ✓ | ? | ? | ? | ? | ✗ | |
| quantai_production_observability | ✓ | ? | ? | ? | ? | ✗ | |
| quantai_production_observability_analytics | ✓ | ? | ? | ? | ? | ✗ | |
| quantai_production_model_registry | ✓ | ? | ? | ? | ? | ✗ | |
| quantai_production_model_registry_integration | ✓ | ? | ? | ? | ? | ✗ | |
| quantai_production_model_runtime_binding | ✓ | ? | ? | ? | ? | ✗ | |
| quantai_production_model_runtime_execution | ✓ | ? | ? | ? | ? | ✗ | |
| quantai_production_model_runtime_incident_management | ✓ | ? | ? | ? | ? | ✗ | |
| quantai_production_model_runtime_lifecycle_recovery_coordination | ✓ | ? | ? | ? | ? | ✗ | |
| quantai_production_model_runtime_monitoring | ✓ | ? | ? | ? | ? | ✗ | |
| quantai_production_model_runtime_monitoring_integration | ✓ | ? | ? | ? | ? | ✗ | |
| quantai_production_model_runtime_recovery | ✓ | ? | ? | ? | ? | ✗ | |
| quantai_production_model_runtime_recovery_integration | ✓ | ? | ? | ? | ? | ✗ | |
| ai_strategy_research_lab | ✓ | ? | ? | ? | ? | ✗ | |
| advanced_strategy_architecture | ✓ | ? | ? | ? | ? | ✗ | |
| microstructure_intelligence | ✓ | ? | ? | ? | ? | ✗ | VPIN/Kyle/Liq stubs |
| order_flow_intelligence | ✓ | ? | ? | ? | ? | ✗ | |
| liquidation_intelligence | ✓ | ? | ? | ? | ? | ✗ | |
| liquidation_heatmap_engine | ✓ | ? | ? | ? | ? | ✗ | |
| sentiment_analysis_engine | ✓ | ? | ? | ? | ? | ✗ | |
| sentiment_divergence_intelligence | ✓ | ? | ? | ? | ? | ✗ | |
| sentiment_information_intelligence | ✓ | ? | ? | ? | ? | ✗ | |
| social_attention_intelligence | ✓ | ? | ? | ? | ? | ✗ | |
| alternative_data | ✓ | ? | ? | ? | ? | ✗ | LunarCrush/Funding/OI stubs |
| market_regime_intelligence | ✓ | ? | ? | ? | ? | ✗ | |
| unified_market_intelligence | ✓ | ? | ? | ? | ? | ✗ | |
| ai_memory | ✓ | ? | ? | ? | ? | ✗ | |
| signal_diagnostics | ✓ | ? | ? | ? | ? | ✗ | |
| signal_quality_analyzer | ✓ | ? | ? | ? | ? | ✗ | |
| strategy_bank | ✓ | ? | ? | ? | ? | ✗ | |
| strategy_genome | ✓ | ? | ? | ? | ? | ✗ | |
| strategy_tournament | ✓ | ? | ? | ? | ? | ✗ | |
| strategy_champion | ✓ | ? | ? | ? | ? | ✗ | |
| capital_allocator | ✓ | ? | ? | ? | ? | ✗ | |
| correlation_risk_engine | ✓ | ? | ? | ? | ? | ✗ | |
| portfolio_exposure_engine | ✓ | ? | ? | ? | ? | ✗ | |
| portfolio_risk_engine | ✓ | ? | ? | ? | ? | ✗ | |
| portfolio_stress_monte_carlo | ✓ | ? | ? | ? | ? | ✗ | |
| kelly_sizer | ✓ | ? | ? | ? | ? | ✗ | |
| dynamic_risk_budget | ✓ | ? | ? | ? | ? | ✗ | |
| cross_margin | ✓ | ? | ? | ? | ? | ✗ | |
| risk_aggregator | ✓ | ? | ? | ? | ? | ✗ | |
| risk_engine | ✓ | ? | ? | ? | ? | ✗ | |
| risk_profile_manager | ✓ | ? | ? | ? | ? | ✗ | |
| risk_rule_engine | ✓ | ? | ? | ? | ? | ✗ | |
| event_risk_intelligence | ✓ | ? | ? | ? | ? | ✗ | |
| price_oi_divergence_intelligence | ✓ | ? | ? | ? | ? | ✗ | |
| open_interest_divergence_engine | ✓ | ? | ? | ? | ? | ✗ | |
| futures_data_engine | ✓ | ? | ? | ? | ? | ✗ | |
| futures_derivatives_intelligence | ✓ | ? | ? | ? | ? | ✗ | |
| liquidity_liquidation_zones | ✓ | ? | ? | ? | ? | ✗ | |
| monte_carlo_engine | ✓ | ? | ? | ? | ? | ✗ | |
| regime_walk_forward | ✓ | ? | ? | ? | ? | ✗ | |
| research_dashboard | ✓ | ? | ? | ? | ? | ✗ | |
| trading_activity_monitor | ✓ | ? | ? | ? | ? | ✗ | |
| trading_activity_optimizer | ✓ | ? | ? | ? | ? | ✗ | |
| trading_activity_policy | ✓ | ? | ? | ? | ? | ✗ | |
| stop_loss_manager | ✓ | ? | ? | ? | ? | ✗ | |
| trailing_stop_engine | ✓ | ? | ? | ? | ? | ✗ | |
| dataset_builder | ✓ | ? | ? | ? | ? | ✗ | |
| ml_engine | ✓ | ✓ | ✓ | ✓ | ? | ✗ | PurgedKFold CV |
| ml_walk_forward | ✓ | ? | ? | ? | ? | ✗ | |
| quantai_model_selection | ✓ | ? | ? | ? | ? | ✗ | |
| quantai_ml_walk_forward_performance_analytics | ✓ | ? | ? | ? | ? | ✗ | |
| fast_vector_backtester | ✓ | ? | ? | ? | ? | ✗ | |
| backtest_engine (legacy) | ✓ | ? | ? | ? | ? | ✗ | Duplicate? |

---

## Orphan Modules (Written but not Integrated)

These modules exist, have tests, but **no production path uses them**:

| Module | Likely Purpose | Blockers |
|--------|----------------|----------|
| champion_* (14) | Strategy lifecycle | No live champion yet |
| quantai_production_* (20+) | Production runtime | Not deployed |
| *_intelligence (10+) | Market signals | Not wired to Strategy |
| ai_memory | Pattern storage | Not used |
| signal_diagnostics | Signal analysis | Not in pipeline |
| strategy_bank/genome/tournament | Strategy evolution | Research only |
| kelly_sizer/dynamic_risk_budget | Advanced sizing | Not in RiskOrchestrator |
| cross_margin | Margin modes | Binance adapter only |
| monte_carlo_engine | Stress testing | Not in validation |
| fast_vector_backtester | Vectorized BT | Not used |

---

## Priority Actions

### Phase 0.5 (Governance Setup)
- [ ] Complete this registry for all 90+ modules
- [ ] Add Architecture Decision Records for key choices
- [ ] Define interface contracts (docs/integration/)

### Phase 1 (Core Hardening)
- [ ] Wire microstructure intelligence → FeatureEngine → Strategy
- [ ] Wire market_regime_intelligence → Strategy (regime filter)
- [ ] Integrate Kelly/dynamic sizing into RiskOrchestrator
- [ ] Complete ExecutionEngine DRY_RUN/LIVE testing

### Phase 2 (Champion Pipeline)
- [ ] Activate Champion Engine with paper trading
- [ ] Define admission/promotion thresholds
- [ ] Build Governance Engine decision loop

### Phase 3 (Production)
- [ ] Production Runtime safe startup
- [ ] Observability + incident management
- [ ] Model registry + rollback

---

**Next Update**: After Phase 0.5 governance docs complete.