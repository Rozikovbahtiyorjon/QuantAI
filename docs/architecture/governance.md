# QuantAI Governance & Meta Architecture

## Level 4 — Governance Layer

```
Champion Engine
    ↓
Governance Engine
    ↓
Production Runtime
    ↓
Strategy Research
    ↓
Market Intelligence
    ↓
AI Memory
```

## Purpose

Manage **strategy lifecycle** from research → validation → production → retirement.

---

## Champion Engine (`src/champion_*.py`)

### Components

| Module | Responsibility |
|--------|----------------|
| `champion_registry.py` | Strategy registration, versioning |
| `champion_lifecycle.py` | State machine: CANDIDATE → CHALLENGER → CHAMPION → RETIRED |
| `champion_evaluator.py` | Metric computation (Sharpe, Calmar, DD, stability) |
| `champion_governance_engine.py` | Promotion/demotion decisions |
| `champion_admission_controller.py` | Entry criteria for CANDIDATE |
| `champion_promotion_engine.py` | CHALLENGER → CHAMPION logic |
| `champion_replacement_guard.py` | Prevent thrashing |
| `champion_rollback_guard.py` | Emergency rollback |
| `champion_stability_monitor.py` | Drift detection |
| `champion_improvement_cycle.py` | Retraining triggers |
| `champion_performance_feedback.py` | Post-trade attribution |
| `champion_history.py` | Audit trail |
| `champion_transition_decision.py` | State transition logic |
| `champion_transition_executor.py` | Atomic transitions |

### Status Contract

```
CANDIDATE      →  Backtest + WF pass
    ↓
CHALLENGER     →  Paper trading 7+ days, metrics ≥ threshold
    ↓
CHAMPION       →  Live allocation, monitored
    ↓
RETIRED        →  Performance decay or risk breach
```

### Promotion Criteria

| Metric | Challenger → Champion |
|--------|----------------------|
| Paper days | ≥ 30 |
| Sharpe | ≥ 1.0 |
| Max DD | ≤ 8% |
| Win rate | 45-55% (with 2:1 RR) |
| Profit factor | ≥ 1.3 |
| Stability | ≤ 20% metric variance across weeks |
| Risk compliance | 0 violations |

### Demotion Triggers

| Trigger | Action |
|---------|--------|
| Live DD > 10% | Immediate → RETIRED |
| 30-day Sharpe < 0.5 | → CHALLENGER (re-eval) |
| Risk violation | Immediate → RETIRED |
| Regime failure | → CHALLENGER (adapt) |

---

## Governance Engine (`src/champion_governance_engine.py`)

**Decision Loop** (daily):
1. Collect metrics from all CHAMPION/CHALLENGER
2. Evaluate stability, risk, performance
3. Compute transition decisions
4. Execute transitions (atomic)
5. Log audit trail

**Contract**: All decisions recorded with timestamp, rationale, metrics snapshot.

---

## Production Runtime (`src/quantai_production_*.py`)

### Components

| Module | Responsibility |
|--------|----------------|
| `quantai_production_runtime.py` | Main runtime coordinator |
| `quantai_production_runtime_control.py` | Start/stop/pause/resume |
| `quantai_production_runtime_supervisor.py` | Health monitoring, auto-restart |
| `quantai_production_runtime_lifecycle.py` | Deployment lifecycle |
| `quantai_production_safe_startup_controller.py` | Pre-flight checks |
| `quantai_production_deployment_preparation.py` | Config validation, migration |
| `quantai_production_readiness_gate.py` | Gate: all systems green |
| `quantai_production_observability.py` | Metrics, logging, tracing |
| `quantai_production_observability_analytics.py` | Anomaly detection |
| `quantai_production_model_registry.py` | Model versioning, rollback |
| `quantai_production_model_runtime_*.py` | Model binding, execution, incidents, recovery |

### Deployment Gates

```
PRE_FLIGHT
    ↓ (config valid, models loaded, connections OK)
READINESS_GATE
    ↓ (all health checks pass)
RUNTIME_START
    ↓ (supervision active)
LIVE
```

### Incident Management

| Severity | Response |
|---------|----------|
| P0 (Capital at risk) | Auto-stop, page, manual investigation |
| P1 (Degraded) | Alert, auto-retry, monitor |
| P2 (Warning) | Log, scheduled review |

---

## Strategy Research (`src/ai_strategy_research_lab.py`, `src/advanced_strategy_architecture.py`)

**Purpose**: Safe sandbox for new strategy ideas.

**Contract**:
- Isolated from production path
- Uses same data/indicators/ML
- Results → Champion Admission Controller
- No live capital exposure

---

## Market Intelligence (`src/*intelligence.py`, `src/*_engine.py`)

| Module | Output |
|--------|--------|
| `microstructure_intelligence.py` | VPIN, Kyle's Lambda, liquidation levels |
| `order_flow_intelligence.py` | Bid/ask pressure, flow toxicity |
| `liquidation_intelligence.py` | Liquidation clusters, heatmap |
| `sentiment_analysis_engine.py` | Social sentiment, news sentiment |
| `alternative_data.py` | LunarCrush, funding rates, OI delta |
| `market_regime_intelligence.py` | Regime classification (trend/flat/volatile) |
| `unified_market_intelligence.py` | Aggregated signals |

**Contract**: All intelligence modules output standardized `Signal` or `Feature` objects consumed by Strategy/Champion.

---

## AI Memory (`src/ai_memory.py`)

**Purpose**: Long-term pattern storage for strategy adaptation.

**Contract**:
- Stores: (market_context, strategy_action, outcome)
- Retrieval: k-NN on market context
- Used by: Strategy Research, Champion Improvement Cycle

---

## Module Status

| Module | Written | Unit | Integrated | E2E | Long-Run | Prod |
|--------|---------|------|------------|-----|----------|------|
| champion_registry | ✓ | ? | ? | ? | ? | ✗ |
| champion_lifecycle | ✓ | ? | ? | ? | ? | ✗ |
| champion_evaluator | ✓ | ? | ? | ? | ? | ✗ |
| champion_governance_engine | ✓ | ? | ? | ? | ? | ✗ |
| champion_admission_controller | ✓ | ? | ? | ? | ? | ✗ |
| champion_promotion_engine | ✓ | ? | ? | ? | ? | ✗ |
| champion_replacement_guard | ✓ | ? | ? | ? | ? | ✗ |
| champion_rollback_guard | ✓ | ? | ? | ? | ? | ✗ |
| champion_stability_monitor | ✓ | ? | ? | ? | ? | ✗ |
| champion_improvement_cycle | ✓ | ? | ? | ? | ? | ✗ |
| champion_performance_feedback | ✓ | ? | ? | ? | ? | ✗ |
| champion_history | ✓ | ? | ? | ? | ? | ✗ |
| champion_transition_decision | ✓ | ? | ? | ? | ? | ✗ |
| champion_transition_executor | ✓ | ? | ? | ? | ? | ✗ |
| quantai_production_runtime | ✓ | ? | ? | ? | ? | ✗ |
| quantai_production_runtime_control | ✓ | ? | ? | ? | ? | ✗ |
| quantai_production_runtime_supervisor | ✓ | ? | ? | ? | ? | ✗ |
| quantai_production_runtime_lifecycle | ✓ | ? | ? | ? | ? | ✗ |
| quantai_production_safe_startup_controller | ✓ | ? | ? | ? | ? | ✗ |
| quantai_production_deployment_preparation | ✓ | ? | ? | ? | ? | ✗ |
| quantai_production_readiness_gate | ✓ | ? | ? | ? | ? | ✗ |
| quantai_production_observability | ✓ | ? | ? | ? | ? | ✗ |
| quantai_production_observability_analytics | ✓ | ? | ? | ? | ? | ✗ |
| quantai_production_model_registry | ✓ | ? | ? | ? | ? | ✗ |
| quantai_production_model_runtime_binding | ✓ | ? | ? | ? | ? | ✗ |
| quantai_production_model_runtime_execution | ✓ | ? | ? | ? | ? | ✗ |
| quantai_production_model_runtime_incident_management | ✓ | ? | ? | ? | ? | ✗ |
| quantai_production_model_runtime_lifecycle_recovery_coordination | ✓ | ? | ? | ? | ? | ✗ |
| quantai_production_model_runtime_monitoring | ✓ | ? | ? | ? | ? | ✗ |
| quantai_production_model_runtime_monitoring_integration | ✓ | ? | ? | ? | ? | ✗ |
| quantai_production_model_runtime_recovery | ✓ | ? | ? | ? | ? | ✗ |
| quantai_production_model_runtime_recovery_integration | ✓ | ? | ? | ? | ? | ✗ |
| ai_strategy_research_lab | ✓ | ? | ? | ? | ? | ✗ |
| advanced_strategy_architecture | ✓ | ? | ? | ? | ? | ✗ |
| microstructure_intelligence | ✓ | ? | ? | ? | ? | ✗ |
| order_flow_intelligence | ✓ | ? | ? | ? | ? | ✗ |
| liquidation_intelligence | ✓ | ? | ? | ? | ? | ✗ |
| sentiment_analysis_engine | ✓ | ? | ? | ? | ? | ✗ |
| alternative_data | ✓ | ? | ? | ? | ? | ✗ |
| market_regime_intelligence | ✓ | ? | ? | ? | ? | ✗ |
| unified_market_intelligence | ✓ | ? | ? | ? | ? | ✗ |
| ai_memory | ✓ | ? | ? | ? | ? | ✗ |

---

## Governance Rules

1. **No module graduates** without passing its level's gates
2. **Champion changes** require Governance Engine decision + audit log
3. **Production deployments** require Safe Startup Controller green
4. **Research modules** never touch live capital path
5. **All decisions** recorded with timestamp, rationale, metrics
6. **Rollback** always possible within 5 min (model, config, strategy)

---

**Status**: Governance architecture defined. Next: Module Status Registry.