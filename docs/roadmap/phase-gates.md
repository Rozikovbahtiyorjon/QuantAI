# QuantAI Phase Gates

**Governance**: Each phase must pass **all** gates before starting next phase.  
**Checkpoint**: Every phase end → full test suite → `py_compile` → commit tag.

---

## Phase 0.5: Governance Foundation (Current)

### Goal
Establish development governance, contracts, and module status baseline.

### Gates

| Gate | Criteria | Verification |
|------|----------|--------------|
| **G0.5-1 Architecture Docs** | `docs/architecture/*.md` complete for Levels 1-4 | Files exist, reviewed |
| **G0.5-2 Module Registry** | `MODULE_STATUS.md` covers all 90+ modules | 100% coverage |
| **G0.5-3 Requirements Template** | `docs/requirements/module-template.md` approved | Team sign-off |
| **G0.5-4 ADR Process** | `docs/decisions/` structure, first ADR written | ADR-0001 exists |
| **G0.5-5 Contract Definitions** | `docs/integration/interface-contracts.md` for core pipeline | All Level 1 interfaces documented |
| **G0.5-6 CI Gates** | Validation gates YAML defined | `.github/workflows/validation.yml` |
| **G0.5-7 Full Test Suite** | All existing tests pass | `pytest tests/ -x` green |

### Deliverables
- [x] Architecture docs (4 levels)
- [x] Module status registry
- [x] Requirements template
- [ ] ADR-0001: ML CV Strategy
- [ ] ADR-0002: Risk Orchestrator Pattern
- [ ] ADR-0003: Signal Fusion Rules
- [ ] Interface contracts doc
- [ ] CI validation workflow

### Checkpoint Commit
```bash
git tag phase-0.5-governance-complete
```

---

## Phase 1: Strategy Redesign (Weeks 1-4)

### Goal
Replace monolithic `strategy.py` with modular, regime-aware pipeline.

### Gates

| Gate | Criteria | Verification |
|------|----------|--------------|
| **G1-1 Modular Split** | `strategy/` package with 5 modules | `src/strategy/__init__.py` exports all |
| **G1-2 AI Analyzer** | Trend/Momentum/Volume/Volatility as separate classes | Unit tests each |
| **G1-3 ML Fusion** | Configurable fusion rules (not hardcoded) | Parametrized tests |
| **G1-4 Order Flow Gate** | Real VPIN/Kyle/Liquidation integration | FeatureEngine outputs real values |
| **G1-5 Signal Generator** | Assembles final SignalResult | Integration test |
| **G1-6 Regime Detection** | `market_regime_intelligence` consumed | Regime filter in pipeline |
| **G1-7 Dynamic SL/TP** | ATR multipliers adapt to volatility regime | Backtest comparison |
| **G1-8 Config Driven** | All thresholds from `settings.yaml` | No magic numbers in code |
| **G1-9 Backtest Parity** | New strategy ≥ old on full dataset | Metrics comparison |
| **G1-10 Walk-Forward** | New strategy passes Gate 3 | `pytest tests/test_walk_forward_*.py` |

### Deliverables
- [ ] `src/strategy/ai_analyzer.py`
- [ ] `src/strategy/ml_fusion.py`
- [ ] `src/strategy/order_flow_gate.py`
- [ ] `src/strategy/signal_generator.py`
- [ ] `src/strategy/sl_tp_calculator.py`
- [ ] `src/strategy/__init__.py` (public API)
- [ ] Updated `generate_signal_result()` signature
- [ ] All tests pass

### Checkpoint Commit
```bash
git tag phase-1-strategy-redesign-complete
```

---

## Phase 2: Risk & Portfolio (Weeks 5-7)

### Goal
Multi-position, correlation-aware, adaptive risk management.

### Gates

| Gate | Criteria | Verification |
|------|----------|--------------|
| **G2-1 Multi-Position** | `max_open_positions` configurable (3-5) | TradeEngine supports N |
| **G2-2 Correlation Limits** | `max_correlation=0.85`, `max_correlated=2` | RiskOrchestrator enforces |
| **G2-3 Kelly Sizing** | `position_sizer_method=kelly` option | Integrated in RiskOrchestrator |
| **G2-4 Volatility Sizing** | `position_sizer_method=volatility_adjusted` | Integrated |
| **G2-5 Pyramiding** | Add to winners at 1R, 2R | TradeEngine logic |
| **G2-6 Partial TP** | Close 33% at 1R, 33% at 2R, 34% at 3R | Configurable |
| **G2-7 Portfolio Risk** | Total portfolio DD, correlation stress | Monte Carlo integration |
| **G2-8 Cross-Asset** | BTC/ETH/SOL simultaneous | Data pipeline multi-symbol |
| **G2-9 Stress Test** | `-20%` crash + API lag scenario | `stress_test_engine` |
| **G2-10 Backtest** | Multi-asset backtest passes Gate 1-3 | Metrics per asset + portfolio |

### Deliverables
- [ ] Enhanced `RiskOrchestrator`
- [ ] `PortfolioRiskEngine` integration
- [ ] `CorrelationMatrix` real-time
- [ ] `KellySizer` / `VolatilityAdjustedSizer`
- [ ] `stress_test_engine` scenarios
- [ ] Updated config schema

### Checkpoint Commit
```bash
git tag phase-2-risk-portfolio-complete
```

---

## Phase 3: ML Enhancement (Weeks 8-11)

### Goal
Production-grade ML with walk-forward retraining, calibration, ensembles.

### Gates

| Gate | Criteria | Verification |
|------|----------|--------------|
| **G3-1 WF Retraining** | Retrain every N windows (configurable) | `ml_walk_forward` integration |
| **G3-2 Optuna HPO** | Bayesian optimization per retrain | `optuna` study persisted |
| **G3-3 Probability Calibration** | Platt/Isotonic on validation fold | `CalibratedClassifierCV` |
| **G3-4 Ensemble** | XGBoost + LightGBM + CatBoost voting | `VotingClassifier` |
| **G3-5 Meta-Labeling** | Predict trade outcome, not direction | Primary model → meta-model |
| **G3-6 Feature Stability** | SHAP importance stability > 0.8 | Per-window tracking |
| **G3-7 Model Registry** | Versioned, rollback capable | `quantai_production_model_registry` |
| **G3-8 Champion ML** | ML model as Challenger → Champion | Governance Engine integration |
| **G3-9 OOS Validation** | Ensemble beats single model on Gate 3 | Statistical significance |
| **G3-10 Inference Latency** | < 5ms per prediction | Benchmark test |

### Deliverables
- [ ] `MLWalkForwardRetrainer`
- [ ] `OptunaOptimizer` with study persistence
- [ ] `CalibratedEnsemble`
- [ ] `MetaLabeler`
- [ ] SHAP stability monitoring
- [ ] Model registry integration
- [ ] Updated `MLEngine` API

### Checkpoint Commit
```bash
git tag phase-3-ml-enhancement-complete
```

---

## Phase 4: Validation & Production Hardening (Weeks 12-14)

### Goal
Statistical rigor, cross-asset validation, production readiness.

### Gates

| Gate | Criteria | Verification |
|------|----------|--------------|
| **G4-1 Monte Carlo** | 1000+ sims with parameter noise | `monte_carlo_engine` |
| **G4-2 Bootstrap CI** | 95% CI on Sharpe, Calmar, DD | `performance_analyzer` |
| **G4-3 Synthetic Data** | Known-edge vs random validation | Generated datasets |
| **G4-4 Cross-Asset** | BTC, ETH, SOL, BNB all pass Gate 3 | Separate datasets |
| **G4-5 Multi-Timeframe** | 5m, 15m, 1h, 4h, 1d all tested | Timeframe parameter sweep |
| **G4-6 Regime Coverage** | Bull/bear/sideways/volatile all tested | Historical periods |
| **G4-7 Paper 30 Days** | Live paper trading 30+ days | `PaperTradingRunner` |
| **G4-8 DRY_RUN** | Binance testnet, 7 days | `ExecutionEngine(DRY_RUN)` |
| **G4-9 Security Audit** | No secrets, rate limits, auth | Checklist |
| **G4-10 Ops Runbook** | Deploy, monitor, rollback, incident | `docs/operations/` |

### Deliverables
- [ ] Monte Carlo stress test results
- [ ] Bootstrap confidence intervals
- [ ] Cross-asset validation report
- [ ] 30-day paper trading log
- [ ] DRY_RUN testnet log
- [ ] Security audit checklist
- [ ] Operations runbook
- [ ] Runbook: deploy/monitor/rollback/incident

### Checkpoint Commit
```bash
git tag phase-4-validation-production-complete
```

---

## Phase 5: Live Deployment (Week 15+)

### Goal
Controlled live deployment with progressive capital allocation.

### Gates

| Gate | Criteria | Verification |
|------|----------|--------------|
| **G5-1 Micro Capital** | $100-500 live, 14 days | Real PnL tracking |
| **G5-2 Scale 1** | $1,000-5,000, 30 days | DD < 5% |
| **G5-3 Scale 2** | $10,000+, 60 days | DD < 8%, Sharpe > 1 |
| **G5-4 Full Capital** | Target allocation | All metrics green |

### Deliverables
- [ ] Live deployment checklist
- [ ] Capital allocation schedule
- [ ] Daily/weekly review process
- [ ] Quarterly strategy review

---

## Summary: Phase Dependencies

```
Phase 0.5 (Governance)
       ↓
Phase 1 (Strategy) ──────────────────┐
       ↓                              │
Phase 2 (Risk) ←─────────────────────┤
       ↓                              │
Phase 3 (ML) ←───────────────────────┤
       ↓                              │
Phase 4 (Validation) ←───────────────┤
       ↓                              │
Phase 5 (Live) ←─────────────────────┘
```

**Critical Path**: 0.5 → 1 → 2 → 3 → 4 → 5 (sequential)

**Parallelizable**: 
- Docs/ADRs during any phase
- Cross-asset data collection during Phase 1-2
- Synthetic data generation during Phase 3

---

**Status**: Phase 0.5 in progress. Next: Phase 1 kickoff after governance complete.