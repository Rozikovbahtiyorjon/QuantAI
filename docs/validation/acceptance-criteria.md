# QuantAI Acceptance Criteria

**Version**: 1.0  
**Scope**: All validation gates

---

## Gate 1: Backtest Acceptance

### Pre-conditions
- [ ] Prepared data passes `BacktestEngine.validate_data()`
- [ ] Indicators computed via `add_indicators(df, core_only=True)`
- [ ] Strategy module imports without error

### Criteria

| Criterion | Threshold | Measurement |
|-----------|-----------|-------------|
| **Determinism** | Identical results | Same seed → byte-for-byte identical trades |
| **No Crashes** | 0 unhandled exceptions | Full dataset run completes |
| **Trade Math** | PnL = gross - commission | `test_trade_engine_math.py` all pass |
| **Risk Math** | Position sizing correct | `test_trade_engine_risk.py` all pass |
| **SL/TP Logic** | Conservative (SL first) | `test_trade_engine_behavior.py` all pass |
| **Edge Cases** | Empty, NaN, gaps handled | `test_backtest_engine_edge_cases.py` all pass |
| **Data Validation** | Rejects bad input | `test_backtest_engine.py::test_validate_data_*` pass |

### Sign-off
- [ ] All pytest tests in `tests/test_backtest_*.py` pass
- [ ] All pytest tests in `tests/test_trade_engine_*.py` pass
- [ ] No flaky tests (3 consecutive runs green)

---

## Gate 2: Performance Acceptance

### Pre-conditions
- [ ] Gate 1 passed
- [ ] Full dataset backtest completes

### Minimum Thresholds (Full Dataset)

| Metric | Minimum | Target | Measurement |
|--------|---------|--------|-------------|
| Net Profit | > 0 | > 10% | `BacktestResult.net_profit` |
| Win Rate | > 40% | > 50% | `BacktestResult.win_rate` |
| Profit Factor | > 1.2 | > 1.5 | Gross profit / Gross loss |
| Max Drawdown | < 15% | < 10% | Peak-to-trough equity |
| Sharpe (ann.) | > 0.5 | > 1.0 | `PerformanceAnalyzer.sharpe` |
| Calmar | > 0.5 | > 1.5 | Return / Max DD |
| Recovery Factor | > 1.0 | > 2.0 | Net profit / Max DD |
| Expectancy | > 0 | > 0.5R | Avg win/loss × win rate |

### Regime Robustness

| Regime | Min Win Rate | Min Profit Factor |
|--------|--------------|-------------------|
| Trending (ADX>25) | 45% | 1.3 |
| Ranging (ADX<20) | 40% | 1.2 |
| High Vol (ATR%>4%) | 35% | 1.1 |

### Sign-off
- [ ] `src/performance_analyzer.py` report meets all minimums
- [ ] Regime breakdown shows no catastrophic regime failure
- [ ] Equity curve visual inspection: no cliff drops

---

## Gate 3: Walk-Forward Acceptance

### Pre-conditions
- [ ] Gate 2 passed
- [ ] `WalkForwardEngine` completes all windows

### Criteria

| Criterion | Threshold | Measurement |
|-----------|-----------|-------------|
| Windows Tested | ≥ 10 | `WalkForwardResult.total_windows` |
| Profitable Windows | ≥ 60% | Windows with net_profit > 0 |
| OOS Net Profit | > 0 | Aggregate test window profit |
| OOS Sharpe | > 0.5 | Annualized on test windows |
| Parameter Stability | CV < 30% | Metric std/mean across windows |
| No Look-Ahead | PurgedKFold enforced | `ml_engine` CV config |

### ML-Specific (if ML enabled)

| Criterion | Threshold |
|-----------|-----------|
| CV Balanced Accuracy | > 0.52 |
| CV F1 (macro) | > 0.30 |
| Fold Stability | Balanced Acc CV < 20% |
| Feature Stability | Top-10 SHAP overlap > 80% |

### Sign-off
- [ ] `WalkForwardEngine` report meets all thresholds
- [ ] Parameter stability report generated
- [ ] ML CV metrics logged per fold

---

## Gate 4: Paper Trading Acceptance (7+ Days)

### Pre-conditions
- [ ] Gate 3 passed
- [ ] `PaperTradingRunner` configured with risk controls
- [ ] Exchange connectivity verified (testnet)

### Criteria

| Criterion | Threshold | Measurement |
|-----------|-----------|-------------|
| Runtime | ≥ 7 calendar days | Continuous operation |
| API Latency | < 500ms avg | `ExecutionEngine` metrics |
| Fill Rate | 100% | Simulated fills vs signals |
| Reconciliation Drift | 0 | Positions/balance match backtest logic |
| Risk Violations | 0 | `RiskOrchestrator` 0 blocks |
| Daily PnL Variance | Within 2σ of backtest | Statistical test |
| Kill Switch | Tested functional | Manual trigger test |

### ML Quality Gate (if enabled)

| Criterion | Threshold |
|-----------|-----------|
| Model Load | Success every session |
| Prediction Latency | < 5ms |
| Quality Gate | Passes configured thresholds |
| Retrain Trigger | Works if configured |

### Sign-off
- [ ] 7-day paper trading log reviewed
- [ ] No unexplained discrepancies vs backtest
- [ ] Risk controls verified in live conditions
- [ ] ML quality gate functional (if enabled)

---

## Gate 5: Long-Run Acceptance (30+ Days Paper)

### Pre-conditions
- [ ] Gate 4 passed
- [ ] Continuous 30-day paper trading

### Thresholds

| Metric | Minimum | Target |
|--------|---------|--------|
| Total Return | > 0 | > 5% |
| Max Drawdown | < 10% | < 8% |
| Win Rate | 45-55% | 48-52% (with 2:1 RR) |
| Profit Factor | > 1.3 | > 1.5 |
| Sharpe (ann.) | > 1.0 | > 1.5 |
| Calmar | > 1.5 | > 2.0 |
| Regime Coverage | All 4 tested | Bull/Bear/Sideways/Volatile |
| OOS ≈ IS | Performance gap < 20% | Backtest vs Paper |

### Catastrophic Failure Triggers (Auto-Stop)

| Trigger | Action |
|---------|--------|
| DD > 12% | Immediate stop, investigation |
| 7-day Sharpe < 0 | Stop, model review |
| Risk violation | Stop, immediate fix |
| API errors > 1% | Stop, infra review |

### Sign-off
- [ ] 30-day paper trading complete
- [ ] All thresholds met
- [ ] No catastrophic regime failure
- [ ] Performance consistent with backtest
- [ ] Ready for DRY_RUN

---

## Acceptance Record Template

```markdown
## Gate X Acceptance Record

**Gate**: [1-5]
**Date**: YYYY-MM-DD
**Evaluator**: @username
**Dataset**: [description]
**Commit**: git sha

### Results
| Criterion | Threshold | Actual | Pass/Fail |
|-----------|-----------|--------|-----------|
| ... | ... | ... | ✓/✗ |

### Notes
- Any deviations explained
- Mitigation for failures

### Decision
- [ ] PASS — Proceed to next gate
- [ ] CONDITIONAL PASS — Fix [items] before next gate
- [ ] FAIL — Return to development

### Sign-off
**Quant Researcher**: @username
**ML Engineer**: @username  
**Risk Manager**: @username
**Senior Developer**: @username
```