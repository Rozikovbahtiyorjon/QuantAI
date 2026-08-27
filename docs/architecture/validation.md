# QuantAI Validation Architecture

## Level 2 — Validation Pipeline

```
Backtest
    ↓
Performance Analyzer
    ↓
Walk-Forward Validation
    ↓
Paper Trading
    ↓
Long-Run Validation (30+ days)
```

## Validation Gates

Each stage must pass **all** gates before promotion.

---

### Gate 1: Backtest Validation (`tests/test_backtest_*.py`)

| Criterion | Threshold | Test |
|-----------|-----------|------|
| Determinism | Same seed → identical results | `test_backtest_engine_determinism.py` |
| Data Validation | Rejects invalid input | `test_backtest_engine.py::test_validate_data_*` |
| Trade Math | PnL = gross - commission | `test_trade_engine_math.py` |
| Risk Math | Position size = risk/stop_dist | `test_trade_engine_risk.py` |
| SL/TP Logic | Conservative (SL first) | `test_trade_engine_behavior.py` |
| Edge Cases | Empty, NaN, gaps handled | `test_backtest_engine_edge_cases.py` |

**Pass Condition**: All pytest tests green, no flaky tests.

---

### Gate 2: Performance Analysis (`src/performance_analyzer.py`)

| Metric | Minimum | Target |
|--------|---------|--------|
| Net Profit | > 0 | > 10% |
| Win Rate | > 40% | > 50% |
| Profit Factor | > 1.2 | > 1.5 |
| Max Drawdown | < 15% | < 10% |
| Sharpe (annualized) | > 0.5 | > 1.0 |
| Calmar | > 0.5 | > 1.5 |
| Recovery Factor | > 1.0 | > 2.0 |
| Expectancy | > 0 | > 0.5R |

**Pass Condition**: All minimums met on **full dataset** (not cherry-picked).

---

### Gate 3: Walk-Forward Validation (`src/walk_forward_engine.py`)

| Criterion | Threshold |
|-----------|-----------|
| Windows tested | ≥ 10 |
| Profitable windows | ≥ 60% |
| Out-of-sample profit | > 0 |
| OOS Sharpe | > 0.5 |
| Parameter stability | CV < 30% across windows |
| No look-ahead bias | PurgedKFold enforced |

**Pass Condition**: Aggregate OOS metrics meet minimums, parameter stability OK.

---

### Gate 4: Paper Trading (`src/paper_trading_runner.py`, `src/paper_trading_session.py`)

| Criterion | Threshold |
|-----------|-----------|
| Runtime | ≥ 7 days continuous |
| API latency | < 500ms avg |
| Order fill rate | 100% (simulated) |
| Reconciliation | 0 drift vs backtest logic |
| Risk controls | 0 violations |
| Daily PnL variance | Consistent with backtest |

**Pass Condition**: 7+ days clean, metrics within 2σ of backtest.

---

### Gate 5: Long-Run Validation (30+ days paper)

| Criterion | Threshold |
|-----------|-----------|
| Runtime | ≥ 30 calendar days |
| Total return | > 0 |
| Max DD | < 10% |
| Win rate | 45-55% (with 2:1 RR) |
| Profit factor | > 1.3 |
| Sharpe | > 1.0 |
| Calmar | > 1.5 |
| No regime failure | Works in trend/flat/volatile |
| No overfitting | OOS ≈ IS performance |

**Pass Condition**: All thresholds met, no catastrophic regime failure.

---

## Regression Gates (CI)

```yaml
# .github/workflows/validation.yml (conceptual)
on: [push, pull_request]
jobs:
  gate-1-backtest:
    runs: pytest tests/test_backtest_*.py tests/test_trade_engine_*.py -v
  gate-2-performance:
    runs: python -m src.performance_analyzer --data data/prepared.csv
  gate-3-walkforward:
    runs: python -m src.walk_forward_runner --data data/prepared.csv
  gate-4-paper:
    runs: pytest tests/test_paper_trading_*.py -v
  gate-contracts:
    runs: pytest tests/test_contracts.py -v  # interface contracts
```

**All gates must pass** before merge to main.

---

## Acceptance Criteria Template

For any new module or change:

```markdown
## Module: <name>
## Change: <description>

### Pre-conditions
- [ ] Dependencies updated
- [ ] Config schema updated

### Unit Tests
- [ ] All existing tests pass
- [ ] New tests for new logic
- [ ] Edge cases covered

### Integration Tests
- [ ] Backtest runs without error
- [ ] Walk-forward completes
- [ ] Paper trading simulation passes

### Contracts
- [ ] Input schema documented
- [ ] Output schema documented
- [ ] Error modes documented

### Performance
- [ ] No regression in key metrics
- [ ] Latency within budget
- [ ] Memory stable

### Risk
- [ ] RiskOrchestrator integration verified
- [ ] Drawdown/Exposure limits tested
- [ ] Kill switch functional

### Sign-off
- [ ] Code review
- [ ] Docs updated
- [ ] Checkpoint commit
```

---

## Module Status Gates

| Status | Requirements |
|--------|--------------|
| **WRITTEN** | Code exists, `py_compile` clean |
| **UNIT_TESTED** | ≥ 80% coverage, all tests pass |
| **INTEGRATED** | Used by ≥1 consumer, no import errors |
| **E2E_VALIDATED** | Passes Gate 1-3 |
| **LONG_RUN_VALIDATED** | Passes Gate 4-5 |
| **PRODUCTION_APPROVED** | All gates + security audit + ops runbook |

---

**Status**: Validation gates defined. Next: Execution Boundary (Level 3).