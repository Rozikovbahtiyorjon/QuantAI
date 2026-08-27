# QuantAI Long-Run Validation Criteria

**Version**: 1.0  
**Purpose**: 30-day paper trading acceptance criteria for production promotion

---

## Overview

Long-run validation is the **final gate** before live deployment. It validates:
1. Strategy robustness across market regimes
2. Operational reliability (API, risk, execution)
3. Performance consistency vs backtest
4. No hidden failure modes

---

## Duration & Scope

| Parameter | Value |
|-----------|-------|
| **Minimum Duration** | 30 calendar days |
| **Target Duration** | 60 calendar days (major releases) |
| **Capital** | Paper (simulated) |
| **Exchange** | Binance Testnet (or live with $0.01 via dust) |
| **Symbols** | BTC/USDT (primary), ETH/USDT (secondary) |
| **Timeframe** | 15m (primary) |
| **Risk Config** | Production settings (1% risk, 10% DD, 60% exposure) |

---

## Regime Coverage Requirements

The 30-day window **must** include meaningful exposure to:

| Regime | Definition | Minimum Days | Validation |
|--------|------------|--------------|------------|
| **Bull Trend** | ADX > 25, EMA alignment bullish | 7 | Profitable, controlled DD |
| **Bear Trend** | ADX > 25, EMA alignment bearish | 7 | Short profits or flat |
| **Sideways** | ADX < 20, BB width narrow | 7 | Low trade freq, no chop losses |
| **High Volatility** | ATR% > 4% (90th percentile) | 3 | SL not hunted, size reduces |

**Failure**: Any regime missing → extend paper trading until covered.

---

## Performance Thresholds

### Core Metrics (All Must Pass)

| Metric | 30-Day Minimum | 60-Day Target | Measurement |
|--------|----------------|---------------|-------------|
| **Total Return** | > 0% | > 5% | Net PnL / Starting equity |
| **Max Drawdown** | < 10% | < 8% | Peak-to-trough equity |
| **Win Rate** | 45-55% | 48-52% | Winning trades / Total |
| **Profit Factor** | > 1.3 | > 1.5 | Gross profit / Gross loss |
| **Sharpe (ann.)** | > 1.0 | > 1.5 | Return / Vol × √252 |
| **Calmar** | > 1.5 | > 2.0 | Return / Max DD |
| **Recovery Factor** | > 2.0 | > 3.0 | Net profit / Max DD |
| **Expectancy** | > 0.5R | > 0.8R | Avg PnL per trade in R |

### Trade Quality

| Metric | Threshold |
|--------|-----------|
| **Avg Win / Avg Loss** | ≥ 1.8 (target 2.0) |
| **Max Consecutive Losses** | ≤ 5 |
| **Max Consecutive Wins** | No limit |
| **Avg Trade Duration** | 4-48 hours (15m TF) |
| **Time in Market** | 20-60% |

---

## Operational Requirements

### Zero-Tolerance Failures

| Failure | Action |
|---------|---------|
| **Risk violation** (DD > 10%, exposure > 60%) | Immediate stop, root cause |
| **Kill switch activation** | Immediate stop, infra review |
| **Reconciliation drift** > tolerance | Immediate stop, fix |
| **API error rate** > 1% | Infra investigation |
| **Order fill failure** > 0 | Execution path review |
| **ML quality gate fail** | Model retrain or disable |

### Monitoring Requirements

| Check | Frequency | Alert Threshold |
|-------|-----------|-----------------|
| **Equity/DD** | Every candle | DD > 8% |
| **Open positions** | Every candle | > max_open |
| **Exposure** | Every candle | > 50% |
| **API latency** | Every request | > 500ms |
| **WS connection** | Continuous | Any disconnect |
| **Model predictions** | Every signal | Quality gate fail |
| **Balance** | Every fill | Drift > $1 |

---

## Statistical Validation

### Bootstrap Confidence Intervals (10,000 resamples)

| Metric | 95% CI Must Contain |
|--------|---------------------|
| **Sharpe** | > 0.5 |
| **Calmar** | > 1.0 |
| **Win Rate** | 40-60% |
| **Profit Factor** | > 1.0 |

### Monte Carlo Stress Test (1,000 paths)

| Scenario | Pass Criteria |
|----------|---------------|
| **Parameter noise** (±10% on all thresholds) | 90% paths profitable |
| **Slippage 2x** | 85% paths profitable |
| **Commission 2x** | 85% paths profitable |
| **Random trade drop** (10%) | 90% paths profitable |
| **Volatility regime shift** | 80% paths profitable |

---

## Comparison vs Backtest

| Metric | Backtest | Paper | Max Gap |
|--------|----------|-------|---------|
| Net Profit | X% | Y% | < 20% |
| Max DD | A% | B% | < 3% absolute |
| Win Rate | C% | D% | < 5% |
| Profit Factor | E | F | < 0.3 |
| Sharpe | G | H | < 0.3 |
| Trade Count | N | M | < 30% |

**Failure**: Any gap > threshold → investigate, extend paper trading.

---

## Regime Attribution

Must provide breakdown:

| Regime | Days | Trades | Net PnL | Win Rate | Max DD | Notes |
|--------|------|--------|---------|----------|--------|-------|
| Bull Trend |  |  |  |  |  |  |
| Bear Trend |  |  |  |  |  |  |
| Sideways |  |  |  |  |  |  |
| High Vol |  |  |  |  |  |  |

**Requirement**: No single regime with catastrophic loss (>50% of max DD).

---

## ML Model Validation (if enabled)

| Criterion | Threshold |
|-----------|-----------|
| **Model uptime** | 100% |
| **Prediction latency** | < 5ms (p99) |
| **Quality gate passes** | 100% |
| **Retrain triggers** | Works if configured |
| **Feature drift** | PSI < 0.1 on top-20 features |
| **Prediction distribution** | Stable (KS test p > 0.05) |

---

## Acceptance Report Template

```markdown
# Long-Run Validation Report

**Period**: YYYY-MM-DD to YYYY-MM-DD (N days)
**Symbols**: BTC/USDT, ETH/USDT
**Timeframe**: 15m
**Config**: Production (risk=1%, DD=10%, exposure=60%)
**Commit**: git sha

## Regime Coverage
| Regime | Days | Coverage |
|--------|------|----------|
| Bull Trend | N | ✓/✗ |
| Bear Trend | N | ✓/✗ |
| Sideways | N | ✓/✗ |
| High Vol | N | ✓/✗ |

## Performance
| Metric | Threshold | Actual | Pass |
|--------|-----------|--------|------|
| Total Return | > 0% | X% | ✓/✗ |
| Max DD | < 10% | Y% | ✓/✗ |
| Win Rate | 45-55% | Z% | ✓/✗ |
| Profit Factor | > 1.3 | W | ✓/✗ |
| Sharpe | > 1.0 | V | ✓/✗ |
| Calmar | > 1.5 | U | ✓/✗ |

## Trade Quality
| Metric | Threshold | Actual |
|--------|-----------|--------|
| Avg Win/Loss | ≥ 1.8 | X |
| Max Consec Losses | ≤ 5 | Y |
| Avg Duration | 4-48h | Z |

## Operational
- [ ] Zero risk violations
- [ ] Zero kill switch activations
- [ ] Zero reconciliation drift
- [ ] API latency < 500ms avg
- [ ] ML quality gate 100% pass (if enabled)

## Statistical Validation
- [ ] Bootstrap CI: Sharpe > 0.5, Calmar > 1.0
- [ ] Monte Carlo: 90% paths profitable under noise
- [ ] Backtest gaps < 20% on all metrics

## Regime Attribution
| Regime | Days | Trades | PnL | Win% | Max DD |
|--------|------|--------|-----|------|--------|
| Bull |  |  |  |  |  |
| Bear |  |  |  |  |  |
| Sideways |  |  |  |  |  |
| High Vol |  |  |  |  |  |

## Backtest Comparison
| Metric | Backtest | Paper | Gap | Pass |
|--------|----------|-------|-----|------|
| Return |  |  |  | ✓/✗ |
| DD |  |  |  | ✓/✗ |
| Win Rate |  |  |  | ✓/✗ |

## Decision
- [ ] **PASS** — Promote to DRY_RUN / Live
- [ ] **EXTEND** — Continue paper for N more days (reason: )
- [ ] **FAIL** — Return to development (reason: )

**Sign-off**:
- Quant Researcher: @username
- ML Engineer: @username
- Risk Manager: @username
- Senior Developer: @username
**Date**: YYYY-MM-DD
```

---

## Escalation Path

| Situation | Escalation |
|-----------|------------|
| Single metric failure | Quant Researcher → Root cause → Fix or extend |
| Multiple failures | Team review → Architecture decision |
| Catastrophic (DD > 12%) | Immediate stop → Incident response → Post-mortem |
| ML model degradation | ML Engineer → Retrain or disable ML gate |

---

**Status**: Criteria defined. Ready for Gate 5 execution.