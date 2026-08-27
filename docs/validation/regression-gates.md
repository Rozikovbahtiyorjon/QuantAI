# QuantAI Regression Gates

**Version**: 1.0  
**Purpose**: Automated CI/CD gates preventing performance/functional regressions

---

## CI Pipeline Structure

```yaml
# .github/workflows/regression.yml
name: Regression Gates

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  gate-0-syntax:
    name: Syntax & Imports
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: '3.11'}
      - run: pip install -e .
      - run: python -m py_compile $(find src -name "*.py")
      - run: python -c "import src; print('Imports OK')"

  gate-1-unit:
    name: Unit Tests
    runs-on: ubuntu-latest
    timeout-minutes: 15
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: '3.11'}
      - run: pip install -e ".[test]"
      - run: pytest tests/test_*_math.py tests/test_position_sizer.py tests/test_risk_*.py -v --tb=short

  gate-2-integration:
    name: Integration Tests
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: '3.11'}
      - run: pip install -e ".[test]"
      - run: pytest tests/test_backtest_*.py tests/test_trade_engine_*.py tests/test_strategy*.py -v --tb=short

  gate-3-contracts:
    name: Contract Tests
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: '3.11'}
      - run: pip install -e ".[test]"
      - run: pytest tests/test_contracts.py -v --tb=short

  gate-4-performance:
    name: Performance Regression
    runs-on: ubuntu-latest
    timeout-minutes: 60
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: '3.11'}
      - run: pip install -e ".[test]"
      - run: python -m src.performance_analyzer --data data/prepared.csv --gate
      - run: python -m src.walk_forward_runner --data data/prepared.csv --gate

  gate-5-ml:
    name: ML Regression
    runs-on: ubuntu-latest
    timeout-minutes: 120
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: '3.11'}
      - run: pip install -e ".[test]"
      - run: pytest tests/test_ml_engine.py -v --tb=short
      - run: python -c "
          from src.ml_engine import MLEngine, MLConfig
          import pandas as pd
          df = pd.read_parquet('data/dataset.parquet')
          engine = MLEngine(MLConfig(n_splits=3))
          result = engine.train(df)
          assert result.balanced_accuracy >= 0.52, f'BA dropped: {result.balanced_accuracy}'
          assert result.f1 >= 0.30, f'F1 dropped: {result.f1}'
        "

  gate-6-drift:
    name: Data/Config Drift
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: '3.11'}
      - run: |
          # Check no hardcoded magic numbers in strategy
          grep -r "70\|60\|55\|45\|40\|30\|1.5\|3.0\|2.0" src/strategy.py || true
          # Check all config from settings
          python -c "
            from config.settings import settings
            assert settings.indicators.rsi_overbought == 70
            assert settings.indicators.rsi_oversold == 30
            assert settings.risk.atr_stop_multiplier == 1.5
            assert settings.risk.atr_take_multiplier == 3.0
          "
```

---

## Gate Thresholds (Automated)

### Functional Gates (Must Pass)

| Gate | Test Suite | Pass Criteria |
|------|------------|---------------|
| **Syntax** | `py_compile` + imports | 0 errors |
| **Unit Math** | `test_*_math.py`, `test_position_sizer.py`, `test_risk_*.py` | 100% pass |
| **Integration** | `test_backtest_*.py`, `test_trade_engine_*.py`, `test_strategy*.py` | 100% pass |
| **Contracts** | `test_contracts.py` | 100% pass |

### Performance Gates (Warning on Regression)

| Metric | Baseline | Regression Threshold | Action |
|--------|----------|---------------------|--------|
| Backtest Net Profit | $X | < 0.9 * baseline | Warning |
| Backtest Max DD | Y% | > 1.1 * baseline | Warning |
| Backtest Sharpe | Z | < 0.9 * baseline | Warning |
| Walk-Forward OOS Profit | > 0 | ≤ 0 | **Fail** |
| Walk-Forward Sharpe | > 0.5 | < 0.4 | **Fail** |
| ML Balanced Accuracy | ≥ 0.52 | < 0.50 | **Fail** |
| ML F1 (macro) | ≥ 0.30 | < 0.25 | **Fail** |

### ML-Specific Gates

```python
# Automated in gate-5-ml
MIN_BALANCED_ACCURACY = 0.52
MIN_F1_MACRO = 0.30
MIN_PRECISION_MACRO = 0.25
MIN_RECALL_MACRO = 0.25
MAX_FOLD_CV = 0.30  # Coefficient of variation across folds
```

---

## Pre-commit Hooks (Local)

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 24.3.0
    hooks:
      - id: black
        args: [--line-length=100]
  
  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
        args: [--profile=black]
  
  - repo: https://github.com/pycqa/flake8
    rev: 7.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=100, --ignore=E203,W503]
  
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.9.0
    hooks:
      - id: mypy
        args: [--ignore-missing-imports, --strict]
        additional_dependencies: [pandas-stubs, numpy-stubs]
  
  - repo: local
    hooks:
      - id: no-magic-numbers
        name: Check for magic numbers in strategy
        entry: bash -c 'grep -n "70\|60\|55\|45\|40\|30\|1.5\|3.0\|2.0\|0.15" src/strategy.py | grep -v "settings\." && exit 1 || exit 0'
        language: system
        types: [python]
```

---

## Nightly Jobs

```yaml
# .github/workflows/nightly.yml
name: Nightly Validation

on:
  schedule:
    - cron: '0 2 * * *'  # 2 AM UTC daily

jobs:
  full-backtest:
    name: Full Backtest
    runs-on: ubuntu-latest
    timeout-minutes: 120
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: '3.11'}
      - run: pip install -e ".[test]"
      - run: python -m src.backtest_runner --data data/prepared.csv --output artifacts/backtest.json
      - uses: actions/upload-artifact@v4
        with:
          name: backtest-report
          path: artifacts/backtest.json

  walk-forward:
    name: Walk-Forward
    runs-on: ubuntu-latest
    timeout-minutes: 180
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: '3.11'}
      - run: pip install -e ".[test]"
      - run: python -m src.walk_forward_runner --data data/prepared.csv --output artifacts/wf.json
      - uses: actions/upload-artifact@v4
        with:
          name: walkforward-report
          path: artifacts/wf.json

  ml-training:
    name: ML Retraining
    runs-on: ubuntu-latest
    timeout-minutes: 240
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: '3.11'}
      - run: pip install -e ".[test]"
      - run: python -m src.ml_walk_forward --data data/prepared.csv --output artifacts/ml.json
      - uses: actions/upload-artifact@v4
        with:
          name: ml-report
          path: artifacts/ml.json

  compare:
    name: Regression Comparison
    runs-on: ubuntu-latest
    needs: [full-backtest, walk-forward, ml-training]
    steps:
      - uses: actions/download-artifact@v4
      - run: python scripts/compare_regression.py
        env:
          BASELINE_BRANCH: main
```

---

## Comparison Script (`scripts/compare_regression.py`)

```python
#!/usr/bin/env python3
"""Compare current metrics against baseline (main branch)."""

import json
import sys
from pathlib import Path

def load_metrics(path):
    with open(path) as f:
        return json.load(f)

def check_regression(current, baseline, metric, threshold=0.95, higher_is_better=True):
    """Check if metric regressed beyond threshold."""
    if higher_is_better:
        ratio = current / baseline if baseline != 0 else float('inf')
        return ratio >= threshold, ratio
    else:
        ratio = baseline / current if current != 0 else float('inf')
        return ratio >= threshold, ratio

def main():
    artifacts = Path("artifacts")
    
    # Load current
    bt_current = load_metrics(artifacts / "backtest.json")
    wf_current = load_metrics(artifacts / "wf.json")
    ml_current = load_metrics(artifacts / "ml.json")
    
    # Load baseline (from main branch artifact storage)
    # In practice: download from GitHub Actions artifacts of main branch
    # For now, use stored baselines
    baselines = {
        "backtest": {
            "net_profit": 150.0,
            "max_drawdown_pct": 8.5,
            "sharpe": 1.2,
            "win_rate": 0.52,
        },
        "walkforward": {
            "oos_profit": 80.0,
            "oos_sharpe": 0.8,
            "profitable_windows_pct": 0.65,
        },
        "ml": {
            "balanced_accuracy": 0.55,
            "f1_macro": 0.35,
        }
    }
    
    checks = [
        # Backtest
        ("Backtest Net Profit", bt_current["net_profit"], baselines["backtest"]["net_profit"], 0.90, True),
        ("Backtest Max DD", bt_current["max_drawdown_pct"], baselines["backtest"]["max_drawdown_pct"], 0.90, False),
        ("Backtest Sharpe", bt_current["sharpe"], baselines["backtest"]["sharpe"], 0.90, True),
        ("Backtest Win Rate", bt_current["win_rate"], baselines["backtest"]["win_rate"], 0.95, True),
        
        # Walk-Forward
        ("WF OOS Profit", wf_current["net_profit"], baselines["walkforward"]["oos_profit"], 1.0, True),
        ("WF OOS Sharpe", wf_current["sharpe"], baselines["walkforward"]["oos_sharpe"], 0.85, True),
        ("WF Profitable Windows", wf_current["profitable_windows_pct"], baselines["walkforward"]["profitable_windows_pct"], 0.90, True),
        
        # ML
        ("ML Balanced Acc", ml_current["balanced_accuracy"], baselines["ml"]["balanced_accuracy"], 0.95, True),
        ("ML F1 Macro", ml_current["f1"], baselines["ml"]["f1_macro"], 0.90, True),
    ]
    
    failed = []
    warned = []
    
    for name, current, baseline, threshold, higher_better in checks:
        passed, ratio = check_regression(current, baseline, threshold, higher_better)
        status = "PASS" if passed else ("WARN" if ratio > 0.8 else "FAIL")
        
        print(f"{name}: current={current:.4f}, baseline={baseline:.4f}, ratio={ratio:.3f} [{status}]")
        
        if not passed:
            if ratio > 0.8:
                warned.append(name)
            else:
                failed.append(name)
    
    print(f"\nSummary: {len(checks) - len(warned) - len(failed)} passed, {len(warned)} warned, {len(failed)} failed")
    
    if failed:
        print("❌ REGRESSION DETECTED - Blocking merge")
        sys.exit(1)
    elif warned:
        print("⚠️ PERFORMANCE WARNING - Review recommended")
        sys.exit(0)
    else:
        print("✅ ALL GATES PASSED")
        sys.exit(0)

if __name__ == "__main__":
    main()
```

---

## Local Development Commands

```bash
# Quick syntax check
python -m py_compile $(find src -name "*.py")

# Unit tests only (fast)
pytest tests/test_*_math.py tests/test_position_sizer.py tests/test_risk_*.py -v

# Integration tests (medium)
pytest tests/test_backtest_*.py tests/test_trade_engine_*.py tests/test_strategy*.py -v

# Full regression (slow)
pytest tests/ -v --tb=short

# Performance gate (manual)
python -m src.performance_analyzer --data data/prepared.csv --gate

# ML gate (manual)
python -c "
from src.ml_engine import MLEngine, MLConfig
import pandas as pd
df = pd.read_parquet('data/dataset.parquet')
engine = MLEngine(MLConfig(n_splits=5))
result = engine.train(df)
print(f'BA: {result.balanced_accuracy:.4f}, F1: {result.f1:.4f}')
assert result.balanced_accuracy >= 0.52
assert result.f1 >= 0.30
"
```

---

## Release Criteria

Before tagging a release:

- [ ] All CI gates green on `main`
- [ ] Nightly jobs green for 7 consecutive days
- [ ] Gate 4 (7-day paper) passed
- [ ] Gate 5 (30-day paper) passed for major releases
- [ ] Security audit clean
- [ ] Documentation updated
- [ ] CHANGELOG.md updated