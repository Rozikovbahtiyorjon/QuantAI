# QuantAI Module Requirements Template

**Use this template for every new module or significant change.**

---

## 1. Module Specification

| Field | Value |
|-------|-------|
| **Module Name** | `src/<module_name>.py` |
| **Level** | 1 (Core) / 2 (Validation) / 3 (Execution) / 4 (Governance) |
| **Owner** | @username |
| **Created** | YYYY-MM-DD |
| **Status** | WRITTEN / UNIT_TESTED / INTEGRATED / E2E_VALIDATED / LONG_RUN_VALIDATED / PRODUCTION_APPROVED |

---

## 2. Purpose & Scope

### Problem Statement
> What problem does this module solve? Why does it need to exist?

### Use Case
> Which production path consumes this?
> - Backtest → `BacktestEngine`
> - Paper Trading → `PaperTradingRunner`
> - Live → `ExecutionEngine`
> - Research → `StrategyResearchLab`
> - Champion → `GovernanceEngine`

### Success Criteria
> How do we know this module works correctly in production?

---

## 3. Interface Contracts

### Inputs
| Parameter | Type | Required | Validation | Description |
|-----------|------|----------|------------|-------------|
| `param1` | `type` | Yes/No | `validator_fn` | Description |

### Outputs
| Return | Type | Description |
|--------|------|-------------|
| `result` | `Type` | Description |

### Side Effects
- [ ] None (pure function)
- [ ] State mutation (describe)
- [ ] I/O (files, network, DB)
- [ ] External calls (exchange, API)

---

## 4. Dependencies

### Internal (QuantAI)
| Module | Version | Purpose |
|--------|---------|---------|
| `src.indicators` | — | EMA/RSI/ATR |
| `src.risk` | — | Position sizing |

### External
| Package | Version | Purpose |
|---------|---------|---------|
| `pandas` | ≥2.0 | DataFrames |
| `xgboost` | ≥2.0 | ML |

---

## 5. Configuration

| Setting | Source | Default | Description |
|---------|--------|---------|-------------|
| `param` | `settings.module.param` | `value` | Description |

---

## 6. Testing Requirements

### Unit Tests (`tests/test_<module>.py`)
- [ ] Happy path
- [ ] Edge cases (empty, NaN, inf, boundary)
- [ ] Error modes (invalid input, missing deps)
- [ ] Determinism (same input → same output)
- [ ] Performance (latency budget)

### Integration Tests
- [ ] Backtest runs with module
- [ ] Walk-forward completes
- [ ] Paper trading simulation

### Contract Tests
- [ ] Input schema validation
- [ ] Output schema validation
- [ ] Error response format

---

## 7. Risk & Safety

| Risk | Mitigation |
|------|------------|
| Financial loss | Position limits, kill switch |
| Data leakage | PurgedKFold, no future data |
| Overfitting | Walk-forward, OOS validation |
| Model drift | Champion stability monitor |
| Execution failure | Reconciliation, emergency stop |

---

## 8. Observability

| Metric | Type | Labels |
|--------|------|--------|
| `module_latency_seconds` | Histogram | `operation`, `status` |
| `module_errors_total` | Counter | `error_type` |
| `module_output_distribution` | Histogram | `output_field` |

---

## 9. Documentation

- [ ] Docstrings on all public functions
- [ ] Architecture doc updated (`docs/architecture/`)
- [ ] Module status updated (`docs/architecture/MODULE_STATUS.md`)
- [ ] ADR created if architectural decision (`docs/decisions/ADR-*.md`)

---

## 10. Checklist for Status Promotion

### → UNIT_TESTED
- [ ] All unit tests pass
- [ ] Coverage ≥ 80%
- [ ] `py_compile` clean

### → INTEGRATED
- [ ] Imported by consumer without error
- [ ] No circular imports
- [ ] Config schema validated

### → E2E_VALIDATED
- [ ] Gate 1: Backtest tests pass
- [ ] Gate 2: Performance metrics meet minimum
- [ ] Gate 3: Walk-forward passes

### → LONG_RUN_VALIDATED
- [ ] Gate 4: 7-day paper clean
- [ ] Gate 5: 30-day paper meets thresholds

### → PRODUCTION_APPROVED
- [ ] All above
- [ ] Security audit
- [ ] Ops runbook written
- [ ] Rollback tested
- [ ] On-call rotation defined

---

## 11. Architecture Decision Record (if applicable)

**ADR Reference**: `docs/decisions/ADR-XXXX.md`

**Decision**: Brief description of what was decided.

**Rationale**: Why this approach over alternatives.

**Consequences**: Trade-offs accepted.

---

**Template Version**: 1.0  
**Last Updated**: 2026-08-25