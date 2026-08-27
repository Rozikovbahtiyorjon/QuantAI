# ADR-0001: PurgedKFold Cross-Validation for ML Training

**Status**: Accepted  
**Date**: 2026-08-25  
**Authors**: QuantAI Team  
**Deciders**: Quant Researcher, ML Engineer, Risk Manager

---

## Context

QuantAI uses XGBoost for 3-class classification (SELL/HOLD/BUY) on cryptocurrency OHLCV data. Standard k-fold CV introduces **look-ahead bias** because:

1. Financial time series have autocorrelation
2. Random splits leak future information into training
3. Overfitting to temporal patterns that don't persist

Previous implementation used only the **first fold** of PurgedKFold, discarding 80% of data and producing optimistically biased metrics.

## Decision

**Use full PurgedKFold cross-validation with embargo for all ML training.**

### Implementation

```python
# In MLEngine.train():
cv = get_purged_cv(
    cv_type="purged",
    n_splits=5,
    embargo_pct=0.01,    # 1% gap between train/test
    purge_pct=0.0        # No label overlap removal (conservative)
)

# Train on each fold, aggregate predictions
all_predictions = []
all_y_true = []
for train_idx, test_idx in cv.split(X_full, y_full):
    fold_model = self._create_model()
    fold_model.fit(X_full.iloc[train_idx], y_full.iloc[train_idx])
    pred = fold_model.predict(X_full.iloc[test_idx])
    all_predictions.extend(pred)
    all_y_true.extend(y_full.iloc[test_idx])

# Final model: train on FULL dataset
self.model = self._create_model()
self.model.fit(X_full, y_full)
```

### Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `n_splits` | 5 | Standard, balances bias/variance |
| `embargo_pct` | 0.01 (1%) | ~10 bars gap on 1000-bar dataset |
| `purge_pct` | 0.0 | Conservative: keep all labels |

## Consequences

### Positive
- **No look-ahead bias**: Test data strictly after train + embargo
- **Statistically valid**: Uses all data, proper CV metrics
- **Realistic performance**: Metrics reflect true OOS expectation
- **Parameter stability**: Can measure metric variance across folds

### Negative
- **Slower training**: 5x model fits + final fit
- **More complex**: Requires proper aggregation logic
- **Embargo reduces effective data**: ~5% loss per fold

## Alternatives Considered

| Alternative | Rejected Because |
|-------------|------------------|
| Standard KFold | Look-ahead bias guaranteed |
| TimeSeriesSplit (no embargo) | Adjacent train/test leak via autocorrelation |
| Walk-forward only (no CV) | No hyperparameter tuning, single path |
| Combinatorial PurgedKFold | Overkill for current data size |

## Validation

- Compare CV metrics vs single-split metrics (expect 10-20% degradation)
- Verify metric stability across folds (CV < 30%)
- Ensure final model on full data outperforms fold models

## References

- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Ch. 7.
- `src/validation/purged_kfold.py` implementation

---

**Related**: ADR-0002 (Risk Orchestrator), ADR-0003 (Signal Fusion)