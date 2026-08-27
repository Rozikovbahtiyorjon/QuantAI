# ADR-0003: AI + ML Signal Fusion v2 Rules

**Status**: Accepted  
**Date**: 2026-08-25  
**Authors**: QuantAI Team  
**Deciders**: Quant Researcher, ML Engineer, Portfolio Manager

---

## Context

QuantAI combines two signal sources:
- **AI (Confidence Engine)**: Rule-based technical analysis (trend, momentum, volume, volatility)
- **ML (XGBoost)**: Learned patterns from historical data

Previous fusion was ad-hoc. Need explicit, auditable rules for combining signals.

## Decision

**Adopt explicit fusion rules (v2) with configurable weights and conflict handling.**

### Fusion Rules

| AI Signal | ML Signal | Result | Confidence | Rationale |
|-----------|-----------|--------|------------|-----------|
| HOLD | HOLD | HOLD | AI_conf | No signal from either |
| HOLD | BUY/SELL | HOLD | AI_conf | **AI HOLD blocks all** — conservative |
| BUY/SELL | HOLD | HOLD | AI_conf | **ML HOLD blocks AI** — ML uncertainty veto |
| BUY/SELL | Same | Signal | `0.6*AI + 0.4*ML` | Agreement → weighted confirmation |
| BUY/SELL | Opposite | HOLD | `0.7*AI` | Conflict → penalize AI, no trade |

### Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `MIN_CONFIDENCE` | 60% | Minimum combined confidence to trade |
| `AI_WEIGHT` | 0.60 | AI weight in agreement |
| `ML_WEIGHT` | 0.40 | ML weight in agreement |
| `CONFLICT_PENALTY` | 0.70 | AI multiplier on conflict |

### Implementation

```python
def fuse_ai_ml(ai_signal, ai_confidence, ml_signal, ml_probability):
    ai_signal = normalize(ai_signal)
    ml_signal = normalize(ml_signal)
    ai_conf = clamp(ai_confidence)  # [0, 100]
    ml_prob = clamp(ml_probability) # [0, 100]

    if ai_signal == "HOLD" and ml_signal == "HOLD":
        return ("HOLD", ai_conf, False, "AI HOLD + ML HOLD")

    if ai_signal == "HOLD":
        return ("HOLD", ai_conf, False, f"AI HOLD blocks ML {ml_signal}")

    if ml_signal == ai_signal:
        combined = ai_conf * AI_WEIGHT + ml_prob * ML_WEIGHT
        approved = combined >= MIN_CONFIDENCE
        return (ai_signal, combined, approved, f"ML confirms {ml_signal}")

    if ml_signal == "HOLD":
        return ("HOLD", ai_conf, False, f"ML HOLD blocks AI {ai_signal}")

    # Conflict
    penalized = ai_conf * CONFLICT_PENALTY
    return ("HOLD", penalized, False, f"ML disagreement: AI={ai_signal}, ML={ml_signal}")
```

## Consequences

### Positive
- **Explicit veto logic**: HOLD from either source blocks trade
- **Conservative by design**: False positives (bad trades) cost more than false negatives
- **Auditable**: Every decision has clear rule trace
- **Tunable**: Weights/thresholds in config, not hardcoded
- **ML uncertainty respected**: ML HOLD = "I don't know" → safe to block

### Negative
- **May miss trades**: Double HOLD veto reduces trade frequency
- **ML dependency**: If ML model degrades (always HOLD), AI signals blocked
- **Weight sensitivity**: 60/40 split needs validation

## Alternatives Considered

| Alternative | Rejected Because |
|-------------|------------------|
| ML overrides AI | ML less interpretable, higher overfitting risk |
| Simple majority vote | Ignores confidence magnitudes |
| No ML gate (AI only) | Loses learned pattern advantage |
| Probabilistic fusion (Bayesian) | Overkill, hard to debug |

## Validation

### Metrics to Track
- Trade frequency (should be reasonable, not zero)
- Win rate with fusion vs AI-only vs ML-only
- False positive reduction (conflict veto effectiveness)
- ML HOLD rate (should be < 50% ideally)

### A/B Test Plan
- Variant A: Current fusion rules
- Variant B: Relaxed (ML HOLD doesn't block, only conflicts)
- Measure: Net profit, Sharpe, trade count over 30-day paper

## Configuration

All parameters in `config/settings.yaml`:
```yaml
strategy:
  min_confidence: 60.0
  ai_weight: 0.60
  ml_weight: 0.40
  conflict_penalty: 0.70
```

## Order Flow Gate (Post-Fusion)

After AI+ML fusion, Order Flow acts as final gate:

```python
# If Strategy says BUY but OrderFlow shows strong ASK pressure → HOLD
# If Strategy says SELL but OrderFlow shows strong BID pressure → HOLD
# BALANCED OrderFlow = neutral (pressure=0, score=0.5)
```

Threshold: `ORDER_FLOW_CONFLICT_THRESHOLD = 0.15`

## References

- `src/strategy.py` — `fuse_ai_ml()`, `apply_order_flow_gate()`
- `src/confidence_engine.py` — AI signal generation
- `src/ml_engine.py` — ML prediction
- `src/order_flow_intelligence.py` — OrderFlowSignal

---

**Related**: ADR-0001 (ML CV), ADR-0002 (Risk Orchestrator)