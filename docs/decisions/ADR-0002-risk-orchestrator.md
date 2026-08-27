# ADR-0002: Risk Orchestrator Pattern

**Status**: Accepted  
**Date**: 2026-08-25  
**Authors**: QuantAI Team  
**Deciders**: Risk Manager, Quant Researcher, Senior Python Developer

---

## Context

QuantAI has multiple risk components developed independently:
- `DrawdownGuard` — equity peak tracking, max DD limits
- `ExposureManager` — total/position exposure caps
- `PositionSizer` — risk-based sizing with leverage
- `calculate_sl_tp()` in `risk_manager.py` — SL/TP calculation

Previously, each was called separately with duplicated logic. No unified decision interface existed.

## Decision

**Implement `RiskOrchestrator` as a single unified facade coordinating all risk components.**

### Architecture

```
SignalResult + Equity + Current Exposure
                ↓
        DrawdownGuard
                ↓
        PositionSizer
                ↓
        ExposureManager
                ↓
        RiskDecision (unified)
```

### Interface

```python
@dataclass(frozen=True)
class RiskDecision:
    allowed: bool
    quantity: float
    stop_loss: float
    take_profit: float
    reason: str
    drawdown_result: DrawdownGuardResult
    exposure_result: ExposureResult
    position_size_result: PositionSizeResult
    metadata: dict

class RiskOrchestrator:
    def evaluate(
        self,
        signal: SignalResult,
        equity: float,
        current_exposure: float = 0.0,
        risk_percent: float | None = None,
        leverage: float | None = None,
    ) -> RiskDecision:
        ...
```

### Decision Flow

1. **Drawdown Check** — Block if DD > max_drawdown_pct (10%)
2. **Position Sizing** — qty = risk_amount / stop_distance
3. **Per-Position Cap** — qty ≤ max_position_capital / entry_price
4. **Total Exposure Check** — current + new ≤ max_total_exposure_pct (60%)
5. **Allow** — Return approved quantity with diagnostics

### Configuration

```python
RiskOrchestrator(
    drawdown_guard=DrawdownGuard(max_drawdown_percent=10.0),
    exposure_manager=ExposureManager(
        max_total_exposure_percent=60.0,
        max_position_exposure_percent=5.0,
    ),
    position_sizer=PositionSizer(min_leverage=1.0, max_leverage=50.0),
    default_risk_percent=1.0,
    default_leverage=1.0,
)
```

## Consequences

### Positive
- **Single source of truth**: All risk logic in one place
- **Composable**: Each component testable independently
- **Extensible**: Easy to add new checks (correlation, volatility, etc.)
- **Auditable**: Full decision trail in `RiskDecision.metadata`
- **Configurable**: All thresholds from `settings.yaml`

### Negative
- **Additional abstraction layer**: Slight indirection
- **All-or-nothing**: Can't easily skip one check (by design)

## Alternatives Considered

| Alternative | Rejected Because |
|-------------|------------------|
| Direct calls in Strategy | Duplicated logic, inconsistent decisions |
| Risk as Strategy mixin | Violates SRP, hard to test |
| Global risk singleton | Hidden dependencies, test pollution |
| Policy-based (chain of responsibility) | Over-engineered for current needs |

## Validation

- Unit tests for each component (`test_drawdown_guard.py`, `test_exposure_manager.py`, `test_position_sizer.py`)
- Integration tests for Orchestrator (`test_risk_orchestrator.py`)
- Backtest verification: DD never exceeds 10%, exposure never exceeds 60%

## References

- `src/risk/risk_orchestrator.py`
- `src/risk/drawdown_guard.py`
- `src/risk/exposure_manager.py`
- `src/risk/position_sizer.py`
- `config/settings.py` — RiskSettings

---

**Related**: ADR-0001 (ML CV), ADR-0003 (Signal Fusion)