# QuantAI Interface Contracts

**Version**: 1.0  
**Scope**: Level 1 Core Trading Pipeline  
**Status**: Draft for Review

---

## Contract Philosophy

Every module interface is a **contract**. Changes require:
1. Updated contract doc
2. Consumer notification
3. Backward compatibility or migration plan
4. ADR if breaking

---

## 1. Market Data → Indicators

### Provider: `src/data_loader.load_binance_data()` / `src/exchange_market_data.py`
### Consumer: `src/indicators.add_indicators()`

```python
# Input Contract
DataFrame[ts, open, high, low, close, volume]
    ts: datetime64[ns, UTC], monotonic increasing, no duplicates
    open/high/low/close: float64, > 0, no NaN
    volume: float64, >= 0, no NaN

# Output Contract (same DataFrame + indicators)
DataFrame[ts, open, high, low, close, volume,
          ema_fast, ema_slow, ema_trend,
          rsi, atr, volume_ratio,
          ...extended...]
```

### Validation
```python
def validate_market_data(df: pd.DataFrame) -> None:
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    assert set(df.columns) >= required
    assert df["timestamp"].is_monotonic_increasing
    assert df[["open","high","low","close","volume"]].notna().all().all()
    assert (df[["open","high","low","close"]] > 0).all().all()
```

---

## 2. Indicators → Feature Engine

### Provider: `src/indicators.add_indicators(df, core_only=True)`
### Consumer: `src/feature_engine.build_features()`

```python
# Input: DataFrame with core indicators
DataFrame[ts, open, high, low, close, volume,
          ema_fast, ema_slow, ema_trend,
          rsi, atr, volume_ratio]

# Output: FeatureVector dict (30+ features)
{
    "ema_fast_distance": float,
    "ema_slow_distance": float,
    "ema_trend_distance": float,
    "ema_fast_slow_spread": float,
    "ema_slow_trend_spread": float,
    "atr_percent": float,
    "relative_volume": float,
    "rsi_normalized": float,      # rsi / 100
    "rsi_distance_50": float,     # (rsi - 50) / 50
    "rsi_overbought": float,      # 1.0 if rsi > 70 else 0
    "rsi_oversold": float,        # 1.0 if rsi < 30 else 0
    # Microstructure (stubs return 0)
    "vpin": 0.0,
    "vpin_toxicity": 0.0,
    "kyle_lambda": 0.0,
    "kyle_lambda_rsq": 0.0,
    "kyle_lambda_confidence": 0.0,
    "nearest_support_dist": 100.0,
    "nearest_resistance_dist": 100.0,
    "support_strength": 0.0,
    "resistance_strength": 0.0,
    # Alternative data (stubs)
    "lunar_galaxy_score": 50.0,
    ...
}
```

### Validation
```python
def validate_feature_vector(fv: dict) -> None:
    required = {"ema_fast_distance", "ema_slow_distance", "ema_trend_distance",
                "ema_fast_slow_spread", "ema_slow_trend_spread",
                "atr_percent", "relative_volume",
                "rsi_normalized", "rsi_distance_50",
                "rsi_overbought", "rsi_oversold"}
    assert set(fv.keys()) >= required
    assert all(isinstance(v, (int, float)) for v in fv.values())
```

---

## 3. Feature Engine → ML Engine

### Provider: `src/feature_engine.build_features(df)`
### Consumer: `src/ml_engine.MLEngine.predict_probabilities()`

```python
# Input: FeatureVector dict
# Output: DataFrame[features] matching training schema

# Training-time contract (MLEngine.prepare_dataset)
X_train, X_test, y_train, y_test, cv, X_full, y_full = prepare_dataset(dataset)
    X: DataFrame[feature_names] — all numeric, no NaN/inf
    y: Series[int] — values in {0,1,2} mapping to {SELL,HOLD,BUY}
    
# Inference-time contract
features = build_features(df)
probas = model.predict_proba(pd.DataFrame([features]))[0]
# Returns: {"SELL": p0*100, "HOLD": p1*100, "BUY": p2*100}
```

### Feature Name Stability
- `MLEngine.feature_names` frozen after first `fit()`
- Inference validates: `missing = [f for f in feature_names if f not in features]`
- Schema changes → model version bump + retrain

---

## 4. ML Engine → Strategy

### Provider: `src/strategy.predict_ml(df, model)`
### Consumer: `src/strategy.generate_signal_result()`

```python
# Input: DataFrame with indicators, trained model (or None)
# Output: Tuple[signal, probability, probabilities_dict]

signal: Literal["SELL", "HOLD", "BUY"]
probability: float  # 0-100, max class probability
probabilities: Dict[int, float]  # {0: sell_pct, 1: hold_pct, 2: buy_pct}

# If model is None or error:
return ("HOLD", 0.0, {})
```

### Contract Guarantees
- Never raises (catches all exceptions, returns HOLD)
- Probabilities sum to ~100
- Model classes: 0=SELL, 1=HOLD, 2=BUY (XGBoost native)

---

## 5. Confidence Engine → Strategy

### Provider: `src/confidence_engine.ConfidenceEngine.evaluate()`
### Consumer: `src/strategy.evaluate_market()`

```python
# Input: add_component(name, score, description)
# Weights (fixed):
#   trend: 1.5, momentum: 1.2, volume: 1.1, volatility: 1.0
#   liquidity: 1.4, structure: 1.3, regime: 1.5

# Output: ConfidenceResult
{
    "total_score": float,      # weighted average
    "confidence": float,       # 50 + 10*|score|, clamped [0,100]
    "probability": float,      # == confidence
    "decision": Literal["HOLD", "BUY", "SELL"],
    "components": List[ScoreComponent],
    "reasons": List[str]
}

# Decision rule:
#   HOLD if confidence < 60 OR |score| < 1.0
#   BUY  if score >= 1.0
#   SELL if score <= -1.0
```

---

## 6. Strategy → Risk

### Provider: `src/strategy.generate_signal_result()`
### Consumer: `src/risk/risk_orchestrator.RiskOrchestrator.evaluate()`

```python
# Input: SignalResult
{
    "signal": "BUY" | "SELL" | "HOLD",
    "entry": float,
    "stop_loss": float,
    "take_profit": float,
    "confidence": float,
    "reasons": List[str],
    # ... diagnostics
}

# + equity: float
# + current_exposure: float
# + risk_percent: float (default 1.0)
# + leverage: float (default 1.0)

# Output: RiskDecision
{
    "allowed": bool,
    "quantity": float,
    "stop_loss": float,
    "take_profit": float,
    "reason": str,
    "drawdown_result": DrawdownGuardResult,
    "exposure_result": ExposureResult,
    "position_size_result": PositionSizeResult,
    "metadata": dict
}
```

### Risk Decision Logic
1. DrawdownGuard: block if DD > 10%
2. PositionSizer: qty = risk_amount / stop_distance
3. ExposureManager: cap qty by max_position_exposure (5%)
4. ExposureManager: check total_exposure + new < max_total_exposure (60%)

---

## 7. Strategy → Trade Engine

### Provider: `src/strategy.generate_signal_result()`
### Consumer: `src/trade_engine.TradeEngine.run()`

```python
# TradeEngine calls generate_signal_result(history) per candle
# If signal != HOLD and can_open_position():
#   open_position(candle, signal)

# SignalResult fields used:
signal.entry          # entry price (close)
signal.stop_loss      # SL price
signal.take_profit    # TP price
signal.confidence     # for position sizing (via Risk)
signal.reasons        # logged
```

### TradeEngine Contract
- **SL/TP checked FIRST** on candle (before BE/trailing)
- **BE/Trailing updates active NEXT candle** (no retroactive)
- **Commission**: 0.04% per side
- **Slippage**: 0.02% (BUY higher, SELL lower)
- **Position sizing**: via `risk_manager.calculate_position_size()`

---

## 8. Paper Trading → Strategy

### Provider: `src/paper_trading_runner.PaperTradingRunner.process_signal()`
### Consumer: `src/strategy.generate_signal_result()` (via ML injection)

```python
# PaperTradingRunner injects ML model:
ml_model = self._ml_engine.model if self.enable_ml else None
signal = generate_signal_result(window, model=ml_model)

# ML Quality Gate (if enabled):
if ml_quality_gate_passed:
    ml_signal, ml_conf = self._get_ml_prediction(df)
    if strategy_signal != ml_signal and ml_signal != "HOLD":
        block_trade("ML disagrees")
```

---

## 9. Execution Engine (Future)

### OrderIntent Contract
```python
@dataclass
class OrderIntent:
    symbol: str
    side: OrderSide        # BUY/SELL
    order_type: OrderType  # MARKET/LIMIT
    quantity: float
    price: float | None    # Required for LIMIT
    reduce_only: bool = False
    time_in_force: TimeInForce = GTC
    client_order_id: str   # UUID
    metadata: dict         # strategy_id, confidence, risk_decision_ref
```

### Safety Guards (pre-submit)
```python
def safety_check(intent: OrderIntent) -> bool:
    # Kill switch: daily_pnl > -5%, drawdown < 10%
    # Position limit: open_positions < max_open
    # Exposure: RiskOrchestrator.can_open()
    # Balance: balance > min_notional
    # Sanity: qty * price < max_notional
```

---

## Contract Evolution Rules

| Change Type | Process |
|-------------|---------|
| Add optional field | Backward compatible, no ADR |
| Add required field | ADR + migration period (2 phases) |
| Remove field | ADR + deprecation (2 phases) |
| Change type/meaning | ADR + new contract version |
| New module | Full contract doc + ADR |

---

**Next**: Implement contract validation tests (`tests/test_contracts.py`)