# QuantAI Core Trading Architecture

## Level 1 — Core Trading Pipeline

```
Market Data
    ↓
Indicators
    ↓
Feature Engine
    ↓
Model Manager
    ↓
Confidence Engine
    ↓
Strategy
    ↓
Risk
    ↓
Trade Engine
```

## Module Contracts

### 1. Market Data (`src/data_loader.py`, `src/exchange_market_data.py`)
- **Input**: symbol, timeframe, limit
- **Output**: DataFrame[ts, open, high, low, close, volume]
- **Contract**: No NaN in OHLCV, monotonically increasing timestamps
- **Used by**: Indicators, Feature Engine

### 2. Indicators (`src/indicators.py`)
- **Input**: DataFrame with OHLCV
- **Output**: DataFrame + [ema_fast, ema_slow, ema_trend, rsi, atr, volume_ratio, ...]
- **Contract**: All numeric columns cleaned (no inf, NaN forward-filled)
- **Functions**: `add_indicators(df, core_only=False)` — single entry point

### 3. Feature Engine (`src/feature_engine.py`)
- **Input**: DataFrame with indicators
- **Output**: FeatureVector dict (30+ features)
- **Contract**: Last row only, normalized to [0,1] or z-score
- **Used by**: ML Engine, Strategy

### 4. Model Manager (`src/model_manager.py`)
- **Input**: Model path or trained model object
- **Output**: Loaded sklearn/XGBoost model with `predict_proba()`
- **Contract**: `model.classes_` = [0,1,2] mapping to [SELL,HOLD,BUY]
- **Used by**: Strategy (via dependency injection)

### 5. Confidence Engine (`src/confidence_engine.py`)
- **Input**: Component scores {trend, momentum, volume, volatility, ...}
- **Output**: ConfidenceResult(score, confidence, probability, decision, components)
- **Contract**: 
  - Score = weighted sum (trend=1.5, momentum=1.2, volume=1.1, volatility=1.0)
  - Confidence = 50 + 10*|score|, clamped [0,100]
  - Decision = HOLD if confidence<60 or |score|<1, else BUY/SELL
- **Used by**: Strategy

### 6. Strategy (`src/strategy.py`)
- **Input**: DataFrame with indicators, optional OrderFlowSignal, optional ML model
- **Output**: SignalResult(signal, confidence, entry, sl, tp, reasons, diagnostics)
- **Pipeline**: 
  1. evaluate_market() → ConfidenceResult (AI)
  2. predict_ml() → ML signal + probabilities
  3. fuse_ai_ml() → Fusion (6 rules)
  4. apply_order_flow_gate() → Final gate
- **Contract**: MIN_CONFIDENCE=60%, AI_WEIGHT=0.6, ML_WEIGHT=0.4, CONFLICT_PENALTY=0.7
- **Used by**: TradeEngine, PaperTradingRunner, BacktestEngine

### 7. Risk (`src/risk/`)
- **RiskOrchestrator** — unified facade
- **DrawdownGuard** — equity peak tracking, max DD limit
- **ExposureManager** — total/position exposure caps
- **PositionSizer** — risk-based sizing with leverage
- **Input**: SignalResult + equity + current_exposure
- **Output**: RiskDecision(allowed, quantity, sl, tp, reason, diagnostics)
- **Contract**: max_drawdown=10%, max_total_exposure=60%, max_position_exposure=5%, risk_per_trade=1%
- **Used by**: PaperTradingRunner, TradeEngine (future)

### 8. Trade Engine (`src/trade_engine.py`)
- **Input**: DataFrame with indicators
- **Output**: TradeEngine with closed_positions, balance, equity
- **Contract**:
  - SL/TP checked FIRST on candle (before BE/trailing)
  - BE/trailing updates active from NEXT candle
  - Position sizing via RiskOrchestrator
  - Commission=0.04%, Slippage=0.02%
- **Used by**: BacktestEngine, WalkForwardEngine

## Data Flow Invariants

| Stage | Input Validation | Output Guarantee |
|-------|------------------|------------------|
| Market Data | CCXT response | Clean OHLCV DataFrame |
| Indicators | OHLCV present | All indicators + cleanup |
| Feature Engine | Indicators present | FeatureVector dict |
| ML Predict | Model loaded, features match | Probabilities [SELL,HOLD,BUY] |
| Confidence | Component scores | ConfidenceResult |
| Strategy | DataFrame + indicators | SignalResult (HOLD/BUY/SELL) |
| Risk | Signal + equity | RiskDecision (allowed + qty) |
| Trade Engine | Signal + data | Executed trades with PnL |

## Configuration Source

All parameters from `config.settings.Settings`:
- `settings.indicators` — EMA/RSI/ATR/Volume periods
- `settings.risk` — DD limits, exposure caps, sizing
- `settings.backtest` — trailing/BE/partial settings
- `settings.ml` — CV config, XGBoost params

## Integration Points

```
┌─────────────────────────────────────────────────────────────┐
│                    BACKTEST / WALK-FORWARD                    │
├─────────────────────────────────────────────────────────────┤
│  Prepared Data → BacktestEngine → TradeEngine → Strategy    │
└─────────────────────────────────────────────────────────────┘
                            ↑
                    uses same Strategy
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    PAPER TRADING                              │
├─────────────────────────────────────────────────────────────┤
│  Live Data → PaperTradingRunner → RiskOrchestrator → Strategy │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    LIVE (future)                              │
├─────────────────────────────────────────────────────────────┤
│  Signal → ExecutionEngine → OrderManager → BinanceAdapter   │
│                    ↓                                           │
│              ReconciliationEngine                              │
└─────────────────────────────────────────────────────────────┘
```

## Module Status

| Module | Written | Unit | Integrated | E2E | Long-Run | Prod |
|--------|---------|------|------------|-----|----------|------|
| data_loader | ✓ | ✓ | ✓ | ✓ | ? | — |
| indicators | ✓ | ✓ | ✓ | ✓ | ? | — |
| feature_engine | ✓ | ✓ | ✓ | ✓ | ? | — |
| model_manager | ✓ | ✓ | ✓ | ✓ | ? | — |
| confidence_engine | ✓ | ✓ | ✓ | ✓ | ? | ✗ |
| strategy | ✓ | ✓ | ✓ | ✓ | ? | ✗ |
| risk_orchestrator | ✓ | ✓ | ✓ | ✓ | ? | ✗ |
| drawdown_guard | ✓ | ✓ | ✓ | ✓ | ? | ✗ |
| exposure_manager | ✓ | ✓ | ✓ | ? | ? | ✗ |
| position_sizer | ✓ | ✓ | ✓ | ✓ | ? | ✗ |
| trade_engine | ✓ | ✓ | ✓ | ✓ | ? | ✗ |
| backtest_engine | ✓ | ✓ | ✓ | ✓ | ? | ✗ |
| walk_forward_engine | ✓ | ✓ | ✓ | ✓ | ? | ✗ |
| paper_trading_runner | ✓ | ✓ | ✓ | ✓ | ? | ✗ |
| execution_engine | ✓ | ✓ | ? | ? | ? | ✗ |
| reconciliation | ✓ | ✓ | ? | ? | ? | ✗ |

---

**Status**: Architecture contracts defined. Next: Validation Gates (Level 2).