# QuantAI Project Context

## Project Overview
**QuantAI** — AI-driven cryptocurrency trading platform for systematic quantitative trading.

**Author:** Бахтиёржон  
**Start Date:** 27 июля 2026 года  
**Current Version:** 5.1.0  
**Status:** Research Phase — validation of strategies, search for robust edge

---

## Architecture (from README)

```
src/
├── strategy/           # Trading strategies & signal generation
├── ml/                 # ML pipeline (train, walk-forward, ensemble)
├── risk/               # Risk management (position sizing, drawdown, exposure)
├── execution/          # Order execution, reconciliation
├── validation/         # PurgedKFold, Walk-Forward validation
├── monitoring/         # Metrics, logging, health checks
└── data/               # Data loading, indicators, feature engineering
```

---

## Core Technologies
- **Python:** 3.12+
- **ML:** XGBoost, LightGBM, CatBoost, scikit-learn, Optuna
- **Data:** pandas, NumPy, pyarrow (Parquet)
- **Exchange:** CCXT (Binance, Bybit, Kraken)
- **Config:** Pydantic v2, python-dotenv
- **CLI:** Typer, Rich
- **Infra:** Redis, asyncpg, Prometheus, Telegram Bot API
- **Validation:** PurgedKFold, Walk-Forward, Nested CV

---

## Key Modules (observed from file structure)

### Strategy & Signals
- `src/strategy/` — Breakout, Mean Reversion, ML Overlay, Meta-Labeling, Order Flow Gate
- `src/strategy_genome.py` — Strategy genome/evolution
- `src/strategy_tournament.py` — Strategy competition
- `src/strategy_champion.py` — Champion selection
- `src/strategy_bank.py` — Strategy repository

### ML Pipeline
- `src/ml_engine.py` — Core ML training
- `src/ml_ensemble.py` — Heterogeneous ensemble (XGB, LGBM, CatBoost)
- `src/ml_walk_forward.py` — Walk-forward training
- `src/ml_regime.py` — Regime detection
- `src/ml_config.py` — ML configuration

### Risk Management
- `src/risk/` — Policies, orchestrator, context, correlation, Kelly sizer
- `src/risk_manager.py` — Main risk manager
- `src/risk/dynamic_risk_budget.py` — Dynamic risk allocation
- `src/risk/cross_margin.py` — Cross-margin mechanics
- `src/position_sizer.py` — Position sizing
- `src/exposure_manager.py` — Portfolio exposure limits

### Validation Gates
- `src/validation/gate.py` — Engineering gate
- `src/validation/purged_kfold.py` — Purged K-Fold CV
- `src/validation/nested_walk_forward.py` — Nested WF
- `src/validation/long_run.py` — Long-run validation
- `src/validation/paper_30d.py` — 30-day paper trading gate
- `src/validation/cost_stress.py` — Cost stress testing
- `src/validation/bootstrap.py` — Bootstrap validation

### Walk-Forward System
- `src/walk/` — Walk-forward validation pipeline
- `src/walk_forward_engine.py` — WF engine
- `src/walk_forward_validator.py` — WF validator

### Paper Trading
- `src/paper_trading_*.py` — Engine, runner, monitor, validator, quality gate, pipeline

### Production Runtime
- `src/quantai_production_*.py` — Observability, model registry, runtime lifecycle, incident management, disaster recovery

### Market Intelligence
- `src/unified_market_intelligence.py` — Unified market data
- `src/order_flow_intelligence.py` — Order flow analysis
- `src/liquidation_intelligence.py` — Liquidation tracking
- `src/microstructure_intelligence.py` — Microstructure features

### Feature Engineering
- `src/feature_engine.py` — Feature generation
- `src/feature_store/` — Feature store with drift detection
- `src/indicators.py` — Technical indicators

### Telegram Integration
- `src/telegram/` — Bot, LLM client, handlers, agent registry

---

## Risk Rules (from AGENTS.md)
1. **1% risk per trade** (max 3% per single trade)
2. **Max 5% total exposure** across all open trades
3. **Max 5% per asset**
4. **40% absolute reserve**
5. **Rule 3-5-7**: max 3% risk/trade, max 5% total exposure, profitable trades ≥7% more than losing

---

## Validation Criteria
- **Engineering Gate:** Compilation, tests, no-lookahead, risk invariants
- **Trading Readiness Gate:** PF > 1.1, Expectancy > 0, DD < 35%, Bankrupt = false
- **Paper Trading:** 30+ days, Sharpe > 1.0, Max DD < 8%

---

## Data Files
- Parquet files in `data/` for SOL, XRP, UNI, TRX, SUI (1H, 15m, 4H)
- Registry JSON files in `data/registry/`

---

## Logs & Artifacts
- Tournament logs: `tournament*.log`, `tournament_final.log`
- Walk-forward logs: `wf4.log`, `triple_wf2.log`
- Optuna logs: `optuna_breakout.log`, `tune_gate.log`
- Supervisor logs: `supervisor_run.log`, `supervisor_real.log`
- Trade exports: `trades.csv`

---

## Docker & Deployment
- `Dockerfile` — Container build
- `docker-compose.testnet.yml` — Testnet deployment
- `deploy_testnet.sh` — Deployment script
- `TESTNET_DEPLOYMENT.md` — Deployment guide

---

## BMAD Skills Installed
Located in `.claude/skills/` — full BMAD toolkit for analysis, architecture, build, review, PRD, sprint planning, etc.

---

## Key Entry Points (CLI)
```bash
# Data download
python -m quantai data download --symbol BTC/USDT --timeframe 4h

# Indicators
python -m quantai indicators build --input data/btcusdt_4h.parquet --output data/btcusdt_4h_prepared.parquet

# Backtest
python -m quantai backtest run --prepared data/btcusdt_4h_prepared.parquet

# Walk-forward
python -m quantai ml walk-forward --prepared data/btcusdt_4h_prepared.parquet

# Validation gate
python -m src.validation.gate
```

---

## Configuration Details

### Settings (config/settings.py) — Pydantic v2
All config via nested settings classes with env override support (`__` delimiter):

| Class | Key Parameters |
|-------|----------------|
| `ExchangeSettings` | exchange=binance, symbol=BTC/USDT, timeframe=15m, testnet=False, mode=PAPER |
| `RiskSettings` (canonical) | risk_per_trade=1%, max_total_exposure=60%, max_position_exposure=5%, max_open_positions=1, drawdown_limit=10%, kelly/volatility sizing, ATR SL/TP multipliers |
| `StrategySettings` | min_confidence=0.60, ai_weight=0.60, ml_weight=0.40, orderflow_enabled=True, weighted_gate_threshold=0.75, sl_tp_method=atr_adaptive |
| `MLSettings` | cv_type=combinatorial (PurgedKFold), n_splits=5, embargo_pct=1%, regime_aware=False, XGB params (n_est=300, depth=6, lr=0.05) |
| `IndicatorSettings` | EMA(20,50,200), RSI(14, buy=55/sell=45), MACD(12,26,9), ADX(14, min=25), BB(20,2.0), ATR(14, SL=1.5/TP=3.0/trail=2.0) |
| `TelegramSettings` | office_enabled=False, LLM routing (openai/ollama/groq), per-agent tokens |

**Config drift guard:** `AccountSettings` risk fields are deprecated; `RiskSettings` is canonical. Startup warns + syncs if drift detected.

### Testnet Overrides (config/testnet_settings.py)
- Base URL: `testnet.binancefuture.com`
- Conservative: risk_per_trade=0.5%, max_drawdown=5%, max_exposure=30%, max_positions=2
- Dry-run by default, paper_trading=True, initial_balance=10000
- ML enabled with purged CV

### Feature Schema (config/FEATURE_SCHEMA.json v5.2.0)
- **25 active features** (all causal, normalized, used by model)
- Categories: EMA distances/spreads (5), ATR% (1), Relative volume (1), RSI variants (6), Trend/ADX/DI (4), MACD (4), Bollinger Bands (4), SuperTrend (2), Volume/Volatility anomalies (2)
- **5 PLANNED features** (microstructure/alternative): VPIN, Kyle Lambda, Liquidation Proximity, LunarCrush Galaxy Score, Funding Rate — skipped until live feeds wired

---

## Key Source Files to Review

| Module | Purpose |
|--------|---------|
| `src/strategy/signal_generator.py` | Main signal fusion (AI + ML + OrderFlow) — Regime-adaptive dual strategy (Trend LONG/SHORT, Range mean-reversion) |
| `src/strategy/breakout_signal.py` / `mean_reversion_signal.py` | Core signal logic |
| `src/strategy/ml_overlay.py` / `ml_fusion.py` | ML model integration |
| `src/strategy/order_flow_gate.py` | VPIN/Kyle/liquidation filtering |
| `src/strategy/sl_tp_calculator.py` | Regime-adaptive SL/TP with liquidation data |
| `src/strategy/confidence_engine.py` | Component scoring + continuous probability |
| `src/strategy/ai_analyzer.py` | Technical component analysis (trend, momentum, volume, volatility) |
| `src/feature_engine.py` | Feature generation (25 active causal features) |
| `src/ml_engine.py` | Training pipeline (XGB/LGBM/CatBoost ensemble) |
| `src/ml_ensemble.py` | Heterogeneous ensemble |
| `src/risk/risk_orchestrator.py` | **Unified risk facade**: DrawdownGuard → PositionSizer → ExposureManager → Correlation-adjusted exposure (15% factor limit) |
| `src/risk/kelly_sizer.py` / `dynamic_risk_budget.py` | Position sizing |
| `src/drawdown_guard.py` | Equity peak tracking, drawdown limits |
| `src/exposure_manager.py` | Position/notional exposure caps |
| `src/position_sizer.py` | Risk-based sizing (fixed_fractional, kelly, volatility_adjusted) |
| `src/walk/walk_forward_engine.py` | WF validation pipeline |
| `src/validation/gate.py` | Engineering + Trading Readiness gates |
| `src/paper_trading_engine.py` | Paper trading execution |
| `src/quantai_production_runtime.py` | Production runtime supervisor |
| `src/unified_market_intelligence.py` | Market data aggregation |

---

## Signal Generation Pipeline (src/strategy/signal_generator.py)

**Flow:**
1. **AIAnalyzer** → `MarketComponents` (trend, momentum, volume, volatility scores)
2. **ConfidenceEngine** → AI signal + confidence + continuous probability
3. **MLEngine** → ML signal + class probabilities (if `use_ml=True`)
4. **RegimeFilter** → Classify: `TREND_UP`, `TREND_DOWN`, `RANGE` (ADX-based with hysteresis)
5. **Regime-Adaptive Strategy**:
   - `TREND_UP`: Trend-following LONG (EMA alignment + ADX > 25), lower gate threshold (0.70), confidence boost 1.2x
   - `TREND_DOWN`: Trend-following SHORT, same as TREND_UP
   - `RANGE`: Mean-reversion (BB squeeze + RSI extremes), gate threshold 0.65, confidence boost 1.0x
6. **WeightedGate** (3 instances: default/trend/range) → Approve/reject with probability threshold
7. **OrderFlowGate** → Microstructure filter (VPIN toxic >0.8, Kyle lambda, liquidation proximity)
8. **SLTPCalculator** → Regime-adaptive SL/TP (ATR-based, volatility regime aware)

**Key Config (SignalConfig.from_settings):**
- `min_confidence`: 60% (0..1 canonical, handles legacy 60.0)
- `use_ml`: from `settings.ml.ml_enabled` (default False)
- `use_order_flow`: from `settings.strategy.orderflow_enabled` (default True)
- `use_regime_adaptive`: True (Phase 1)
- `trend_adx_min`: 25.0, `range_adx_max`: 25.0
- BB squeeze width: 0.02, RSI oversold/overbought: 30/70

**SignalResult** — rich diagnostics: AI/ML/Fusion/OrderFlow/SLTP signals, confidence, reasons, metadata

---

## Risk Orchestrator (src/risk/risk_orchestrator.py)

**Decision Pipeline:**
```
SignalResult + Equity + Current Exposure
    ↓
DrawdownGuard (equity health, max DD %)
    ↓
PositionSizer (risk-based size: risk% × equity / stop_distance × leverage)
    ↓
ExposureManager (per-position cap: max 5% equity, total cap: max 60% equity)
    ↓
Correlation-Adjusted Exposure (15% factor limit, PCA on correlation matrix) — Audit §26-27
    ↓
RiskDecision (allowed, quantity, SL/TP, reasons, full diagnostics)
```

**Key Limits (from RiskSettings):**
- `max_drawdown_percent`: 10% (testnet: 5%)
- `max_total_exposure_percent`: 60% (testnet: 30%)
- `max_position_exposure_percent`: 5% (testnet: 3%)
- `risk_per_trade`: 1% (testnet: 0.5%)
- Correlation factor limit: 15% (enforced if RiskContext provides correlation_matrix)

**RiskContext (R0.1)** — carries projected exposure for flip-aware limit checks, open positions dict, correlation matrix

**Fail-open on correlation calc errors** — never blocks on correlation exceptions

---

## Notes for Future Sessions
- Project is in active research/development phase
- Multiple parallel validation systems exist (walk/, validation/, paper_trading_*)
- Production runtime modules are extensive but may not be fully integrated
- Strategy tournament/evolution system exists for automated strategy discovery
- Feature store with drift detection for production monitoring
- Cross-margin risk management implemented

---

## Last Updated
2026-09-01 — Initial context creation