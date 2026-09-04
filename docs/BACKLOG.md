# QuantAI Backlog (отложенные задачи)

Сформирован по итогам аудита 2026-08-27. Порядок — приоритет.
**Корректная последовательность**: `PHASE 0 Engineering+Risk → 1 Trusted Evidence+Control Plane → 2 OOS/Nested WF/Research Integrity → 3 Robustness+Statistical → 4 Alpha Research (P3) → 5 Paper (30–90д) → 6 Testnet → 7 Small Live → 8 Autonomous Evolution (P4+P5.8)`

> **FREEZE до `phase-1-control-plane-complete`**: `LLM trading signals`, `Deep RL`, `Transformer`, `GAN/VAE`, `десятки индикаторов`, `массовая Optuna 1000+`, `LunarCrush`, `сложная genetic evolution` — любые PR из этого списка → `BLOCKED` гейт (см. `docs/roadmap/phase-gates.md`).

> **P3 Advanced Alpha только после P0/P1/P2**, **P4 Autonomous только после Research Integrity (Phase 2)**, **P5 Production только после Phase 3**.

## P0 — до первого запуска с реальными ключами
- [ ] **Маскирование секретов**: `run.py config show`, логи — печатать только `...last4` ключей (`api_key/api_secret/token/password`)
- [ ] **gitleaks** в CI на каждый push (+ bandit уже есть weekly)

## P0-autonomy — автономный режим (перед фоновым запуском)
- [ ] **Telegram-notifier**: события сделка/ребаланс/incident/gate-вердикт (~100 строк, конфиг готов)
- [ ] **Daemon-обвязка**: systemd-unit / compose-service для `portfolio_long_run` + watchdog по свежести `state.updated_at`
- [ ] **Ежедневная сводка** в Telegram (CSV-итоги дня; Sheets — опционально позже)
- [ ] **Kill-switch файл** (`STOP` в каталоге сессии → graceful flat перед следующим баром)

## P1 — перед live-торговлей
- [ ] Ключи: env с `chmod 600`, ротация после любого контакта вне сервера; IP-whitelist обязателен
- [ ] ML: переобучить модели с `purge_pct ≥ горизонт лейблов`; прогнать CPCV; иначе `ml_enabled=False` навсегда
- [ ] Подключить `BinanceRateLimiter` в DRY/LIVE путь ExecutionBridge (hook готов)
- [ ] Vault (Fernet+Postgres) — решить: подключать или удалить вместе с зависимостями
- [ ] **Автовыбор плеча 3–10×**: `LeverageSelector` в `RiskOrchestrator` (формула `floor(0.8*entry/(entry-SL))` clamped [3,10], set {3,5,7,10}; ликвидация ≥20% за стопом; кросс-маржа + STOP_MARKET reduceOnly) — после R3-гейта
- [ ] **Кросс-маржа дефолт + явный marginType**: `RiskSettings.margin_mode="CROSS"` (портфель) / `ISOLATED` опция для single-asset aggressive; `PositionSizer` → `OrderIntentData(marginType)` → `BinanceAdapter` ставит `POST /fapi/v1/marginType`; paper-брокер считает общий equity vs maintenance

## P1-optimization — Бритва Оккама (до live)
- [ ] **Лимит индикаторов 2–3**: аудит `IndicatorSettings` (сейчас 8 активных), WF-абляция каждого индикатора на P-C1/P-C2 данных; удалить вклад <0.02 Sharpe без потери PF; запрет новых индикаторов без A/B через банк кандидатов

## P1-data — Комплексный анализ данных (источники для генерации альфа-сигналов)

- [ ] **Фундаментальный анализ**: RSS/News API (CryptoPanic, CoinDesk, CoinTelegraph) + календарь событий (CoinMarketCal); парсинг отчетов (10-Q/20-F для публичных компаний, on-chain treasury для протоколов). Нормализация → эмбеддинги → ежедневный вектор «фундаментальная сила».
- [ ] **Сентимент-анализ (Social NLP)**: Twitter/X API v2 (filtered по кастомному списку влиятельных аккаунтов/ключевых слов), Reddit API (r/CryptoCurrency, r/Bitcoin, специфические сабреддиты), Telegram-каналы (через TDLib/bot) → очистка спама/ботов → sentiment score per symbol/день → интеграция в Feature Engine как доп. фича `sentiment_score`. LunarCrush API (Galaxy Score, AltRank) — опционально, если лимиты позволяют.
- [ ] **Технический анализ расширенный**:
    - Паттерны свечей (candle patterns: engulfing, doji, hammer, harami) → binary фичи
    - Price Action: HH/HL/LH/LL, order blocks, fair value gaps, breaker blocks
- [ ] **L2 / Order Book — для maker-исполнения (детально, после Long-Run гейта)**:
    - WS `depth@100ms` + REST snapshot → локальный L2-кеш (`src/market_data/fanout.py` → `OrderBookMarketData`): bid/ask heaps, TTL 1с, sequence-gap handling (snapshot+delta), queue-position estimation
    - Фичи: bid-ask spread, order book imbalance (OIB), depth slope, microprice
    - Order Flow: cumulative delta (CVD), absorption, footprint-кластеры → `OrderFlowGate` (сейчас `None`, активировать после кэша)
    - Тесты на исторических depth-снепшотах; метрика `order_book_imbalance` в Feature Engine
- [ ] **Деривативы / Futures — как фичи (детально, после L2-кеша)**:
    - Open Interest (OI): агрегатный + по биржам/контрактам — `fetchOpenInterest` (CCXT) + `GET /fapi/v1/openInterest` fallback; OI delta (1h/1d change rate)
    - Funding Rate (8h) + предиктор next funding: `GET /fapi/v1/fundingRate` + `basis = (mark - index)/index`; `funding_basis` фича
    - Liquidations: `forceOrder` stream (WS) + `GET /fapi/v1/allForceOrders` REST — кластеризация по ценам → `liquidation_clusters` фича
    - Long/Short Ratio: `GET /futures/data/globalLongShortAccountRatio` + `topLongShortPositionRatio` — `long_short_ratio` фича
    - Basis-сигналы: `oi_delta + funding_basis` → конкатенация в Feature Engine
- [ ] Интеграция в Feature Engine: все выше → нормализованные признаки → конкатенация с core-индикаторами → выборка для ML/Strategies

## P1-portfolio — Диверсификация 60/40 для кросс-секционного брокера
- [ ] **60% депозита / 40% резерв + 3–5% на актив**: `CrossSectionParams(reserve_ratio=0.40, max_position_pct=0.05)` → `per_name = equity*(1-reserve)/top_k` capped; `top_k=2` (30% — пометка «концентрированный») vs `top_k=12` (5% честный); weekly-оборот уже есть (142 ребаланса)

## P3 — Advanced Alpha Research — ТОЛЬКО после P0/P1/P2 (`phase-3-robustness-complete`)

- [ ] **P3.1 Breakout + regime filter** (`src/research/breakout_research_branch.py:RegimeFilteredBreakout`, `RegimeFilter adx 22/18 hysteresis`)
- [ ] **P3.2 Breakout + ML meta-labeling** (`Primary → Candidate → ML Meta → TAKE/REJECT`, `FilteredGenerator` + `ExpectedReturnModel E[net]`, `src/strategy/meta_label.py`)
- [ ] **P3.3 Multi-timeframe** `4h regime (HTF EMA 50) → 1h structure → 15m entry` (`MultiTFConfirm tf_bars=4`, `src/strategy/meta_label.py`)
- [ ] **P3.4 Cross-sectional momentum** `BTC/ETH/SOL` sort `return 20d` long top/short bottom (`src/strategies/cross_sectional.py`, `src/portfolio/multi_symbol.py`)
- [ ] **P3.5 Volatility breakout** `ATR 14*1.5 + range breakout` (`src/strategy/breakout_signal.py`)
- [ ] **P3.6 Alternative data** `funding/OI/liquidation/L2/VPIN/Kyle Lambda` (`src/microstructure_intelligence.py`, `src/alternative_data.py` — stub до L2-cache, активируется после `phase-3`)
- [ ] **P3.7 HMM/regime clustering** `HMM 3-state` + `KMeans 7` → `REGIMES` (`src/ml_regime.py`, `src/research/regime_stability.py`)

## P4 — Autonomous Intelligence — ТОЛЬКО после Research Integrity (Phase 2)

- [ ] **P4.1 Autonomous Hypothesis Generator** `Supervisor → hypothesis` (`src/control_plane/supervisor.py:_research_loop_on_reject`)
- [ ] **P4.2 Autonomous Experiment Generator** `hypothesis → experiment spec` (`ExperimentRegistry` + `TaskManager`)
- [ ] **P4.3 Autonomous Coding** `task → code → tests → repair` (`RetryEngine` + `ruff`/`py_compile`)
- [ ] **P4.4 Autonomous Research Loop** `experiment → evaluate → reject/improve → new experiment` (`_execute_cycle` OBSERVE→GATE)
- [ ] **P4.5 Research Budget Enforcement** `experiments/Optuna trials/mutations/OOS touches/retries` durable `AtomicResearchLedger` (`src/research/research_budget.py`)
- [ ] **P4.6 Autonomous Champion Search** только `verified evidence + robust OOS + statistical PASS` (`src/champion/pipeline.py` + `ResearchIntegrity`)

## P5 — Production — И только здесь

- [ ] **P5.1 30–90 дней Paper Trading** (`src/validation/long_run.py`, `min_trades 30/90d`, gate `BLOCKED` до 30д)
- [ ] **P5.2 Testnet full lifecycle** (`ExecutionEngine(DRY_RUN)` + `verify_no_withdraw_permission`)
- [ ] **P5.3 Failure injection (8)** `exchange unavailable / websocket disconnect / REST timeout / Redis failure / database failure / stale market data / corrupted model / unexpected position` (`tests/test_quantai_production_runtime_*`)
- [ ] **P5.4 Disaster recovery drill** (`emergency_stop` → `resume_from_halt` 7 шагов)
- [ ] **P5.5 Reconciliation drill** (`ReconciliationEngine` ghost/stuck)
- [ ] **P5.6 Emergency-stop drill** (`HALTED → flatten → verify flat`)
- [ ] **P5.7 Small-capital live** `$100–500` 14д, `max_exposure 5%`, `reserve 40%`
- [ ] **P5.8 Gradual capital scaling** `+20%` каждые 30д при `PF>1.2`, `DD<10%` (после P4.6)

## P5-AutoML — Самообучение / «Конструктор стратегий» — ТОЛЬКО после Phase 8

- [ ] **StrategyComposer** (`src/auto_ml/strategy_composer.py`):
    - `propose(market: MarketContext, constraints: Constraints) -> List[CandidateSpec]`
    - 1. Retrieve relevant competences from StrategyBank (similar regime/assets)
    - 2. Compose candidate genomes via genetic programming / Bayesian optimization
    - 3. Return top-K CandidateSpec for ChampionPipeline evaluation
- [ ] **Knowledge Graph** (`src/auto_ml/knowledge_graph.py`):
    - Semantic links: "RSI<30 + TrendUp → BUY works on BTC 1h, not ETH 4h"
    - Graph DB for transfer learning across assets/regimes
- [ ] **Meta-Learning** (`src/auto_ml/meta_learner.py`):
    - `ModelSelector` learns which genome works best on new asset/regime (few-shot adaptation)
- [ ] **AutoML Pipeline** (orchestrator):
    - Generate N candidates → ChampionPipeline evaluation → top-K → auto-push to StrategyBank
    - Continuous learning loop: Live/Paper telemetry → Performance Feedback → Candidate stats → Mutation → Re-evaluation
- [ ] **P2-diversification: A(18 Binance) → B-1(3 криптобиржи) → B-2(PAXG/SPX via Alpaca)**: этап A (18 альтов, cross-sectional уже чемпион) валидируется; B-1 — redundancy + funding-basis (Binance+Bybit+OKX, один API CCXT); B-2 — первый кросс-актив через PAXG (золото на Binance, без новой биржи) и SPX (Alpaca/OANDA) для decorrelation

## P2-aggressive — MAX-PROFIT пресет (изолированная ветка, вне 3-5-7)
- [ ] **Агрессивный кандидат `xs_aggressive`**: `risk 5–7%` (Kelly×1.0), `total 95%`, `position 30%`, `leverage 15–20×` (изолированная), `top_k=1`, `rebalance 3д`, `DD soft 30% / hard 50%`, `weighting equal` (без vol-target), без Hard Stop — ожидается +800–1200%/3г при DD −90%+ и риске слива ~40%; только как отдельный кандидат банка, никогда дефолтом

## FREEZE — Что НЕ делать сейчас (до `phase-1-control-plane-complete`)

| Заморожено | Почему | Разморозка |
|------------|--------|------------|
| LLM trading signals | Нужна калибровка OOS, иначе утечка | `phase-3` |
| Deep RL | Требует симуляцию + risk safety | `phase-3` |
| Transformer | Сложность без профита в P0 | `phase-4` |
| GAN/VAE | Сложность без профита | `phase-4` |
| Десятки индикаторов | Бритва Оккама: лимит 2–3, WF-абляция | `phase-2` |
| Массовая Optuna 1000+ | Переоптимизация, бюджет `max_optuna 50` | `phase-4` |
| LunarCrush Galaxy Score | Внешняя зависимость, лимиты | `phase-4` alt data |
| Сложная genetic evolution | Требует verified champion search | `phase-8` |

> Любой PR из FREEZE до `phase-1` → `BLOCKED` гейт. Проверка: `grep -r "Transformer\|LunarCrush\|GAN" src --include="*.py" | gate`.

## Известные ограничения (не баги)
- OrderFlowGate получает None в оффлайн-режимах (нет L2-истории) — активировать после data-infra
- Stability-кап (std≤120%) валит сильнейшего кандидата lb14_k1 — вопрос калибровки решит Long-Run телеметрия
