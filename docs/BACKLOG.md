# QuantAI Backlog (отложенные задачи)

Сформирован по итогам аудита 2026-08-27. Порядок — приоритет.

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

## P2 — архитектурные долги
- [ ] Портфельный брокер Long-Run провести через RiskOrchestrator/ExecutionBridge (сейчас собственный леджер с параметрическим риском)
- [ ] WF-неймспейс: довести facades до полного отказа от корневых импортов в тестах
- [ ] Regime-aware риск-слой для кросс-секционных кандидатов (де-риски с учётом фазы рынка)
- [ ] MC/Stress скоры в Tournament (хуки 0.5 нейтральные готовы)
- [ ] Entry Engine (EV-гейт + Setup/Trigger/TTL + maker-зоны) — см. ADR-0004, только как кандидаты банка
- [ ] **Мартингейл/усреднение — только experimental с Hard Stop**: `AveragingEngine` max 2 шага ×1.5, aggregate SL на весь пакет, риск пакета ≤20% от 40% резерва (≤8% депозита), gate `/stress-test -20% + лаг API + активная сетка` обязателен, дефолт OFF, never в автономном ядре
- [ ] **Hard Stop 10% портфеля / 25% флэт-тайм-аут**: `HardStopGuard` — принудительный flat всех позиций по DD от пика, независимо от усреднения; метрика `risk_kill_switch_active=1`, ручной reset; обязателен до live с плечом
- [ ] **Авто-стейкинг (bet sizing)**: Kelly-fraction × confidence_score; `confidence≥0.8` → 3% риск, `0.6–0.8` → 1.5%, `<0.6` → 0.5%; `0.5%` минималка, кап 5% депозита на сделку
- [ ] **Плечо авто 3/5/10/20/50×**: `LeverageSelector` в `RiskOrchestrator` — формула `floor(0.8*entry/(entry-SL))` clamped [3,5,10,20,50]; ликвидация ≥20% за стопом; кросс-маржа + STOP_MARKET reduceOnly; после R3-гейта
- [ ] **P2-diversification: A(18 Binance) → B-1(3 криптобиржи) → B-2(PAXG/SPX via Alpaca)**: этап A (18 альтов, cross-sectional уже чемпион) валидируется; B-1 — redundancy + funding-basis (Binance+Bybit+OKX, один API CCXT); B-2 — первый кросс-актив через PAXG (золото на Binance, без новой биржи) и SPX (Alpaca/OANDA) для decorrelation
- [ ] **P2-diversification: A(18 Binance) → B-1(3 криптобиржи) → B-2(PAXG/SPX via Alpaca)**: этап A (18 альтов, cross-sectional уже чемпион) валидируется; B-1 — redundancy + funding-basis (Binance+Bybit+OKX, один API CCXT); B-2 — первый кросс-актив через PAXG (золото на Binance, без новой биржи) и SPX (Alpaca/OANDA) для decorrelation

## P2-aggressive — MAX-PROFIT пресет (изолированная ветка, вне 3-5-7)
- [ ] **Агрессивный кандидат `xs_aggressive`**: `risk 5–7%` (Kelly×1.0), `total 95%`, `position 30%`, `leverage 15–20×` (изолированная), `top_k=1`, `rebalance 3д`, `DD soft 30% / hard 50%`, `weighting equal` (без vol-target), без Hard Stop — ожидается +800–1200%/3г при DD −90%+ и риске слива ~40%; только как отдельный кандидат банка, никогда дефолтом

## Известные ограничения (не баги)
- OrderFlowGate получает None в оффлайн-режимах (нет L2-истории) — активировать после data-infra
- Stability-кап (std≤120%) валит сильнейшего кандидата lb14_k1 — вопрос калибровки решит Long-Run телеметрия
