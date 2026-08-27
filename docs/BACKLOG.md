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

## P1-optimization — Бритва Оккама (до live)
- [ ] **Лимит индикаторов 2–3**: аудит `IndicatorSettings` (сейчас 8 активных), WF-абляция каждого индикатора на P-C1/P-C2 данных; удалить вклад <0.02 Sharpe без потери PF; запрет новых индикаторов без A/B через банк кандидатов

## P1-data — Комплексный анализ данных (источники для генерации альфа-сигналов)

- [ ] **Фундаментальный анализ**: RSS/News API (CryptoPanic, CoinDesk, CoinTelegraph) + календарь событий (CoinMarketCal); парсинг отчетов (10-Q/20-F для публичных компаний, on-chain treasury для протоколов). Нормализация → эмбеддинги → ежедневный вектор «фундаментальная сила».
- [ ] **Сентимент-анализ (Social NLP)**: Twitter/X API v2 (filtered по кастомному списку влиятельных аккаунтов/ключевых слов), Reddit API (r/CryptoCurrency, r/Bitcoin, специфические сабреддиты), Telegram-каналы (через TDLib/bot) → очистка спама/ботов → sentiment score per symbol/день → интеграция в Feature Engine как доп. фича `sentiment_score`. LunarCrush API (Galaxy Score, AltRank) — опционально, если лимиты позволяют.
- [ ] **Технический анализ расширенный**:
    - Паттерны свечей (candle patterns library: engulfing, doji, hammer, harami, etc.) → binary features
    - Price Action: структура рынка (HH/HL/LH/LL), order blocks, fair value gaps, breaker blocks
    - Order Book / L2: микроструктурные фичи (bid-ask spread, order book imbalance, depth slope, microprice)
    - L2 Data / Order Flow: cumulative delta, CVD, absorption, footprint clusters
- [ ] **Деривативные метрики (On-chain + Exchange)**:
    - Open Interest (OI) — агрегатный и по биржам/контрактам
    - Funding Rate (8h) + предиктор next funding (basis)
    - Liquidations (биржевые вебхуки/REST) — кластеризация ликвидаций по ценам
    - Long/Short Ratio (accounts + positions) — top trader positions (Binance/Bybit API)
    - Open Interest change rate (OI delta) + Funding Rate basis → basis trading signals
- [ ] Интеграция в Feature Engine: все выше → нормализованные признаки → конкатенация с core-индикаторами → выборка для ML/Strategies

## P2 — архитектурные долги
- [ ] Портфельный брокер Long-Run провести через RiskOrchestrator/ExecutionBridge (сейчас собственный леджер с параметрическим риском)
- [ ] WF-неймспейс: довести facades до полного отказа от корневых импортов в тестах
- [ ] Regime-aware риск-слой для кросс-секционных кандидатов (де-риски с учётом фазы рынка)
- [ ] MC/Stress скоры в Tournament (хуки 0.5 нейтральные готовы)
- [ ] Entry Engine (EV-гейт + Setup/Trigger/TTL + maker-зоны) — см. ADR-0004, только как кандидаты банка

## Известные ограничения (не баги)
- OrderFlowGate получает None в оффлайн-режимах (нет L2-истории) — активировать после data-infra
- Stability-кап (std≤120%) валит сильнейшего кандидата lb14_k1 — вопрос калибровки решит Long-Run телеметрия
