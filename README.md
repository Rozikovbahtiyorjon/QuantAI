# QuantAI

## AI Trading Platform

QuantAI — это модульная платформа для анализа финансовых рынков с использованием искусственного интеллекта.

### Основные цели проекта

- Анализ криптовалютного рынка
- Генерация торговых сигналов
- Backtesting торговых стратегий
- Управление рисками
- Интеграция с Telegram
- Поддержка Binance, Bybit, Kraken и других бирж
- Подготовка к автоматической торговле

### Архитектура

```
src/
├── strategy/           # Торговые стратегии и генерация сигналов
├── ml/                 # ML pipeline (train, walk-forward, ensemble)
├── risk/               # Risk management (position sizing, drawdown, exposure)
├── execution/          # Order execution, reconciliation
├── validation/         # PurgedKFold, Walk-Forward validation
├── monitoring/         # Metrics, logging, health checks
└── data/               # Data loading, indicators, feature engineering
```

### Быстрый старт

```bash
# Установка
pip install -e .

# Скачать данные
python -m quantai data download --symbol BTC/USDT --timeframe 4h

# Подготовить индикаторы
python -m quantai indicators build --input data/btcusdt_4h.parquet --output data/btcusdt_4h_prepared.parquet

# Запустить бэктест
python -m quantai backtest run --prepared data/btcusdt_4h_prepared.parquet

# Walk-forward валидация
python -m quantai ml walk-forward --prepared data/btcusdt_4h_prepared.parquet

# Запустить валидационный гейт
python -m src.validation.gate
```

### Конфигурация

Основные настройки в `config/settings.py`:
- `StrategyConfig` — параметры стратегии (thresholds, weights, regime params)
- `MLConfig` — ML параметры (PurgedKFold, ensemble, hyperparameters)
- `RiskConfig` — риск-менеджмент (position sizing, drawdown limits, exposure)

### Валидация

Проект использует многоуровневую валидацию:
1. **Engineering Gate** — компиляция, тесты, no-lookahead, risk invariants
2. **Trading Readiness Gate** — PF > 1.1, expectancy > 0, DD < 35%, bankrupt = false
3. **Paper Trading** — 30+ дней, Sharpe > 1.0, max DD < 8%

### Статус проекта

Текущий статус: **Research Phase** — валидация стратегий, поиск устойчивого edge.

---

Автор: Бахтиёржон

Дата начала проекта: 27 июля 2026 года