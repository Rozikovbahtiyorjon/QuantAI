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

## P2 — архитектурные долги
- [ ] Портфельный брокер Long-Run провести через RiskOrchestrator/ExecutionBridge (сейчас собственный леджер с параметрическим риском)
- [ ] WF-неймспейс: довести facades до полного отказа от корневых импортов в тестах
- [ ] Regime-aware риск-слой для кросс-секционных кандидатов (де-риски с учётом фазы рынка)
- [ ] MC/Stress скоры в Tournament (хуки 0.5 нейтральные готовы)
- [ ] Entry Engine (EV-гейт + Setup/Trigger/TTL + maker-зоны) — см. ADR-0004, только как кандидаты банка

## Известные ограничения (не баги)
- OrderFlowGate получает None в оффлайн-режимах (нет L2-истории) — активировать после data-infra
- Stability-кап (std≤120%) валит сильнейшего кандидата lb14_k1 — вопрос калибровки решит Long-Run телеметрия
