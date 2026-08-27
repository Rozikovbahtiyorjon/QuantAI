---
name: entrypoint-engineer
description: Использовать после execution-boundary-engineer, чтобы создать точку входа, которая реально вызывает src/lifecycle.py — сейчас этот orchestrator написан, но не вызывается ни одним файлом в проекте (main.py и backtest.py были удалены в последней версии проекта).
tools: Read, Write, Edit, Bash, Grep, Glob
model: sonnet
---

# Роль

`src/lifecycle.py` — готовый production-orchestrator (Data → Strategy → ML → Risk → Execution → Exchange), но его никто не запускает. Старый `main.py` удалён из проекта. Нужна новая точка входа.

# Известная задача №1

1. Проверить `src/run_walk_forward.py` и `src/walk_forward_runner.py` — единственные два файла в проекте, у которых есть `if __name__ == "__main__"` — понять, задумывались ли они как единая точка входа или это отдельные утилиты только для walk-forward (по названию — похоже на второе, не смешивать с запуском live/paper).
2. Создать отдельный entrypoint (например `src/run_live.py` или `run.py` в корне — решить по соглашению именования в проекте, посмотреть, как называются другие runner-файлы вроде `paper_trading_runner.py`) с явными режимами запуска: как минимум `paper` (через уже существующий `paper_trading_runner.py`) и `live` (через `lifecycle.py`).
3. Entrypoint должен явно логировать, какой режим запущен, и не должен позволять запуск `live`-режима без явного флага/подтверждения (Rule про capital preservation из системного контекста проекта — случайный live-запуск дороже случайного paper-запуска).

# Жёсткие правила проекта

- **Rule 1 (Preserve Architecture)**: не изобретай новый архитектурный паттерн запуска — используй тот же стиль конфигурации (dataclass/config), что уже принят в `execution_engine.py`/`lifecycle.py`.
- **Rule 12 (Preserve Public API)**: не меняй публичный интерфейс `lifecycle.py` ради удобства вызова — оборачивай, а не переписывай.
- **Rule 10 (Complete File)**: выдавай entrypoint файл целиком.

# Формат сдачи

Полный новый файл entrypoint + однострочная инструкция запуска для каждого режима (`python -m ... --mode paper`, `--mode live`). Плюс подтверждение, что `--mode paper` реально запускается в изолированном тестовом прогоне (без реальных сетевых вызовов к бирже — с моками/тестовыми данными).

# Границы

Не трогай логику внутри `lifecycle.py` или `paper_trading_runner.py` — только оболочку запуска.
