"""
Handlers — business logic for each command / agent dispatch.

Safe by design:
- No live trading without explicit --live flag.
- /kill only sets a flag file, does NOT kill process directly — supervisor reads it.
- All handlers are async and catch exceptions.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from .agent_registry import AGENTS, get_agent, list_agents

# Optional: integrate with real engines if available
try:
    from src.monitoring.health import HealthChecker
except Exception:
    HealthChecker = None  # type: ignore


HELP_TEXT = """🏢 *QuantAI Virtual Office — говори как с людьми*

Просто пиши человеческим языком, бот сам выберет агента:

{agents}

*Примеры человеческого общения:*
`Привет, как там риски?` → 🛡️ Risk Manager
`Проверь архитектуру, есть мусор?` → 🏛️ Architect
`Запусти обучение модели` → 🤖 ML
`Останови все` → 🛑 Kill
`Как дела у офиса?` → 📊 Status

Команды тоже работают, но не обязательны: /help /status /agents /whoami /kill
_В группе отвечу, если обратишься ко мне или напишешь в личку._
"""

KILL_FLAG = Path("logs/TELEGRAM_KILL.flag")


def _format_agents_help() -> str:
    lines = []
    for a in list_agents():
        cmds = ", ".join(a.commands)
        lines.append(f"• {a.emoji} *{a.name}* `@{a.key}` ({cmds}) — {a.description}")
    return "\n".join(lines)


async def handle_help() -> str:
    return HELP_TEXT.format(agents=_format_agents_help())


async def handle_whoami(user_id: int, chat_id: int, username: str | None) -> str:
    return (
        f"👤 *Whoami*\n"
        f"user_id: `{user_id}`\n"
        f"chat_id: `{chat_id}`\n"
        f"username: @{username or '—'}\n\n"
        f"Добавь `TELEGRAM__ADMIN_IDS={user_id}` и `TELEGRAM__CHAT_ID={chat_id}` в `.env` чтобы запереть офис."
    )


async def handle_status() -> str:
    """Lightweight status without importing heavy engines."""
    lines = [
        "📊 *QuantAI Status*",
        f"Время: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
    ]
    # Try to read last trades / logs if exist
    for p in [Path("trades.csv"), Path("logs/quantai.log")]:
        if p.exists():
            try:
                size = p.stat().st_size
                mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).strftime("%H:%M:%S")
                lines.append(f"• `{p.name}`: {size} bytes, mtime {mtime} UTC")
            except Exception:
                pass
    if KILL_FLAG.exists():
        lines.append("🚨 *KILL FLAG активен* — торговля остановлена")
    else:
        lines.append("✅ Kill flag: не активен")
    lines.append("\n_Для полного health: /health_")
    return "\n".join(lines)


async def handle_health() -> str:
    if HealthChecker is None:
        return "⚠️ HealthChecker не доступен (импорт не удался)."
    try:
        hc = HealthChecker(version="5.1.0")
        result = await hc.liveness()
        # run_checks would need real engines — keep liveness only for office bot
        return (
            f"🏥 *Health*\n"
            f"```json\n{json.dumps(result, indent=2, ensure_ascii=False)}\n```\n"
            f"Uptime: {result.get('uptime_seconds', 0):.1f}s"
        )
    except Exception as e:
        return f"❌ Health error: `{e}`"


async def handle_agents() -> str:
    return "🤖 *Агенты офиса:*\n\n" + _format_agents_help()


async def handle_kill(user_id: int) -> str:
    try:
        KILL_FLAG.parent.mkdir(parents=True, exist_ok=True)
        KILL_FLAG.write_text(f"killed by {user_id} at {datetime.now(timezone.utc).isoformat()}\n", encoding="utf-8")
        return (
            "🛑 *KILL активирован*\n"
            f"Флаг: `{KILL_FLAG}`\n"
            "Супервизор должен остановить торговлю при следующем тике.\n"
            "Снять: удали файл `logs/TELEGRAM_KILL.flag` и перезапусти."
        )
    except Exception as e:
        return f"❌ Не удалось создать kill-flag: `{e}`"


async def handle_agent(agent_key: str, clean_text: str, user_id: int) -> str:
    agent = get_agent(agent_key)
    if not agent:
        return f"❓ Агент `@{agent_key}` не найден. /agents — список."

    prompt = clean_text.strip() or "Привет! Что ты умеешь?"

    # Try LLM first if enabled
    try:
        from .llm_client import is_llm_available, llm_agent_reply

        if is_llm_available():
            llm_text = await llm_agent_reply(agent_key, prompt)
            if llm_text:
                # LLM already replies as agent — wrap with header
                return f"{agent.emoji} *{agent.name}*\n{llm_text}"
    except Exception:
        pass

    # Fallback — deterministic stub
    extra = ""
    if agent_key == "architect" and ("orphan" in prompt.lower() or "аудит" in prompt.lower()):
        extra = "\n\n_Подсказка: полный граф — `python -m src.telegram.handlers --audit`._"
    if agent_key == "risk" and ("просад" in prompt.lower() or "drawdown" in prompt.lower() or "риск" in prompt.lower()):
        extra = "\n\n_Лимиты: 1%/трейд, 5% экспозиция, 40% резерв (config/settings.py:RiskSettings)._"

    return (
        f"{agent.emoji} *{agent.name}* `@{agent.key}`\n"
        f"_{agent.description}_\n\n"
        f"📨 _{prompt}_\n\n"
        f"✅ Принято. Включи LLM для умных ответов: `TELEGRAM__LLM_ENABLED=true` + `OPENAI_API_KEY` в `.env`."
        f"{extra}"
    )


async def handle_office_chat(text: str, user_id: int) -> str:
    """General conversation when no agent matched — office receptionist."""
    # Try LLM receptionist
    try:
        from .llm_client import is_llm_available

        if is_llm_available():
            from .llm_client import _get_client, SYSTEM_ROUTER  # reuse client
            from .agent_registry import list_agents

            client = _get_client()
            if client:
                from config.settings import settings

                agents_desc = "\n".join(f"- {a.name} ({a.key}): {a.description}" for a in list_agents())
                resp = await client.chat.completions.create(
                    model=settings.telegram.llm_model,
                    messages=[
                        {
                            "role": "system",
                            "content": f"Ты — ресепшн офиса QuantAI. Отвечай дружелюбно, кратко, на языке пользователя. Агенты:\n{agents_desc}\nЕсли спрашивают про риски/торговлю — позови Risk. Про код — Reviewer. Про модели — ML.",
                        },
                        {"role": "user", "content": text[:2000]},
                    ],
                    temperature=0.5,
                    max_tokens=500,
                )
                t = (resp.choices[0].message.content or "").strip()
                if t:
                    return f"🏢 *Офис*\n{t}"
    except Exception:
        pass

    # Fallback — heuristics + help
    low = text.lower()
    if any(w in low for w in ["привет", "здравствуй", "хай", "hello", "hi "]):
        return "Привет! 👋 Я офис QuantAI. Спроси: `как риски?`, `проверь архитектуру`, `запусти модель` — я позову нужного агента."
    if any(w in low for w in ["как дела", "что нового", "статус", "status"]):
        return await handle_status()
    if any(w in low for w in ["кто ты", "что умеешь", "агенты", "помощь", "help"]):
        return await handle_help()
    return (
        "Понял 👍 Попробуй сказать конкретнее:\n"
        "• `как там риски и просадка?` → Risk\n"
        "• `проверь архитектуру` → Architect\n"
        "• `запусти обучение` → ML\n"
        "• `останови торговлю` → Kill\n"
        "Или напиши `/agents` — покажу всех."
    )


async def handle_office_command(command: str, rest: str, user_id: int, chat_id: int, username: str | None) -> str | None:
    """Returns response or None if not an office command."""
    cmd = command.lower()
    if cmd in ("/help", "/start"):
        return await handle_help()
    if cmd == "/status":
        return await handle_status()
    if cmd == "/health":
        return await handle_health()
    if cmd == "/agents":
        return await handle_agents()
    if cmd == "/whoami":
        return await handle_whoami(user_id, chat_id, username)
    if cmd == "/kill":
        return await handle_kill(user_id)
    return None
