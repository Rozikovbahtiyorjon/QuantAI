"""
LLM client for natural-language routing + agent replies.

Free options:
- ollama (local, no key, no quota) — LLM_PROVIDER=ollama, model=qwen2.5:3b, base_url=http://localhost:11434/v1
- groq (cloud free tier) — LLM_PROVIDER=groq, GROQ_API_KEY=gsk_..., model=llama-3.1-8b-instant
- openai (paid) — LLM_PROVIDER=openai, OPENAI_API_KEY=sk-...

Security: keys read ONLY from env/.env, never from chat, never logged.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

log = logging.getLogger("telegram-office.llm")

try:
    from openai import AsyncOpenAI  # type: ignore

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    AsyncOpenAI = None  # type: ignore

from config.settings import settings
from .agent_registry import AGENTS

SYSTEM_ROUTER = """Ты — ресепшн виртуального офиса QuantAI. Твоя задача — по сообщению пользователя выбрать одного агента.

Агенты:
{agents}

Правила:
- Ответь СТРОГО JSON: {{"agent": "risk"|"architect"|"ml"|"qa"|"paper"|"prod"|"intel"|"config"|"execution"|"entrypoint"|"git"|"reviewer"|"office", "reason": "коротко почему"}}
- "office" — если вопрос общий, приветствие, или не подходит никому.
- Не выдумывай агентов вне списка.
- Учитывай синонимы на русском и английском.
"""

SYSTEM_AGENT = """Ты — {name} ({key}) виртуального офиса QuantAI.
Твоя роль: {description}
Файл с инструкциями: {md_file}

Отвечай как этот агент: кратко, по делу, на языке пользователя.
Если запрос вне твоей зоны — скажи честно и предложи позвать другого агента.
Не выдумывай цифры портфеля — если данных нет, скажи что проверишь.
"""


def _get_provider() -> str:
    try:
        return (settings.telegram.llm_provider or "openai").lower().strip()  # type: ignore[attr-defined]
    except Exception:
        return os.getenv("TELEGRAM__LLM_PROVIDER", "openai").lower().strip()


def _get_client() -> Optional["AsyncOpenAI"]:
    if not HAS_OPENAI:
        return None
    provider = _get_provider()

    # Ollama — local, no key needed
    if provider == "ollama":
        base = (settings.telegram.llm_base_url or os.getenv("TELEGRAM__LLM_BASE_URL", "") or "http://localhost:11434/v1").strip()  # type: ignore[attr-defined]
        try:
            return AsyncOpenAI(base_url=base, api_key="ollama")
        except Exception as e:
            log.warning("Ollama client init failed: %s", e)
            return None

    # Groq — OpenAI-compatible
    if provider == "groq":
        key = ""
        try:
            key = (settings.telegram.groq_api_key or "").strip()  # type: ignore[attr-defined]
        except Exception:
            pass
        if not key:
            key = os.getenv("GROQ_API_KEY", "").strip()
            if not key:
                # also check .env manually
                try:
                    from pathlib import Path

                    for line in Path(".env").read_text(encoding="utf-8").splitlines():
                        if line.strip().startswith("GROQ_API_KEY="):
                            key = line.split("=", 1)[1].strip().strip('"').strip("'")
                            break
                except Exception:
                    pass
        if not key:
            return None
        base = (settings.telegram.llm_base_url or "https://api.groq.com/openai/v1").strip()  # type: ignore[attr-defined]
        try:
            return AsyncOpenAI(base_url=base, api_key=key)
        except Exception as e:
            log.warning("Groq client init failed: %s", e)
            return None

    # OpenAI — default
    try:
        key = settings.telegram.get_llm_key().strip()  # type: ignore[attr-defined]
    except Exception:
        key = ""
    if not key:
        key = os.getenv("OPENAI_API_KEY", "").strip() or os.getenv("TELEGRAM__LLM_API_KEY", "").strip()
    if not key:
        return None
    try:
        base = (settings.telegram.llm_base_url or "").strip()  # type: ignore[attr-defined]
        if base:
            return AsyncOpenAI(base_url=base, api_key=key)
        return AsyncOpenAI(api_key=key)
    except Exception as e:
        log.warning("LLM client init failed: %s", e)
        return None


def is_llm_available() -> bool:
    if not settings.telegram.llm_enabled:  # type: ignore[attr-defined]
        return False
    if not HAS_OPENAI:
        log.warning("LLM enabled but openai not installed: pip install openai")
        return False
    provider = _get_provider()
    if provider == "ollama":
        # check ollama reachable (quick)
        return True
    if provider == "groq":
        try:
            k = (settings.telegram.groq_api_key or "").strip()  # type: ignore[attr-defined]
        except Exception:
            k = ""
        if not k:
            k = os.getenv("GROQ_API_KEY", "").strip()
        return bool(k)
    # openai
    try:
        key = settings.telegram.get_llm_key().strip()  # type: ignore[attr-defined]
    except Exception:
        key = ""
    if not key:
        key = os.getenv("OPENAI_API_KEY", "").strip() or os.getenv("TELEGRAM__LLM_API_KEY", "").strip()
    return bool(key)


async def llm_route(text: str) -> tuple[Optional[str], str]:
    client = _get_client()
    if not client:
        return None, "llm unavailable"
    agents_desc = "\n".join(f"- {k}: {v.name} — {v.description} (примеры: {', '.join(v.commands)})" for k, v in AGENTS.items())
    system = SYSTEM_ROUTER.format(agents=agents_desc)
    model = settings.telegram.llm_model  # type: ignore[attr-defined]
    # default models per provider
    provider = _get_provider()
    if provider == "ollama" and model == "gpt-4o-mini":
        model = "qwen2.5:3b"
    if provider == "groq" and model == "gpt-4o-mini":
        model = "llama-3.1-8b-instant"
    try:
        kwargs: dict = dict(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": text[:2000]},
            ],
            temperature=0.1,
            max_tokens=80,
        )
        # ollama/groq may not support response_format json_object reliably — keep without for ollama
        if provider == "openai":
            kwargs["response_format"] = {"type": "json_object"}  # type: ignore
        resp = await client.chat.completions.create(**kwargs)  # type: ignore[arg-type]
        import json

        raw = resp.choices[0].message.content or "{}"
        # try to extract JSON even if model added text
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end > start:
            raw = raw[start:end]
        data = json.loads(raw)
        agent = str(data.get("agent", "office")).lower().strip()
        reason = str(data.get("reason", ""))[:120]
        if agent not in AGENTS and agent != "office":
            agent = "office"
        return agent, reason
    except Exception as e:
        log.warning("LLM route failed: %s", e)
        return None, f"error: {e}"


async def llm_agent_reply(agent_key: str, user_text: str) -> Optional[str]:
    client = _get_client()
    if not client:
        return None
    from .agent_registry import get_agent

    agent = get_agent(agent_key)
    if not agent:
        return None
    md_hint = ""
    try:
        from pathlib import Path

        p = Path(agent.md_file)
        if p.exists():
            md_hint = p.read_text(encoding="utf-8")[:4000]
    except Exception:
        pass
    system = SYSTEM_AGENT.format(name=agent.name, key=agent.key, description=agent.description, md_file=agent.md_file)
    if md_hint:
        system += f"\n\nКонтекст из {agent.md_file} (сокращено):\n{md_hint[:2000]}"
    model = settings.telegram.llm_model  # type: ignore[attr-defined]
    provider = _get_provider()
    if provider == "ollama" and model == "gpt-4o-mini":
        model = "qwen2.5:3b"
    if provider == "groq" and model == "gpt-4o-mini":
        model = "llama-3.1-8b-instant"
    try:
        resp = await client.chat.completions.create(  # type: ignore[arg-type]
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_text[:3000]},
            ],
            temperature=0.4,
            max_tokens=700,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        log.warning("LLM agent reply failed (%s): %s", agent_key, e)
        return None
