"""
Router — natural-language first, commands second.

Priority:
1. /command  -> deterministic (still works, but not required)
2. @mention -> deterministic
3. Natural language -> keyword heuristics (no LLM) -> LLM classifier (if enabled)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .agent_registry import AGENTS, get_agent, mention_to_key


@dataclass
class RouteResult:
    agent_key: str | None  # e.g. "risk", "ml", None=office/general
    command: str | None  # e.g. "/risk", "/status"
    clean_text: str
    is_command: bool
    method: str  # "command" | "mention" | "keyword" | "llm" | "office"


_COMMAND_TO_AGENT: dict[str, str] = {}
for _k, _a in AGENTS.items():
    for _c in _a.commands:
        _COMMAND_TO_AGENT[_c] = _k

_OFFICE_COMMANDS = {"/start", "/help", "/status", "/health", "/agents", "/whoami", "/kill"}

# Keyword heuristics for natural language (works without LLM)
_KEYWORDS: dict[str, list[str]] = {
    "risk": ["риск", "просад", "drawdown", "экспозиц", "exposure", "лимит", "limit", "стоп", "stop", "ликвид", "margin", "плеч", "убыток", "sl ", "tp ", "3-5-7"],
    "architect": ["архитектур", "orphan", "зависимост", "dependency", "граф", "дубль", "duplicate", "импорт", "import", "структур"],
    "ml": ["модель", "model", "обучи", "train", "walk-forward", "walk forward", "wf ", "прогноз", "predict", "фич", "feature", "переобуч"],
    "qa": ["тест", "test", "qa ", "gate", "проверь код", "coverage"],
    "paper": ["paper", "бумаг", "демо", "симуляц"],
    "prod": ["продакш", "production", "депло", "deploy", "готовность", "readiness", "релиз"],
    "intel": ["рынок", "market", "сигнал", "signal", "тренд", "волатильн", "объем", "volume"],
    "reviewer": ["ревью", "review", "код ревью", "проверь файл", "линт"],
    "git": ["git", "коммит", "checkpoint", "чекпоинт", "ветк"],
    "config": ["конфиг", "config", "env ", ".env", "настройк"],
    "execution": ["исполнен", "execution", "ордер", "order", "проскальз", "slippage", "maker", "taker"],
    "entrypoint": ["entrypoint", "lifecycle", "запуск", "startup"],
}


def parse_mention(text: str) -> str | None:
    return mention_to_key(text)


def _keyword_route(text: str) -> str | None:
    low = text.lower()
    scores: dict[str, int] = {}
    for k, words in _KEYWORDS.items():
        s = sum(1 for w in words if w in low)
        if s:
            scores[k] = s
    if not scores:
        return None
    # highest score wins; ties -> first
    return max(scores, key=lambda k: scores[k])  # type: ignore


def route_message(text: str) -> RouteResult:
    if not text:
        return RouteResult(None, None, "", False, "office")
    t = text.strip()

    # 1. Slash command (still supported, but not required)
    m = re.match(r"^(/[a-zA-Z0-9_\-]+)(?:@\w+)?\b\s*(.*)", t, re.DOTALL)
    if m:
        cmd = m.group(1).lower()
        rest = (m.group(2) or "").strip()
        if cmd in _OFFICE_COMMANDS:
            return RouteResult(None, cmd, rest, True, "command")
        agent_key = _COMMAND_TO_AGENT.get(cmd)
        if agent_key:
            return RouteResult(agent_key, cmd, rest, True, "command")
        return RouteResult(None, cmd, rest, True, "command")

    # 2. @mention
    key = mention_to_key(t)
    if key and key in AGENTS:
        clean = re.sub(r"@\w+\b", "", t, count=1).strip(" ,:—-")
        return RouteResult(key, None, clean, False, "mention")

    # 3. Keyword heuristics (no LLM needed)
    kw = _keyword_route(t)
    if kw:
        return RouteResult(kw, None, t, False, "keyword")

    # 4. Fallback -> office will try LLM if enabled, else general reply
    return RouteResult(None, None, t, False, "office")


async def route_natural(text: str) -> RouteResult:
    """
    Natural-language routing: keyword first, then LLM if available.
    Call this from office_bot instead of route_message when you want LLM.
    """
    base = route_message(text)
    # command/mention/keyword already resolved -> return
    if base.agent_key is not None or base.method in ("command", "mention", "keyword"):
        return base
    # office fallback -> try LLM
    try:
        from .llm_client import is_llm_available, llm_route

        if is_llm_available():
            agent, _reason = await llm_route(text)
            if agent and agent != "office" and agent in AGENTS:
                return RouteResult(agent, None, text, False, "llm")
    except Exception:
        pass
    return base
