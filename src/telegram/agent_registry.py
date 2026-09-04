"""
Agent registry — maps Telegram mentions to QuantAI AGENTS/*.md.

Single-bot mode: mentions like @risk, @architect are routed inside one bot.
Multi-bot mode: each agent can have its own TOKEN (via TELEGRAM__AGENT_TOKENS_JSON).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentDef:
    key: str  # short key used in @mention
    name: str  # display name
    emoji: str
    md_file: str  # AGENTS file
    description: str
    commands: tuple[str, ...]  # commands this agent handles


AGENTS: dict[str, AgentDef] = {
    "architect": AgentDef(
        key="architect",
        name="Architecture Auditor",
        emoji="🏛️",
        md_file="AGENTS/01-architecture-auditor.md",
        description="Граф зависимостей, orphan-модули, дубли",
        commands=("/audit", "/graph", "/orphans", "/architect", "/architecture"),
    ),
    "config": AgentDef(
        key="config",
        name="Config Fixer",
        emoji="🔧",
        md_file="AGENTS/02-config-dependency-fixer.md",
        description="Фиксы config / namespace",
        commands=("/config", "/fix-config"),
    ),
    "execution": AgentDef(
        key="execution",
        name="Execution Engineer",
        emoji="⚡",
        md_file="AGENTS/03-execution-boundary-engineer.md",
        description="Границы execution, лимиты",
        commands=("/execution", "/limits"),
    ),
    "entrypoint": AgentDef(
        key="entrypoint",
        name="Entrypoint Engineer",
        emoji="🚪",
        md_file="AGENTS/04-entrypoint-engineer.md",
        description="Entrypoint / lifecycle",
        commands=("/entrypoint",),
    ),
    "risk": AgentDef(
        key="risk",
        name="Risk Manager",
        emoji="🛡️",
        md_file="AGENTS/05-risk-integration-engineer.md",
        description="Риски 1%/3%/5%/40%, 3-5-7",
        commands=("/risk", "/risk-matrix", "/kill", "/stress-test", "/exposure"),
    ),
    "qa": AgentDef(
        key="qa",
        name="QA Gate",
        emoji="✅",
        md_file="AGENTS/06-qa-gate.md",
        description="Quality gate, тесты",
        commands=("/qa", "/test", "/gate"),
    ),
    "paper": AgentDef(
        key="paper",
        name="Paper Validator",
        emoji="📄",
        md_file="AGENTS/07-paper-trading-validator.md",
        description="Paper trading валидация",
        commands=("/paper", "/validate-paper"),
    ),
    "ml": AgentDef(
        key="ml",
        name="ML WalkForward",
        emoji="🤖",
        md_file="AGENTS/08-ml-walkforward-engineer.md",
        description="ML, Walk-Forward, PurgedKFold",
        commands=("/ml", "/wf", "/train"),
    ),
    "intel": AgentDef(
        key="intel",
        name="Market Intel",
        emoji="📊",
        md_file="AGENTS/09-market-intelligence-evaluator.md",
        description="Market intelligence оценка",
        commands=("/intel", "/market"),
    ),
    "prod": AgentDef(
        key="prod",
        name="Production Readiness",
        emoji="🚀",
        md_file="AGENTS/10-production-readiness-engineer.md",
        description="Готовность к проду",
        commands=("/prod", "/readiness"),
    ),
    "git": AgentDef(
        key="git",
        name="Git Checkpoint",
        emoji="📦",
        md_file="AGENTS/11-git-checkpoint-manager.md",
        description="Git чекпоинты",
        commands=("/git", "/checkpoint"),
    ),
    "reviewer": AgentDef(
        key="reviewer",
        name="Code Reviewer",
        emoji="🔍",
        md_file="AGENTS/12-code-reviewer.md",
        description="Code review",
        commands=("/review", "/code-review"),
    ),
}

# Aliases — what user can type
_ALIASES: dict[str, str] = {
    # risk
    "risk": "risk",
    "risk-manager": "risk",
    "risk_manager": "risk",
    "риск": "risk",
    # architect
    "arch": "architect",
    "architect": "architect",
    "архитектор": "architect",
    # ml
    "ml": "ml",
    "walkforward": "ml",
    "walk-forward": "ml",
    # qa
    "qa": "qa",
    "gate": "qa",
    # paper
    "paper": "paper",
    # prod
    "prod": "prod",
    "production": "prod",
    # intel
    "intel": "intel",
    "intelligence": "intel",
    # others
    "config": "config",
    "execution": "execution",
    "entrypoint": "entrypoint",
    "git": "git",
    "reviewer": "reviewer",
    "review": "reviewer",
}


def get_agent(key: str) -> AgentDef | None:
    key = key.lower().strip().lstrip("@")
    canonical = _ALIASES.get(key, key)
    return AGENTS.get(canonical)


def list_agents() -> list[AgentDef]:
    return list(AGENTS.values())


def mention_to_key(text: str) -> str | None:
    """Extract first @mention as agent key."""
    import re

    m = re.search(r"@([a-zA-Z0-9_\-]+)", text)
    if not m:
        return None
    raw = m.group(1).lower()
    # strip bot suffix like @risk_bot
    raw = raw.replace("_bot", "").replace("bot", "")
    raw = raw.strip("_-")
    return _ALIASES.get(raw, raw) if raw in _ALIASES or raw in AGENTS else None
