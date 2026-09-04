"""
QuantAI Telegram Office — public package exports.
"""

from .agent_registry import AGENTS, get_agent, list_agents
from .permissions import is_admin, check_access
from .router import route_message, parse_mention

__all__ = ["AGENTS", "get_agent", "list_agents", "is_admin", "check_access", "route_message", "parse_mention"]
