"""
Permissions — whitelist for Telegram Office.

- If TELEGRAM__ADMIN_IDS is empty → allow everyone (dev mode on local PC).
- If set → only those IDs can run commands / talk to agents.
- Group chat_id lockdown optional.
"""

from __future__ import annotations

import logging
import os

from config.settings import settings

logger = logging.getLogger(__name__)


def _is_production() -> bool:
    """Detect production/live mode — fail-closed required."""
    # BINANCE_TESTNET=false → live production
    testnet = os.getenv("BINANCE_TESTNET", "true").strip().lower()
    if testnet in ("false", "0", "no", "live", "production"):
        return True
    env = os.getenv("ENV", "").strip().lower()
    if env in ("production", "prod", "live"):
        return True
    # Explicit flag to require admin lockdown
    if os.getenv("TELEGRAM__REQUIRE_ADMIN", "").strip().lower() in ("1", "true", "yes"):
        return True
    return False


def is_admin(user_id: int) -> bool:
    admin_ids = settings.telegram.admin_id_list
    if not admin_ids:
        # SECURITY (Task 11): No admin fallback in production — fail-closed.
        # Empty ADMIN_IDS is allowed ONLY in dev/testnet with explicit opt-in.
        if _is_production():
            logger.warning(
                "TELEGRAM__ADMIN_IDS not set in production — denying admin access (fail-closed). "
                "Set TELEGRAM__ADMIN_IDS to restrict bot access."
            )
            return False
        # Dev/testnet: open mode but warn; can be locked via TELEGRAM__ALLOW_OPEN_MODE=false
        allow_open = os.getenv("TELEGRAM__ALLOW_OPEN_MODE", "true").strip().lower()
        if allow_open in ("false", "0", "no"):
            logger.warning("TELEGRAM__ADMIN_IDS empty but TELEGRAM__ALLOW_OPEN_MODE=false — denying access.")
            return False
        logger.warning(
            "TELEGRAM__ADMIN_IDS empty — open mode (dev/testnet only). "
            "Set TELEGRAM__ADMIN_IDS for production; will fail-closed in production."
        )
        return True  # open mode for local dev/testnet
    return user_id in admin_ids


def is_allowed_chat(chat_id: int) -> bool:
    """If TELEGRAM__CHAT_ID is set, lock to that group."""
    configured = settings.telegram.chat_id.strip()
    if not configured:
        return True
    try:
        return str(chat_id) == configured.strip()
    except Exception:
        return False


def check_access(user_id: int, chat_id: int) -> tuple[bool, str]:
    if not is_allowed_chat(chat_id):
        return False, f"⛔ Чат {chat_id} не в whitelist. Проверь TELEGRAM__CHAT_ID."
    if not is_admin(user_id):
        return False, "⛔ Нет доступа. Твой ID не в TELEGRAM__ADMIN_IDS."
    return True, "ok"
