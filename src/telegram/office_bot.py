#!/usr/bin/env python
"""
QuantAI Telegram Office Bot — 24/7 control plane.

Single-bot mode (default): one TOKEN, routing via @mention / /command.
Multi-bot mode: TELEGRAM__AGENT_TOKENS_JSON — each agent gets its own bot (optional).

Run locally (this PC):
    python -m src.telegram.office_bot --polling
    # or
    python src/telegram/office_bot.py

Run via Docker:
    docker compose --profile office up -d telegram-office

Env (.env):
    TELEGRAM__ENABLED=true
    TELEGRAM__TOKEN=123456:AAF...
    TELEGRAM__CHAT_ID=-100123...   # optional lock to one group
    TELEGRAM__ADMIN_IDS=123456789  # comma-separated, empty = allow all (dev)
    TELEGRAM__OFFICE_ENABLED=true
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Ensure project root on sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import settings

from src.telegram.handlers import handle_agent, handle_office_chat, handle_office_command
from src.telegram.permissions import check_access
from src.telegram.router import route_message, route_natural

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("telegram-office")

# Lazy import so dry-run without deps still works
try:
    from telegram import Update
    from telegram.constants import ParseMode
    from telegram.ext import Application, CommandHandler, MessageHandler, filters

    HAS_TG = True
except ImportError:
    HAS_TG = False
    Update = object  # type: ignore


WELCOME = (
    "🏢 *QuantAI Office Bot* онлайн\n\n"
    "Просто пиши как человеку — я сам выберу агента.\n"
    "Примеры: `как риски?`, `проверь архитектуру`, `запусти модель`, `останови все`\n"
    "Команды тоже работают: /help /status /agents /whoami"
)


async def on_start(update: Update, context) -> None:  # type: ignore
    user = update.effective_user
    chat = update.effective_chat
    ok, msg = check_access(user.id, chat.id)
    if not ok:
        await update.message.reply_text(msg)
        return
    # also handle via office handler for consistency
    text = await handle_office_command("/help", "", user.id, chat.id, user.username)
    await update.message.reply_text(WELCOME + "\n\n" + (text or ""), parse_mode=ParseMode.MARKDOWN)


async def on_help(update: Update, context) -> None:  # type: ignore
    user = update.effective_user
    chat = update.effective_chat
    ok, msg = check_access(user.id, chat.id)
    if not ok:
        await update.message.reply_text(msg)
        return
    args = " ".join(context.args) if context.args else ""
    # if user typed /help @risk etc — route
    resp = await handle_office_command("/help", args, user.id, chat.id, user.username)
    await update.message.reply_text(resp or "help error", parse_mode=ParseMode.MARKDOWN)


async def on_status(update: Update, context) -> None:  # type: ignore
    from src.telegram.handlers import handle_status

    user = update.effective_user
    chat = update.effective_chat
    ok, msg = check_access(user.id, chat.id)
    if not ok:
        await update.message.reply_text(msg)
        return
    resp = await handle_status()
    await update.message.reply_text(resp, parse_mode=ParseMode.MARKDOWN)


async def on_health(update: Update, context) -> None:  # type: ignore
    from src.telegram.handlers import handle_health

    user = update.effective_user
    chat = update.effective_chat
    ok, msg = check_access(user.id, chat.id)
    if not ok:
        await update.message.reply_text(msg)
        return
    resp = await handle_health()
    await update.message.reply_text(resp, parse_mode=ParseMode.MARKDOWN)


async def on_agents(update: Update, context) -> None:  # type: ignore
    from src.telegram.handlers import handle_agents

    user = update.effective_user
    chat = update.effective_chat
    ok, msg = check_access(user.id, chat.id)
    if not ok:
        await update.message.reply_text(msg)
        return
    resp = await handle_agents()
    await update.message.reply_text(resp, parse_mode=ParseMode.MARKDOWN)


async def on_whoami(update: Update, context) -> None:  # type: ignore
    from src.telegram.handlers import handle_whoami

    user = update.effective_user
    chat = update.effective_chat
    resp = await handle_whoami(user.id, chat.id, user.username)
    # whoami is allowed even without access — so user can discover ID
    await update.message.reply_text(resp, parse_mode=ParseMode.MARKDOWN)


async def on_kill(update: Update, context) -> None:  # type: ignore
    from src.telegram.handlers import handle_kill

    user = update.effective_user
    chat = update.effective_chat
    ok, msg = check_access(user.id, chat.id)
    if not ok:
        await update.message.reply_text(msg)
        return
    resp = await handle_kill(user.id)
    await update.message.reply_text(resp, parse_mode=ParseMode.MARKDOWN)


async def on_text(update: Update, context) -> None:  # type: ignore
    """Natural-language router — understands human speech."""
    if not update.message or not update.message.text:
        return
    text = update.message.text.strip()
    if not text:
        return
    if update.effective_user and update.effective_user.is_bot:
        return

    user = update.effective_user
    chat = update.effective_chat
    log.info("INCOMING chat=%s user=%s text=%r", chat.id, user.id, text[:120])
    ok, msg = check_access(user.id, chat.id)
    if not ok:
        log.warning("ACCESS DENIED user=%s chat=%s", user.id, chat.id)
        await update.message.reply_text(msg)
        return

    # In groups: respond to all messages if LLM enabled, else only when addressed
    # For now: always respond in private, in groups respond if natural routing finds an agent
    #         or if bot is mentioned / reply to bot.
    is_private = chat.type == "private"
    bot_username = (context.bot.username or "").lower()
    is_mentioned = bot_username and bot_username in text.lower()
    is_reply_to_bot = bool(update.message.reply_to_message and update.message.reply_to_message.from_user and update.message.reply_to_message.from_user.id == context.bot.id)

    # Use natural routing (keyword -> LLM)
    try:
        route = await route_natural(text)
    except Exception as e:
        log.exception("route_natural failed: %s", e)
        route = route_message(text)
    log.info("ROUTE text=%r -> agent=%s method=%s cmd=%s", text[:80], route.agent_key, route.method, route.command)

    # 1. Explicit command still wins
    if route.is_command and route.command:
        office_resp = await handle_office_command(route.command, route.clean_text, user.id, chat.id, user.username)
        if office_resp is not None:
            await update.message.reply_text(office_resp, parse_mode=ParseMode.MARKDOWN)
            return
        if route.agent_key:
            resp = await handle_agent(route.agent_key, route.clean_text, user.id)
            await update.message.reply_text(resp, parse_mode=ParseMode.MARKDOWN)
            return
        await update.message.reply_text(f"❓ Не понял `{route.command}`. Попробуй сказать словами: `как риски?`", parse_mode=ParseMode.MARKDOWN)
        return

    # 2. Routed to agent (mention / keyword / llm)
    if route.agent_key:
        if not is_private and route.method == "keyword" and not is_mentioned and not is_reply_to_bot:
            if route.method != "llm":
                log.info("SKIP group keyword without mention")
                return
        try:
            resp = await handle_agent(route.agent_key, route.clean_text, user.id)
        except Exception as e:
            log.exception("handle_agent failed")
            resp = f"❌ Ошибка агента {route.agent_key}: {e}"
        log.info("REPLY agent=%s len=%s", route.agent_key, len(resp))
        try:
            await update.message.reply_text(resp, parse_mode=ParseMode.MARKDOWN)
        except Exception:
            await update.message.reply_text(resp)
        return

    # 3. No agent matched -> office small talk (receptionist)
    if is_private or is_mentioned or is_reply_to_bot:
        try:
            resp = await handle_office_chat(route.clean_text, user.id)
        except Exception as e:
            log.exception("handle_office_chat failed")
            resp = f"❌ Ошибка офиса: {e}"
        log.info("REPLY office len=%s", len(resp))
        try:
            await update.message.reply_text(resp, parse_mode=ParseMode.MARKDOWN)
        except Exception:
            await update.message.reply_text(resp)
    elif route.method == "llm":
        try:
            resp = await handle_office_chat(route.clean_text, user.id)
        except Exception as e:
            log.exception("handle_office_chat llm failed")
            resp = f"❌ Ошибка: {e}"
        try:
            await update.message.reply_text(resp, parse_mode=ParseMode.MARKDOWN)
        except Exception:
            await update.message.reply_text(resp)


def build_app(token: str):
    if not HAS_TG:
        raise RuntimeError("python-telegram-bot not installed. pip install python-telegram-bot==21.*")
    app = Application.builder().token(token).build()
    # Commands
    app.add_handler(CommandHandler("start", on_start))
    app.add_handler(CommandHandler("help", on_help))
    app.add_handler(CommandHandler("status", on_status))
    app.add_handler(CommandHandler("health", on_health))
    app.add_handler(CommandHandler("agents", on_agents))
    app.add_handler(CommandHandler("whoami", on_whoami))
    app.add_handler(CommandHandler("kill", on_kill))
    # Catch-all for agent commands like /audit, /risk, /ml, /review etc — route via text handler
    # Also handle plain text with mentions
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.add_handler(MessageHandler(filters.COMMAND, on_text))
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="QuantAI Telegram Office Bot")
    parser.add_argument("--polling", action="store_true", help="Run polling (default)")
    parser.add_argument("--token", type=str, default=None, help="Override TELEGRAM__TOKEN")
    args = parser.parse_args()

    token = args.token or settings.telegram.token
    if not token:
        log.error("TELEGRAM__TOKEN пустой. Задай в .env: TELEGRAM__TOKEN=123:ABC")
        # Dry-run mode — validate imports without token
        log.info("Dry-run: imports OK, router OK, handlers OK. Укажи токен для запуска.")
        sys.exit(2)

    if not HAS_TG:
        log.error("Зависимость не установлена: pip install python-telegram-bot==21.*")
        sys.exit(3)

    log.info("Starting Telegram Office… chat_id=%s admins=%s", settings.telegram.chat_id or "any", settings.telegram.admin_id_list or "any")
    app = build_app(token)
    # Blocking polling — for 24/7 on this PC
    app.run_polling(
        allowed_updates=Update.ALL_TYPES if HAS_TG else None,
        drop_pending_updates=True,
    )


if __name__ == "__main__":
    main()
