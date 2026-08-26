"""
====================================================
QuantAI Professional
Structured Logging Configuration
====================================================

JSON logging with correlation IDs, context enrichment,
and structured output for log aggregation.
====================================================
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

# Context variables for correlation tracking
correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")
user_id_var: ContextVar[str] = ContextVar("user_id", default="")
session_id_var: ContextVar[str] = ContextVar("session_id", default="")
request_id_var: ContextVar[str] = ContextVar("request_id", default="")


@dataclass
class LogContext:
    """Structured log context for correlation."""
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    user_id: str = ""
    session_id: str = ""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    component: str = ""
    operation: str = ""
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "correlation_id": self.correlation_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "component": self.component,
            "operation": self.operation,
            **self.metadata,
        }
    
    def set_contextvars(self):
        """Set context variables for automatic inclusion."""
        correlation_id_var.set(self.correlation_id)
        user_id_var.set(self.user_id)
        session_id_var.set(self.session_id)
        request_id_var.set(self.request_id)


class CorrelationFilter(logging.Filter):
    """Add correlation context to all log records."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = correlation_id_var.get("")
        record.user_id = user_id_var.get("")
        record.session_id = session_id_var.get("")
        record.request_id = request_id_var.get("")
        return True


class JSONFormatter(logging.Formatter):
    """JSON log formatter with structured fields."""
    
    def __init__(self, include_extra: bool = True):
        super().__init__()
        self.include_extra = include_extra
    
    def format(self, record: logging.LogRecord) -> str:
        # Base log entry
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Correlation IDs
        if record.correlation_id:
            log_entry["correlation_id"] = record.correlation_id
        if record.user_id:
            log_entry["user_id"] = record.user_id
        if record.session_id:
            log_entry["session_id"] = record.session_id
        if record.request_id:
            log_entry["request_id"] = record.request_id
        
        # Exception info
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Extra fields
        if self.include_extra:
            extra_fields = {}
            for key, value in record.__dict__.items():
                if key not in {
                    "name", "msg", "args", "created", "filename", "funcName",
                    "levelname", "levelno", "lineno", "module", "msecs",
                    "message", "name", "pathname", "process", "processName",
                    "relativeCreated", "thread", "threadName", "exc_info",
                    "exc_text", "stack_info", "correlation_id", "user_id",
                    "session_id", "request_id"
                }:
                    extra_fields[key] = value
            if extra_fields:
                log_entry["extra"] = extra_fields
        
        return json.dumps(log_entry, ensure_ascii=False)


def setup_logging(
    level: str = "INFO",
    json_format: bool = True,
    log_file: Optional[str] = None,
    include_extra: bool = True,
) -> logging.Logger:
    """
    Configure structured logging for the application.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: Use JSON formatter
        log_file: Optional file path for file logging
        include_extra: Include extra fields in JSON output
    
    Returns:
        Root logger
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Add correlation filter
    correlation_filter = CorrelationFilter()
    root_logger.addFilter(correlation_filter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    
    if json_format:
        console_handler.setFormatter(JSONFormatter(include_extra=include_extra))
    else:
        console_handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        ))
    console_handler.addFilter(correlation_filter)
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(JSONFormatter(include_extra=include_extra))
        file_handler.addFilter(correlation_filter)
        root_logger.addHandler(file_handler)
    
    # Reduce noise from third-party loggers
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("ccxt").setLevel(logging.WARNING)
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get logger with correlation context support."""
    return logging.getLogger(name)


def log_with_context(
    logger: logging.Logger,
    level: int,
    message: str,
    context: Optional[LogContext] = None,
    **kwargs,
):
    """Log message with structured context."""
    if context:
        context.set_contextvars()
        extra = context.to_dict()
        extra.update(kwargs)
    else:
        extra = kwargs
    
    logger.log(level, message, extra=extra)


# Convenience functions
def log_debug(logger: logging.Logger, message: str, **kwargs):
    log_with_context(logger, logging.DEBUG, message, **kwargs)

def log_info(logger: logging.Logger, message: str, **kwargs):
    log_with_context(logger, logging.INFO, message, **kwargs)

def log_warning(logger: logging.Logger, message: str, **kwargs):
    log_with_context(logger, logging.WARNING, message, **kwargs)

def log_error(logger: logging.Logger, message: str, **kwargs):
    log_with_context(logger, logging.ERROR, message, **kwargs)

def log_critical(logger: logging.Logger, message: str, **kwargs):
    log_with_context(logger, logging.CRITICAL, message, **kwargs)


# Context managers for correlation
class correlation_context:
    """Context manager for correlation ID."""
    
    def __init__(self, correlation_id: Optional[str] = None, **kwargs):
        self.correlation_id = correlation_id or str(uuid.uuid4())[:8]
        self.context = LogContext(correlation_id=self.correlation_id, **kwargs)
        self.token = None
    
    def __enter__(self):
        self.context.set_contextvars()
        return self.context
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        correlation_id_var.set("")
        user_id_var.set("")
        session_id_var.set("")
        request_id_var.set("")


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "LogContext",
    "CorrelationFilter",
    "JSONFormatter",
    "setup_logging",
    "get_logger",
    "log_with_context",
    "log_debug",
    "log_info",
    "log_warning",
    "log_error",
    "log_critical",
    "correlation_context",
    "correlation_id_var",
    "user_id_var",
    "session_id_var",
    "request_id_var",
]