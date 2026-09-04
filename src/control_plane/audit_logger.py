"""
QuantAI Audit Logger
Centralized audit logging for all supervisor activities
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .supervisor import SupervisorState


class AuditLevel(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEntry:
    """Single audit log entry"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    level: AuditLevel = AuditLevel.INFO
    message: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    source: str = "supervisor"
    correlation_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "message": self.message,
            "context": self.context,
            "source": self.source,
            "correlation_id": self.correlation_id
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


class AuditLogger:
    """
    Centralized audit logging for the AI Supervisor.
    Provides structured, queryable audit trail of all supervisor activities.
    """

    def __init__(
        self,
        log_path: str = "logs/audit",
        max_file_size_mb: int = 100,
        max_files: int = 10,
        buffer_size: int = 100
    ):
        self.log_path = Path(log_path)
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.max_file_size_mb = max_file_size_mb
        self.max_files = max_files
        self.buffer_size = buffer_size

        self._buffer: List[AuditEntry] = []
        self._buffer_lock = asyncio.Lock()
        self._flush_task: Optional[asyncio.Task] = None
        self._running = False
        self._current_file: Optional[Path] = None
        self._current_file_size = 0
        self._file_count = 0

        # Correlate related entries
        self._correlation_context: Optional[str] = None

    async def start(self) -> None:
        """Start the audit logger"""
        self._running = True
        self._flush_task = asyncio.create_task(self._flush_loop())
        await self._open_new_log_file()

    async def stop(self) -> None:
        """Stop the audit logger"""
        self._running = False
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        await self._flush_buffer()
        await self._close_current_file()

    async def _flush_loop(self) -> None:
        """Periodically flush buffer to disk"""
        while True:
            try:
                await asyncio.sleep(5)  # Flush every 5 seconds
                await self._flush_buffer()
            except asyncio.CancelledError:
                break
            except Exception:
                pass

    async def _open_new_log_file(self) -> None:
        """Open a new log file"""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._current_file = self.log_path / f"audit_{timestamp}.jsonl"
        self._current_file_size = 0
        self._file_count += 1

        # Rotate if too many files
        files = sorted(self.log_path.glob("audit_*.jsonl"))
        while len(list(self.log_path.glob("audit_*.jsonl"))) >= 10:
            oldest = min(self.log_path.glob("audit_*.jsonl"), key=lambda f: f.stat().st_mtime)
            try:
                oldest.unlink()
            except Exception:
                pass

    async def _close_current_file(self) -> None:
        self._current_file = None
        self._current_file_size = 0

    async def _flush_buffer(self) -> None:
        """Flush buffer to disk"""
        if not self._buffer:
            return

        async with asyncio.Lock():
            if not self._buffer:
                return

            entries = self._buffer[:]
            self._buffer.clear()

            if self._current_file:
                try:
                    with open(self._current_file, 'a', encoding='utf-8') as f:
                        for entry in entries:
                            f.write(entry.to_json() + '\n')
                        self._current_file_size += sum(len(e.to_json()) + 1 for e in entries)

                        # Check if file is too large
                        if self._current_file_size > self.max_file_size_mb * 1024 * 1024:
                            await self._rotate_log_file()

                        return
                except Exception as e:
                    # Log error but don't fail
                    print(f"Error writing to audit log: {e}")

            # If no current file, open new one
            await self._open_new_log_file()
            await self._flush_buffer()

    async def _rotate_log_file(self) -> None:
        """Rotate to new log file"""
        await self._close_current_file()
        await self._open_new_log_file()

    def log(
        self,
        level: AuditLevel,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        source: str = "supervisor",
        correlation_id: Optional[str] = None
    ) -> None:
        """Log an audit entry"""
        entry = AuditEntry(
            level=level,
            message=message,
            context=context or {},
            source=source,
            correlation_id=correlation_id or self._correlation_context
        )

        self._buffer.append(entry)

    def debug(self, message: str, **kwargs) -> None:
        self.log(AuditLevel.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs) -> None:
        self.log(AuditLevel.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        self.log(AuditLevel.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs) -> None:
        self.log(AuditLevel.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs) -> None:
        self.log(AuditLevel.CRITICAL, message, **kwargs)

    def set_correlation_id(self, correlation_id: str) -> None:
        """Set correlation ID for related log entries"""
        self._correlation_context = correlation_id

    def clear_correlation_id(self) -> None:
        self._correlation_context = None

    async def _open_new_log_file(self) -> None:
        """Open a new log file"""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._current_file = self.log_path / f"audit_{timestamp}.jsonl"
        self._current_file_size = 0
        self._file_count += 1

        # Rotate if too many files
        files = sorted(self.log_path.glob("audit_*.jsonl"))
        while len(list(self.log_path.glob("audit_*.jsonl"))) >= 10:
            oldest = min(self.log_path.glob("audit_*.jsonl"), key=lambda f: f.stat().st_mtime)
            try:
                oldest.unlink()
            except Exception:
                pass

    async def _close_current_file(self) -> None:
        self._current_file = None
        self._current_file_size = 0

    async def _flush_buffer(self) -> None:
        """Flush buffer to disk"""
        if not self._buffer:
            return

        async with asyncio.Lock():
            if not self._buffer:
                return

            entries = self._buffer[:]
            self._buffer.clear()

            if self._current_file:
                try:
                    with open(self._current_file, 'a', encoding='utf-8') as f:
                        for entry in entries:
                            f.write(entry.to_json() + '\n')
                        self._current_file_size += sum(len(e.to_json()) + 1 for e in entries)

                        # Check if file is too large
                        if self._current_file_size > self.max_file_size_mb * 1024 * 1024:
                            await self._rotate_log_file()

                        return
                except Exception as e:
                    # Log error but don't fail
                    print(f"Error writing to audit log: {e}")

            # If no current file, open new one
            await self._open_new_log_file()
            await self._flush_buffer()

    async def _rotate_log_file(self) -> None:
        """Rotate to new log file"""
        await self._close_current_file()
        await self._open_new_log_file()

    def log(
        self,
        level: AuditLevel,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        source: str = "supervisor",
        correlation_id: Optional[str] = None
    ) -> None:
        """Log an audit entry"""
        entry = AuditEntry(
            level=level,
            message=message,
            context=context or {},
            source=source,
            correlation_id=correlation_id or self._correlation_context
        )

        self._buffer.append(entry)

    def debug(self, message: str, **kwargs) -> None:
        self.log(AuditLevel.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs) -> None:
        self.log(AuditLevel.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        self.log(AuditLevel.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs) -> None:
        self.log(AuditLevel.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs) -> None:
        self.log(AuditLevel.CRITICAL, message, **kwargs)

    def set_correlation_id(self, correlation_id: str) -> None:
        """Set correlation ID for related log entries"""
        self._correlation_context = correlation_id

    def clear_correlation_id(self) -> None:
        self._correlation_context = None

    async def query_logs(
        self,
        level: Optional[AuditLevel] = None,
        source: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Query audit logs"""
        results = []

        # Search current file
        if self._current_file and self._current_file.exists():
            try:
                with open(self._current_file, 'r') as f:
                    for line in f:
                        entry = json.loads(line)
                        if self._matches_filter(entry, level, source, start_time, end_time):
                            results.append(entry)
                            if len(results) >= limit:
                                return results
            except Exception:
                pass

        # Search rotated files
        for log_file in sorted(self.log_path.glob("audit_*.jsonl"), key=lambda f: f.stat().st_mtime, reverse=True):
            if len(results) >= limit:
                break
            try:
                with open(log_file, 'r') as f:
                    for line in f:
                        entry = json.loads(line)
                        if self._matches_filter(entry, level, source, start_time, end_time):
                            results.append(entry)
                            if len(results) >= limit:
                                return results
            except Exception:
                pass

        return results

    def _matches_filter(
        self,
        entry: Dict[str, Any],
        level: Optional[AuditLevel],
        source: Optional[str],
        start_time: Optional[datetime],
        end_time: Optional[datetime]
    ) -> bool:
        if level and entry.get("level") != level.value:
            return False
        if source and entry.get("source") != source:
            return False
        if start_time:
            entry_time = datetime.fromisoformat(entry["timestamp"])
            if entry_time < start_time:
                return False
        if end_time:
            entry_time = datetime.fromisoformat(entry["timestamp"])
            if entry_time > end_time:
                return False
        return True

    def get_correlation_trail(self, correlation_id: str) -> List[Dict[str, Any]]:
        """Get all log entries for a correlation ID"""
        return [e for e in self.query_logs() if e.get("correlation_id") == correlation_id]


__all__ = [
    "AuditLevel",
    "AuditEntry",
    "AuditLogger",
]