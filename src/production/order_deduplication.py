"""
====================================================
QuantAI Professional
Order Deduplication & Idempotency
====================================================

Prevents duplicate order submissions and ensures
exactly-once execution semantics.

Features:
- ClientOrderId tracking with TTL
- Idempotency key generation
- Duplicate detection with TTL cleanup
- Race condition prevention
- Audit trail for all order attempts

====================================================
"""

from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, Optional, Set

import asyncio


class OrderStatus(str, Enum):
    """Order status for deduplication tracking."""
    PENDING = "PENDING"          # Submitted, awaiting response
    ACCEPTED = "ACCEPTED"        # Accepted by exchange
    REJECTED = "REJECTED"        # Rejected by exchange
    FILLED = "FILLED"            # Fully filled
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELED = "CANCELED"        # Canceled by user
    EXPIRED = "EXPIRED"          # Expired (TTL)
    DUPLICATE = "DUPLICATE"      # Detected as duplicate


@dataclass(frozen=True)
class OrderAttempt:
    """Record of an order submission attempt."""
    client_order_id: str
    idempotency_key: str
    symbol: str
    side: str
    quantity: float
    price: Optional[float]
    order_type: str
    timestamp: datetime
    status: OrderStatus = OrderStatus.PENDING
    exchange_order_id: Optional[str] = None
    error_message: Optional[str] = None
    retry_count: int = 0


class IdempotencyKeyGenerator:
    """
    Generates deterministic idempotency keys for order requests.
    
    Key is derived from order parameters to ensure
    identical orders produce the same key.
    """

    @staticmethod
    def generate(
        symbol: str,
        side: str,
        quantity: float,
        price: Optional[float],
        order_type: str,
        timestamp: Optional[datetime] = None,
    ) -> str:
        """
        Generate deterministic idempotency key.
        
        Key components:
        - Symbol
        - Side (BUY/SELL)
        - Quantity (rounded to 8 decimals)
        - Price (if provided, rounded to 8 decimals)
        - Order type
        - Minute-precision timestamp (for time-bound idempotency)
        
        Args:
            symbol: Trading symbol
            side: BUY or SELL
            quantity: Order quantity
            price: Order price (None for market orders)
            order_type: Order type (LIMIT, MARKET, etc.)
            timestamp: Optional timestamp (uses current time if not provided)
            
        Returns:
            SHA256 hash as idempotency key
        """
        ts = timestamp or datetime.now(timezone.utc)
        # Round to minute for time-bound idempotency (1-minute window)
        minute_bucket = ts.replace(second=0, microsecond=0)
        
        # Build deterministic string
        parts = [
            symbol.upper(),
            side.upper(),
            f"{quantity:.8f}",
            f"{price:.8f}" if price is not None else "MARKET",
            order_type.upper(),
            minute_bucket.isoformat(),
        ]
        
        key_string = "|".join(parts)
        return hashlib.sha256(key_string.encode()).hexdigest()[:32]


class OrderDeduplicator:
    """
    Manages order deduplication with TTL-based cleanup.
    
    Features:
    - LRU cache with TTL for recent orders
    - Idempotency key tracking
    - Automatic cleanup of expired entries
    - Thread-safe async operations
    """

    def __init__(
        self,
        max_entries: int = 10000,
        ttl_seconds: int = 3600,  # 1 hour TTL
        cleanup_interval: int = 300,  # Cleanup every 5 minutes
    ):
        """
        Args:
            max_entries: Maximum entries in cache
            ttl_seconds: Time-to-live for order records
            cleanup_interval: Cleanup interval in seconds
        """
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")

        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.cleanup_interval = cleanup_interval

        # OrderedDict for LRU behavior
        self._orders: OrderedDict[str, OrderAttempt] = OrderedDict()
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        """Start background cleanup task."""
        if self._running:
            return
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self) -> None:
        """Stop background cleanup task."""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

    async def check_and_register(
        self,
        idempotency_key: str,
        order_params: dict,
    ) -> tuple[bool, Optional[str]]:
        """
        Check if order is duplicate and register if not.
        
        Args:
            idempotency_key: Idempotency key from IdempotencyKeyGenerator
            order_params: Order parameters for creating OrderAttempt
            
        Returns:
            (is_new, existing_order_id)
            - is_new: True if new order, False if duplicate
            - existing_order_id: ID of existing order if duplicate
        """
        async with self._lock:
            now = datetime.now(timezone.utc)
            
            # Check for existing order
            if idempotency_key in self._orders:
                existing = self._orders[idempotency_key]
                # Check if expired
                if (datetime.now(timezone.utc) - existing.timestamp).total_seconds() < self.ttl_seconds:
                    # Mark as duplicate
                    return (False, existing.client_order_id)
                else:
                    # Expired, remove and allow new order
                    del self._orders[idempotency_key]
            
            # Check capacity
            if len(self._orders) >= self.max_entries:
                # Remove oldest entry (LRU)
                self._orders.popitem(last=False)
            
            # Create new order attempt
            attempt = OrderAttempt(
                client_order_id=uuid.uuid4().hex[:16],
                idempotency_key=idempotency_key,
                **order_params,
                timestamp=datetime.now(timezone.utc),
            )
            
            self._orders[idempotency_key] = attempt
            return (True, None)

    async def update_status(
        self,
        idempotency_key: str,
        status: OrderStatus,
        exchange_order_id: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> bool:
        """
        Update order status.
        
        Returns True if found and updated, False if not found.
        """
        async with self._lock:
            if idempotency_key not in self._orders:
                return False
            
            attempt = self._orders[idempotency_key]
            # Create new frozen dataclass with updated status
            updated = OrderAttempt(
                client_order_id=attempt.client_order_id,
                idempotency_key=attempt.idempotency_key,
                symbol=attempt.symbol,
                side=attempt.side,
                quantity=attempt.quantity,
                price=attempt.price,
                order_type=attempt.order_type,
                timestamp=attempt.timestamp,
                status=status,
                exchange_order_id=exchange_order_id or attempt.exchange_order_id,
                error_message=error_message or attempt.error_message,
                retry_count=attempt.retry_count,
            )
            self._orders[idempotency_key] = updated
            return True

    async def get_order(self, idempotency_key: str) -> Optional[OrderAttempt]:
        """Get order attempt by idempotency key."""
        async with self._lock:
            return self._orders.get(idempotency_key)

    async def get_by_client_order_id(self, client_order_id: str) -> Optional[OrderAttempt]:
        """Find order by client order ID."""
        async with self._lock:
            for attempt in self._orders.values():
                if attempt.client_order_id == client_order_id:
                    return attempt
            return None

    async def increment_retry(self, idempotency_key: str) -> bool:
        """Increment retry count for an order."""
        async with self._lock:
            if idempotency_key not in self._orders:
                return False
            attempt = self._orders[idempotency_key]
            updated = OrderAttempt(
                client_order_id=attempt.client_order_id,
                idempotency_key=attempt.idempotency_key,
                symbol=attempt.symbol,
                side=attempt.side,
                quantity=attempt.quantity,
                price=attempt.price,
                order_type=attempt.order_type,
                timestamp=attempt.timestamp,
                status=attempt.status,
                exchange_order_id=attempt.exchange_order_id,
                error_message=attempt.error_message,
                retry_count=attempt.retry_count + 1,
            )
            self._orders[idempotency_key] = updated
            return True

    def _cleanup_expired(self) -> int:
        """Remove expired entries. Returns count removed."""
        now = datetime.now(timezone.utc)
        removed = 0
        keys_to_remove = []
        
        for key, attempt in self._orders.items():
            if (datetime.now(timezone.utc) - attempt.timestamp).total_seconds() > self.ttl_seconds:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self._orders[key]
            removed += 1
        
        return removed

    async def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                removed = self._cleanup_expired()
                if __debug__:
                    print(f"[OrderDeduplicator] Cleaned up {removed} expired orders")
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[OrderDeduplicator] Cleanup error: {e}")

    def get_stats(self) -> dict:
        """Get deduplicator statistics."""
        now = datetime.now(timezone.utc)
        active = 0
        expired = 0
        for attempt in self._orders.values():
            if (datetime.now(timezone.utc) - attempt.timestamp).total_seconds() < self.ttl_seconds:
                active += 1
            else:
                expired += 1
        
        return {
            "total_entries": len(self._orders),
            "active": _,
            "expired": expired,
            "max_entries": self.max_entries,
            "ttl_seconds": self.ttl_seconds,
        }

    def clear(self) -> int:
        """Clear all entries. Returns count cleared."""
        count = len(self._orders)
        self._orders.clear()
        return count


# ============================================================
# EXECUTION CONTEXT MANAGER
# ============================================================

class OrderExecutionContext:
    """
    Context manager for atomic order execution with deduplication.
    
    Usage:
        async with OrderExecutionContext(deduplicator, order_params) as ctx:
            if ctx.is_duplicate:
                return  # Skip duplicate
            # Execute order
            result = await exchange.place_order(...)
            await ctx.complete(exchange_order_id="12345")
    """

    def __init__(
        self,
        deduplicator: 'OrderDeduplicator',
        idempotency_key: str,
        order_params: dict,
    ):
        self.deduplicator = deduplicator
        self.idempotency_key = idempotency_key
        self.order_params = order_params
        self.is_duplicate = False
        self.existing_order_id: Optional[str] = None
        self._completed = False

    async def __aenter__(self):
        is_new, existing_id = await self.deduplicator.check_and_register(
            self.idempotency_key,
            self.order_params,
        )
        self.is_duplicate = not is_new
        self.existing_order_id = existing_id
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._completed:
            return
        
        if not self.is_duplicate:
            # Mark as completed (will be updated by caller)
            pass

    async def complete(
        self,
        status: str = "ACCEPTED",
        exchange_order_id: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Mark order as completed with final status."""
        if self._completed:
            return
        self._completed = True
        
        if not self.is_duplicate:
            await self.deduplicator.update_status(
                self.idempotency_key,
                status=status,
                exchange_order_id=exchange_order_id,
                error_message=error_message,
            )

    async def mark_failed(self, error_message: str) -> None:
        """Mark order as failed."""
        await self.complete(status="REJECTED", error_message=error_message)


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "OrderStatus",
    "OrderAttempt",
    "IdempotencyKeyGenerator",
    "OrderDeduplicator",
    "OrderExecutionContext",
]