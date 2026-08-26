"""
====================================================
QuantAI Professional
Rate Limiter - Binance API Compliance
====================================================

Token bucket rate limiter for Binance API compliance.
Handles:
- Request weight limits (1200/minute)
- Order rate limits (50 orders/second, 160000/day)
- IP and API key based limits
- Automatic backoff and retry

====================================================
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Deque, Dict, Optional, Tuple
from collections import deque as Deque

import asyncio


class LimitType(Enum):
    """Rate limit categories."""
    REQUEST_WEIGHT = "REQUEST_WEIGHT"      # 1200/min
    ORDERS_PER_SECOND = "ORDERS_PER_SEC"   # 50/sec
    ORDERS_PER_DAY = "ORDERS_PER_DAY"      # 160000/day
    RAW_REQUESTS = "RAW_REQUESTS"          # 6100/min (IP)


@dataclass
class RateLimitConfig:
    """Configuration for a rate limit."""
    limit: int
    window_seconds: float
    burst_allowance: float = 1.0  # Allow burst up to this multiplier


@dataclass
class RateLimitState:
    """Current state of a rate limit."""
    limit: int
    window_seconds: float
    requests: deque = field(default_factory=deque)
    last_reset: float = field(default_factory=time.time)


class TokenBucket:
    """
    Token bucket rate limiter for smooth rate limiting.
    
    Allows burst traffic up to bucket capacity while maintaining
    average rate over time.
    """

    def __init__(
        self,
        rate: float,      # Tokens per second
        capacity: int,    # Maximum bucket size
        initial_tokens: Optional[int] = None,
    ):
        """
        Args:
            rate: Tokens added per second
            capacity: Maximum bucket capacity
            initial_tokens: Initial tokens (defaults to capacity)
        """
        if rate <= 0:
            raise ValueError("rate must be positive")
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        
        self.rate = float(rate)
        self.capacity = int(capacity)
        self._tokens = float(initial_tokens if initial_tokens is not None else capacity)
        self._last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """
        Acquire tokens from the bucket.
        
        Args:
            tokens: Number of tokens to acquire
            timeout: Maximum time to wait (None = wait forever)
            
        Returns:
            True if acquired, False if timeout
        """
        if tokens <= 0:
            return True
        
        if tokens > self.capacity:
            raise ValueError(f"Cannot acquire {tokens} tokens, capacity is {self.capacity}")
        
        async with self._lock:
            start_time = time.monotonic()
            
            while True:
                self._refill()
                
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return True
                
                if timeout is not None:
                    elapsed = time.monotonic() - start_time
                    if elapsed >= timeout:
                        return False
                
                # Calculate wait time for next token
                wait_time = (tokens - self._tokens) / self.rate
                wait_time = min(wait_time, 0.1)  # Cap wait time
                
                await asyncio.sleep(wait_time)
                
                # Check timeout
                if timeout is not None:
                    elapsed = time.monotonic() - start_time
                    if elapsed >= timeout:
                        return False

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_update
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
        self._last_update = now

    def available_tokens(self) -> float:
        """Get current available tokens (refills first)."""
        self._refill()
        return self._tokens

    def try_acquire(self, tokens: int = 1) -> bool:
        """Try to acquire tokens without waiting."""
        if tokens <= 0:
            return True
        if tokens > self.capacity:
            return False
        
        self._refill()
        if self._tokens >= tokens:
            self._tokens -= tokens
            return True
        return False

    def reset(self) -> None:
        """Reset bucket to full capacity."""
        self._tokens = float(self.capacity)
        self._last_update = time.monotonic()


class SlidingWindowRateLimiter:
    """
    Sliding window rate limiter with precise counting.
    
    More accurate than token bucket for strict limits,
    but slightly more complex.
    """

    def __init__(self, limit: int, window_seconds: float):
        """
        Args:
            limit: Maximum requests in window
            window_seconds: Time window in seconds
        """
        if limit <= 0:
            raise ValueError("limit must be positive")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")

        self.limit = limit
        self.window_seconds = window_seconds
        self._requests: Deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self, timeout: Optional[float] = None) -> bool:
        """Acquire permission to make a request."""
        start_time = time.monotonic()
        
        while True:
            async with self._lock:
                self._cleanup_old()
                
                if len(self._requests) < self.limit:
                    self._requests.append(time.monotonic())
                    return True
                
                if timeout is not None:
                    elapsed = time.monotonic() - start_time
                    if elapsed >= timeout:
                        return False
                
                # Wait until oldest request expires
                if self._requests:
                    wait_time = self.window_seconds - (time.monotonic() - self._requests[0])
                    wait_time = max(0, min(wait_time, 0.1))
                else:
                    wait_time = 0.1
            
            await asyncio.sleep(wait_time)
            
            if timeout is not None:
                elapsed = time.monotonic() - start_time
                if elapsed >= timeout:
                    return False

    def try_acquire(self) -> bool:
        """Try to acquire without waiting."""
        self._cleanup_old()
        if len(self._requests) < self.limit:
            self._requests.append(time.monotonic())
            return True
        return False

    def _cleanup_old(self) -> None:
        """Remove expired entries."""
        cutoff = time.monotonic() - self.window_seconds
        while self._requests and self._requests[0] < cutoff:
            self._requests.popleft()

    def available_slots(self) -> int:
        """Get available slots in current window."""
        self._cleanup_old()
        return max(0, self.limit - len(self._requests))

    def current_usage(self) -> int:
        """Get current request count in window."""
        self._cleanup_old()
        return len(self._requests)

    def time_until_slot(self) -> float:
        """Time until next slot available."""
        self._cleanup_old()
        if len(self._requests) < self.limit:
            return 0.0
        return max(0, self.window_seconds - (time.monotonic() - self._requests[0]))

    def reset(self) -> None:
        """Reset all request history."""
        self._requests.clear()


class MultiLimitRateLimiter:
    """
    Composite rate limiter managing multiple limits simultaneously.
    
    Handles Binance-specific limits:
    - Request weight: 1200/min
    - Orders/second: 50/sec
    - Orders/day: 160000/day
    - Raw requests (IP): 6100/min
    """

    def __init__(self):
        # Binance Futures limits (as of 2024)
        self.limits = {
            LimitType.REQUEST_WEIGHT: SlidingWindowRateLimiter(1200, 60.0),
            LimitType.ORDERS_PER_SECOND: SlidingWindowRateLimiter(50, 1.0),
            LimitType.ORDERS_PER_DAY: SlidingWindowRateLimiter(160000, 86400.0),
            LimitType.RAW_REQUESTS: SlidingWindowRateLimiter(6100, 60.0),
        }
        
        # Weight per endpoint (Binance API)
        self.endpoint_weights = {
            # Market data (weight 1)
            "/fapi/v1/ping": 1,
            "/fapi/v1/time": 1,
            "/fapi/v1/exchangeInfo": 10,
            "/fapi/v1/depth": 1,
            "/fapi/v1/ticker/24hr": 1,
            "/fapi/v1/ticker/price": 1,
            "/fapi/v1/ticker/bookTicker": 1,
            "/fapi/v1/klines": 1,
            "/fapi/v1/premiumIndex": 1,
            "/fapi/v1/fundingRate": 1,
            "/fapi/v1/openInterest": 1,
            
            # Account (weight 5-10)
            "/fapi/v2/account": 5,
            "/fapi/v2/balance": 5,
            "/fapi/v1/positionRisk": 5,
            "/fapi/v1/order": 1,  # Order placement
            "/fapi/v1/batchOrders": 5,
            "/fapi/v1/order": 1,  # Cancel order
            "/fapi/v1/allOpenOrders": 5,
            "/fapi/v1/allOrders": 5,
            "/fapi/v1/openOrders": 1,
            
            # User Data Stream
            "/fapi/v1/listenKey": 1,
        }
        
        self._lock = asyncio.Lock()

    def get_weight(self, endpoint: str) -> int:
        """Get weight for an endpoint."""
        # Match by prefix
        for endpoint_pattern, weight in self.endpoint_weights.items():
            if endpoint.startswith(endpoint_pattern):
                return weight
        return 1  # Default weight

    async def acquire(
        self,
        endpoint: str,
        weight: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> bool:
        """
        Acquire permission for an API request.
        
        Args:
            endpoint: API endpoint path
            weight: Override weight (auto-detected if None)
            timeout: Maximum wait time (None = wait forever)
            
        Returns:
            True if acquired, False if timeout
        """
        weight = weight or self.get_weight(endpoint)
        
        # Need to acquire for both REQUEST_WEIGHT and RAW_REQUESTS
        weight_limit = self.limits[LimitType.REQUEST_WEIGHT]
        raw_limit = self.limits[LimitType.RAW_REQUESTS]
        
        # For weight > 1, need multiple slots
        weight_needed = weight
        
        async def try_acquire():
            # Try to acquire weight slots
            for _ in range(weight_needed):
                if not self.limits[LimitType.REQUEST_WEIGHT].try_acquire():
                    # Rollback
                    for _ in range(_):
                        self.limits[LimitType.REQUEST_WEIGHT]._requests.pop()
                    return False
            
            if not self.limits[LimitType.RAW_REQUESTS].try_acquire():
                # Rollback weight
                for _ in range(weight_needed):
                    self.limits[LimitType.REQUEST_WEIGHT]._requests.pop()
                return False
            
            return True
        
        start = time.monotonic()
        while True:
            async with asyncio.Lock():
                if await asyncio.get_event_loop().run_in_executor(None, lambda: None):
                    pass
                if weight_needed <= 1:
                    if self.limits[LimitType.REQUEST_WEIGHT].try_acquire() and \
                       self.limits[LimitType.RAW_REQUESTS].try_acquire():
                        return True
                else:
                    # For weight > 1, need to check if we have enough slots
                    if (self.limits[LimitType.REQUEST_WEIGHT].available_slots() >= weight_needed and
                        self.limits[LimitType.RAW_REQUESTS].available_slots() >= weight_needed):
                        for _ in range(weight_needed):
                            self.limits[LimitType.REQUEST_WEIGHT]._requests.append(time.monotonic())
                        self.limits[LimitType.RAW_REQUESTS]._requests.append(time.monotonic())
                        return True
                
                return False
            
            if timeout is not None:
                elapsed = time.monotonic() - start_time
                if elapsed >= timeout:
                    return False
            
            await asyncio.sleep(0.05)
    
    def try_acquire(self, endpoint: str, weight: Optional[int] = None) -> bool:
        """Try to acquire without waiting."""
        weight = weight or self.get_weight(endpoint)
        
        if not self.limits[LimitType.REQUEST_WEIGHT].try_acquire():
            return False
        if not self.limits[LimitType.RAW_REQUESTS].try_acquire():
            # Rollback
            self.limits[LimitType.REQUEST_WEIGHT]._requests.pop()
            return False
        return True

    def release(self, endpoint: str, weight: Optional[int] = None) -> None:
        """Release a request slot (for manual release)."""
        weight = weight or self.get_weight(endpoint)
        for _ in range(weight):
            if self.limits[LimitType.REQUEST_WEIGHT]._requests:
                self.limits[LimitType.REQUEST_WEIGHT]._requests.pop()
            if self.limits[LimitType.RAW_REQUESTS]._requests:
                self.limits[LimitType.RAW_REQUESTS]._requests.pop()

    def get_remaining(self, limit_type: LimitType) -> int:
        """Get remaining slots for a limit type."""
        if limit_type in self.limits:
            return self.limits[limit_type].available_slots()
        return 0

    def get_usage(self) -> Dict[str, Any]:
        """Get current usage stats for all limits."""
        return {
            limit_type.value: {
                "used": self.limits[lt].current_usage(),
                "limit": self.limits[lt].limit,
                "remaining": self.limits[lt].available_slots(),
                "window_seconds": self.limits[lt].window_seconds,
            }
            for lt, lt in self.limits.items()
        }

    def reset_all(self) -> None:
        """Reset all limiters."""
        for limiter in self.limits.values():
            limiter.reset()


class BinanceRateLimiter:
    """
    Binance-specific rate limiter with endpoint-aware weighting.
    
    Combines:
    - Request weight limiter (1200/min)
    - Order rate limits (50/sec, 160000/day)
    - Raw request limit (6100/min)
    - Automatic weight detection by endpoint
    """

    def __init__(self):
        self.multi_limiter = MultiLimitRateLimiter()
        self._order_count_today = 0
        self._day_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

    async def acquire_for_endpoint(
        self,
        endpoint: str,
        weight: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> bool:
        """Acquire rate limit slot for Binance endpoint."""
        return await self.multi_limiter.acquire(endpoint, weight, timeout)

    def try_acquire_for_endpoint(self, endpoint: str, weight: Optional[int] = None) -> bool:
        """Try to acquire without waiting."""
        return self.multi_limiter.try_acquire(endpoint, weight)

    def record_order(self, count: int = 1) -> None:
        """Record order placement for daily limit tracking."""
        # Reset daily counter at midnight
        now = datetime.now(timezone.utc)
        if now >= self._day_start + timedelta(days=1):
            self._order_count_today = 0
            self._day_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        
        self._order_count_today += count
        
        # Check daily limit
        if self._order_count_today >= 160000:
            raise RuntimeError("Daily order limit (160,000) exceeded")

    def get_usage_report(self) -> dict:
        """Get current usage report for all limiters."""
        return self.multi_limiter.get_usage_report()

    def reset_all(self) -> None:
        """Reset all rate limiters."""
        self.multi_limiter.reset_all()
        self._order_count_today = 0
        self._day_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "LimitType",
    "TokenBucket",
    "SlidingWindowRateLimiter",
    "MultiLimitRateLimiter",
    "BinanceRateLimiter",
]