"""
QuantAI Market Data Fan-out

Centralized market data distribution via Redis Pub/Sub.

Architecture:
- Single MarketDataService maintains WebSocket connections to exchanges
- Publishes normalized market data to Redis channels
- Multiple consumers (strategies, risk engines, etc.) subscribe to channels
- Eliminates duplicate exchange connections and rate limit issues

Channels:
- market:tick:{symbol} - Real-time tick data
- market:kline:{symbol}:{timeframe} - Kline/candlestick updates
- market:depth:{symbol} - Order book depth updates
- market:liquidation:{symbol} - Liquidation events
- market:funding:{symbol} - Funding rate updates
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set
from urllib.parse import urlparse

import redis.asyncio as redis
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ============================================================
# DATA MODELS
# ============================================================

class TickData(BaseModel):
    """Real-time tick data."""
    symbol: str
    timestamp: int  # Unix timestamp in milliseconds
    price: float
    volume: float
    side: str  # "BUY" or "SELL"


class KlineData(BaseModel):
    """Kline/candlestick data."""
    symbol: str
    timeframe: str
    timestamp: int  # Open time
    open: float
    high: float
    low: float
    close: float
    volume: float
    closed: bool  # Whether the kline is closed


class DepthData(BaseModel):
    """Order book depth update."""
    symbol: str
    timestamp: int
    bids: List[List[float]]  # [[price, quantity], ...]
    asks: List[List[float]]  # [[price, quantity], ...]


class LiquidationData(BaseModel):
    """Liquidation event."""
    symbol: str
    timestamp: int
    side: str  # "LONG" or "SHORT"
    price: float
    quantity: float
    notional: float


class FundingData(BaseModel):
    """Funding rate update."""
    symbol: str
    timestamp: int
    funding_rate: float
    next_funding_time: int


@dataclass
class ChannelConfig:
    """Configuration for a Redis channel."""
    name: str
    pattern: str  # e.g., "market:tick:*"
    max_len: int = 10000  # Max messages to keep in stream


# ============================================================
# MARKET DATA SERVICE
# ============================================================

class MarketDataService:
    """
    Centralized market data service with Redis Pub/Sub fan-out.
    
    Maintains single WebSocket connections per exchange/symbol
    and fans out data to Redis channels for multiple consumers.
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        exchange_configs: Optional[Dict[str, Any]] = None,
    ):
        self.redis_url = redis_url
        self.exchange_configs = exchange_configs or {}
        
        self._redis: Optional[redis.Redis] = None
        self._pubsub: Optional[redis.client.PubSub] = None
        self._running = False
        self._tasks: List[asyncio.Task] = []
        
        # Exchange connections
        self._ws_connections: Dict[str, Any] = {}
        
        # Channel configurations
        self.channel_configs = {
            "tick": ChannelConfig("market:tick", "market:tick:*"),
            "kline": ChannelConfig("market:kline", "market:kline:*"),
            "depth": ChannelConfig("market:depth", "market:depth:*"),
            "liquidation": ChannelConfig("market:liquidation", "market:liquidation:*"),
            "funding": ChannelConfig("market:funding", "market:funding:*"),
        }
        
        # Local cache for latest data
        self._latest_tick: Dict[str, Any] = {}
        self._latest_kline: Dict[str, Any] = {}
        self._latest_depth: Dict[str, Any] = {}
    
    async def start(self) -> None:
        """Start the market data service."""
        if self._running:
            return
            
        self._running = True
        
        # Connect to Redis
        self._redis = redis.from_url(self.redis_url, decode_responses=True)
        await self._redis.ping()
        logger.info("Connected to Redis")
        
        # Start exchange connections
        for exchange_name, config in self.exchange_configs.items():
            task = asyncio.create_task(self._run_exchange_connection(exchange_name, config))
            self._tasks.append(task)
        
        logger.info("Market Data Service started")
    
    async def stop(self) -> None:
        """Stop the market data service."""
        self._running = False
        
        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        # Close WebSocket connections
        for conn in self._ws_connections.values():
            if hasattr(conn, 'close'):
                await conn.close()
        
        # Close Redis
        if self._redis:
            await self._redis.close()
        
        logger.info("Market Data Service stopped")
    
    async def _run_exchange_connection(self, exchange_name: str, config: Dict[str, Any]) -> None:
        """Run WebSocket connection for an exchange."""
        # This would be implemented with actual exchange WebSocket clients
        # For now, this is a placeholder
        logger.info(f"Starting exchange connection: {exchange_name}")
        
        # Example for Binance:
        # ws = await connect_binance_ws(config)
        # async for msg in ws:
        #     await self._process_message(exchange_name, msg)
        
        while True:
            await asyncio.sleep(60)  # Heartbeat
            if not self._running:
                break
    
    async def _process_message(self, exchange: str, message: Dict[str, Any]) -> None:
        """Process incoming exchange message and publish to Redis."""
        try:
            # Parse message based on exchange format
            # This is exchange-specific
            pass
        except Exception as e:
            logger.error(f"Error processing message from {exchange}: {e}")
    
    # ============================================================
    # PUBLISH METHODS
    # ============================================================
    
    async def publish_tick(self, tick: TickData) -> None:
        """Publish tick data to Redis channel."""
        channel = f"market:tick:{tick.symbol}"
        data = tick.model_dump_json()
        await self._redis.xadd(f"{channel}:stream", {"data": data}, maxlen=10000)
        await self._redis.publish(channel, data)
        
        # Update local cache
        self._latest_tick[tick.symbol] = tick
    
    async def publish_kline(self, kline: KlineData) -> None:
        """Publish kline data to Redis channel."""
        channel = f"market:kline:{kline.symbol}:{kline.timeframe}"
        data = kline.model_dump_json()
        await self._redis.xadd(f"{channel}:stream", {"data": data}, maxlen=5000)
        await self._redis.publish(channel, data)
        
        key = f"{kline.symbol}:{kline.timeframe}"
        self._latest_kline[key] = kline
    
    async def publish_depth(self, depth: DepthData) -> None:
        """Publish order book depth to Redis channel."""
        channel = f"market:depth:{depth.symbol}"
        data = depth.model_dump_json()
        await self._redis.xadd(f"{channel}:stream", {"data": data}, maxlen=1000)
        await self._redis.publish(channel, data)
        
        self._latest_depth[depth.symbol] = depth
    
    async def publish_liquidation(self, liquidation: LiquidationData) -> None:
        """Publish liquidation event to Redis channel."""
        channel = f"market:liquidation:{liquidation.symbol}"
        data = liquidation.model_dump_json()
        await self._redis.xadd(f"{channel}:stream", {"data": data}, maxlen=5000)
        await self._redis.publish(channel, data)
    
    async def publish_funding(self, funding: FundingData) -> None:
        """Publish funding rate to Redis channel."""
        channel = f"market:funding:{funding.symbol}"
        data = funding.model_dump_json()
        await self._redis.xadd(f"{channel}:stream", {"data": data}, maxlen=1000)
        await self._redis.publish(channel, data)
    
    # ============================================================
    # SUBSCRIBE METHODS (for consumers)
    # ============================================================
    
    async def subscribe_ticks(self, symbols: List[str], callback: Callable[[TickData], None]) -> asyncio.Task:
        """Subscribe to tick updates for symbols."""
        channels = [f"market:tick:{s}" for s in symbols]
        return await self._subscribe_channels(channels, callback, TickData)
    
    async def subscribe_klines(self, symbols: List[str], timeframe: str, callback: Callable[[KlineData], None]) -> asyncio.Task:
        """Subscribe to kline updates for symbols."""
        channels = [f"market:kline:{s}:{timeframe}" for s in symbols]
        return await self._subscribe_channels(channels, callback, KlineData)
    
    async def subscribe_depth(self, symbols: List[str], callback: Callable[[DepthData], None]) -> asyncio.Task:
        """Subscribe to depth updates for symbols."""
        channels = [f"market:depth:{s}" for s in symbols]
        return await self._subscribe_channels(channels, callback, DepthData)
    
    async def subscribe_liquidations(self, symbols: List[str], callback: Callable[[LiquidationData], None]) -> asyncio.Task:
        """Subscribe to liquidation events for symbols."""
        channels = [f"market:liquidation:{s}" for s in symbols]
        return await self._subscribe_channels(channels, callback, LiquidationData)
    
    async def _subscribe_channels(
        self,
        channels: List[str],
        callback: Callable,
        model_class: type,
    ) -> asyncio.Task:
        """Subscribe to Redis channels and call callback with parsed data."""
        pubsub = self._redis.pubsub()
        await pubsub.subscribe(*channels)
        
        async def listener():
            async for message in pubsub.listen():
                if message["type"] == "message":
                    try:
                        data = model_class.model_validate_json(message["data"])
                        await callback(data)
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
        
        task = asyncio.create_task(listener())
        return task
    
    # ============================================================
    # QUERY METHODS (for latest data)
    # ============================================================
    
    def get_latest_tick(self, symbol: str) -> Optional[TickData]:
        """Get latest cached tick for symbol."""
        return self._latest_tick.get(symbol)
    
    def get_latest_kline(self, symbol: str, timeframe: str) -> Optional[KlineData]:
        """Get latest cached kline for symbol/timeframe."""
        return self._latest_kline.get(f"{symbol}:{timeframe}")
    
    def get_latest_depth(self, symbol: str) -> Optional[DepthData]:
        """Get latest cached depth for symbol."""
        return self._latest_depth.get(symbol)
    
    async def get_recent_ticks(self, symbol: str, count: int = 100) -> List[TickData]:
        """Get recent ticks from Redis stream."""
        channel = f"market:tick:{symbol}:stream"
        messages = await self._redis.xrevrange(f"{channel}:stream", count=count)
        return [TickData.model_validate_json(m[1]["data"]) for m in messages]
    
    async def get_recent_klines(self, symbol: str, timeframe: str, count: int = 100) -> List[KlineData]:
        """Get recent klines from Redis stream."""
        channel = f"market:kline:{symbol}:{timeframe}:stream"
        messages = await self._redis.xrevrange(f"{channel}:stream", count=count)
        return [KlineData.model_validate_json(m[1]["data"]) for m in messages]


# ============================================================
# CONSUMER BASE CLASS
# ============================================================

class MarketDataConsumer:
    """
    Base class for market data consumers.
    
    Strategies, risk engines, and other components can inherit
    from this to easily consume market data.
    """
    
    def __init__(self, service: MarketDataService):
        self.service = service
        self._tasks: List[asyncio.Task] = []
    
    async def start(self) -> None:
        """Start consuming market data."""
        pass
    
    async def stop(self) -> None:
        """Stop consuming market data."""
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    
    async def _subscribe_ticks(self, symbols: List[str], handler: Callable[[TickData], None]) -> None:
        task = await self.service.subscribe_ticks(symbols, handler)
        self._tasks.append(task)
    
    async def _subscribe_klines(self, symbols: List[str], timeframe: str, handler: Callable[[KlineData], None]) -> None:
        task = await self.service.subscribe_klines(symbols, timeframe, handler)
        self._tasks.append(task)


# ============================================================
# FACTORY FUNCTION
# ============================================================

async def create_market_data_service(
    redis_url: str = "redis://localhost:6379",
    exchange_configs: Optional[Dict[str, Any]] = None,
) -> MarketDataService:
    """Create and start market data service."""
    service = MarketDataService(redis_url=redis_url, exchange_configs=exchange_configs)
    await service.start()
    return service


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "TickData",
    "KlineData",
    "DepthData",
    "LiquidationData",
    "FundingData",
    "ChannelConfig",
    "MarketDataService",
    "MarketDataConsumer",
    "create_market_data_service",
]