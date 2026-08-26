"""
QuantAI Market Data Package

Market Data Fan-out via Redis Pub/Sub.
"""

from .fanout import (
    TickData,
    KlineData,
    DepthData,
    LiquidationData,
    FundingData,
    ChannelConfig,
    MarketDataService,
    MarketDataConsumer,
    create_market_data_service,
)

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