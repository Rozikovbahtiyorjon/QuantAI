"""
QuantAI Market Data Package

Market Data Fan-out via Redis Pub/Sub.
"""

try:
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
except ImportError:
    # redis not available in test env — fanout optional
    TickData = KlineData = DepthData = LiquidationData = FundingData = ChannelConfig = MarketDataService = MarketDataConsumer = create_market_data_service = None  # type: ignore

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