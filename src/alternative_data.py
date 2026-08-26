"""
====================================================
QuantAI Professional
Alternative Data Integration
====================================================

Alternative data sources for enhanced ML features:
- LunarCrush (Galaxy Score, AltRank, Social Metrics)
- Funding Rate Tracking (cross-exchange)
- Open Interest Delta per Candle
- News/Sentiment Analysis (NLP-based)

====================================================
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional
from urllib.parse import urlencode

import aiohttp
import numpy as np
import pandas as pd


# ============================================================
# LUNARCRUSH INTEGRATION
# ============================================================

@dataclass(frozen=True)
class LunarCrushData:
    """LunarCrush data for a symbol."""
    symbol: str
    galaxy_score: float          # 0-100, overall health
    alt_rank: int                # Rank among all coins
    social_volume: float         # Total mentions
    social_engagement: float     # Likes + retweets + comments + views
    social_dominance: float      # % of total crypto discussion
    sentiment: float             # Bullish/Bearish ratio (0-1)
    price_score: float           # Price momentum score
    correlation_rank: int        # Correlation rank
    volatility_rank: int         # Volatility rank
    timestamp: datetime


class LunarCrushClient:
    """
    LunarCrush API Client for social sentiment data.
    
    Provides:
    - Galaxy Score (0-100): Overall coin health
    - AltRank: Ranking by social + price activity
    - Social Volume/Engagement/Dominance
    - Sentiment (Bullish/Bearish)
    """

    BASE_URL = "https://api.lunarcrush.com/v2"
    
    def __init__(
        self,
        api_key: str,
        timeout: float = 10.0,
        rate_limit_per_minute: int = 100,
    ):
        """
        Args:
            api_key: LunarCrush API key
            timeout: Request timeout in seconds
            rate_limit_per_minute: Max requests per minute
        """
        self.api_key = api_key
        self.timeout = timeout
        self.rate_limit = rate_limit_per_minute
        
        self._session: Optional[aiohttp.ClientSession] = None
        self._rate_limiter = asyncio.Semaphore(rate_limit_per_minute)
        self._last_request_time = 0.0
        self._request_times: deque = deque(maxlen=rate_limit_per_minute)

    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=30.0)
        self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30.0))
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            await self._session.close()

    async def _rate_limit(self):
        """Enforce rate limiting."""
        now = time.time()
        # Remove old timestamps
        while self._request_times and now - self._request_times[0] > 60:
            self._request_times.popleft()
        
        if len(self._request_times) >= 100:  # Conservative limit
            wait_time = 60 - (time.time() - self._request_times[0])
            if wait_time > 0:
                await asyncio.sleep(wait_time)
        
        self._request_times.append(time.time())

    async def get_coin_data(
        self,
        symbol: str,
        interval: str = "1d",
    ) -> Optional[LunarCrushData]:
        """
        Get LunarCrush data for a symbol.
        
        Args:
            symbol: Coin symbol (e.g., "BTC", "ETH")
            interval: Time interval ("1h", "1d", "7d", "30d")
            
        Returns:
            LunarCrushData or None if error
        """
        await self._rate_limit()
        
        if not self._session:
            return None
            
        try:
            params = {
                "key": self.api_key,
                "symbol": symbol.upper(),
                "interval": interval,
            }
            
            url = f"{self.BASE_URL}/assets"
            async with self._session.get(url, params=params) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                
            if not data.get("data"):
                return None
                
            coin = data["data"][0]
            return LunarCrushData(
                symbol=symbol.upper(),
                galaxy_score=float(coin.get("galaxy_score", 0)),
                alt_rank=int(coin.get("alt_rank", 9999)),
                social_volume=float(coin.get("social_volume_24h", 0)),
                social_engagement=float(coin.get("social_engagement_24h", 0)),
                social_dominance=float(coin.get("social_dominance", 0)),
                sentiment=float(coin.get("sentiment", 0.5)),
                price_score=float(coin.get("price_score", 0)),
                correlation_rank=int(coin.get("correlation_rank", 9999)),
                volatility_rank=int(coin.get("volatility_rank", 9999)),
                timestamp=datetime.now(timezone.utc),
            )
        except Exception:
            return None

    async def get_multiple_coins(
        self,
        symbols: list[str],
        interval: str = "1d",
    ) -> dict[str, LunarCrushData]:
        """Get data for multiple symbols."""
        results = {}
        for symbol in symbols:
            data = await self.get_coin_data(symbol, interval)
            if data:
                results[symbol.upper()] = data
            await asyncio.sleep(0.1)  # Small delay between requests
        return results


# ============================================================
# FUNDING RATE TRACKER
# ============================================================

@dataclass(frozen=True)
class FundingRateData:
    """Funding rate data for a symbol."""
    symbol: str
    exchange: str
    funding_rate: float          # Current funding rate (e.g., 0.0001 = 0.01%)
    funding_rate_8h: float       # 8h funding rate
    funding_rate_24h: float      # 24h average funding rate
    next_funding_time: datetime  # Next funding timestamp
    predicted_rate: float        # Predicted next funding rate
    annualized_rate: float       # Annualized rate (funding_rate * 365 * 3)
    timestamp: datetime


class FundingRateTracker:
    """
    Tracks funding rates across exchanges for arbitrage and signal generation.
    
    Features:
    - Cross-exchange funding rate comparison
    - Funding rate prediction
    - Annualized yield calculation
    - Arbitrage opportunity detection
    """

    def __init__(
        self,
        binance_rest: Any,
        bybit_rest: Optional[Any] = None,
        okx_rest: Optional[Any] = None,
    ):
        self.binance = binance_rest
        self.bybit = bybit_rest
        self.okx = okx_rest
        
        self._cache: dict[str, FundingRateData] = {}
        self._history: dict[str, deque] = {}

    async def get_funding_rate(
        self,
        symbol: str,
        exchange: str = "binance",
    ) -> Optional[FundingRateData]:
        """Get current funding rate for symbol from exchange."""
        if exchange == "binance" and self.binance:
            return await self._get_binance_funding(symbol)
        # Add other exchanges as needed
        return None

    async def _get_binance_funding(self, symbol: str) -> Optional[FundingRateData]:
        """Get funding rate from Binance."""
        try:
            # Get current funding rate
            premium = await self._get_binance_premium_index(symbol)
            funding = await self._get_binance_funding_rate(symbol)
            
            if not funding:
                return None
                
            rate = float(funding.get("fundingRate", 0))
            next_time = int(funding.get("nextFundingTime", 0))
            
            # Get 24h average
            history = await self._get_funding_history(symbol)
            rates_24h = [float(h.get("fundingRate", 0)) for h in history[-3:] if h.get("fundingRate")]
            avg_24h = np.mean(rates_24h) if rates_24h else rate
            
            predicted = rate  # Simple prediction
            annualized = rate * 3 * 365  # 3 funding periods per day * 365
            
            return FundingRateData(
                symbol=symbol,
                exchange="binance",
                funding_rate=rate,
                funding_rate_8h=rate,
                funding_rate_24h=avg_24h,
                next_funding_time=datetime.fromtimestamp(next_time / 1000, tz=timezone.utc),
                predicted_rate=predicted,
                annualized_rate=annualized,
                timestamp=datetime.now(timezone.utc),
            )
        except Exception:
            return None

    async def _get_binance_premium_index(self, symbol: str) -> dict:
        """Get premium index from Binance."""
        if hasattr(self, 'binance') and self.binance:
            return await self.binance._get("/fapi/v1/premiumIndex", {"symbol": symbol})
        return {}

    async def _get_binance_funding_rate(self, symbol: str) -> dict:
        """Get current funding rate from Binance."""
        if hasattr(self, 'binance') and self.binance:
            return await self.binance._get("/fapi/v1/fundingRate", {"symbol": symbol, "limit": 1})
        return {}

    async def _get_funding_history(self, symbol: str) -> list:
        """Get funding rate history."""
        if hasattr(self, 'binance') and self.binance:
            return await self.binance._get("/fapi/v1/fundingRate", {"symbol": symbol, "limit": 10})
        return []

    async def get_all_rates(self, symbol: str) -> dict[str, FundingRateData]:
        """Get funding rates from all available exchanges."""
        results = {}
        
        if self.binance:
            rate = await self._get_binance_funding(symbol)
            if rate:
                results["binance"] = rate
                
        return results

    def get_arbitrage_opportunities(
        self,
        symbol: str,
        min_spread: float = 0.0001,
    ) -> list[dict]:
        """Find funding rate arbitrage opportunities."""
        # This would compare rates across exchanges
        # Placeholder for future implementation
        return []


# ============================================================
# OPEN INTEREST DELTA TRACKER
# ============================================================

@dataclass(frozen=True)
class OIData:
    """Open Interest data point."""
    symbol: str
    open_interest: float       # Total open interest (contracts)
    open_interest_usd: float   # Open interest in USD
    oi_delta: float            # Change in OI from previous candle
    oi_delta_pct: float        # Percentage change
    timestamp: datetime


class OIDeltaTracker:
    """
    Tracks Open Interest delta per candle for momentum signals.
    
    Features:
    - Per-candle OI delta calculation
    - OI delta percentage
    - OI/Price divergence detection
    - OI momentum signals
    """

    def __init__(
        self,
        binance_rest,
        lookback_candles: int = 100,
    ):
        self.binance = binance_rest
        self.lookback = lookback_candles
        self._oi_history: dict[str, deque] = {}  # symbol -> deque of OIData

    async def update(
        self,
        symbol: str,
    ) -> Optional[OIData]:
        """Update OI data for symbol and return latest delta."""
        try:
            # Get current OI
            oi_data = await self.binance._get("/fapi/v1/openInterest", {"symbol": symbol})
            
            if not oi_data:
                return None
                
            oi = float(oi_data.get("openInterest", 0))
            oi_usd = oi * await self._get_mark_price(symbol)
            
            # Get previous OI from history
            if symbol not in self._oi_history:
                self._oi_history[symbol] = deque(maxlen=1000)
            
            history = self._oi_history[symbol]
            prev_oi = history[-1].open_interest if history else oi
            
            oi_delta = oi - prev_oi
            oi_delta_pct = (oi_delta / prev_oi * 100) if prev_oi > 0 else 0.0
            
            data = OIData(
                symbol=symbol,
                open_interest=oi,
                open_interest_usd=oi_usd,
                oi_delta=oi_delta,
                oi_delta_pct=oi_delta_pct,
                timestamp=datetime.now(timezone.utc),
            )
            
            history.append(data)
            return data
            
        except Exception:
            return None

    async def _get_mark_price(self, symbol: str) -> float:
        """Get mark price for USD conversion."""
        try:
            data = await self._get_mark_price_raw(symbol)
            return float(data.get("markPrice", 0))
        except Exception:
            return 0.0

    async def _get_mark_price_raw(self, symbol: str) -> dict:
        return {}  # placeholder

    def get_oi_momentum(self, symbol: str, periods: int = 10) -> dict:
        """Get OI momentum indicators."""
        if symbol not in self._oi_history:
            return {}
            
        history = list(self._oi_history[symbol])[-periods:]
        if len(history) < 2:
            return {}
            
        oi_values = [h.open_interest for h in history]
        oi_deltas = [h.oi_delta for h in history]
        oi_delta_pcts = [h.oi_delta_pct for h in history]
        
        return {
            "oi_trend": "increasing" if oi_values[-1] > oi_values[0] else "decreasing",
            "oi_momentum": np.mean(oi_deltas) if oi_deltas else 0,
            "oi_volatility": np.std(oi_deltas) if oi_deltas else 0,
            "avg_delta_pct": np.mean(oi_delta_pcts) if oi_delta_pcts else 0,
            "consecutive_increasing": self._consecutive_increasing(oi_values),
        }

    def _consecutive_increasing(self, values: list) -> int:
        """Count consecutive increasing values from end."""
        count = 0
        for i in range(len(values) - 1, 0, -1):
            if values[i] > values[i-1]:
                count += 1
            else:
                break
        return count

    def get_oi_divergence(self, symbol: str, price_data: list[float]) -> dict:
        """
        Detect OI/Price divergence.
        
        Returns:
            - divergence_type: "bullish", "bearish", "none"
            - strength: 0-1
        """
        if symbol not in self._oi_history or len(self._oi_history[symbol]) < 10:
            return {"divergence_type": "none", "strength": 0.0}
            
        history = list(self._oi_history[symbol])[-20:]
        oi_values = [h.open_interest for h in history]
        
        # Need price data - placeholder
        # In production, pass price_data as parameter
        
        return {"divergence_type": "none", "strength": 0.0}


# ============================================================
# ALTERNATIVE DATA MANAGER (Unified Interface)
# ============================================================

class AlternativeDataManager:
    """
    Unified manager for all alternative data sources.
    
    Coordinates:
    - LunarCrush social sentiment
    - Funding rate tracking
    - Open Interest delta
    - Microstructure features (VPIN, Kyle's Lambda, Liquidation)
    """

    def __init__(
        self,
        lunar_api_key: Optional[str] = None,
        binance_rest=None,
        bybit_rest=None,
        okx_rest=None,
    ):
        self.lunar_client = LunarCrushClient(lunar_api_key) if lunar_api_key else None
        self.funding_tracker = FundingRateTracker(binance_rest=None)  # Will be set later
        self.oi_tracker = None  # Initialized with binance_rest
        
        self._initialized = False

    async def initialize(
        self,
        binance_rest=None,
        bybit_rest=None,
        okx_rest=None,
    ):
        """Initialize all data sources with exchange connections."""
        if self._initialized:
            return
            
        if self.lunar_client:
            self.lunar_client._session = aiohttp.ClientSession()
        
        self.funding_tracker = FundingRateTracker(
            binance_rest=None,  # Will be set by caller
        )
        
        # OI tracker needs binance_rest
        # self.oi_tracker = OIDeltaTracker(binance_rest)
        
        self._initialized = True

    async def get_all_features(
        self,
        symbol: str,
    ) -> dict:
        """
        Get all alternative data features for a symbol.
        
        Returns combined feature dict for ML model.
        """
        features = {}
        
        # LunarCrush
        if self.lunar_client:
            lunar = await self.lunar_client.get_coin_data(symbol)
            if lunar:
                features.update({
                    "lunar_galaxy_score": lunar.galaxy_score,
                    "lunar_alt_rank": lunar.alt_rank,
                    "lunar_social_volume": lunar.social_volume,
                    "lunar_social_engagement": lunar.social_engagement,
                    "lunar_social_dominance": lunar.social_dominance,
                    "lunar_sentiment": lunar.sentiment,
                    "lunar_price_score": lunar.price_score,
                })
        
        # Funding rate
        # (requires exchange connection)
        
        # OI delta
        # (requires exchange connection)
        
        return features

    async def close(self):
        """Close all connections."""
        if self.lunar_client and self.lunar_client._session:
            await self.lunar_client._session.close()


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "LunarCrushData",
    "LunarCrushClient",
    "FundingRateData",
    "FundingRateTracker",
    "OIData",
    "OIDeltaTracker",
    "AlternativeDataManager",
]