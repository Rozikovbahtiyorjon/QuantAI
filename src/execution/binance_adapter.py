"""
====================================================
QuantAI Professional
Binance Exchange Adapter (REST + WebSocket)
====================================================

Production-ready Binance adapter with:
- REST API: account, orders, market data, exchange info
- WebSocket: User Data Stream + Market Data
- Rate limiting, retry logic, precision handling
- Testnet/Mainnet support
====================================================
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from typing import Any, Callable, Optional
from urllib.parse import urlencode

import aiohttp
import websockets

from src.execution.orders import (
    Fill,
    Order,
    OrderIntentData,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class BinanceConfig:
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = True
    recv_window: int = 5000
    
    # Rate limits (weights per minute)
    max_weight_per_minute: int = 1200
    max_orders_per_second: int = 10
    max_orders_per_day: int = 200000
    
    # Timeouts
    rest_timeout: float = 10.0
    ws_ping_interval: float = 20.0
    ws_ping_timeout: float = 10.0
    ws_reconnect_delay: float = 5.0
    ws_max_reconnect_delay: float = 60.0
    
    # Endpoints
    @property
    def base_url(self) -> str:
        return "https://testnet.binancefuture.com" if self.testnet else "https://fapi.binance.com"
    
    @property
    def ws_base_url(self) -> str:
        return "wss://stream.binancefuture.com" if self.testnet else "wss://fstream.binance.com"

    @property
    def spot_api_url(self) -> str:
        # apiRestrictions lives on the spot REST host for both envs
        return "https://testnet.binance.vision" if self.testnet else "https://api.binance.com"


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class SymbolInfo:
    symbol: str
    base_asset: str
    quote_asset: str
    price_precision: int
    quantity_precision: int
    min_qty: Decimal
    max_qty: Decimal
    step_size: Decimal
    min_notional: Decimal
    min_price: Decimal
    max_price: Decimal
    tick_size: Decimal
    
    def round_price(self, price: float) -> float:
        """Round price to tick size."""
        d = Decimal(str(price))
        tick = self.tick_size
        rounded = (d / tick).quantize(Decimal('1'), rounding=ROUND_DOWN) * tick
        return float(rounded.quantize(Decimal(f'1.{"0" * self.price_precision}')))
    
    def round_qty(self, qty: float) -> float:
        """Round quantity to step size."""
        d = Decimal(str(qty))
        step = self.step_size
        rounded = (d / step).quantize(Decimal('1'), rounding=ROUND_DOWN) * step
        return float(rounded.quantize(Decimal(f'1.{"0" * self.quantity_precision}')))
    
    def validate_qty(self, qty: float) -> bool:
        d = Decimal(str(qty))
        return self.min_qty <= d <= self.max_qty
    
    def validate_notional(self, qty: float, price: float) -> bool:
        notional = Decimal(str(qty)) * Decimal(str(price))
        return notional >= self.min_notional
    
    def validate_price(self, price: float) -> bool:
        d = Decimal(str(price))
        return self.min_price <= d <= self.max_price


@dataclass
class AccountBalance:
    asset: str
    wallet_balance: float
    available_balance: float
    unrealized_pnl: float
    margin_balance: float
    max_withdraw_amount: float


@dataclass
class Position:
    symbol: str
    position_amt: float
    entry_price: float
    mark_price: float
    unrealized_pnl: float
    liquidation_price: float
    leverage: int
    isolated_wallet: float
    is_isolated: bool


# ============================================================
# RATE LIMITER
# ============================================================

class RateLimiter:
    """Token bucket rate limiter for Binance API."""
    
    def __init__(self, max_per_minute: int = 1200, max_per_second: int = 10):
        self.max_per_minute = max_per_minute
        self.max_per_second = max_per_second
        self.minute_tokens = max_per_minute
        self.second_tokens = max_per_second
        self.last_minute = time.time()
        self.last_second = time.time()
        self._lock = asyncio.Lock()
    
    async def acquire(self, weight: int = 1):
        async with self._lock:
            now = time.time()
            
            # Refill minute bucket
            if now - self.last_minute >= 60:
                self.minute_tokens = self.max_per_minute
                self.last_minute = now
            
            # Refill second bucket
            if now - self.last_second >= 1:
                self.second_tokens = self.max_per_second
                self.last_second = now
            
            # Wait if needed
            while self.minute_tokens < weight or self.second_tokens < 1:
                await asyncio.sleep(0.1)
                now = time.time()
                if now - self.last_minute >= 60:
                    self.minute_tokens = self.max_per_minute
                    self.last_minute = now
                if now - self.last_second >= 1:
                    self.second_tokens = self.max_per_second
                    self.last_second = now
            
            self.minute_tokens -= weight
            self.second_tokens -= 1


# ============================================================
# BINANCE REST ADAPTER
# ============================================================

class BinanceRestAdapter:
    """Binance Futures REST API adapter."""
    
    def __init__(
        self, 
        config: BinanceConfig,
        rate_limiter: Optional[Any] = None,
    ):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        # Use injected rate_limiter or create internal one as fallback
        self.rate_limiter = rate_limiter or RateLimiter(
            max_per_minute=config.max_weight_per_minute,
            max_per_second=config.max_orders_per_second,
        )
        self.symbols: dict[str, SymbolInfo] = {}
        self._listen_key: Optional[str] = None
        self._listen_key_task: Optional[asyncio.Task] = None
    
    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=self.config.rest_timeout)
        self.session = aiohttp.ClientSession(timeout=timeout)
        await self.load_exchange_info()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._listen_key_task:
            self._listen_key_task.cancel()
        if self.session:
            await self.session.close()
    
    # ========================================================
    # LOW-LEVEL HTTP
    # ========================================================
    
    def _sign_params(self, params: dict) -> str:
        """Sign parameters for signed endpoints."""
        query_string = urlencode(params)
        signature = hmac.new(
            self.config.api_secret.encode(),
            query_string.encode(),
            hashlib.sha256
        ).hexdigest()
        return f"{query_string}&signature={signature}"
    
    def _headers(self) -> dict:
        return {"X-MBX-APIKEY": self.config.api_key}
    
    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[dict] = None,
        signed: bool = False,
        weight: int = 1,
    ) -> Any:
        # Use injected rate limiter (MultiLimitRateLimiter) or internal fallback
        if hasattr(self.rate_limiter, 'acquire_for_endpoint'):
            # Production MultiLimitRateLimiter
            await self.rate_limiter.acquire_for_endpoint(endpoint, weight)
        else:
            # Internal RateLimiter fallback
            await self.rate_limiter.acquire(weight)
        
        url = f"{self.config.base_url}{endpoint}"
        headers = self._headers()
        
        if params is None:
            params = {}
        
        params["timestamp"] = int(time.time() * 1000)
        params["recvWindow"] = self.config.recv_window
        
        if signed:
            query = self._sign_params(params)
            url = f"{url}?{query}"
            request_params = None
        else:
            request_params = params
        
        async with self.session.request(method, url, params=request_params, headers=headers) as resp:
            data = await resp.json()
            
            if resp.status != 200:
                raise BinanceAPIError(resp.status, data)
            
            # Update rate limit info from headers
            if "X-MBX-USED-WEIGHT-1M" in resp.headers:
                pass  # Could track used weight
            
            return data
    
    async def _get(self, endpoint: str, params: Optional[dict] = None, signed: bool = False, weight: int = 1):
        return await self._request("GET", endpoint, params, signed, weight)
    
    async def _post(self, endpoint: str, params: Optional[dict] = None, signed: bool = False, weight: int = 1):
        return await self._request("POST", endpoint, params, signed, weight)
    
    async def _delete(self, endpoint: str, params: Optional[dict] = None, signed: bool = False, weight: int = 1):
        return await self._request("DELETE", endpoint, params, signed, weight)
    
    # ========================================================
    # EXCHANGE INFO
    # ========================================================
    
    async def verify_no_withdraw_permission(self) -> dict:
        """
        SECURITY GUARD (R1): refuse to operate with a key that can withdraw.

        Queries GET /sapi/v1/account/apiRestrictions (signed, read perms).
        Fail-closed: any ambiguity raises. Testnet keys without the spot
        endpoint may skip via allow_unverified=True explicitly.
        """
        params = {"timestamp": int(time.time() * 1000),
                  "recvWindow": self.config.recv_window}
        url = f"{self.config.spot_api_url}/sapi/v1/account/apiRestrictions?{self._sign_params(params)}"

        async with self.session.get(url, headers=self._headers()) as resp:
            data = await resp.json(content_type=None)
            if resp.status != 200:
                raise PermissionError(
                    f"SECURITY: cannot verify key permissions "
                    f"(HTTP {resp.status}): {data}"
                )

        if data.get("canWithdraw"):
            raise PermissionError(
                "SECURITY: API key has WITHDRAW permission - "
                "regenerate key WITHOUT withdrawals before live trading"
            )
        for perm in data.get("permissions", []):
            if str(perm).upper() in {"WITHDRAW", "WITHDRAWALS"}:
                raise PermissionError(
                    f"SECURITY: key permission '{perm}' allows withdrawals"
                )

        return {
            "can_trade": data.get("canTrade"),
            "can_withdraw": data.get("canWithdraw", False),
            "permissions": data.get("permissions", []),
        }

    async def load_exchange_info(self) -> None:
        """Load symbol precision and filters."""
        data = await self._get("/fapi/v1/exchangeInfo", weight=10)
        
        for s in data.get("symbols", []):
            if s["status"] != "TRADING":
                continue
            
            symbol = s["symbol"]
            price_precision = s["pricePrecision"]
            qty_precision = s["quantityPrecision"]
            
            filters = {f["filterType"]: f for f in s["filters"]}
            
            lot_size = filters.get("LOT_SIZE", {})
            price_filter = filters.get("PRICE_FILTER", {})
            min_notional = filters.get("MIN_NOTIONAL", {})
            
            self.symbols[symbol] = SymbolInfo(
                symbol=symbol,
                base_asset=s["baseAsset"],
                quote_asset=s["quoteAsset"],
                price_precision=price_precision,
                quantity_precision=qty_precision,
                min_qty=Decimal(lot_size.get("minQty", "0")),
                max_qty=Decimal(lot_size.get("maxQty", "0")),
                step_size=Decimal(lot_size.get("stepSize", "0")),
                min_notional=Decimal(min_notional.get("notional", "0")),
                min_price=Decimal(price_filter.get("minPrice", "0")),
                max_price=Decimal(price_filter.get("maxPrice", "0")),
                tick_size=Decimal(price_filter.get("tickSize", "0")),
            )
    
    def get_symbol_info(self, symbol: str) -> Optional[SymbolInfo]:
        return self.symbols.get(symbol)
    
    def validate_order(self, symbol: str, qty: float, price: Optional[float] = None) -> tuple[bool, str]:
        """Validate order quantity and price against symbol filters."""
        info = self.symbols.get(symbol)
        if not info:
            return False, f"Unknown symbol: {symbol}"
        
        if not info.validate_qty(qty):
            return False, f"Invalid qty {qty}: min={info.min_qty}, max={info.max_qty}, step={info.step_size}"
        
        if price is not None and not info.validate_price(price):
            return False, f"Invalid price {price}: min={info.min_price}, max={info.max_price}"
        
        if price and not info.validate_notional(qty, price):
            return False, f"Notional too small: {qty}*{price} < {info.min_notional}"
        
        return True, "OK"
    
    # ========================================================
    # ACCOUNT
    # ========================================================
    
    async def get_account(self) -> dict:
        return await self._get("/fapi/v2/account", signed=True, weight=5)
    
    async def get_balance(self) -> list[AccountBalance]:
        data = await self.get_account()
        balances = []
        for b in data.get("assets", []):
            balances.append(AccountBalance(
                asset=b["asset"],
                wallet_balance=float(b["walletBalance"]),
                available_balance=float(b["availableBalance"]),
                unrealized_pnl=float(b["unrealizedProfit"]),
                margin_balance=float(b["marginBalance"]),
                max_withdraw_amount=float(b["maxWithdrawAmount"]),
            ))
        return balances
    
    async def get_positions(self) -> list[Position]:
        data = await self.get_account()
        positions = []
        for p in data.get("positions", []):
            amt = float(p["positionAmt"])
            if amt == 0:
                continue
            positions.append(Position(
                symbol=p["symbol"],
                position_amt=amt,
                entry_price=float(p["entryPrice"]),
                mark_price=float(p["markPrice"]),
                unrealized_pnl=float(p["unRealizedProfit"]),
                liquidation_price=float(p["liquidationPrice"]),
                leverage=int(p["leverage"]),
                isolated_wallet=float(p["isolatedWallet"]),
                is_isolated=p["isolated"],
            ))
        return positions
    
    async def change_leverage(self, symbol: str, leverage: int) -> dict:
        return await self._post("/fapi/v1/leverage", {"symbol": symbol, "leverage": leverage}, signed=True, weight=1)
    
    async def change_margin_type(self, symbol: str, margin_type: str) -> dict:
        return await self._post("/fapi/v1/marginType", {"symbol": symbol, "marginType": margin_type}, signed=True, weight=1)
    
    # ========================================================
    # ORDERS
    # ========================================================
    
    def _build_order_params(self, intent) -> dict:
        """Convert OrderIntentData to Binance params."""
        params = {
            "symbol": intent.symbol,
            "side": intent.side.value,
            "type": intent.order_type.value,
            "quantity": self.symbols[intent.symbol].round_qty(intent.quantity) if intent.symbol in self.symbols else intent.quantity,
        }
        
        if intent.order_type in {OrderType.LIMIT, OrderType.STOP_LIMIT, OrderType.TAKE_PROFIT_LIMIT}:
            params["price"] = self.symbols[intent.symbol].round_price(intent.price) if intent.symbol in self.symbols else intent.price
            params["timeInForce"] = intent.time_in_force.value
        
        if intent.order_type in {OrderType.STOP_MARKET, OrderType.STOP_LIMIT, OrderType.TAKE_PROFIT_MARKET, OrderType.TAKE_PROFIT_LIMIT}:
            params["stopPrice"] = self.symbols[intent.symbol].round_price(intent.stop_price) if intent.symbol in self.symbols else intent.stop_price
        
        if intent.reduce_only:
            params["reduceOnly"] = "true"
        
        return params
    
    async def place_order(self, intent) -> dict:
        params = self._build_order_params(intent)
        params["newClientOrderId"] = intent.metadata.get("client_order_id", f"quantai_{uuid.uuid4().hex[:12]}")
        
        return await self._post("/fapi/v1/order", params, signed=True, weight=1)
    
    async def place_order_with_client_id(self, intent, client_order_id: str) -> dict:
        """Place order with explicit clientOrderId for idempotency."""
        params = self._build_order_params(intent)
        params["newClientOrderId"] = client_order_id
        
        return await self._post("/fapi/v1/order", params, signed=True, weight=1)
    
    async def cancel_order(self, symbol: str, order_id: Optional[str] = None, client_order_id: Optional[str] = None) -> dict:
        params = {"symbol": symbol}
        if order_id:
            params["orderId"] = order_id
        if client_order_id:
            params["origClientOrderId"] = client_order_id
        return await self._delete("/fapi/v1/order", params, signed=True, weight=1)
    
    async def cancel_all_orders(self, symbol: str) -> dict:
        return await self._delete("/fapi/v1/allOpenOrders", {"symbol": symbol}, signed=True, weight=1)
    
    async def get_order(self, symbol: str, order_id: Optional[str] = None, client_order_id: Optional[str] = None) -> dict:
        params = {"symbol": symbol}
        if order_id:
            params["orderId"] = order_id
        if client_order_id:
            params["origClientOrderId"] = client_order_id
        return await self._get("/fapi/v1/order", params, signed=True, weight=1)
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> list:
        params = {}
        if symbol:
            params["symbol"] = symbol
        return await self._get("/fapi/v1/openOrders", params, signed=True, weight=1)
    
    # ========================================================
    # MARKET DATA
    # ========================================================
    
    async def get_ticker(self, symbol: str) -> dict:
        return await self._get("/fapi/v1/ticker/24hr", {"symbol": symbol}, weight=1)
    
    async def get_klines(self, symbol: str, interval: str, limit: int = 500) -> list:
        return await self._get("/fapi/v1/klines", {"symbol": symbol, "interval": interval, "limit": limit}, weight=2)
    
    async def get_mark_price(self, symbol: str) -> dict:
        return await self._get("/fapi/v1/premiumIndex", {"symbol": symbol}, weight=1)
    
    # ========================================================
    # USER DATA STREAM (Listen Key)
    # ========================================================
    
    async def start_user_stream(self) -> str:
        data = await self._post("/fapi/v1/listenKey", signed=True, weight=1)
        self._listen_key = data["listenKey"]
        return self._listen_key
    
    async def keepalive_user_stream(self) -> dict:
        return await self._put("/fapi/v1/listenKey", signed=True, weight=1)
    
    async def close_user_stream(self) -> dict:
        if self._listen_key:
            try:
                return await self._delete("/fapi/v1/listenKey", signed=True, weight=1)
            finally:
                self._listen_key = None
        return {}
    
    async def _put(self, endpoint: str, params: Optional[dict] = None, signed: bool = False, weight: int = 1):
        return await self._request("PUT", endpoint, params, signed, weight)
    
    async def keepalive_loop(self):
        """Background task to keep listen key alive."""
        while True:
            try:
                await asyncio.sleep(30 * 60)  # Every 30 min
                await self.keepalive_user_stream()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[Binance] Listen key keepalive error: {e}")


# ============================================================
# BINANCE WEBSOCKET ADAPTER
# ============================================================

class BinanceWebSocketAdapter:
    """Binance WebSocket adapter for User Data Stream and Market Data."""
    
    def __init__(self, config: BinanceConfig, rest_adapter: BinanceRestAdapter):
        self.config = config
        self.rest = rest_adapter
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self._running = False
        self._reconnect_delay = config.ws_reconnect_delay
        
        # Callbacks
        self.on_order_update: Optional[Callable[[dict], None]] = None
        self.on_account_update: Optional[Callable[[dict], None]] = None
        self.on_position_update: Optional[Callable[[dict], None]] = None
        self.on_balance_update: Optional[Callable[[dict], None]] = None
        self.on_trade: Optional[Callable[[dict], None]] = None
        self.on_kline: Optional[Callable[[dict], None]] = None
        self.on_ticker: Optional[Callable[[dict], None]] = None
        self.on_depth: Optional[Callable[[dict], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None
        self.on_connect: Optional[Callable[[], None]] = None
        self.on_disconnect: Optional[Callable[[], None]] = None
    
    async def connect_user_stream(self) -> None:
        """Connect to User Data Stream."""
        if not self.rest._listen_key:
            await self.rest.start_user_stream()
        
        url = f"{self.config.ws_base_url}/ws/{self.rest._listen_key}"
        await self._connect(url, "user")
    
    async def connect_market_stream(self, streams: list[str]) -> None:
        """Connect to market data streams (e.g., ['btcusdt@trade', 'btcusdt@kline_1m'])."""
        stream_path = "/".join(streams)
        url = f"{self.config.ws_base_url}/stream?streams={stream_path}"
        await self._connect(url, "market")
    
    async def _connect(self, url: str, stream_type: str):
        self._running = True
        self._reconnect_delay = self.config.ws_reconnect_delay
        
        while self._running:
            try:
                async with websockets.connect(
                    url,
                    ping_interval=self.config.ws_ping_interval,
                    ping_timeout=self.config.ws_ping_timeout,
                ) as ws:
                    self.ws = ws
                    if self.on_connect:
                        self.on_connect()
                    
                    async for message in ws:
                        if not self._running:
                            break
                        try:
                            data = json.loads(message)
                            self._handle_message(data, stream_type)
                        except Exception as e:
                            print(f"[Binance WS] Message parse error: {e}")
                    
            except websockets.exceptions.ConnectionClosed as e:
                if self.on_disconnect:
                    self.on_disconnect()
                if not self._running:
                    break
                print(f"[Binance WS] Connection closed: {e}, reconnecting in {self._reconnect_delay}s...")
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, self.config.ws_max_reconnect_delay)
            except Exception as e:
                if self.on_error:
                    self.on_error(e)
                if not self._running:
                    break
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, self.config.ws_max_reconnect_delay)
    
    def _handle_message(self, data: dict, stream_type: str):
        if stream_type == "user":
            self._handle_user_event(data)
        else:
            self._handle_market_event(data)
    
    def _handle_user_event(self, data: dict):
        event_type = data.get("e")
        
        if event_type == "ORDER_TRADE_UPDATE":
            if self.on_order_update:
                self.on_order_update(data["o"])
        elif event_type == "ACCOUNT_UPDATE":
            if self.on_account_update:
                self.on_account_update(data["a"])
            # Also trigger balance/position updates
            for pos in data["a"].get("P", []):
                if self.on_position_update:
                    self.on_position_update(pos)
            for bal in data["a"].get("B", []):
                if self.on_balance_update:
                    self.on_balance_update(bal)
        elif event_type == "ACCOUNT_CONFIG_UPDATE":
            pass  # Leverage/margin type changes
    
    def _handle_market_event(self, data: dict):
        stream = data.get("stream", "")
        event = data.get("data", {})
        
        if "@trade" in stream and self.on_trade:
            self.on_trade(event)
        elif "@kline" in stream and self.on_kline:
            self.on_kline(event["k"])
        elif "@ticker" in stream and self.on_ticker:
            self.on_ticker(event)
        elif "@depth" in stream and self.on_depth:
            self.on_depth(event)
    
    async def close(self):
        self._running = False
        if self.ws:
            await self.ws.close()
            self.ws = None
    
    # ========================================================
    # MARKET DATA SUBSCRIPTIONS
    # ========================================================
    
    async def subscribe_trades(self, symbols: list[str]):
        streams = [f"{s.lower()}@trade" for s in symbols]
        await self.connect_market_stream(streams)
    
    async def subscribe_klines(self, symbols: list[str], interval: str):
        streams = [f"{s.lower()}@kline_{interval}" for s in symbols]
        await self.connect_market_stream(streams)
    
    async def subscribe_tickers(self, symbols: list[str]):
        streams = [f"{s.lower()}@ticker" for s in symbols]
        await self.connect_market_stream(streams)
    
    async def subscribe_depth(self, symbols: list[str], levels: int = 20):
        streams = [f"{s.lower()}@depth{levels}@100ms" for s in symbols]
        await self.connect_market_stream(streams)


# ============================================================
# ERROR CLASSES
# ============================================================

class BinanceAPIError(Exception):
    def __init__(self, status: int, data: dict):
        self.status = status
        self.data = data
        self.code = data.get("code")
        self.msg = data.get("msg", "Unknown error")
        super().__init__(f"Binance API Error {status}: {self.code} - {self.msg}")


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "BinanceConfig",
    "SymbolInfo",
    "AccountBalance",
    "Position",
    "BinanceRestAdapter",
    "BinanceWebSocketAdapter",
    "BinanceAPIError",
    "RateLimiter",
]