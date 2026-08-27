# QuantAI Execution Boundary Architecture

## Level 3 — Safety-Critical Path

```
Final Signal (Strategy)
    ↓
OrderIntent (Risk Approved)
    ↓
Safety Guards
    ↓
Exchange Adapter
    ↓
Exchange
    ↓
Reconciliation
```

## Component Contracts

### 1. Signal → OrderIntent (`src/execution/orders.py`)

```python
@dataclass
class OrderIntent:
    symbol: str
    side: OrderSide       # BUY/SELL
    order_type: OrderType # MARKET/LIMIT
    quantity: float
    price: float | None   # required for LIMIT
    reduce_only: bool = False
    time_in_force: TimeInForce = GTC
    client_order_id: str  # UUID for idempotency
    metadata: dict        # strategy_id, confidence, risk_decision_ref
```

**Invariants**:
- `quantity > 0`
- `LIMIT` → `price > 0`
- `client_order_id` unique per intent
- `metadata` contains full traceability

---

### 2. OrderManager (`src/execution/order_manager.py`)

**Responsibilities**:
- Intent → Order lifecycle (NEW → SUBMITTED → PARTIAL/FILLED/CANCELED/REJECTED/EXPIRED)
- Retry logic (max 3, exponential backoff)
- Order expiration (max 1hr default)
- State persistence (SQLite/Redis)

**Callbacks** (set by ExecutionEngine):
```python
_submit_callback: Callable[[Order], Awaitable[bool]]
_cancel_callback: Callable[[Order], Awaitable[bool]]
```

**Contract**:
- `submit_intent(intent)` → `Order` (immediate, local state)
- `cancel_order(order_id)` → `bool`
- `get_order(order_id)` → `Order | None`
- `get_active_orders(symbol?)` → `List[Order]`

---

### 3. Safety Guards (pre-submission)

Executed **inside ExecutionEngine** before OrderManager:

| Guard | Check | Failure Action |
|-------|-------|----------------|
| Kill Switch | `daily_pnl > -max_daily_loss%` AND `drawdown < max_dd%` | Reject intent, alert |
| Position Limit | `open_positions < max_open` | Reject intent |
| Exposure Limit | `RiskOrchestrator.can_open()` | Reject intent |
| Balance Check | `balance > min_notional` | Reject intent |
| Sanity Check | `quantity * price < max_notional` | Reject intent |

**Contract**: All guards MUST pass. Fail = `ExecutionError`, intent rejected, logged.

---

### 4. Binance Adapter (`src/execution/binance_adapter.py`)

#### REST Adapter (`BinanceRestAdapter`)
- `place_order(intent)` → `dict` with `orderId`
- `cancel_order(symbol, order_id)` → `dict`
- `get_balance()` → `List[Balance]`
- `get_positions()` → `List[Position]`
- `get_mark_price(symbol)` → `float`
- `validate_order(symbol, qty, price)` → `(bool, str)`
- `keepalive_loop()` → maintain listenKey

#### WebSocket Adapter (`BinanceWebSocketAdapter`)
- User data stream: order updates, account, position, balance
- Auto-reconnect with exponential backoff
- Callbacks: `on_order_update`, `on_account_update`, `on_position_update`, `on_balance_update`, `on_error`

**Contract**: 
- All network calls timeout=10s
- Rate limit: respect `X-MBX-USED-WEIGHT` header
- Signature: HMAC-SHA256 with API secret

---

### 5. ReconciliationEngine (`src/execution/reconciliation_engine.py`)

**Runs every 30s** (configurable):

| Check | Tolerance | Fix Action |
|-------|-----------|------------|
| Position qty | ±0.001 | Overwrite local with exchange |
| Balance (USDT) | ±$1.0 | Overwrite local with exchange |
| Ghost positions | any | Add to local cache |
| Open orders | any | Sync status from exchange |

**Contract**: 
- Reports: `{"timestamp", "fixes_applied", "issues", "positions", "balance"}`
- Callback: `on_reconciliation_fix(symbol, {"type", "old", "new"})`
- Never auto-flattens positions (manual only)

---

### 6. ExecutionEngine (`src/execution/execution_engine.py`)

**Modes**: `PAPER` | `DRY_RUN` | `LIVE`

**Flow**:
```
submit_intent(intent)
    ↓
safety_check(intent) → bool
    ↓
order_manager.submit_intent(intent) → Order
    ↓
OrderManager._submit_callback → BinanceAdapter / PaperEngine
    ↓
WS callbacks → order_manager.on_order_update / on_fill
    ↓
reconciliation_loop() periodic
```

**Contract**:
- `start()` → initializes all components, gets initial balance
- `submit_intent(intent)` → `Order` (trackable)
- `stop()` → cancels all, closes WS, saves state
- `emergency_stop()` → cancels all, returns `{canceled_orders, timestamp}`
- `get_stats()` → `{intents, orders, fills, reconciliations, errors}`
- `get_position(symbol)` → `float`
- `get_balance()` → `float`

---

## Mode Differences

| Aspect | PAPER | DRY_RUN | LIVE |
|--------|-------|---------|------|
| Orders | PaperTradingEngine | Binance validate only | Binance real |
| Fills | Immediate at signal price | Simulated at mark price | Real |
| Balance | Paper engine | Exchange (read-only) | Exchange |
| Positions | Paper engine | Exchange (read-only) | Exchange |
| Reconciliation | N/A | Every 30s | Every 30s |
| Kill Switch | Enabled | Enabled | Enabled |
| Fees | Config | Real fee structure | Real |

---

## Error Handling

| Error Type | Handling |
|------------|----------|
| Network timeout | Retry 3x, then reject, alert |
| Rate limit (429) | Backoff 1s→2s→4s, retry |
| Auth error (401) | Kill switch, alert, stop |
| Insufficient margin | Reject intent, alert |
| Order rejected | Log, update Order status=REJECTED |
| WS disconnect | Auto-reconnect, replay missed on reconnect |
| Reconciliation drift > tolerance | Log, fix local, alert |

---

## Observability

**Metrics** (Prometheus):
- `execution_intents_total{mode,result}`
- `execution_orders_total{mode,status}`
- `execution_fills_total{mode,side}`
- `execution_latency_seconds{mode,operation}`
- `reconciliation_fixes_total{type}`
- `kill_switch_activations_total`

**Logs** (structured JSON):
```json
{
  "ts": "2026-08-25T10:00:00Z",
  "level": "INFO",
  "component": "ExecutionEngine",
  "event": "intent_submitted",
  "intent_id": "uuid",
  "symbol": "BTC/USDT",
  "side": "BUY",
  "qty": 0.01,
  "mode": "LIVE"
}
```

---

## Module Status

| Module | Written | Unit | Integrated | E2E | Long-Run | Prod |
|--------|---------|------|------------|-----|----------|------|
| orders.py | ✓ | ✓ | ✓ | ? | ? | ✗ |
| order_manager.py | ✓ | ✓ | ✓ | ? | ? | ✗ |
| binance_adapter.py | ✓ | ✓ | ✓ | ? | ? | ✗ |
| execution_engine.py | ✓ | ✓ | ? | ? | ? | ✗ |
| reconciliation_engine.py | ✓ | ✓ | ? | ? | ? | ✗ |

---

**Status**: Execution boundary contracts defined. Next: Governance (Level 4).