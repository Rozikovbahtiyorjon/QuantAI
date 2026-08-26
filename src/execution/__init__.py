"""
====================================================
QuantAI Professional
Execution Package
====================================================

Execution boundary components:
- Orders: Intent, Order, Fill definitions
- OrderManager: Order lifecycle management
- BinanceAdapter: REST + WebSocket for Binance Futures
- ExecutionEngine: High-level coordinator
- ReconciliationEngine: State synchronization
====================================================
"""

from src.execution.orders import (
    OrderSide,
    OrderType,
    OrderStatus,
    TimeInForce,
    OrderIntent,
    OrderIntentData,
    Order,
    Fill,
)

from src.execution.order_manager import (
    OrderManagerConfig,
    OrderManager,
)

from src.execution.binance_adapter import (
    BinanceConfig,
    SymbolInfo,
    AccountBalance,
    Position,
    BinanceRestAdapter,
    BinanceWebSocketAdapter,
    BinanceAPIError,
    RateLimiter,
)

from src.execution.execution_engine import (
    ExecutionMode,
    ExecutionConfig,
    ExecutionStats,
    ExecutionEngine,
    ExecutionError,
    create_execution_engine,
)

from src.execution.reconciliation_engine import (
    ReconciliationAction,
    ReconciliationIssue,
    ReconciliationReport,
    ReconciliationConfig,
    ReconciliationEngine,
    create_reconciliation_engine,
)

__all__ = [
    # Orders
    "OrderSide",
    "OrderType",
    "OrderStatus",
    "TimeInForce",
    "OrderIntent",
    "OrderIntentData",
    "Order",
    "Fill",
    
    # Order Manager
    "OrderManagerConfig",
    "OrderManager",
    
    # Binance Adapter
    "BinanceConfig",
    "SymbolInfo",
    "AccountBalance",
    "Position",
    "BinanceRestAdapter",
    "BinanceWebSocketAdapter",
    "BinanceAPIError",
    "RateLimiter",
    
    # Execution Engine
    "ExecutionMode",
    "ExecutionConfig",
    "ExecutionStats",
    "ExecutionEngine",
    "ExecutionError",
    "create_execution_engine",
    
    # Reconciliation
    "ReconciliationAction",
    "ReconciliationIssue",
    "ReconciliationReport",
    "ReconciliationConfig",
    "ReconciliationEngine",
    "create_reconciliation_engine",
]

__version__ = "1.0.0"