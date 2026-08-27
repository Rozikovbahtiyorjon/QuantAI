"""
====================================================
QuantAI Professional
Application Lifecycle Management
====================================================

Startup and shutdown orchestration for all system components.
====================================================
"""

from __future__ import annotations

import asyncio
import signal
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from config.settings import Settings, settings
from src.monitoring.logging_config import setup_logging, get_logger, log_info, log_warning, log_error, log_critical
from src.monitoring.config_validation import validate_config_or_exit, run_startup_validation
from src.monitoring.health import HealthChecker, create_health_checker, HealthStatus
from src.monitoring.metrics import metrics, quantai_registry, metrics_endpoint
from src.execution.execution_engine import ExecutionEngine, ExecutionMode, ExecutionConfig
from src.execution.order_manager import OrderManager, OrderManagerConfig
from src.execution.binance_adapter import BinanceConfig, BinanceRestAdapter, BinanceWebSocketAdapter
from src.execution.reconciliation_engine import ReconciliationEngine, ReconciliationConfig, create_reconciliation_engine
from src.risk.risk_orchestrator import RiskOrchestrator, create_default_orchestrator
from src.paper_trading_engine import PaperTradingEngine
from src.ml_engine import MLEngine, MLConfig
from src.model_manager import ModelManager
from src.strategy import generate_signal_result
from src.production.disaster_recovery import CheckpointManager, StateSerializer
from src.production.order_deduplication import OrderDeduplicator
from src.production.rate_limiter import MultiLimitRateLimiter


@dataclass
class AppState:
    """Application state container."""
    settings: Settings
    logger: Any
    health_checker: Optional[Any] = None
    execution_engine: Optional[Any] = None
    binance_rest: Optional[Any] = None
    binance_ws: Optional[Any] = None
    order_manager: Optional[Any] = None
    risk_orchestrator: Optional[Any] = None
    reconciliation_engine: Optional[Any] = None
    paper_engine: Optional[Any] = None
    ml_engine: Optional[Any] = None
    model_manager: Optional[Any] = None
    # Production guards
    checkpoint_manager: Optional[Any] = None
    order_deduplicator: Optional[Any] = None
    rate_limiter: Optional[Any] = None
    start_time: float = field(default_factory=lambda: time.time())
    shutdown_event: asyncio.Event = field(default_factory=asyncio.Event)


# Global state
_state: Optional[AppState] = None


def get_state() -> AppState:
    """Get global application state."""
    if _state is None:
        raise RuntimeError("Application not initialized")
    return _state


async def initialize_logging() -> Any:
    """Initialize structured logging."""
    logger = setup_logging(
        level=settings.config.logging.log_level,
        json_format=True,
        log_file=settings.config.logging.trades_file,
    )
    log_info(logger, "Logging initialized", level=settings.config.logging.log_level)
    return logger


async def validate_configuration(logger) -> None:
    """Validate configuration on startup."""
    log_info(logger, "Validating configuration...")
    validate_config_or_exit(settings, logger)
    log_info(logger, "Configuration validation passed")


async def initialize_paper_engine(logger) -> PaperTradingEngine:
    """Initialize paper trading engine."""
    log_info(logger, "Initializing paper trading engine...")
    engine = PaperTradingEngine(
        initial_balance=settings.config.account.initial_balance,
        commission=settings.config.commission.commission,
    )
    log_info(logger, "Paper trading engine initialized")
    return engine


async def initialize_ml_engine(logger) -> tuple[MLEngine, ModelManager]:
    """Initialize ML engine and model manager."""
    log_info(logger, "Initializing ML engine...")
    
    ml_config = MLConfig(
        cv_type=settings.config.ml.cv_type,
        n_splits=settings.config.ml.n_splits,
        embargo_pct=settings.config.ml.embargo_pct,
        purge_pct=settings.config.ml.purge_pct,
        n_test_folds=settings.config.ml.n_test_folds,
        n_estimators=settings.config.ml.n_estimators,
        max_depth=settings.config.ml.max_depth,
        learning_rate=settings.config.ml.learning_rate,
        subsample=settings.config.ml.subsample,
        colsample_bytree=settings.config.ml.colsample_bytree,
        use_class_weights=settings.config.ml.use_class_weights,
    )
    
    model_manager = ModelManager()
    model_manager.model_path = Path(settings.config.ml.model_path)
    
    ml_engine = MLEngine(
        config=ml_config,
        load_existing=True,
    )
    
    if ml_engine.model is not None:
        log_info(logger, "ML model loaded from disk")
    else:
        log_warning(logger, "No ML model found, will need training")
    
    return ml_engine, model_manager


async def initialize_binance(logger, state: AppState) -> tuple[Optional[BinanceRestAdapter], Optional[BinanceWebSocketAdapter]]:
    """Initialize Binance adapters for DRY_RUN/LIVE modes."""
    if state.settings.config.mode == "PAPER":
        log_info(logger, "PAPER mode: skipping Binance initialization")
        return None, None
    
    log_info(logger, "Initializing Binance adapters...")
    
    binance_config = BinanceConfig(
        api_key=state.settings.config.binance.api_key,
        api_secret=state.settings.config.binance.api_secret,
        testnet=state.settings.config.binance.testnet,
        recv_window=state.settings.config.binance.recv_window,
    )
    
    binance_rest = BinanceRestAdapter(binance_config)
    await binance_rest.__aenter__()
    log_info(logger, "Binance REST adapter initialized")
    
    binance_ws = BinanceWebSocketAdapter(binance_config, binance_rest)
    log_info(logger, "Binance WebSocket adapter initialized")
    
    return binance_rest, binance_ws


async def initialize_order_manager(logger, binance_rest, binance_ws) -> OrderManager:
    """Initialize order manager with exchange callbacks."""
    log_info(logger, "Initializing order manager...")
    
    om_config = OrderManagerConfig(
        max_open_orders=settings.config.risk.max_open_positions or 100,
        max_order_age_seconds=3600,
        enable_order_expiration=True,
    )
    
    order_manager = OrderManager(om_config)
    
    # Set exchange callbacks
    if binance_rest and binance_ws:
        order_manager._submit_callback = binance_rest.place_order
        order_manager._cancel_callback = lambda o: binance_rest.cancel_order(
            symbol=o.intent.symbol,
            order_id=o.exchange_order_id,
        )
    
    log_info(logger, "Order manager initialized")
    return order_manager


async def initialize_risk_orchestrator(logger) -> RiskOrchestrator:
    """Initialize risk orchestrator."""
    log_info(logger, "Initializing risk orchestrator...")
    
    risk_orchestrator = create_default_orchestrator(
        max_drawdown_percent=settings.config.risk.drawdown_limit_pct,
        max_total_exposure_percent=settings.config.risk.max_total_exposure_pct,
        max_position_exposure_percent=settings.config.risk.max_position_exposure_pct,
        min_leverage=1.0,
        max_leverage=50.0,
        default_risk_percent=settings.config.account.risk_per_trade * 100,
        default_leverage=1.0,
    )
    
    log_info(logger, "Risk orchestrator initialized")
    return risk_orchestrator


async def initialize_reconciliation_engine(logger, binance_rest, order_manager) -> Optional[ReconciliationEngine]:
    """Initialize reconciliation engine."""
    if not binance_rest:
        return None
    
    log_info(logger, "Initializing reconciliation engine...")
    
    rec_config = ReconciliationConfig(
        interval_seconds=30.0,
        position_tolerance=0.001,
        balance_tolerance=1.0,
        enable_auto_fix=True,
        enable_ghost_detection=True,
        enable_stuck_order_detection=True,
    )
    
    reconciliation_engine = ReconciliationEngine(
        config=rec_config,
        binance_rest=binance_rest,
        order_manager=order_manager,
    )
    
    await reconciliation_engine.start()
    log_info(logger, "Reconciliation engine started")
    return reconciliation_engine


async def initialize_execution_engine(
    logger,
    state: AppState,
    paper_engine: PaperTradingEngine,
    binance_rest: Optional[BinanceRestAdapter],
    binance_ws: Optional[BinanceWebSocketAdapter],
    order_manager: OrderManager,
    risk_orchestrator: RiskOrchestrator,
    reconciliation_engine: Optional[ReconciliationEngine],
    rate_limiter: Optional[MultiLimitRateLimiter] = None,
    order_deduplicator: Optional[OrderDeduplicator] = None,
) -> ExecutionEngine:
    """Initialize execution engine."""
    log_info(logger, "Initializing execution engine...")
    
    # Determine mode
    mode_map = {
        "PAPER": ExecutionMode.PAPER,
        "DRY_RUN": ExecutionMode.DRY_RUN,
        "LIVE": ExecutionMode.LIVE,
    }
    mode = mode_map.get(settings.config.mode, ExecutionMode.PAPER)
    
    exec_config = ExecutionConfig(
        mode=mode,
        binance=BinanceConfig(
            api_key=settings.config.binance.api_key,
            api_secret=settings.config.binance.api_secret,
            testnet=settings.config.binance.testnet,
        ) if binance_rest else None,
        max_open_orders=settings.config.risk.max_open_positions or 100,
        reconciliation_interval_seconds=30.0,
        kill_switch_enabled=True,
        max_drawdown_pct=settings.config.risk.drawdown_limit_pct,
        max_daily_loss_pct=5.0,
        paper_fallback=True,
    )
    
    execution_engine = ExecutionEngine(
        config=exec_config,
        paper_engine=paper_engine,
        rate_limiter=rate_limiter,
        order_deduplicator=order_deduplicator,
        on_fill=lambda f: metrics.record_fill_latency(
            f.symbol, "MARKET", f.timestamp.timestamp()  # placeholder
        ),
    )
    
    # Inject dependencies
    execution_engine.order_manager = order_manager
    execution_engine.binance_rest = binance_rest
    execution_engine.binance_ws = binance_ws
    execution_engine.paper_engine = state.paper_engine
    
    # Inject rate_limiter into binance_rest if available
    if binance_rest and rate_limiter:
        binance_rest.rate_limiter = rate_limiter
    
    await execution_engine.start()
    log_info(logger, f"Execution engine started in {mode.value} mode")
    return execution_engine


async def initialize_health_checker(logger, state: AppState) -> HealthChecker:
    """Initialize health checker with all components."""
    log_info(logger, "Initializing health checker...")
    
    health_checker = create_health_checker(
        version="5.0.0",
        binance_rest=state.binance_rest,
        binance_ws=state.binance_ws,
        ml_engine=state.ml_engine,
        paper_engine=state.paper_engine,
        risk_orchestrator=state.risk_orchestrator,
        reconciliation_engine=state.reconciliation_engine,
    )
    
    # Mark startup complete after all init
    health_checker.mark_startup_complete()
    log_info(logger, "Health checker initialized")
    return health_checker


async def initialize_metrics_endpoint(logger) -> None:
    """Initialize metrics endpoint (for Prometheus scraping)."""
    log_info(logger, "Metrics endpoint available at /metrics")
    # The actual HTTP server would be set up by the web framework


async def initialize_checkpoint_manager(logger, state: AppState) -> CheckpointManager:
    """Initialize disaster recovery checkpoint manager."""
    log_info(logger, "Initializing checkpoint manager...")
    
    checkpoint_dir = Path("data/checkpoints")
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        interval_seconds=60.0,
        max_checkpoints=10,
    )
    
    # Register critical components for checkpointing
    if state.execution_engine:
        checkpoint_manager.register_component(
            "execution_engine",
            lambda: state.execution_engine.get_stats() if hasattr(state.execution_engine, 'get_stats') else {},
            priority=10,
        )
    if state.risk_orchestrator:
        checkpoint_manager.register_component(
            "risk_orchestrator",
            lambda: {"drawdown": state.risk_orchestrator.drawdown_guard.get_current_drawdown() if hasattr(state.risk_orchestrator, 'drawdown_guard') else 0},
            priority=5,
        )
    if state.paper_engine:
        checkpoint_manager.register_component(
            "paper_engine",
            lambda: {"balance": state.paper_engine.balance, "position": state.paper_engine.position},
            priority=10,
        )
    
    await checkpoint_manager.start()
    log_info(logger, "Checkpoint manager initialized")
    return checkpoint_manager


async def initialize_order_deduplicator(logger) -> OrderDeduplicator:
    """Initialize order deduplicator."""
    log_info(logger, "Initializing order deduplicator...")
    
    order_deduplicator = OrderDeduplicator(
        max_entries=10000,
        ttl_seconds=3600,
        cleanup_interval=300,
    )
    
    await order_deduplicator.start()
    log_info(logger, "Order deduplicator initialized")
    return order_deduplicator


async def initialize_rate_limiter(logger) -> MultiLimitRateLimiter:
    """Initialize rate limiter."""
    log_info(logger, "Initializing rate limiter...")
    
    rate_limiter = MultiLimitRateLimiter()
    
    log_info(logger, "Rate limiter initialized")
    return rate_limiter


async def startup() -> AppState:
    """
    Main startup sequence.
    
    Order:
    1. Logging
    2. Config validation
    3. Paper engine
    4. ML engine
    4. Binance (if not PAPER)
    5. Order manager
    6. Risk orchestrator
    7. Reconciliation engine
    8. Execution engine
    9. Health checker
    10. Metrics endpoint
    11. Checkpoint manager
    12. Order deduplicator
    13. Rate limiter
    """
    global _state
    
    import time
    start_time = time.time()
    
    # 1. Logging
    logger = await initialize_logging()
    log_info(logger, "=" * 60)
    log_info(logger, "QuantAI v5.0.0 Starting Up")
    log_info(logger, "=" * 60)
    log_info(logger, f"Mode: {settings.config.mode}")
    log_info(logger, f"Symbol: {settings.config.binance.symbol}")
    log_info(logger, f"Timeframe: {settings.config.binance.timeframe}")
    
    # 2. Config validation
    await validate_configuration(logger)
    
    # Create state container
    state = AppState(settings=settings, logger=logger)
    _state = state
    
    # 3. Paper engine
    state.paper_engine = await initialize_paper_engine(logger)
    
    # 4. ML engine
    state.ml_engine, state.model_manager = await initialize_ml_engine(logger)
    
    # 4. Binance (if not PAPER)
    state.binance_rest, state.binance_ws = await initialize_binance(logger, state)
    
    # 5. Order manager
    state.order_manager = await initialize_order_manager(logger, state.binance_rest, state.binance_ws)
    
    # 6. Risk orchestrator
    state.risk_orchestrator = await initialize_risk_orchestrator(logger)
    
    # 7. Reconciliation engine
    state.reconciliation_engine = await initialize_reconciliation_engine(
        logger, state.binance_rest, state.order_manager
    )
    
    # 8. Order deduplicator (moved before execution engine)
    state.order_deduplicator = await initialize_order_deduplicator(logger)
    
    # 9. Rate limiter (moved before execution engine)
    state.rate_limiter = await initialize_rate_limiter(logger)
    
    # 10. Execution engine
    state.execution_engine = await initialize_execution_engine(
        logger, state, state.paper_engine, state.binance_rest,
        state.binance_ws, state.order_manager, state.risk_orchestrator,
        state.reconciliation_engine, state.rate_limiter, state.order_deduplicator
    )
    
    # 11. Health checker
    state.health_checker = await initialize_health_checker(logger, state)
    
    # 12. Metrics endpoint
    await initialize_metrics_endpoint(logger)
    
    # 13. Checkpoint manager (disaster recovery)
    state.checkpoint_manager = await initialize_checkpoint_manager(logger, state)
    
    elapsed = time.time() - start_time
    log_info(logger, "=" * 60)
    log_info(logger, f"Startup Complete in {elapsed:.2f}s")
    log_info(logger, "=" * 60)
    
    return state


async def shutdown(state: Optional[AppState] = None) -> None:
    """
    Graceful shutdown sequence.
    
    Order (reverse of startup):
    1. Stop accepting new requests
    2. Cancel all open orders
    3. Stop reconciliation
    4. Close Binance connections
    5. Flush metrics/logs
    """
    if state is None:
        state = _state
    
    if state is None:
        return
    
    logger = state.logger
    log_info(logger, "=" * 60)
    log_info(logger, "Shutting Down...")
    log_info(logger, "=" * 60)
    
    state.shutdown_event.set()
    
    # 1. Stop execution engine
    if state.execution_engine:
        log_info(logger, "Stopping execution engine...")
        await state.execution_engine.stop()
    
    # 2. Stop reconciliation
    if state.reconciliation_engine:
        log_info(logger, "Stopping reconciliation engine...")
        await state.reconciliation_engine.stop()
    
    # 3. Stop rate limiter
    if state.rate_limiter:
        log_info(logger, "Stopping rate limiter...")
        # Rate limiter has no explicit stop, just let it go out of scope
    
    # 4. Stop order deduplicator
    if state.order_deduplicator:
        log_info(logger, "Stopping order deduplicator...")
        await state.order_deduplicator.stop()
    
    # 5. Stop checkpoint manager (save final checkpoint)
    if state.checkpoint_manager:
        log_info(logger, "Stopping checkpoint manager...")
        await state.checkpoint_manager.stop()
    
    # 6. Stop reconciliation
    if state.reconciliation_engine:
        log_info(logger, "Stopping reconciliation engine...")
        await state.reconciliation_engine.stop()
    
    # 7. Close Binance WebSocket
    if state.binance_ws:
        log_info(logger, "Closing Binance WebSocket...")
        await state.binance_ws.close()
    
    # 8. Close Binance REST
    if state.binance_rest:
        log_info(logger, "Closing Binance REST adapter...")
        await state.binance_rest.__aexit__(None, None, None)
    
    # 9. Health checker
    if state.health_checker:
        log_info(logger, "Stopping health checker...")
        await state.health_checker.stop()
    
    elapsed = time.time() - state.start_time
    log_info(logger, "=" * 60)
    log_info(logger, f"Shutdown Complete (uptime: {elapsed:.2f}s)")
    log_info(logger, "=" * 60)
    
    # Clear global state
    _state = None


def setup_signal_handlers(state: AppState):
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(sig, frame):
        log_info(state.logger, f"Received signal {sig}, initiating shutdown...")
        asyncio.create_task(shutdown(state))
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


@asynccontextmanager
async def lifespan():
    """Async context manager for application lifespan."""
    state = await startup()
    try:
        yield state
    finally:
        await shutdown(state)


def run_app():
    """Run the application with proper lifecycle management."""
    async def main():
        state = await startup()
        setup_signal_handlers(state)
        
        try:
            # Keep running until shutdown
            await state.shutdown_event.wait()
        except asyncio.CancelledError:
            pass
        finally:
            await shutdown(state)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "AppState",
    "get_state",
    "startup",
    "shutdown",
    "setup_signal_handlers",
    "lifespan",
    "run_app",
]


# ============================================================
# MAIN ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    run_app()