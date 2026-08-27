"""
====================================================
QuantAI Professional
Prometheus Metrics
====================================================

Prometheus metrics for trading system observability.
====================================================
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Optional

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Summary,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST,
    REGISTRY,
)


# ============================================================
# CUSTOM REGISTRY (optional isolation)
# ============================================================

quantai_registry = CollectorRegistry()


# ============================================================
# METRIC DEFINITIONS
# ============================================================

# --- System Metrics ---

system_info = Gauge(
    "quantai_system_info",
    "System information",
    ["version", "mode", "environment"],
    registry=quantai_registry,
)

system_uptime_seconds = Gauge(
    "quantai_system_uptime_seconds",
    "System uptime in seconds",
    registry=quantai_registry,
)

# --- Execution Metrics ---

execution_intents_total = Counter(
    "quantai_execution_intents_total",
    "Total number of execution intents received",
    ["mode", "symbol", "side", "intent_type"],
    registry=quantai_registry,
)

execution_orders_created_total = Counter(
    "quantai_execution_orders_created_total",
    "Total orders created",
    ["mode", "symbol", "side", "order_type"],
    registry=quantai_registry,
)

execution_orders_submitted_total = Counter(
    "quantai_execution_orders_submitted_total",
    "Total orders submitted to exchange",
    ["mode", "symbol", "exchange"],
    registry=quantai_registry,
)

execution_orders_filled_total = Counter(
    "quantai_execution_orders_filled_total",
    "Total orders filled",
    ["mode", "symbol", "side"],
    registry=quantai_registry,
)

execution_orders_canceled_total = Counter(
    "quantai_execution_orders_canceled_total",
    "Total orders canceled",
    ["mode", "symbol", "reason"],
    registry=quantai_registry,
)

execution_orders_rejected_total = Counter(
    "quantai_execution_orders_rejected_total",
    "Total orders rejected",
    ["mode", "symbol", "reason"],
    registry=quantai_registry,
)

execution_orders_failed_total = Counter(
    "quantai_execution_orders_failed_total",
    "Total orders failed (submission error)",
    ["mode", "symbol", "error_type"],
    registry=quantai_registry,
)

execution_fill_latency_seconds = Histogram(
    "quantai_execution_fill_latency_seconds",
    "Time from order submission to fill",
    ["mode", "symbol", "order_type"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    registry=quantai_registry,
)

execution_submission_latency_seconds = Histogram(
    "quantai_execution_submission_latency_seconds",
    "Time from intent to order submission",
    ["mode", "symbol"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
    registry=quantai_registry,
)

execution_open_orders = Gauge(
    "quantai_execution_open_orders",
    "Number of currently open orders",
    ["mode", "symbol", "side"],
    registry=quantai_registry,
)

execution_position_qty = Gauge(
    "quantai_execution_position_qty",
    "Current position quantity",
    ["mode", "symbol", "side"],
    registry=quantai_registry,
)

execution_position_notional = Gauge(
    "quantai_execution_position_notional",
    "Current position notional value (USDT)",
    ["mode", "symbol"],
    registry=quantai_registry,
)

# --- Risk Metrics ---

risk_drawdown_pct = Gauge(
    "quantai_risk_drawdown_pct",
    "Current drawdown percentage",
    ["mode"],
    registry=quantai_registry,
)

risk_daily_pnl = Gauge(
    "quantai_risk_daily_pnl",
    "Daily P&L",
    ["mode"],
    registry=quantai_registry,
)

risk_daily_pnl_pct = Gauge(
    "quantai_risk_daily_pnl_pct",
    "Daily P&L percentage",
    ["mode"],
    registry=quantai_registry,
)

risk_position_exposure_pct = Gauge(
    "quantai_risk_position_exposure_pct",
    "Position exposure as percentage of equity",
    ["mode", "symbol"],
    registry=quantai_registry,
)

risk_total_exposure_pct = Gauge(
    "quantai_risk_total_exposure_pct",
    "Total portfolio exposure percentage",
    ["mode"],
    registry=quantai_registry,
)

risk_margin_ratio = Gauge(
    "quantai_risk_margin_ratio",
    "Margin ratio (equity / used margin)",
    ["mode"],
    registry=quantai_registry,
)

risk_kill_switch_active = Gauge(
    "quantai_risk_kill_switch_active",
    "Whether kill switch is active (1) or not (0)",
    ["mode"],
    registry=quantai_registry,
)

risk_checks_total = Counter(
    "quantai_risk_checks_total",
    "Total risk checks performed",
    ["mode", "check_type", "result"],
    registry=quantai_registry,
)

# --- ML Metrics ---

ml_model_loaded = Gauge(
    "quantai_ml_model_loaded",
    "Whether ML model is loaded (1) or not (0)",
    ["model_path"],
    registry=quantai_registry,
)

ml_predictions_total = Counter(
    "quantai_ml_predictions_total",
    "Total ML predictions made",
    ["model", "signal", "quality_gate_passed"],
    registry=quantai_registry,
)

ml_prediction_latency_seconds = Histogram(
    "quantai_ml_prediction_latency_seconds",
    "ML prediction latency",
    ["model"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
    registry=quantai_registry,
)

ml_confidence_distribution = Histogram(
    "quantai_ml_confidence_distribution",
    "ML prediction confidence distribution",
    ["model", "signal"],
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    registry=quantai_registry,
)

ml_quality_gate_checks = Counter(
    "quantai_ml_quality_gate_checks_total",
    "ML quality gate checks",
    ["model", "gate", "result"],
    registry=quantai_registry,
)

ml_model_retrains = Counter(
    "quantai_ml_model_retrains_total",
    "Number of model retrains",
    ["model", "trigger"],
    registry=quantai_registry,
)

ml_disagreement_blocks = Counter(
    "quantai_ml_disagreement_blocks_total",
    "ML disagreement blocks (ML vs Strategy)",
    ["strategy_signal", "ml_signal"],
    registry=quantai_registry,
)

# --- Feature Store Metrics ---

feature_store_features_logged_total = Counter(
    "quantai_feature_store_features_logged_total",
    "Total live features logged to Feature Store",
    ["view"],
    registry=quantai_registry,
)

feature_store_versions_total = Counter(
    "quantai_feature_store_versions_total",
    "Total Feature Store versions materialized",
    ["view"],
    registry=quantai_registry,
)

feature_store_drift_alerts_total = Counter(
    "quantai_feature_store_drift_alerts_total",
    "Total drift alerts fired",
    ["view", "feature"],
    registry=quantai_registry,
)

feature_store_buffer_size = Gauge(
    "quantai_feature_store_buffer_size",
    "Current live feature buffer size",
    ["view"],
    registry=quantai_registry,
)

feature_store_psi = Gauge(
    "quantai_feature_store_psi",
    "Population Stability Index per feature",
    ["view", "feature"],
    registry=quantai_registry,
)

feature_store_ks_pvalue = Gauge(
    "quantai_feature_store_ks_pvalue",
    "KS-test p-value per feature",
    ["view", "feature"],
    registry=quantai_registry,
)

feature_store_drift_check_duration_seconds = Histogram(
    "quantai_feature_store_drift_check_duration_seconds",
    "Drift check duration",
    ["view"],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
    registry=quantai_registry,
)

feature_store_last_materialize_timestamp = Gauge(
    "quantai_feature_store_last_materialize_timestamp",
    "Unix timestamp of last materialization",
    ["view"],
    registry=quantai_registry,
)

# --- Reconciliation Metrics ---

reconciliation_runs_total = Counter(
    "quantai_reconciliation_runs_total",
    "Total reconciliation runs",
    ["result"],
    registry=quantai_registry,
)

reconciliation_duration_seconds = Histogram(
    "quantai_reconciliation_duration_seconds",
    "Reconciliation cycle duration",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
    registry=quantai_registry,
)

reconciliation_issues_found = Counter(
    "quantai_reconciliation_issues_found_total",
    "Issues found during reconciliation",
    ["issue_type", "severity"],
    registry=quantai_registry,
)

reconciliation_fixes_applied = Counter(
    "quantai_reconciliation_fixes_applied_total",
    "Fixes applied during reconciliation",
    ["fix_type"],
    registry=quantai_registry,
)

reconciliation_position_mismatches = Gauge(
    "quantai_reconciliation_position_mismatches",
    "Current position mismatches",
    ["symbol"],
    registry=quantai_registry,
)

reconciliation_balance_mismatch = Gauge(
    "quantai_reconciliation_balance_mismatch",
    "Balance mismatch amount",
    ["asset"],
    registry=quantai_registry,
)

reconciliation_stuck_orders = Gauge(
    "quantai_reconciliation_stuck_orders",
    "Number of stuck orders detected",
    ["symbol"],
    registry=quantai_registry,
)

# --- Paper Trading Metrics ---

paper_balance = Gauge(
    "quantai_paper_balance",
    "Paper trading account balance",
    ["mode"],
    registry=quantai_registry,
)

paper_equity = Gauge(
    "quantai_paper_equity",
    "Paper trading account equity",
    ["mode"],
    registry=quantai_registry,
)

paper_realized_pnl = Gauge(
    "quantai_paper_realized_pnl",
    "Paper trading realized P&L",
    ["mode"],
    registry=quantai_registry,
)

paper_unrealized_pnl = Gauge(
    "quantai_paper_unrealized_pnl",
    "Paper trading unrealized P&L",
    ["mode"],
    registry=quantai_registry,
)

paper_open_positions = Gauge(
    "quantai_paper_open_positions",
    "Number of open paper positions",
    ["mode"],
    registry=quantai_registry,
)

paper_trades_total = Counter(
    "quantai_paper_trades_total",
    "Total paper trades",
    ["mode", "side", "result"],
    registry=quantai_registry,
)

paper_win_rate = Gauge(
    "quantai_paper_win_rate",
    "Paper trading win rate",
    ["mode"],
    registry=quantai_registry,
)

# --- Exchange/Connectivity Metrics ---

exchange_connectivity = Gauge(
    "quantai_exchange_connectivity",
    "Exchange connectivity status (1=connected, 0=disconnected)",
    ["exchange", "stream"],
    registry=quantai_registry,
)

exchange_latency_seconds = Histogram(
    "quantai_exchange_latency_seconds",
    "Exchange API latency",
    ["exchange", "endpoint", "method"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    registry=quantai_registry,
)

exchange_rate_limit_remaining = Gauge(
    "quantai_exchange_rate_limit_remaining",
    "Exchange rate limit remaining",
    ["exchange", "limit_type"],
    registry=quantai_registry,
)

exchange_rate_limit_exceeded = Counter(
    "quantai_exchange_rate_limit_exceeded_total",
    "Rate limit exceeded events",
    ["exchange", "limit_type"],
    registry=quantai_registry,
)

exchange_errors_total = Counter(
    "quantai_exchange_errors_total",
    "Exchange API errors",
    ["exchange", "endpoint", "error_code"],
    registry=quantai_registry,
)

ws_reconnects_total = Counter(
    "quantai_ws_reconnects_total",
    "WebSocket reconnection count",
    ["exchange", "stream"],
    registry=quantai_registry,
)

ws_message_latency_seconds = Histogram(
    "quantai_ws_message_latency_seconds",
    "WebSocket message processing latency",
    ["exchange", "stream", "message_type"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
    registry=quantai_registry,
)

# --- Strategy/Signal Metrics ---

strategy_signals_total = Counter(
    "quantai_strategy_signals_total",
    "Total strategy signals generated",
    ["signal", "symbol", "timeframe"],
    registry=quantai_registry,
)

strategy_signal_confidence = Histogram(
    "quantai_strategy_signal_confidence",
    "Strategy signal confidence distribution",
    ["signal"],
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    registry=quantai_registry,
)

strategy_fusion_agreement = Counter(
    "quantai_strategy_fusion_agreement_total",
    "AI/ML fusion agreement",
    ["ai_signal", "ml_signal", "result"],
    registry=quantai_registry,
)

order_flow_signals_total = Counter(
    "quantai_order_flow_signals_total",
    "Order flow signals",
    ["context", "symbol"],
    registry=quantai_registry,
)

# --- Performance Metrics ---

performance_sharpe_ratio = Gauge(
    "quantai_performance_sharpe_ratio",
    "Portfolio Sharpe ratio",
    ["mode", "period"],
    registry=quantai_registry,
)

performance_sortino_ratio = Gauge(
    "quantai_performance_sortino_ratio",
    "Portfolio Sortino ratio",
    ["mode", "period"],
    registry=quantai_registry,
)

performance_max_drawdown = Gauge(
    "quantai_performance_max_drawdown",
    "Maximum drawdown",
    ["mode", "period"],
    registry=quantai_registry,
)

performance_total_return = Gauge(
    "quantai_performance_total_return",
    "Total return percentage",
    ["mode", "period"],
    registry=quantai_registry,
)

performance_profit_factor = Gauge(
    "quantai_performance_profit_factor",
    "Profit factor",
    ["mode", "period"],
    registry=quantai_registry,
)


# ============================================================
# METRICS HELPER CLASS
# ============================================================

@dataclass
class MetricsCollector:
    """Convenience class for updating metrics."""
    
    mode: str = "PAPER"
    
    def set_mode(self, mode: str):
        self.mode = mode
    
    # --- Execution ---
    
    def record_intent(self, symbol: str, side: str, intent_type: str):
        execution_intents_total.labels(
            mode=self.mode, symbol=symbol, side=side, intent_type=intent_type
        ).inc()
    
    def record_order_created(self, symbol: str, side: str, order_type: str):
        execution_orders_created_total.labels(
            mode=self.mode, symbol=symbol, side=side, order_type=order_type
        ).inc()
    
    def record_order_submitted(self, symbol: str, exchange: str = "binance"):
        execution_orders_submitted_total.labels(
            mode=self.mode, symbol=symbol, exchange=exchange
        ).inc()
    
    def record_order_filled(self, symbol: str, side: str):
        execution_orders_filled_total.labels(
            mode=self.mode, symbol=symbol, side=side
        ).inc()
    
    def record_order_canceled(self, symbol: str, reason: str):
        execution_orders_canceled_total.labels(
            mode=self.mode, symbol=symbol, reason=reason
        ).inc()
    
    def record_order_rejected(self, symbol: str, reason: str):
        execution_orders_rejected_total.labels(
            mode=self.mode, symbol=symbol, reason=reason
        ).inc()
    
    def record_fill_latency(self, symbol: str, order_type: str, latency: float):
        execution_fill_latency_seconds.labels(
            mode=self.mode, symbol=symbol, order_type=order_type
        ).observe(latency)
    
    def set_open_orders(self, symbol: str, side: str, count: int):
        execution_open_orders.labels(
            mode=self.mode, symbol=symbol, side=side
        ).set(count)
    
    def set_position(self, symbol: str, side: str, qty: float, notional: float = 0.0):
        execution_position_qty.labels(
            mode=self.mode, symbol=symbol, side=side
        ).set(qty)
        if notional:
            execution_position_notional.labels(
                mode=self.mode, symbol=symbol
            ).set(notional)
    
    # --- Risk ---
    
    def set_drawdown(self, pct: float):
        risk_drawdown_pct.labels(mode=self.mode).set(pct)
    
    def set_daily_pnl(self, pnl: float, pct: float = 0.0):
        risk_daily_pnl.labels(mode=self.mode).set(pnl)
        if pct:
            risk_daily_pnl_pct.labels(mode=self.mode).set(pct)
    
    def set_exposure(self, symbol: str, pct: float):
        risk_position_exposure_pct.labels(mode=self.mode, symbol=symbol).set(pct)
    
    def set_total_exposure(self, pct: float):
        risk_total_exposure_pct.labels(mode=self.mode).set(pct)
    
    def set_kill_switch(self, active: bool):
        risk_kill_switch_active.labels(mode=self.mode).set(1 if active else 0)
    
    def record_risk_check(self, check_type: str, passed: bool):
        risk_checks_total.labels(
            mode=self.mode, check_type=check_type, result="pass" if passed else "fail"
        ).inc()
    
    # --- ML ---
    
    def record_ml_prediction(self, model: str, signal: str, quality_gate_passed: bool, latency: float, confidence: float):
        ml_predictions_total.labels(
            model=model, signal=signal, quality_gate_passed=str(quality_gate_passed)
        ).inc()
        ml_prediction_latency_seconds.labels(model=model).observe(latency)
        ml_confidence_distribution.labels(model=model, signal=signal).observe(confidence)
    
    def record_quality_gate(self, model: str, gate: str, passed: bool):
        ml_quality_gate_checks.labels(
            model=model, gate=gate, result="pass" if passed else "fail"
        ).inc()
    
    def record_disagreement(self, strategy_signal: str, ml_signal: str):
        ml_disagreement_blocks.labels(
            strategy_signal=strategy_signal, ml_signal=ml_signal
        ).inc()
    
    # --- Reconciliation ---
    
    def record_reconciliation(self, duration: float, issues: int, fixes: int, success: bool):
        reconciliation_runs_total.labels(result="success" if success else "failure").inc()
        reconciliation_duration_seconds.observe(duration)
        reconciliation_issues_found.labels(
            issue_type="all", severity="all"
        ).inc(issues)
        reconciliation_fixes_applied.labels(fix_type="all").inc(fixes)
    
    def set_position_mismatch(self, symbol: str, diff: float):
        reconciliation_position_mismatches.labels(symbol=symbol).set(diff)
    
    def set_balance_mismatch(self, asset: str, diff: float):
        reconciliation_balance_mismatch.labels(asset=asset).set(diff)
    
    def set_stuck_orders(self, symbol: str, count: int):
        reconciliation_stuck_orders.labels(symbol=symbol).set(count)
    
    # --- Paper ---
    
    def set_paper_balance(self, balance: float, equity: float = 0.0):
        paper_balance.labels(mode=self.mode).set(balance)
        if equity:
            paper_equity.labels(mode=self.mode).set(equity)
    
    def set_paper_pnl(self, realized: float, unrealized: float = 0.0):
        paper_realized_pnl.labels(mode=self.mode).set(realized)
        if unrealized:
            paper_unrealized_pnl.labels(mode=self.mode).set(unrealized)
    
    def record_paper_trade(self, side: str, won: bool):
        paper_trades_total.labels(
            mode=self.mode, side=side, result="win" if won else "loss"
        ).inc()
    
    def set_paper_win_rate(self, rate: float):
        paper_win_rate.labels(mode=self.mode).set(rate)
    
    # --- Exchange ---
    
    def set_connectivity(self, exchange: str, stream: str, connected: bool):
        exchange_connectivity.labels(exchange=exchange, stream=stream).set(1 if connected else 0)
    
    def record_exchange_latency(self, exchange: str, endpoint: str, method: str, latency: float):
        exchange_latency_seconds.labels(
            exchange=exchange, endpoint=endpoint, method=method
        ).observe(latency)
    
    def record_exchange_error(self, exchange: str, endpoint: str, error_code: str):
        exchange_errors_total.labels(
            exchange=exchange, endpoint=endpoint, error_code=error_code
        ).inc()
    
    def record_ws_reconnect(self, exchange: str, stream: str):
        ws_reconnects_total.labels(exchange=exchange, stream=stream).inc()
    
    def record_ws_latency(self, exchange: str, stream: str, msg_type: str, latency: float):
        ws_message_latency_seconds.labels(
            exchange=exchange, stream=stream, message_type=msg_type
        ).observe(latency)
    
    # --- Strategy ---
    
    def record_signal(self, signal: str, symbol: str, timeframe: str, confidence: float):
        strategy_signals_total.labels(
            signal=signal, symbol=symbol, timeframe=timeframe
        ).inc()
        strategy_signal_confidence.labels(signal=signal).observe(confidence)
    
    def record_fusion(self, ai_signal: str, ml_signal: str, result: str):
        strategy_fusion_agreement.labels(
            ai_signal=ai_signal, ml_signal=ml_signal, result=result
        ).inc()
    
    # --- Performance ---
    
    def set_performance(self, sharpe: float, sortino: float, max_dd: float, total_return: float, profit_factor: float, period: str = "all"):
        performance_sharpe_ratio.labels(mode=self.mode, period=period).set(sharpe)
        performance_sortino_ratio.labels(mode=self.mode, period=period).set(sortino)
        performance_max_drawdown.labels(mode=self.mode, period=period).set(max_dd)
        performance_total_return.labels(mode=self.mode, period=period).set(total_return)
        performance_profit_factor.labels(mode=self.mode, period=period).set(profit_factor)


# Global collector instance
metrics = MetricsCollector()


# ============================================================
# DECORATORS
# ============================================================

def measure_latency(metric: Histogram, **labels):
    """Decorator to measure function latency."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                metric.labels(**labels).observe(time.perf_counter() - start)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                metric.labels(**labels).observe(time.perf_counter() - start)
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator


def count_calls(metric: Counter, **labels):
    """Decorator to count function calls."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            metric.labels(**labels).inc()
            return func(*args, **kwargs)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            metric.labels(**labels).inc()
            return await func(*args, **kwargs)
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator


# ============================================================
# HTTP ENDPOINT FOR PROMETHEUS
# ============================================================

def metrics_endpoint() -> tuple[bytes, str]:
    """Generate Prometheus metrics output."""
    return generate_latest(quantai_registry), CONTENT_TYPE_LATEST


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "quantai_registry",
    "MetricsCollector",
    "metrics",
    "measure_latency",
    "count_calls",
    "metrics_endpoint",
]