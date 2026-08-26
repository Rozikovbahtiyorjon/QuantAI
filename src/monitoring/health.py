"""
====================================================
QuantAI Professional
Health Checks
====================================================

Health and readiness endpoints for Kubernetes/container orchestration.
====================================================
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional

from prometheus_client import Gauge, CollectorRegistry

from src.monitoring.metrics import quantai_registry


class HealthStatus(str, Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"


class ComponentType(str, Enum):
    EXCHANGE_REST = "exchange_rest"
    EXCHANGE_WS = "exchange_ws"
    DATABASE = "database"
    CACHE = "cache"
    ML_MODEL = "ml_model"
    RISK_ENGINE = "risk_engine"
    RECONCILIATION = "reconciliation"
    PAPER_ENGINE = "paper_engine"
    STRATEGY = "strategy"


@dataclass
class HealthCheckResult:
    component: ComponentType
    status: HealthStatus
    message: str = ""
    latency_ms: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "component": self.component.value,
            "status": self.status.value,
            "message": self.message,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class SystemHealth:
    overall_status: HealthStatus
    checks: list[HealthCheckResult] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    uptime_seconds: float = 0.0
    version: str = "1.0.0"
    
    def to_dict(self) -> dict:
        return {
            "status": self.overall_status.value,
            "timestamp": self.timestamp.isoformat(),
            "uptime_seconds": self.uptime_seconds,
            "version": self.version,
            "checks": [c.to_dict() for c in self.checks],
        }
    
    @property
    def is_healthy(self) -> bool:
        return self.overall_status == HealthStatus.HEALTHY
    
    @property
    def is_ready(self) -> bool:
        # Ready if not UNHEALTHY
        return self.overall_status != HealthStatus.UNHEALTHY


class HealthChecker:
    """
    Composite health checker for all system components.
    
    Supports:
    - Liveness probe (/health/live) - is process alive
    - Readiness probe (/health/ready) - can serve traffic
    - Startup probe (/health/startup) - initialization complete
    """
    
    def __init__(
        self,
        version: str = "1.0.0",
        startup_timeout_seconds: float = 60.0,
    ):
        self.version = version
        self.startup_timeout = startup_timeout_seconds
        self.start_time = time.time()
        self._startup_complete = False
        self._checks: dict[ComponentType, Callable[[], Any]] = {}
        self._last_health: Optional[SystemHealth] = None
        
        # Prometheus metrics
        self.health_status = Gauge(
            "quantai_health_status",
            "Health status (1=healthy, 0.5=degraded, 0=unhealthy)",
            ["component"],
            registry=quantai_registry,
        )
        self.health_check_duration = Gauge(
            "quantai_health_check_duration_ms",
            "Health check duration in milliseconds",
            ["component"],
            registry=quantai_registry,
        )
        self.startup_complete_gauge = Gauge(
            "quantai_startup_complete",
            "Startup completion status (1=complete, 0=incomplete)",
            registry=quantai_registry,
        )
    
    def register_check(
        self,
        component: ComponentType,
        check_fn: Callable[[], Any],
        timeout_seconds: float = 5.0,
    ):
        """Register a health check function."""
        async def wrapped_check():
            start = time.perf_counter()
            try:
                if asyncio.iscoroutinefunction(check_fn):
                    result = await asyncio.wait_for(check_fn(), timeout=timeout_seconds)
                else:
                    result = await asyncio.wait_for(
                        asyncio.to_thread(check_fn), timeout=timeout_seconds
                    )
                
                latency = (time.perf_counter() - start) * 1000
                
                if result is True or (isinstance(result, dict) and result.get("healthy", True)):
                    return HealthCheckResult(
                        component=component,
                        status=HealthStatus.HEALTHY,
                        message=result.get("message", "OK") if isinstance(result, dict) else "OK",
                        latency_ms=latency,
                        metadata=result if isinstance(result, dict) else {},
                    )
                elif result is False or (isinstance(result, dict) and not result.get("healthy", True)):
                    return HealthCheckResult(
                        component=component,
                        status=HealthStatus.UNHEALTHY,
                        message=result.get("message", "Check failed") if isinstance(result, dict) else "Check failed",
                        latency_ms=latency,
                        metadata=result if isinstance(result, dict) else {},
                    )
                else:
                    return HealthCheckResult(
                        component=component,
                        status=HealthStatus.DEGRADED,
                        message="Check returned unexpected result",
                        latency_ms=latency,
                        metadata={"result": str(result)},
                    )
            except asyncio.TimeoutError:
                return HealthCheckResult(
                    component=component,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check timeout after {timeout_seconds}s",
                    latency_ms=timeout_seconds * 1000,
                )
            except Exception as e:
                return HealthCheckResult(
                    component=component,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check error: {e}",
                    latency_ms=(time.perf_counter() - start) * 1000,
                )
        
        self._checks[component] = wrapped_check
    
    def mark_startup_complete(self):
        """Mark startup as complete."""
        self._startup_complete = True
        self.startup_complete_gauge.set(1)
    
    async def run_checks(self) -> SystemHealth:
        """Run all registered health checks."""
        start = time.time()
        results = []
        
        for component, check_fn in self._checks.items():
            result = await check_fn()
            results.append(result)
            
            # Update Prometheus
            status_map = {HealthStatus.HEALTHY: 1, HealthStatus.DEGRADED: 0.5, HealthStatus.UNHEALTHY: 0}
            self.health_status.labels(component=component.value).set(status_map.get(result.status, 0))
            self.health_check_duration.labels(component=component.value).set(result.latency_ms)
        
        # Determine overall status
        if any(r.status == HealthStatus.UNHEALTHY for r in results):
            overall = HealthStatus.UNHEALTHY
        elif any(r.status == HealthStatus.DEGRADED for r in results):
            overall = HealthStatus.DEGRADED
        else:
            overall = HealthStatus.HEALTHY
        
        health = SystemHealth(
            overall_status=overall,
            checks=results,
            uptime_seconds=time.time() - self.start_time,
            version=self.version,
        )
        
        self._last_health = health
        return health
    
    # ============================================================
    # PROBE ENDPOINTS
    # ============================================================
    
    async def liveness(self) -> dict:
        """Liveness probe - is the process alive?"""
        # Simple check - if we can respond, we're alive
        return {
            "status": "alive",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": time.time() - self.start_time,
        }
    
    async def readiness(self) -> dict:
        """Readiness probe - can serve traffic?"""
        health = await self.run_checks()
        return {
            "ready": health.is_ready,
            "status": health.overall_status.value,
            "timestamp": health.timestamp.isoformat(),
            "checks": [c.to_dict() for c in health.checks],
        }
    
    async def startup(self) -> dict:
        """Startup probe - initialization complete?"""
        if self._startup_complete:
            return {
                "started": True,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        elif time.time() - self.start_time > self.startup_timeout:
            return {
                "started": False,
                "error": "Startup timeout exceeded",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        else:
            return {
                "started": False,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
    
    def get_last_health(self) -> Optional[SystemHealth]:
        return self._last_health


# ============================================================
# PREDEFINED CHECKS
# ============================================================

async def check_exchange_rest(binance_rest) -> bool:
    """Check Binance REST API connectivity."""
    try:
        # Ping endpoint
        await binance_rest._get("/fapi/v1/ping", weight=1)
        return {"healthy": True, "message": "REST API reachable"}
    except Exception as e:
        return {"healthy": False, "message": f"REST error: {e}"}


async def check_exchange_ws(binance_ws) -> bool:
    """Check WebSocket connectivity."""
    # Check if WebSocket is connected
    if binance_ws and binance_ws.ws:
        return {"healthy": True, "message": "WebSocket connected"}
    return {"healthy": False, "message": "WebSocket not connected"}


async def check_ml_model(ml_engine) -> bool:
    """Check ML model is loaded."""
    if ml_engine and ml_engine.model is not None:
        return {"healthy": True, "message": "Model loaded"}
    return {"healthy": False, "message": "No model loaded"}


def check_paper_engine(paper_engine) -> bool:
    """Check paper trading engine."""
    if paper_engine:
        return {"healthy": True, "message": "Paper engine initialized"}
    return {"healthy": False, "message": "Paper engine not available"}


def check_risk_engine(risk_orchestrator) -> bool:
    """Check risk orchestrator."""
    if risk_orchestrator:
        return {"healthy": True, "message": "Risk engine ready"}
    return {"healthy": False, "message": "Risk engine not initialized"}


def check_reconciliation(reconciliation_engine) -> bool:
    """Check reconciliation engine."""
    if reconciliation_engine and reconciliation_engine._running:
        return {"healthy": True, "message": "Reconciliation running"}
    return {"healthy": False, "message": "Reconciliation not running"}


# ============================================================
# FACTORY
# ============================================================

def create_health_checker(
    version: str = "1.0.0",
    binance_rest=None,
    binance_ws=None,
    ml_engine=None,
    paper_engine=None,
    risk_orchestrator=None,
    reconciliation_engine=None,
) -> HealthChecker:
    """Create health checker with standard checks."""
    checker = HealthChecker(version=version)
    
    if binance_rest:
        checker.register_check(
            ComponentType.EXCHANGE_REST,
            lambda: check_exchange_rest(binance_rest),
        )
    
    if binance_ws:
        checker.register_check(
            ComponentType.EXCHANGE_WS,
            lambda: check_exchange_ws(binance_ws),
        )
    
    if ml_engine:
        checker.register_check(
            ComponentType.ML_MODEL,
            lambda: check_ml_model(ml_engine),
        )
    
    if paper_engine:
        checker.register_check(
            ComponentType.PAPER_ENGINE,
            lambda: check_paper_engine(paper_engine),
        )
    
    if risk_orchestrator:
        checker.register_check(
            ComponentType.RISK_ENGINE,
            lambda: check_risk_engine(risk_orchestrator),
        )
    
    if reconciliation_engine:
        checker.register_check(
            ComponentType.RECONCILIATION,
            lambda: check_reconciliation(reconciliation_engine),
        )
    
    return checker


# ============================================================
# HTTP HANDLERS (for aiohttp/FastAPI)
# ============================================================

async def health_liveness_handler(request, checker: HealthChecker):
    """Liveness endpoint handler."""
    result = await checker.liveness()
    return result


async def health_readiness_handler(request, checker: HealthChecker):
    """Readiness endpoint handler."""
    result = await checker.readiness()
    status_code = 200 if result["ready"] else 503
    return {"status_code": status_code, "body": result}


async def health_startup_handler(request, checker: HealthChecker):
    """Startup endpoint handler."""
    result = await checker.startup()
    status_code = 200 if result["started"] else 503
    return {"status_code": status_code, "body": result}


async def health_full_handler(request, checker: HealthChecker):
    """Full health check endpoint."""
    health = await checker.run_checks()
    return health.to_dict()


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "HealthStatus",
    "ComponentType",
    "HealthCheckResult",
    "SystemHealth",
    "HealthChecker",
    "create_health_checker",
    "check_exchange_rest",
    "check_exchange_ws",
    "check_ml_model",
    "check_paper_engine",
    "check_risk_engine",
    "check_reconciliation",
    "health_liveness_handler",
    "health_readiness_handler",
    "health_startup_handler",
    "health_full_handler",
]