"""
====================================================
QuantAI Professional
Monitoring Package
====================================================

Observability components:
- Logging: Structured JSON logging with correlation IDs
- Metrics: Prometheus metrics for all system components
- Health: Liveness/Readiness/Startup probes
- Config Validation: Startup configuration verification
====================================================
"""

from src.monitoring.logging_config import (
    LogContext,
    CorrelationFilter,
    JSONFormatter,
    setup_logging,
    get_logger,
    log_with_context,
    log_debug,
    log_info,
    log_warning,
    log_error,
    log_critical,
    correlation_context,
)

from src.monitoring.metrics import (
    quantai_registry,
    MetricsCollector,
    metrics,
    measure_latency,
    count_calls,
    metrics_endpoint,
)

from src.monitoring.health import (
    HealthStatus,
    ComponentType,
    HealthCheckResult,
    SystemHealth,
    HealthChecker,
    create_health_checker,
    health_liveness_handler,
    health_readiness_handler,
    health_startup_handler,
    health_full_handler,
)

from src.monitoring.config_validation import (
    ValidationResult,
    ConfigValidator,
    validate_config,
    validate_config_or_exit,
    run_startup_validation,
)

__all__ = [
    # Logging
    "LogContext",
    "CorrelationFilter",
    "JSONFormatter",
    "setup_logging",
    "get_logger",
    "log_with_context",
    "log_debug",
    "log_info",
    "log_warning",
    "log_error",
    "log_critical",
    "correlation_context",
    
    # Metrics
    "quantai_registry",
    "MetricsCollector",
    "metrics",
    "measure_latency",
    "count_calls",
    "metrics_endpoint",
    
    # Health
    "HealthStatus",
    "ComponentType",
    "HealthCheckResult",
    "SystemHealth",
    "HealthChecker",
    "create_health_checker",
    "health_liveness_handler",
    "health_readiness_handler",
    "health_startup_handler",
    "health_full_handler",
    
    # Config Validation
    "ValidationResult",
    "ConfigValidator",
    "validate_config",
    "validate_config_or_exit",
    "run_startup_validation",
]

__version__ = "1.0.0"