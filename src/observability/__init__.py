"""
2api.ai - Observability Module

Comprehensive observability stack including:
- Prometheus metrics (Counter, Histogram, Gauge)
- OpenTelemetry distributed tracing
- Structured JSON logging with context injection
- W3C trace context propagation

Usage:
    from src.observability import (
        setup_observability,
        get_metrics,
        get_tracer,
        get_logger,
    )

    # Initialize at startup
    setup_observability(service_name="2api")

    # Use throughout code
    logger = get_logger(__name__)
    tracer = get_tracer()
    metrics = get_metrics()
"""

from .metrics import (
    MetricsCollector,
    get_metrics,
    setup_metrics,
    metrics_endpoint,
)
from .tracing import (
    TracingManager,
    get_tracer,
    setup_tracing,
    trace_context_middleware,
)
from .logging import (
    StructuredLogger,
    get_logger,
    setup_logging,
    LogContext,
)
from .middleware import (
    ObservabilityMiddleware,
    setup_observability,
)

__all__ = [
    # Metrics
    "MetricsCollector",
    "get_metrics",
    "setup_metrics",
    "metrics_endpoint",
    # Tracing
    "TracingManager",
    "get_tracer",
    "setup_tracing",
    "trace_context_middleware",
    # Logging
    "StructuredLogger",
    "get_logger",
    "setup_logging",
    "LogContext",
    # Combined
    "ObservabilityMiddleware",
    "setup_observability",
]
