"""
2api.ai - Prometheus Metrics

Production-grade metrics collection with Prometheus client library.

Metrics exposed:
- twoapi_requests_total: Counter of all requests by endpoint, provider, model, status
- twoapi_request_duration_seconds: Histogram of request latency
- twoapi_tokens_total: Counter of tokens used (input/output)
- twoapi_cost_usd_total: Counter of total cost in USD
- twoapi_active_requests: Gauge of currently active requests
- twoapi_circuit_breaker_state: Gauge of circuit breaker state per provider
- twoapi_rate_limit_hits_total: Counter of rate limit hits

Usage:
    from src.observability.metrics import get_metrics, setup_metrics, metrics_endpoint

    # Setup at startup
    setup_metrics()

    # Record metrics
    metrics = get_metrics()
    metrics.record_request(endpoint="/v1/chat/completions", provider="openai", model="gpt-4", status=200, duration_seconds=1.5)
    metrics.record_tokens(provider="openai", model="gpt-4", input_tokens=100, output_tokens=50)

    # Expose /metrics endpoint
    @app.get("/metrics")
    async def metrics():
        return metrics_endpoint()
"""

import time
from typing import Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Info,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST,
    REGISTRY,
)
from fastapi import Response


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = 0
    HALF_OPEN = 1
    OPEN = 2


@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    endpoint: str
    provider: str
    model: str
    status_code: int
    duration_seconds: float
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    error_type: Optional[str] = None
    is_streaming: bool = False
    fallback_used: bool = False


class MetricsCollector:
    """
    Central metrics collector using Prometheus client.

    Thread-safe, singleton pattern for global access.
    """

    _instance: Optional["MetricsCollector"] = None
    _initialized_registries: set = set()

    def __init__(self, registry: CollectorRegistry = REGISTRY):
        """Initialize metrics collectors."""
        self.registry = registry

        # Check if this registry was already initialized
        registry_id = id(registry)
        if registry_id in MetricsCollector._initialized_registries:
            # Reuse existing metrics from singleton
            if MetricsCollector._instance is not None:
                self._copy_from(MetricsCollector._instance)
                return

        MetricsCollector._initialized_registries.add(registry_id)

        # Service info
        self.info = Info(
            "twoapi",
            "2api.ai service information",
            registry=registry,
        )
        self.info.info({
            "version": "1.0.0",
            "service": "2api-gateway",
        })

        # Request counter
        self.requests_total = Counter(
            "twoapi_requests_total",
            "Total number of requests",
            labelnames=["endpoint", "provider", "model", "status", "error_type", "streaming"],
            registry=registry,
        )

        # Request duration histogram with buckets optimized for AI API calls
        # AI calls typically range from 0.1s to 60s+
        self.request_duration = Histogram(
            "twoapi_request_duration_seconds",
            "Request duration in seconds",
            labelnames=["endpoint", "provider", "model", "streaming"],
            buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0, float("inf")),
            registry=registry,
        )

        # Time to first token (streaming only)
        self.time_to_first_token = Histogram(
            "twoapi_time_to_first_token_seconds",
            "Time to first token in streaming responses",
            labelnames=["provider", "model"],
            buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, float("inf")),
            registry=registry,
        )

        # Token counters
        self.tokens_total = Counter(
            "twoapi_tokens_total",
            "Total tokens used",
            labelnames=["provider", "model", "type", "tenant_id"],  # type = input/output
            registry=registry,
        )

        # Cost counter
        self.cost_total = Counter(
            "twoapi_cost_usd_total",
            "Total cost in USD",
            labelnames=["provider", "model", "tenant_id"],
            registry=registry,
        )

        # Active requests gauge
        self.active_requests = Gauge(
            "twoapi_active_requests",
            "Number of currently active requests",
            labelnames=["endpoint", "provider"],
            registry=registry,
        )

        # Circuit breaker state gauge
        # 0 = closed (healthy), 1 = half-open, 2 = open (unhealthy)
        self.circuit_breaker_state = Gauge(
            "twoapi_circuit_breaker_state",
            "Circuit breaker state (0=closed, 1=half-open, 2=open)",
            labelnames=["provider"],
            registry=registry,
        )

        # Rate limit hits
        self.rate_limit_hits = Counter(
            "twoapi_rate_limit_hits_total",
            "Total rate limit hits",
            labelnames=["tenant_id", "api_key_prefix", "limit_type"],  # limit_type = rpm/tpm/daily
            registry=registry,
        )

        # Provider health check results
        self.health_check_duration = Histogram(
            "twoapi_health_check_duration_seconds",
            "Health check duration",
            labelnames=["provider"],
            buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
            registry=registry,
        )

        self.health_check_success = Counter(
            "twoapi_health_check_success_total",
            "Health check successes",
            labelnames=["provider"],
            registry=registry,
        )

        self.health_check_failure = Counter(
            "twoapi_health_check_failure_total",
            "Health check failures",
            labelnames=["provider"],
            registry=registry,
        )

        # Fallback metrics
        self.fallback_attempts = Counter(
            "twoapi_fallback_attempts_total",
            "Total fallback attempts",
            labelnames=["from_provider", "to_provider", "reason"],
            registry=registry,
        )

        # Routing metrics
        self.routing_decisions = Counter(
            "twoapi_routing_decisions_total",
            "Total routing decisions",
            labelnames=["strategy", "selected_provider", "model"],
            registry=registry,
        )

    @classmethod
    def get_instance(cls) -> "MetricsCollector":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset singleton (for testing)."""
        cls._instance = None
        cls._initialized_registries.clear()

    def _copy_from(self, other: "MetricsCollector"):
        """Copy metrics references from another collector."""
        self.info = other.info
        self.requests_total = other.requests_total
        self.request_duration = other.request_duration
        self.time_to_first_token = other.time_to_first_token
        self.tokens_total = other.tokens_total
        self.cost_total = other.cost_total
        self.active_requests = other.active_requests
        self.circuit_breaker_state = other.circuit_breaker_state
        self.rate_limit_hits = other.rate_limit_hits
        self.health_check_duration = other.health_check_duration
        self.health_check_success = other.health_check_success
        self.health_check_failure = other.health_check_failure
        self.fallback_attempts = other.fallback_attempts
        self.routing_decisions = other.routing_decisions

    def record_request(
        self,
        endpoint: str,
        provider: str,
        model: str,
        status_code: int,
        duration_seconds: float,
        error_type: Optional[str] = None,
        streaming: bool = False,
        tenant_id: str = "unknown",
    ):
        """Record a completed request."""
        error_label = error_type or "none"
        streaming_label = "true" if streaming else "false"

        self.requests_total.labels(
            endpoint=endpoint,
            provider=provider,
            model=model,
            status=str(status_code),
            error_type=error_label,
            streaming=streaming_label,
        ).inc()

        self.request_duration.labels(
            endpoint=endpoint,
            provider=provider,
            model=model,
            streaming=streaming_label,
        ).observe(duration_seconds)

    def record_tokens(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        tenant_id: str = "unknown",
    ):
        """Record token usage."""
        self.tokens_total.labels(
            provider=provider,
            model=model,
            type="input",
            tenant_id=tenant_id,
        ).inc(input_tokens)

        self.tokens_total.labels(
            provider=provider,
            model=model,
            type="output",
            tenant_id=tenant_id,
        ).inc(output_tokens)

    def record_cost(
        self,
        provider: str,
        model: str,
        cost_usd: float,
        tenant_id: str = "unknown",
    ):
        """Record cost."""
        self.cost_total.labels(
            provider=provider,
            model=model,
            tenant_id=tenant_id,
        ).inc(cost_usd)

    def record_time_to_first_token(
        self,
        provider: str,
        model: str,
        ttft_seconds: float,
    ):
        """Record time to first token for streaming requests."""
        self.time_to_first_token.labels(
            provider=provider,
            model=model,
        ).observe(ttft_seconds)

    def track_active_request(self, endpoint: str, provider: str) -> "ActiveRequestTracker":
        """Context manager to track active requests."""
        return ActiveRequestTracker(self, endpoint, provider)

    def set_circuit_breaker_state(self, provider: str, state: CircuitState):
        """Update circuit breaker state."""
        self.circuit_breaker_state.labels(provider=provider).set(state.value)

    def record_rate_limit_hit(
        self,
        tenant_id: str,
        api_key_prefix: str,
        limit_type: str,
    ):
        """Record a rate limit hit."""
        self.rate_limit_hits.labels(
            tenant_id=tenant_id,
            api_key_prefix=api_key_prefix,
            limit_type=limit_type,
        ).inc()

    def record_health_check(
        self,
        provider: str,
        success: bool,
        duration_seconds: float,
    ):
        """Record health check result."""
        self.health_check_duration.labels(provider=provider).observe(duration_seconds)

        if success:
            self.health_check_success.labels(provider=provider).inc()
        else:
            self.health_check_failure.labels(provider=provider).inc()

    def record_fallback(
        self,
        from_provider: str,
        to_provider: str,
        reason: str,
    ):
        """Record a fallback attempt."""
        self.fallback_attempts.labels(
            from_provider=from_provider,
            to_provider=to_provider,
            reason=reason,
        ).inc()

    def record_routing_decision(
        self,
        strategy: str,
        selected_provider: str,
        model: str,
    ):
        """Record a routing decision."""
        self.routing_decisions.labels(
            strategy=strategy,
            selected_provider=selected_provider,
            model=model,
        ).inc()


class ActiveRequestTracker:
    """Context manager for tracking active requests."""

    def __init__(self, collector: MetricsCollector, endpoint: str, provider: str):
        self.collector = collector
        self.endpoint = endpoint
        self.provider = provider

    def __enter__(self):
        self.collector.active_requests.labels(
            endpoint=self.endpoint,
            provider=self.provider,
        ).inc()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.collector.active_requests.labels(
            endpoint=self.endpoint,
            provider=self.provider,
        ).dec()


# Module-level functions for convenience
_metrics_instance: Optional[MetricsCollector] = None


def setup_metrics(registry: CollectorRegistry = REGISTRY) -> MetricsCollector:
    """
    Setup metrics collection.

    Call once at application startup.
    Safe to call multiple times - returns existing instance.
    """
    global _metrics_instance

    # Return existing instance if already setup with same registry
    if _metrics_instance is not None and _metrics_instance.registry is registry:
        return _metrics_instance

    _metrics_instance = MetricsCollector(registry)
    MetricsCollector._instance = _metrics_instance
    return _metrics_instance


def get_metrics() -> MetricsCollector:
    """
    Get the metrics collector instance.

    Raises if setup_metrics() hasn't been called.
    """
    global _metrics_instance
    if _metrics_instance is None:
        # Auto-initialize with defaults
        _metrics_instance = MetricsCollector.get_instance()
    return _metrics_instance


def metrics_endpoint() -> Response:
    """
    Generate Prometheus metrics endpoint response.

    Usage:
        @app.get("/metrics")
        async def metrics():
            return metrics_endpoint()
    """
    content = generate_latest(REGISTRY)
    return Response(
        content=content,
        media_type=CONTENT_TYPE_LATEST,
    )
