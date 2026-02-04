"""
2api.ai - Observability Middleware

Unified middleware that combines metrics, tracing, and logging.

Features:
- Request/response metrics collection
- Distributed tracing with W3C context propagation
- Structured logging with correlation IDs
- Performance tracking

Usage:
    from src.observability import setup_observability, ObservabilityMiddleware

    # Setup at startup
    setup_observability(service_name="2api")

    # Add middleware
    app.add_middleware(ObservabilityMiddleware)
"""

import os
import time
import uuid
from typing import Optional, Dict, Any, Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp

from .metrics import get_metrics, MetricsCollector, setup_metrics
from .tracing import (
    get_tracing_manager,
    TracingManager,
    TraceContext,
    setup_tracing,
)
from .logging import (
    get_logger,
    LogContext,
    setup_logging,
    StructuredLogger,
)

from opentelemetry.trace import SpanKind, Status, StatusCode


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """
    Unified observability middleware.

    Combines:
    - Prometheus metrics
    - OpenTelemetry tracing
    - Structured logging with context

    All correlation IDs are propagated through the request lifecycle.
    """

    # Paths to exclude from detailed observability
    EXCLUDE_PATHS = {"/health", "/ready", "/metrics", "/openapi.json", "/docs", "/redoc"}

    def __init__(
        self,
        app: ASGIApp,
        service_name: str = "2api",
        exclude_paths: Optional[set] = None,
    ):
        super().__init__(app)
        self.service_name = service_name
        self.exclude_paths = exclude_paths or self.EXCLUDE_PATHS
        self.logger = get_logger("observability.middleware")

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Process request with full observability."""

        # Skip observability for excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)

        # Get components
        metrics = get_metrics()
        tracing = get_tracing_manager()

        # Extract headers
        headers = dict(request.headers)

        # Get or generate request ID
        request_id = headers.get("x-request-id", "")
        if not request_id:
            request_id = f"req_{uuid.uuid4().hex[:24]}"

        # Build span name
        span_name = f"{request.method} {request.url.path}"

        # Start timing
        start_time = time.perf_counter()

        # Start server span with context extraction
        with tracing.start_server_span(
            name=span_name,
            headers=headers,
            attributes={
                "http.method": request.method,
                "http.url": str(request.url),
                "http.route": request.url.path,
                "http.scheme": request.url.scheme,
                "http.host": request.url.hostname or "",
                "http.user_agent": headers.get("user-agent", ""),
                "twoapi.request_id": request_id,
            },
        ) as span:
            # Get trace context from span
            trace_ctx = TraceContext.from_span(span)

            # Setup log context
            log_ctx = LogContext(
                request_id=request_id,
                trace_id=trace_ctx.trace_id,
                span_id=trace_ctx.span_id,
                endpoint=request.url.path,
            )
            LogContext.set_current(log_ctx)

            # Store in request state
            request.state.request_id = request_id
            request.state.trace_id = trace_ctx.trace_id
            request.state.span_id = trace_ctx.span_id
            request.state.trace_context = trace_ctx
            request.state.log_context = log_ctx

            # Extract provider/model from path for routing endpoints
            provider = "unknown"
            model = "unknown"

            # Track active request
            endpoint_group = self._get_endpoint_group(request.url.path)

            try:
                with metrics.track_active_request(endpoint_group, provider):
                    # Process request
                    response = await call_next(request)

                    # Calculate duration
                    duration_seconds = time.perf_counter() - start_time

                    # Extract provider/model from response headers if available
                    provider = response.headers.get("x-provider", provider)
                    model = response.headers.get("x-model", model)

                    # Update span attributes
                    span.set_attribute("http.status_code", response.status_code)
                    span.set_attribute("ai.provider", provider)
                    span.set_attribute("ai.model", model)

                    # Set span status based on response
                    if response.status_code >= 500:
                        span.set_status(Status(StatusCode.ERROR, f"HTTP {response.status_code}"))
                    elif response.status_code >= 400:
                        # Client errors are not span errors, but we note them
                        span.set_attribute("http.error", True)
                    else:
                        span.set_status(Status(StatusCode.OK))

                    # Record metrics
                    error_type = None
                    if response.status_code >= 400:
                        error_type = response.headers.get("x-error-code", f"http_{response.status_code}")

                    metrics.record_request(
                        endpoint=endpoint_group,
                        provider=provider,
                        model=model,
                        status_code=response.status_code,
                        duration_seconds=duration_seconds,
                        error_type=error_type,
                        streaming=response.headers.get("content-type", "").startswith("text/event-stream"),
                    )

                    # Log request completion
                    self._log_request(
                        request=request,
                        response=response,
                        duration_ms=duration_seconds * 1000,
                        provider=provider,
                        model=model,
                    )

                    # Add correlation headers to response
                    response.headers["X-Request-Id"] = request_id
                    response.headers["X-Trace-Id"] = trace_ctx.trace_id
                    response.headers["X-Span-Id"] = trace_ctx.span_id

                    return response

            except Exception as e:
                # Calculate duration
                duration_seconds = time.perf_counter() - start_time

                # Record exception in span
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))

                # Record error metrics
                metrics.record_request(
                    endpoint=endpoint_group,
                    provider=provider,
                    model=model,
                    status_code=500,
                    duration_seconds=duration_seconds,
                    error_type=type(e).__name__,
                )

                # Log error
                self.logger.exception(
                    "Request failed with exception",
                    error=str(e),
                    error_type=type(e).__name__,
                    duration_ms=duration_seconds * 1000,
                )

                raise

            finally:
                # Clear log context
                LogContext.clear()

    def _get_endpoint_group(self, path: str) -> str:
        """Group endpoints for metrics aggregation."""
        if path.startswith("/v1/chat"):
            return "/v1/chat/completions"
        elif path.startswith("/v1/embeddings"):
            return "/v1/embeddings"
        elif path.startswith("/v1/images"):
            return "/v1/images/generations"
        elif path.startswith("/v1/models"):
            return "/v1/models"
        elif path.startswith("/v1/"):
            return path.split("/")[2] if len(path.split("/")) > 2 else path
        return path

    def _log_request(
        self,
        request: Request,
        response: Response,
        duration_ms: float,
        provider: str,
        model: str,
    ):
        """Log request completion."""
        status_code = response.status_code

        log_data = {
            "method": request.method,
            "path": request.url.path,
            "status_code": status_code,
            "duration_ms": round(duration_ms, 2),
            "provider": provider,
            "model": model,
            "client_ip": request.client.host if request.client else None,
        }

        if status_code >= 500:
            self.logger.error("Request completed with server error", **log_data)
        elif status_code >= 400:
            self.logger.warning("Request completed with client error", **log_data)
        else:
            self.logger.info("Request completed", **log_data)


_observability_initialized = False


def setup_observability(
    service_name: str = "2api",
    service_version: str = "1.0.0",
    otlp_endpoint: Optional[str] = None,
    log_level: str = "INFO",
    metrics_enabled: bool = True,
    tracing_enabled: bool = True,
    logging_enabled: bool = True,
) -> Dict[str, Any]:
    """
    Setup complete observability stack.

    Call once at application startup. Safe to call multiple times.

    Args:
        service_name: Name of the service
        service_version: Version of the service
        otlp_endpoint: OTLP collector endpoint (optional)
        log_level: Log level (DEBUG, INFO, WARNING, ERROR)
        metrics_enabled: Enable Prometheus metrics
        tracing_enabled: Enable OpenTelemetry tracing
        logging_enabled: Enable structured logging

    Returns:
        Dict with initialized components

    Usage:
        # In lifespan or startup
        components = setup_observability(
            service_name="2api",
            otlp_endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
        )
    """
    global _observability_initialized

    result = {}

    # Read from environment if not provided
    if otlp_endpoint is None:
        otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")

    log_level = os.getenv("LOG_LEVEL", log_level)

    # Setup logging first (other components may log)
    if logging_enabled:
        json_output = os.getenv("LOG_FORMAT", "json").lower() == "json"
        setup_logging(
            level=log_level,
            json_output=json_output,
        )
        result["logging"] = True

    # Setup metrics (skip if already initialized to avoid registry conflicts)
    if metrics_enabled:
        try:
            metrics = setup_metrics()
            result["metrics"] = metrics
        except ValueError as e:
            if "Duplicated timeseries" in str(e):
                # Metrics already registered, get existing instance
                result["metrics"] = get_metrics()
            else:
                raise

    # Setup tracing
    if tracing_enabled:
        console_export = os.getenv("OTEL_CONSOLE_EXPORT", "").lower() == "true"
        tracing = setup_tracing(
            service_name=service_name,
            service_version=service_version,
            otlp_endpoint=otlp_endpoint,
            console_export=console_export,
        )
        result["tracing"] = tracing

    # Log initialization only once
    if not _observability_initialized:
        logger = get_logger("observability")
        logger.info(
            "Observability initialized",
            service_name=service_name,
            service_version=service_version,
            metrics_enabled=metrics_enabled,
            tracing_enabled=tracing_enabled,
            otlp_endpoint=otlp_endpoint or "none",
        )
        _observability_initialized = True

    return result


# Utility functions for use in route handlers

def get_request_context(request: Request) -> Dict[str, str]:
    """
    Get observability context from request.

    Returns dict with request_id, trace_id, span_id.
    """
    return {
        "request_id": getattr(request.state, "request_id", ""),
        "trace_id": getattr(request.state, "trace_id", ""),
        "span_id": getattr(request.state, "span_id", ""),
    }


def set_provider_info(request: Request, provider: str, model: str):
    """
    Set provider/model info on request for metrics.

    Call this in route handlers after routing decision.
    """
    log_ctx = getattr(request.state, "log_context", None)
    if log_ctx:
        log_ctx.provider = provider
        log_ctx.model = model


def record_tokens(
    request: Request,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float = 0.0,
):
    """
    Record token usage for the current request.

    Call this after receiving response from provider.
    """
    metrics = get_metrics()
    log_ctx = getattr(request.state, "log_context", None)

    provider = log_ctx.provider if log_ctx else "unknown"
    model = log_ctx.model if log_ctx else "unknown"
    tenant_id = log_ctx.tenant_id if log_ctx else "unknown"

    metrics.record_tokens(
        provider=provider,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        tenant_id=tenant_id,
    )

    if cost_usd > 0:
        metrics.record_cost(
            provider=provider,
            model=model,
            cost_usd=cost_usd,
            tenant_id=tenant_id,
        )
