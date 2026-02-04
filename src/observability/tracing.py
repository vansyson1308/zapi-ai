"""
2api.ai - OpenTelemetry Distributed Tracing

Production-grade distributed tracing with OpenTelemetry.

Features:
- W3C trace context propagation (traceparent header)
- Automatic span creation for requests
- Provider call instrumentation
- OTLP exporter support (Jaeger, Zipkin, etc.)
- Context injection into logs

Usage:
    from src.observability.tracing import setup_tracing, get_tracer

    # Setup at startup
    setup_tracing(
        service_name="2api",
        otlp_endpoint="http://localhost:4317",  # Optional
    )

    # Create spans
    tracer = get_tracer()
    with tracer.start_as_current_span("operation_name") as span:
        span.set_attribute("key", "value")
        # ... do work
"""

import os
from typing import Optional, Dict, Any, Callable
from contextlib import contextmanager
from dataclasses import dataclass

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider, Span
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.propagate import set_global_textmap, get_global_textmap, inject, extract
from opentelemetry.trace import SpanKind, Status, StatusCode
from opentelemetry.context import Context

# Optional OTLP exporter (requires opentelemetry-exporter-otlp)
try:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    OTLP_AVAILABLE = True
except ImportError:
    OTLP_AVAILABLE = False


@dataclass
class TraceContext:
    """Extracted trace context from incoming request."""
    trace_id: str
    span_id: str
    trace_flags: int = 1
    trace_state: Optional[str] = None

    @classmethod
    def from_span(cls, span: Span) -> "TraceContext":
        """Create TraceContext from current span."""
        ctx = span.get_span_context()
        return cls(
            trace_id=format(ctx.trace_id, "032x"),
            span_id=format(ctx.span_id, "016x"),
            trace_flags=ctx.trace_flags,
            trace_state=str(ctx.trace_state) if ctx.trace_state else None,
        )

    def to_traceparent(self) -> str:
        """Generate W3C traceparent header value."""
        return f"00-{self.trace_id}-{self.span_id}-{self.trace_flags:02x}"


class TracingManager:
    """
    Central tracing manager using OpenTelemetry.

    Singleton pattern for global access.
    """

    _instance: Optional["TracingManager"] = None

    def __init__(
        self,
        service_name: str = "2api",
        service_version: str = "1.0.0",
        otlp_endpoint: Optional[str] = None,
        console_export: bool = False,
        sample_rate: float = 1.0,
    ):
        """
        Initialize tracing.

        Args:
            service_name: Name of the service
            service_version: Version of the service
            otlp_endpoint: OTLP collector endpoint (e.g., http://localhost:4317)
            console_export: Whether to export spans to console (for debugging)
            sample_rate: Sampling rate (0.0 to 1.0)
        """
        self.service_name = service_name
        self.service_version = service_version

        # Create resource
        resource = Resource.create({
            SERVICE_NAME: service_name,
            SERVICE_VERSION: service_version,
            "deployment.environment": os.getenv("MODE", "local"),
        })

        # Create tracer provider
        self.provider = TracerProvider(resource=resource)

        # Add OTLP exporter if endpoint provided
        if otlp_endpoint and OTLP_AVAILABLE:
            otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
            self.provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

        # Add console exporter for debugging
        if console_export:
            self.provider.add_span_processor(
                SimpleSpanProcessor(ConsoleSpanExporter())
            )

        # Set as global provider
        trace.set_tracer_provider(self.provider)

        # Set W3C trace context propagator
        set_global_textmap(TraceContextTextMapPropagator())

        # Get tracer
        self.tracer = trace.get_tracer(service_name, service_version)

    @classmethod
    def get_instance(cls) -> "TracingManager":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset singleton (for testing)."""
        cls._instance = None

    def get_tracer(self) -> trace.Tracer:
        """Get the tracer instance."""
        return self.tracer

    def extract_context(self, headers: Dict[str, str]) -> Context:
        """
        Extract trace context from HTTP headers.

        Args:
            headers: HTTP headers dict (case-insensitive keys)

        Returns:
            OpenTelemetry Context with extracted trace info
        """
        # Normalize header keys to lowercase for extraction
        normalized = {k.lower(): v for k, v in headers.items()}
        return extract(normalized)

    def inject_context(self, headers: Dict[str, str]) -> Dict[str, str]:
        """
        Inject current trace context into HTTP headers.

        Args:
            headers: HTTP headers dict to inject into

        Returns:
            Headers dict with trace context added
        """
        inject(headers)
        return headers

    def start_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
        context: Optional[Context] = None,
    ):
        """
        Start a new span.

        Args:
            name: Span name
            kind: Span kind (SERVER, CLIENT, INTERNAL, PRODUCER, CONSUMER)
            attributes: Initial span attributes
            context: Parent context (uses current if not provided)

        Returns:
            Context manager that yields the span
        """
        return self.tracer.start_as_current_span(
            name,
            kind=kind,
            attributes=attributes,
            context=context,
        )

    def start_server_span(
        self,
        name: str,
        headers: Dict[str, str],
        attributes: Optional[Dict[str, Any]] = None,
    ):
        """
        Start a server span with context extraction from headers.

        Use this for incoming HTTP requests.
        """
        parent_context = self.extract_context(headers)
        return self.tracer.start_as_current_span(
            name,
            kind=SpanKind.SERVER,
            attributes=attributes,
            context=parent_context,
        )

    def start_client_span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        """
        Start a client span for outgoing requests.

        Use this for calls to providers.
        """
        return self.tracer.start_as_current_span(
            name,
            kind=SpanKind.CLIENT,
            attributes=attributes,
        )

    def get_current_span(self) -> Optional[Span]:
        """Get the current active span."""
        return trace.get_current_span()

    def get_current_trace_context(self) -> Optional[TraceContext]:
        """Get current trace context for logging/headers."""
        span = self.get_current_span()
        if span and span.get_span_context().is_valid:
            return TraceContext.from_span(span)
        return None

    def add_span_attributes(self, attributes: Dict[str, Any]):
        """Add attributes to the current span."""
        span = self.get_current_span()
        if span:
            for key, value in attributes.items():
                if value is not None:
                    span.set_attribute(key, value)

    def record_exception(self, exception: Exception):
        """Record an exception on the current span."""
        span = self.get_current_span()
        if span:
            span.record_exception(exception)
            span.set_status(Status(StatusCode.ERROR, str(exception)))

    def set_span_status(self, status_code: StatusCode, description: str = ""):
        """Set status on the current span."""
        span = self.get_current_span()
        if span:
            span.set_status(Status(status_code, description))

    def shutdown(self):
        """Shutdown the tracer provider."""
        self.provider.shutdown()


# Module-level functions for convenience
_tracing_instance: Optional[TracingManager] = None


def setup_tracing(
    service_name: str = "2api",
    service_version: str = "1.0.0",
    otlp_endpoint: Optional[str] = None,
    console_export: bool = False,
) -> TracingManager:
    """
    Setup distributed tracing.

    Call once at application startup.

    Args:
        service_name: Name of the service
        service_version: Version of the service
        otlp_endpoint: OTLP collector endpoint (optional)
        console_export: Enable console export for debugging

    Returns:
        TracingManager instance
    """
    global _tracing_instance

    # Check for environment variables
    if otlp_endpoint is None:
        otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")

    if os.getenv("OTEL_CONSOLE_EXPORT", "").lower() == "true":
        console_export = True

    _tracing_instance = TracingManager(
        service_name=service_name,
        service_version=service_version,
        otlp_endpoint=otlp_endpoint,
        console_export=console_export,
    )
    return _tracing_instance


def get_tracer() -> trace.Tracer:
    """
    Get the tracer instance.

    Auto-initializes with defaults if not setup.
    """
    global _tracing_instance
    if _tracing_instance is None:
        _tracing_instance = TracingManager.get_instance()
    return _tracing_instance.get_tracer()


def get_tracing_manager() -> TracingManager:
    """Get the tracing manager instance."""
    global _tracing_instance
    if _tracing_instance is None:
        _tracing_instance = TracingManager.get_instance()
    return _tracing_instance


async def trace_context_middleware(request, call_next):
    """
    FastAPI middleware for trace context propagation.

    Extracts traceparent from incoming requests and creates server spans.

    Usage:
        app.middleware("http")(trace_context_middleware)
    """
    tracing = get_tracing_manager()

    # Extract headers
    headers = dict(request.headers)

    # Get or generate request/trace IDs
    request_id = headers.get("x-request-id", "")
    if not request_id:
        import uuid
        request_id = f"req_{uuid.uuid4().hex[:24]}"

    # Build span name
    span_name = f"{request.method} {request.url.path}"

    # Start server span with context extraction
    with tracing.start_server_span(
        name=span_name,
        headers=headers,
        attributes={
            "http.method": request.method,
            "http.url": str(request.url),
            "http.route": request.url.path,
            "http.scheme": request.url.scheme,
            "http.host": request.url.hostname,
            "twoapi.request_id": request_id,
        },
    ) as span:
        # Store trace context in request state
        trace_ctx = TraceContext.from_span(span)
        request.state.trace_context = trace_ctx
        request.state.trace_id = trace_ctx.trace_id
        request.state.span_id = trace_ctx.span_id

        try:
            response = await call_next(request)

            # Add response attributes
            span.set_attribute("http.status_code", response.status_code)

            # Set span status based on response
            if response.status_code >= 500:
                span.set_status(Status(StatusCode.ERROR, f"HTTP {response.status_code}"))
            elif response.status_code >= 400:
                span.set_status(Status(StatusCode.ERROR, f"HTTP {response.status_code}"))
            else:
                span.set_status(Status(StatusCode.OK))

            # Add trace headers to response
            response.headers["X-Trace-Id"] = trace_ctx.trace_id
            response.headers["X-Span-Id"] = trace_ctx.span_id

            return response

        except Exception as e:
            tracing.record_exception(e)
            raise


@contextmanager
def trace_provider_call(provider: str, model: str, operation: str = "chat"):
    """
    Context manager for tracing provider API calls.

    Usage:
        with trace_provider_call("openai", "gpt-4", "chat") as span:
            response = await provider.chat(...)
            span.set_attribute("tokens.input", response.usage.input_tokens)
    """
    tracing = get_tracing_manager()

    with tracing.start_client_span(
        name=f"{provider}.{operation}",
        attributes={
            "ai.provider": provider,
            "ai.model": model,
            "ai.operation": operation,
        },
    ) as span:
        yield span
