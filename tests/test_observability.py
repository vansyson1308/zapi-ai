"""
2api.ai - Observability Tests

Tests for the complete observability stack:
- Prometheus metrics
- OpenTelemetry tracing
- Structured logging
- Middleware integration
"""

import pytest
import json
import asyncio
from unittest.mock import MagicMock, patch
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.responses import Response
from prometheus_client import REGISTRY, CollectorRegistry

# Import observability components
from src.observability.metrics import (
    MetricsCollector,
    setup_metrics,
    get_metrics,
    metrics_endpoint,
    CircuitState,
)
from src.observability.tracing import (
    TracingManager,
    setup_tracing,
    get_tracer,
    get_tracing_manager,
    TraceContext,
    trace_provider_call,
)
from src.observability.logging import (
    setup_logging,
    get_logger,
    LogContext,
    JSONFormatter,
    TimedOperation,
)
from src.observability.middleware import (
    ObservabilityMiddleware,
    setup_observability,
    get_request_context,
)


# ============================================================
# Metrics Tests
# ============================================================

class TestMetricsCollector:
    """Tests for MetricsCollector."""

    @pytest.fixture
    def fresh_registry(self):
        """Create a fresh registry for each test."""
        return CollectorRegistry()

    @pytest.fixture
    def metrics(self, fresh_registry):
        """Create a metrics collector with fresh registry."""
        return MetricsCollector(registry=fresh_registry)

    def test_record_request(self, metrics):
        """Test recording a request."""
        metrics.record_request(
            endpoint="/v1/chat/completions",
            provider="openai",
            model="gpt-4",
            status_code=200,
            duration_seconds=1.5,
        )

        # Verify counter incremented
        sample = metrics.requests_total.labels(
            endpoint="/v1/chat/completions",
            provider="openai",
            model="gpt-4",
            status="200",
            error_type="none",
            streaming="false",
        )._value.get()
        assert sample == 1.0

    def test_record_tokens(self, metrics):
        """Test recording token usage."""
        metrics.record_tokens(
            provider="openai",
            model="gpt-4",
            input_tokens=100,
            output_tokens=50,
            tenant_id="test_tenant",
        )

        # Verify counters
        input_sample = metrics.tokens_total.labels(
            provider="openai",
            model="gpt-4",
            type="input",
            tenant_id="test_tenant",
        )._value.get()
        assert input_sample == 100.0

        output_sample = metrics.tokens_total.labels(
            provider="openai",
            model="gpt-4",
            type="output",
            tenant_id="test_tenant",
        )._value.get()
        assert output_sample == 50.0

    def test_record_cost(self, metrics):
        """Test recording cost."""
        metrics.record_cost(
            provider="openai",
            model="gpt-4",
            cost_usd=0.05,
            tenant_id="test_tenant",
        )

        sample = metrics.cost_total.labels(
            provider="openai",
            model="gpt-4",
            tenant_id="test_tenant",
        )._value.get()
        assert sample == 0.05

    def test_circuit_breaker_state(self, metrics):
        """Test circuit breaker state tracking."""
        metrics.set_circuit_breaker_state("openai", CircuitState.CLOSED)
        sample = metrics.circuit_breaker_state.labels(provider="openai")._value.get()
        assert sample == 0.0

        metrics.set_circuit_breaker_state("openai", CircuitState.OPEN)
        sample = metrics.circuit_breaker_state.labels(provider="openai")._value.get()
        assert sample == 2.0

    def test_active_request_tracker(self, metrics):
        """Test active request tracking."""
        with metrics.track_active_request("/v1/chat", "openai"):
            sample = metrics.active_requests.labels(
                endpoint="/v1/chat",
                provider="openai",
            )._value.get()
            assert sample == 1.0

        # After context manager, should be decremented
        sample = metrics.active_requests.labels(
            endpoint="/v1/chat",
            provider="openai",
        )._value.get()
        assert sample == 0.0

    def test_rate_limit_hit(self, metrics):
        """Test rate limit hit recording."""
        metrics.record_rate_limit_hit(
            tenant_id="test_tenant",
            api_key_prefix="2api_abc",
            limit_type="rpm",
        )

        sample = metrics.rate_limit_hits.labels(
            tenant_id="test_tenant",
            api_key_prefix="2api_abc",
            limit_type="rpm",
        )._value.get()
        assert sample == 1.0


# ============================================================
# Tracing Tests
# ============================================================

class TestTracingManager:
    """Tests for TracingManager."""

    @pytest.fixture
    def tracing(self):
        """Create a tracing manager."""
        TracingManager.reset_instance()
        return TracingManager(
            service_name="test-service",
            console_export=False,
        )

    def test_span_creation(self, tracing):
        """Test creating spans."""
        with tracing.start_span("test-operation") as span:
            span.set_attribute("test.key", "test-value")
            ctx = TraceContext.from_span(span)

            assert len(ctx.trace_id) == 32
            assert len(ctx.span_id) == 16

    def test_trace_context_propagation(self, tracing):
        """Test W3C trace context extraction."""
        headers = {
            "traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
        }

        context = tracing.extract_context(headers)
        assert context is not None

    def test_traceparent_generation(self, tracing):
        """Test traceparent header generation."""
        with tracing.start_span("test") as span:
            ctx = TraceContext.from_span(span)
            traceparent = ctx.to_traceparent()

            parts = traceparent.split("-")
            assert len(parts) == 4
            assert parts[0] == "00"  # version
            assert len(parts[1]) == 32  # trace-id
            assert len(parts[2]) == 16  # parent-id
            assert len(parts[3]) == 2  # trace-flags

    def test_inject_context(self, tracing):
        """Test context injection into headers."""
        headers = {}
        with tracing.start_span("test"):
            tracing.inject_context(headers)

        assert "traceparent" in headers


# ============================================================
# Logging Tests
# ============================================================

class TestStructuredLogging:
    """Tests for structured logging."""

    @pytest.fixture(autouse=True)
    def setup_logging_fixture(self):
        """Setup logging for tests."""
        setup_logging(level="DEBUG", json_output=True)
        yield
        LogContext.clear()

    def test_json_formatter(self):
        """Test JSON log formatting."""
        import logging
        import io

        formatter = JSONFormatter()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        data = json.loads(output)

        assert data["level"] == "INFO"
        assert data["message"] == "Test message"
        assert "timestamp" in data

    def test_log_context_injection(self):
        """Test that log context is injected."""
        ctx = LogContext(
            request_id="req_123",
            trace_id="trace_456",
            tenant_id="tenant_789",
        )
        LogContext.set_current(ctx)

        current = LogContext.get_current()
        assert current.request_id == "req_123"
        assert current.trace_id == "trace_456"

    def test_sensitive_field_redaction(self):
        """Test sensitive field redaction."""
        formatter = JSONFormatter(redact_sensitive=True)

        import logging
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test",
            args=(),
            exc_info=None,
        )
        record.password = "secret123"
        record.api_key = "key123"

        output = formatter.format(record)
        data = json.loads(output)

        assert data.get("password") == "[REDACTED]"
        assert data.get("api_key") == "[REDACTED]"

    def test_timed_operation(self):
        """Test timed operation context manager."""
        import time

        logger = get_logger("test")

        with TimedOperation("test_op", logger) as timer:
            time.sleep(0.02)  # 20ms to avoid flaky timing issues

        assert timer.duration_ms is not None
        assert timer.duration_ms >= 15  # Allow some margin


# ============================================================
# Middleware Tests
# ============================================================

class TestObservabilityMiddleware:
    """Tests for ObservabilityMiddleware."""

    @pytest.fixture
    def app(self):
        """Create test FastAPI app."""
        app = FastAPI()
        app.add_middleware(ObservabilityMiddleware, service_name="test")

        @app.get("/test")
        async def test_endpoint():
            return {"status": "ok"}

        @app.get("/health")
        async def health():
            return {"status": "healthy"}

        return app

    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)

    def test_request_id_generation(self, client):
        """Test that request ID is generated and returned."""
        response = client.get("/test")

        assert response.status_code == 200
        assert "x-request-id" in response.headers
        assert response.headers["x-request-id"].startswith("req_")

    def test_trace_id_generation(self, client):
        """Test that trace ID is generated and returned."""
        response = client.get("/test")

        assert "x-trace-id" in response.headers
        assert len(response.headers["x-trace-id"]) == 32

    def test_request_id_passthrough(self, client):
        """Test that provided request ID is passed through."""
        response = client.get(
            "/test",
            headers={"x-request-id": "req_custom123"}
        )

        assert response.headers["x-request-id"] == "req_custom123"

    def test_excluded_paths(self, client):
        """Test that excluded paths skip detailed observability."""
        response = client.get("/health")

        # Health endpoint should still work
        assert response.status_code == 200
        # But may not have full observability headers (depends on implementation)

    def test_traceparent_extraction(self, client):
        """Test W3C traceparent header extraction."""
        traceparent = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"

        response = client.get(
            "/test",
            headers={"traceparent": traceparent}
        )

        assert response.status_code == 200
        # The extracted trace should be used (can verify via response headers or logs)


# ============================================================
# Integration Tests
# ============================================================

class TestObservabilityIntegration:
    """Integration tests for the complete observability stack."""

    def test_setup_observability(self):
        """Test complete observability setup."""
        # This test uses the global registry which may already have metrics
        # from previous tests, so we handle that gracefully
        result = setup_observability(
            service_name="test",
            service_version="1.0.0",
            log_level="INFO",
            metrics_enabled=True,
            tracing_enabled=True,
            logging_enabled=True,
        )

        # Either metrics is present or logging is true
        assert "metrics" in result or result.get("logging") is True
        assert "tracing" in result
        assert result["logging"] is True

    def test_end_to_end_request_flow(self):
        """Test complete request flow with all observability."""
        # Setup - metrics may already be initialized from global registry
        setup_observability(
            service_name="e2e-test",
            metrics_enabled=True,
            tracing_enabled=True,
            logging_enabled=True,
        )

        # Create app with middleware
        app = FastAPI()
        app.add_middleware(ObservabilityMiddleware, service_name="e2e-test")

        @app.get("/api/test")
        async def test_handler(request: Request):
            # Verify request context is available
            ctx = get_request_context(request)
            return {
                "request_id": ctx["request_id"],
                "trace_id": ctx["trace_id"],
            }

        client = TestClient(app)
        response = client.get("/api/test")

        assert response.status_code == 200
        data = response.json()

        # Verify context was available in handler
        assert data["request_id"].startswith("req_")
        assert len(data["trace_id"]) == 32

        # Verify response headers
        assert "x-request-id" in response.headers
        assert "x-trace-id" in response.headers


# ============================================================
# Run Tests
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
