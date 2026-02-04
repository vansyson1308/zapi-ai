"""
2api.ai - Main API Server

FastAPI-based API server for the unified AI interface.
Uses canonical error layer from src/core/errors.py

Supports two modes:
- MODE=local: Development mode with relaxed auth (format-check only)
- MODE=prod: Production mode with full DB-backed auth

Features:
- OpenAI-compatible API endpoints
- Multi-provider support (OpenAI, Anthropic, Google)
- Intelligent routing with fallback
- Usage tracking and cost calculation
- Rate limiting and quotas
- Full observability (metrics, tracing, logging)
"""

import os
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .core.models import Provider
from .core.errors import (
    TwoApiException,
    InfraError,
    ErrorDetails,
    ErrorType,
)
from .adapters.base import AdapterConfig
from .adapters.openai_adapter import OpenAIAdapter
from .adapters.anthropic_adapter import AnthropicAdapter
from .adapters.google_adapter import GoogleAdapter
from .routing.router import Router

# Auth imports
from .auth.middleware import get_auth_context
from .auth.config import get_auth_mode, is_local_mode, is_prod_mode
from .db.models import AuthContext
from .db.connection import init_db, close_db

# API imports - new modular routes
from .api import (
    management_router,
    chat_router,
    embeddings_router,
    images_router,
    models_router,
)

# Observability imports
from .observability import (
    setup_observability,
    ObservabilityMiddleware,
    get_metrics,
    get_logger,
    metrics_endpoint,
)

# Usage tracking
from .usage import get_usage_tracker, set_usage_tracker, UsageTracker


# ============================================================
# Global state
# ============================================================

router_instance: Optional[Router] = None
usage_tracker: Optional[UsageTracker] = None


# ============================================================
# Lifespan management
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown."""
    global router_instance, usage_tracker

    mode = get_auth_mode()

    # Initialize observability first (for logging during startup)
    observability = setup_observability(
        service_name="2api",
        service_version="1.0.0",
        otlp_endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )

    logger = get_logger("server")
    logger.info(f"2api.ai starting in {mode.value.upper()} mode")

    # Initialize database in production mode
    if is_prod_mode():
        try:
            db_url = os.getenv("DATABASE_URL")
            if db_url:
                await init_db(db_url)
                logger.info("Database connected")
            else:
                logger.warning("Database not configured (set DATABASE_URL)")
        except Exception as e:
            logger.error(f"Database connection failed", error=str(e))

    # Initialize usage tracker
    usage_tracker = UsageTracker(
        buffer_size=int(os.getenv("USAGE_BUFFER_SIZE", "100")),
        flush_interval_seconds=float(os.getenv("USAGE_FLUSH_INTERVAL", "5.0"))
    )
    set_usage_tracker(usage_tracker)
    await usage_tracker.start_background_flush()
    logger.info("Usage tracker initialized")

    # Initialize adapters from environment (for local mode or default)
    adapters = {}

    # Initialize OpenAI adapter
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        adapters[Provider.OPENAI] = OpenAIAdapter(AdapterConfig(api_key=openai_key))

    # Initialize Anthropic adapter
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        adapters[Provider.ANTHROPIC] = AnthropicAdapter(AdapterConfig(api_key=anthropic_key))

    # Initialize Google adapter
    google_key = os.getenv("GOOGLE_API_KEY")
    if google_key:
        adapters[Provider.GOOGLE] = GoogleAdapter(AdapterConfig(api_key=google_key))

    if not adapters:
        logger.warning("No providers configured. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY")
    else:
        logger.info(f"Providers initialized: {', '.join(p.value for p in adapters.keys())}")

    router_instance = Router(adapters)

    # Run health check and record metrics
    metrics = get_metrics()
    if adapters:
        health = await router_instance.check_all_health()
        for provider, status in health.items():
            logger.info(
                f"Provider health check",
                provider=provider.value,
                healthy=status.is_healthy,
                latency_ms=status.avg_latency_ms,
            )
            # Record initial circuit breaker state
            from .observability.metrics import CircuitState
            metrics.set_circuit_breaker_state(
                provider.value,
                CircuitState.CLOSED if status.is_healthy else CircuitState.OPEN
            )

    logger.info(
        "2api.ai server ready",
        auth_mode=mode.value,
        providers=list(p.value for p in adapters.keys()),
    )
    if is_local_mode():
        logger.info("Tip: Use any key starting with '2api_' (e.g., 2api_test)")

    yield

    # Shutdown: Stop usage tracker
    if usage_tracker:
        await usage_tracker.stop_background_flush()
        logger.info("Usage tracker flushed and stopped")

    # Shutdown: Close adapters
    for adapter in adapters.values():
        await adapter.close()

    # Close database
    if is_prod_mode():
        await close_db()

    # Shutdown tracing
    if "tracing" in observability:
        observability["tracing"].shutdown()

    logger.info("2api.ai server stopped")


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(
    title="2api.ai",
    description="Unified AI API - Access OpenAI, Anthropic, and Google through a single interface",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add custom middleware (order matters - first added = outermost)
# ObservabilityMiddleware handles metrics, tracing, and logging in one place
app.add_middleware(ObservabilityMiddleware, service_name="2api")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all routers
app.include_router(management_router)
app.include_router(chat_router)
app.include_router(embeddings_router)
app.include_router(images_router)
app.include_router(models_router)


# ============================================================
# Router getter (for dependency injection)
# ============================================================

def get_router_instance() -> Router:
    """Get the router instance for dependency injection."""
    if router_instance is None:
        raise InfraError(
            ErrorDetails(
                code="service_unavailable",
                message="Router not initialized",
                type=ErrorType.INFRA,
                request_id="",
                retryable=True,
                retry_after=5
            ),
            status_code=503
        )
    return router_instance


# Store in dependencies module for routes to access
from .api import dependencies as api_deps
api_deps._router_instance_getter = get_router_instance


# ============================================================
# Core Endpoints (not in routes)
# ============================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    router = get_router_instance()
    health = await router.check_all_health()

    all_healthy = all(h.is_healthy for h in health.values())

    return {
        "status": "healthy" if all_healthy else "degraded",
        "version": "1.0.0",
        "mode": get_auth_mode().value,
        "usage_tracker": {
            "active_requests": get_usage_tracker().get_active_request_count() if usage_tracker else 0
        },
        "providers": {
            provider.value: {
                "status": "healthy" if h.is_healthy else "unhealthy",
                "latency_ms": h.avg_latency_ms
            }
            for provider, h in health.items()
        }
    }


@app.get("/metrics")
async def prometheus_metrics():
    """
    Prometheus metrics endpoint.

    Exposes all collected metrics in Prometheus text format.
    Scrape this endpoint with Prometheus server.
    """
    return metrics_endpoint()


@app.get("/ready")
async def readiness_check():
    """
    Readiness check endpoint.

    Returns 200 if the service is ready to accept requests.
    Used by Kubernetes/load balancers for health checks.
    """
    router = get_router_instance()
    health = await router.check_all_health()

    # Service is ready if at least one provider is healthy
    any_healthy = any(h.is_healthy for h in health.values())

    if not any_healthy and health:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "reason": "No healthy providers available"
            }
        )

    return {"status": "ready"}


@app.get("/v1/usage")
async def get_usage(
    auth: AuthContext = Depends(get_auth_context)
):
    """Get usage statistics for the current tenant."""
    tracker = get_usage_tracker()

    # Get in-memory usage for this tenant
    if auth.tenant_id:
        tenant_usage = tracker.get_tenant_usage(auth.tenant_id)
    else:
        tenant_usage = {
            "tenant_id": None,
            "total_tokens": 0,
            "total_cost_usd": 0.0,
            "request_count": 0
        }

    return JSONResponse(
        content={
            "object": "usage",
            "tenant_id": auth.tenant_id,
            "data": tenant_usage,
            "note": "In-memory usage data. For historical data, query the database."
        },
        headers={
            "X-Request-Id": auth.request_id,
            "X-Trace-Id": auth.trace_id
        }
    )


@app.get("/v1/stats")
async def get_stats(auth: AuthContext = Depends(get_auth_context)):
    """Get routing statistics."""
    router = get_router_instance()
    tracker = get_usage_tracker()

    return JSONResponse(
        content={
            "routing": router.get_stats(),
            "usage": {
                "active_requests": tracker.get_active_request_count()
            }
        },
        headers={
            "X-Request-Id": auth.request_id,
            "X-Trace-Id": auth.trace_id
        }
    )


# ============================================================
# Error handlers
# ============================================================

@app.exception_handler(TwoApiException)
async def twoapi_exception_handler(request: Request, exc: TwoApiException):
    """Handle all 2api.ai canonical errors."""
    headers = {
        "X-Request-Id": exc.error.request_id,
        "X-Trace-Id": exc.error.request_id.replace("req_", "trace_"),
        "X-Error-Type": exc.error.type.value,
        "X-Error-Code": exc.error.code,
    }

    if exc.error.retry_after:
        headers["Retry-After"] = str(exc.error.retry_after)

    if exc.error.provider:
        headers["X-Provider"] = exc.error.provider

    return JSONResponse(
        status_code=exc.status_code,
        content=exc.error.to_dict(),
        headers=headers
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle standard HTTP exceptions."""
    request_id = f"req_{uuid.uuid4().hex[:24]}"

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": "http_error",
                "message": str(exc.detail) if isinstance(exc.detail, str) else exc.detail.get("message", "Unknown error"),
                "type": "semantic_error" if exc.status_code < 500 else "infra_error",
                "request_id": request_id,
                "retryable": exc.status_code >= 500
            }
        },
        headers={
            "X-Request-Id": request_id,
            "X-Trace-Id": request_id.replace("req_", "trace_")
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    request_id = f"req_{uuid.uuid4().hex[:24]}"

    import traceback
    traceback.print_exc()

    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "internal_error",
                "message": "An unexpected error occurred",
                "type": "infra_error",
                "request_id": request_id,
                "retryable": True
            }
        },
        headers={
            "X-Request-Id": request_id,
            "X-Trace-Id": request_id.replace("req_", "trace_")
        }
    )


# ============================================================
# Run server
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
