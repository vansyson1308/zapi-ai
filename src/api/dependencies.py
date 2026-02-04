"""
2api.ai - API Dependencies

Shared dependencies for FastAPI routes.
Provides common functionality used across endpoints.
"""

import time
import uuid
from typing import Any, Dict, Optional

from fastapi import Depends, Header, Request

from ..auth.middleware import get_auth_context
from ..auth.config import is_local_mode
from ..db.models import AuthContext
from ..core.errors import (
    InfraError,
    ErrorDetails,
    ErrorType,
)
from ..routing.router import Router
from ..usage import (
    UsageTracker,
    RequestTracker,
    OperationType,
    start_tracking,
    check_limits,
    record_usage,
)


# Global router instance getter (set by server lifespan)
# This function is set by server.py to avoid circular imports
_router_instance_getter = None


def set_router_getter(getter):
    """Set the function that returns the router instance."""
    global _router_instance_getter
    _router_instance_getter = getter


def get_router() -> Router:
    """
    Get the router instance.

    Dependency that provides access to the AI router.
    Uses a getter function set by the server to avoid circular imports.
    """
    if _router_instance_getter is None:
        raise InfraError(
            ErrorDetails(
                code="service_unavailable",
                message="Router not initialized. Server may be starting up.",
                type=ErrorType.INFRA,
                request_id="",
                retryable=True,
                retry_after=5
            ),
            status_code=503
        )
    return _router_instance_getter()


async def get_request_context(
    request: Request,
    auth: AuthContext = Depends(get_auth_context)
) -> Dict[str, Any]:
    """
    Build complete request context.

    Combines auth context with request metadata.
    """
    return {
        "request_id": auth.request_id,
        "trace_id": auth.trace_id,
        "tenant_id": str(auth.tenant_id),
        "api_key_id": str(auth.api_key_id),
        "client_ip": request.client.host if request.client else None,
        "user_agent": request.headers.get("user-agent"),
        "start_time": time.time(),
    }


async def check_rate_limits(
    auth: AuthContext = Depends(get_auth_context),
    estimated_tokens: int = 0,
    estimated_cost: float = 0.0
):
    """
    Check rate limits for the request.

    Raises if limits are exceeded.
    """
    from ..usage import LimitCheckResult

    if is_local_mode():
        # Skip rate limiting in local mode
        return

    result = await check_limits(
        tenant_id=str(auth.tenant_id),
        tokens=estimated_tokens,
        cost=estimated_cost
    )

    if not result.allowed:
        # Find the first exceeded limit
        exceeded = result.exceeded_limits[0] if result.exceeded_limits else None

        from ..core.errors import RateLimitedError

        raise RateLimitedError(
            ErrorDetails(
                code="rate_limit_exceeded",
                message=f"Rate limit exceeded: {exceeded.limit.name if exceeded else 'unknown'}",
                type=ErrorType.SEMANTIC,
                request_id=auth.request_id,
                retryable=True,
                retry_after=result.retry_after or 60
            ),
            status_code=429
        )


def start_request_tracking(
    auth: AuthContext,
    model: str,
    operation: OperationType = OperationType.CHAT
) -> RequestTracker:
    """
    Start tracking a request.

    Returns a tracker that should be completed when request finishes.
    """
    provider = ""
    if "/" in model:
        provider = model.split("/")[0]

    return start_tracking(
        request_id=auth.request_id,
        tenant_id=str(auth.tenant_id),
        api_key_id=str(auth.api_key_id),
        model=model,
        provider=provider,
        operation=operation
    )


def add_standard_headers(
    response_headers: Dict[str, str],
    auth: AuthContext,
    **extra_headers
) -> Dict[str, str]:
    """
    Add standard response headers.

    Adds request ID, trace ID, and any extra headers.
    """
    headers = {
        "X-Request-Id": auth.request_id,
        "X-Trace-Id": auth.trace_id,
        **response_headers,
        **{k: str(v) for k, v in extra_headers.items() if v is not None}
    }
    return headers


class RequestTimer:
    """
    Context manager for timing requests.

    Usage:
        async with RequestTimer() as timer:
            # do work
        print(f"Took {timer.elapsed_ms}ms")
    """

    def __init__(self):
        self.start_time: float = 0
        self.end_time: float = 0
        self.first_token_time: Optional[float] = None

    async def __aenter__(self):
        self.start_time = time.time()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        return False

    def record_first_token(self):
        """Record time of first token."""
        if self.first_token_time is None:
            self.first_token_time = time.time()

    @property
    def elapsed_ms(self) -> int:
        """Get elapsed time in milliseconds."""
        if self.end_time:
            return int((self.end_time - self.start_time) * 1000)
        return int((time.time() - self.start_time) * 1000)

    @property
    def ttft_ms(self) -> Optional[int]:
        """Get time to first token in milliseconds."""
        if self.first_token_time:
            return int((self.first_token_time - self.start_time) * 1000)
        return None


def validate_model_access(
    model: str,
    auth: AuthContext,
    router: Router
) -> bool:
    """
    Validate that the tenant has access to the model.

    Returns True if access is allowed.
    Raises SemanticError if not.
    """
    from ..core.errors import SemanticError

    # Extract provider
    if "/" in model:
        provider_name = model.split("/")[0]
    else:
        # Default to openai for unqualified model names
        provider_name = "openai"

    # Check if provider is configured
    from ..core.models import Provider
    try:
        provider = Provider(provider_name)
    except ValueError:
        raise SemanticError(
            ErrorDetails(
                code="invalid_provider",
                message=f"Unknown provider: {provider_name}",
                type=ErrorType.SEMANTIC,
                request_id=auth.request_id,
                retryable=False
            ),
            status_code=400
        )

    # Check if adapter is available
    if provider not in router.adapters:
        raise SemanticError(
            ErrorDetails(
                code="provider_not_configured",
                message=f"Provider {provider_name} is not configured. Set {provider_name.upper()}_API_KEY environment variable.",
                type=ErrorType.SEMANTIC,
                request_id=auth.request_id,
                retryable=False
            ),
            status_code=400
        )

    return True


def generate_request_id() -> str:
    """Generate a unique request ID."""
    return f"req_{uuid.uuid4().hex[:24]}"


def generate_trace_id(request_id: str) -> str:
    """Generate trace ID from request ID."""
    return request_id.replace("req_", "trace_")
