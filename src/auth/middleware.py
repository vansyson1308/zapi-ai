"""
2api.ai - Auth Middleware

FastAPI dependency for API key validation and tenant context.
"""

import uuid
from functools import wraps
from typing import Optional
from uuid import UUID

from fastapi import Header, Request, Depends

from ..core.errors import (
    InvalidAPIKeyError,
    MissingAPIKeyError,
    SemanticError,
    ErrorDetails,
    ErrorType,
)
from ..db.models import AuthContext, Tenant, APIKey
from ..db.connection import get_db_optional
from ..db.services import APIKeyService, TenantService
from .config import (
    AuthMode,
    get_auth_mode,
    is_local_mode,
    LOCAL_DEFAULT_TENANT_ID,
    LOCAL_DEFAULT_API_KEY_ID,
)


class RequestContext:
    """
    Request context with tracing information.

    Created at the start of each request for tracking.
    """

    def __init__(self):
        self.request_id = f"req_{uuid.uuid4().hex[:24]}"
        self.trace_id = f"trace_{uuid.uuid4().hex[:24]}"


def _create_local_auth_context(
    api_key: str,
    request_id: str,
    trace_id: str,
) -> AuthContext:
    """
    Create a mock AuthContext for local development mode.

    In local mode, we don't require a database. We just verify
    the key format and create a mock context.
    """
    # Create mock tenant
    tenant = Tenant(
        id=UUID(LOCAL_DEFAULT_TENANT_ID),
        name="Local Development",
        email="local@dev.test",
        plan="pro",
        settings={},
        is_active=True,
    )

    # Create mock API key
    api_key_obj = APIKey(
        id=UUID(LOCAL_DEFAULT_API_KEY_ID),
        tenant_id=UUID(LOCAL_DEFAULT_TENANT_ID),
        key_hash="local",
        key_prefix=api_key[:12] if len(api_key) >= 12 else api_key,
        name="Local Dev Key",
        permissions=["*"],
        rate_limit_per_minute=1000,  # High limit for development
        is_active=True,
    )

    return AuthContext(
        tenant_id=tenant.id,
        api_key_id=api_key_obj.id,
        tenant=tenant,
        api_key=api_key_obj,
        permissions=["*"],
        rate_limit_per_minute=1000,
        request_id=request_id,
        trace_id=trace_id,
    )


async def _validate_api_key_db(
    api_key: str,
    request_id: str,
    trace_id: str,
) -> AuthContext:
    """
    Validate API key against database (production mode).

    Raises:
        InvalidAPIKeyError: If key is invalid, expired, or inactive
    """
    db = get_db_optional()
    if db is None:
        raise SemanticError(
            ErrorDetails(
                code="database_unavailable",
                message="Database not configured. Set DATABASE_URL or use MODE=local.",
                type=ErrorType.SEMANTIC,
                request_id=request_id,
                retryable=False,
            ),
            status_code=503,
        )

    # Validate key format
    if not api_key.startswith("2api_"):
        raise InvalidAPIKeyError(
            message="Invalid API key format. Keys should start with '2api_'",
            request_id=request_id,
        )

    # Look up key in database
    key_service = APIKeyService(db)
    api_key_obj = await key_service.validate_key(api_key)

    if api_key_obj is None:
        raise InvalidAPIKeyError(
            message="Invalid or expired API key",
            request_id=request_id,
        )

    # Get tenant
    tenant_service = TenantService(db)
    tenant = await tenant_service.get_tenant(api_key_obj.tenant_id)

    if tenant is None:
        raise InvalidAPIKeyError(
            message="Tenant not found or inactive",
            request_id=request_id,
        )

    # Update last used timestamp (fire and forget)
    try:
        await key_service.update_last_used(api_key_obj.id)
    except Exception:
        pass  # Don't fail request if update fails

    return AuthContext(
        tenant_id=tenant.id,
        api_key_id=api_key_obj.id,
        tenant=tenant,
        api_key=api_key_obj,
        permissions=api_key_obj.permissions,
        rate_limit_per_minute=api_key_obj.rate_limit_per_minute,
        request_id=request_id,
        trace_id=trace_id,
    )


async def get_auth_context(
    authorization: Optional[str] = Header(None, alias="Authorization"),
    request: Request = None,
) -> AuthContext:
    """
    FastAPI dependency that validates API key and returns auth context.

    Usage:
        @app.get("/v1/models")
        async def list_models(auth: AuthContext = Depends(get_auth_context)):
            print(f"Tenant: {auth.tenant.name}")

    Behavior by mode:
    - LOCAL mode: Just validates key format (2api_*), returns mock context
    - PROD mode: Full database lookup, validates key hash, tenant status

    Raises:
        MissingAPIKeyError: If no Authorization header
        InvalidAPIKeyError: If key is invalid
    """
    # Create request context for tracing
    ctx = RequestContext()

    # Check for Authorization header
    if not authorization:
        raise MissingAPIKeyError(request_id=ctx.request_id)

    # Extract bearer token
    if authorization.startswith("Bearer "):
        api_key = authorization[7:]
    else:
        api_key = authorization

    # Validate based on mode
    if is_local_mode():
        # Local mode: just check format
        if not api_key.startswith("2api_"):
            raise InvalidAPIKeyError(
                message="Invalid API key format. Keys should start with '2api_'",
                request_id=ctx.request_id,
            )
        return _create_local_auth_context(api_key, ctx.request_id, ctx.trace_id)
    else:
        # Production mode: full DB validation
        return await _validate_api_key_db(api_key, ctx.request_id, ctx.trace_id)


async def get_auth_context_optional(
    authorization: Optional[str] = Header(None, alias="Authorization"),
    request: Request = None,
) -> Optional[AuthContext]:
    """
    Optional auth context - returns None instead of raising error.

    Useful for endpoints that work with or without auth.
    """
    if not authorization:
        return None

    try:
        return await get_auth_context(authorization, request)
    except Exception:
        return None


def require_permission(permission: str):
    """
    Decorator to require a specific permission.

    Usage:
        @app.post("/admin/tenants")
        @require_permission("admin:tenants:create")
        async def create_tenant(auth: AuthContext = Depends(get_auth_context)):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, auth: AuthContext = None, **kwargs):
            if auth is None:
                # Try to get from kwargs if passed differently
                auth = kwargs.get("auth")

            if auth is None:
                raise MissingAPIKeyError()

            if not auth.has_permission(permission):
                raise SemanticError(
                    ErrorDetails(
                        code="permission_denied",
                        message=f"Permission required: {permission}",
                        type=ErrorType.SEMANTIC,
                        request_id=auth.request_id,
                        retryable=False,
                    ),
                    status_code=403,
                )

            return await func(*args, auth=auth, **kwargs)

        return wrapper
    return decorator
