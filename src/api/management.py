"""
2api.ai - Management API

Endpoints for tenant and API key management.
These are admin-only endpoints for managing the platform.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr, Field

from ..auth.middleware import get_auth_context
from ..auth.config import is_prod_mode
from ..db.models import AuthContext
from ..db.connection import get_db_optional
from ..db.services import (
    TenantService,
    APIKeyService,
    ProviderConfigService,
    UsageService,
    RoutingPolicyService,
)
from ..core.errors import (
    SemanticError,
    ErrorDetails,
    ErrorType,
)


router = APIRouter(prefix="/v1", tags=["management"])


# ============================================================
# Pydantic Models
# ============================================================

class CreateTenantRequest(BaseModel):
    """Request to create a new tenant."""
    name: str = Field(..., min_length=1, max_length=255)
    email: EmailStr
    plan: str = Field(default="free", pattern="^(free|starter|pro|enterprise)$")
    settings: Optional[Dict[str, Any]] = None


class CreateAPIKeyRequest(BaseModel):
    """Request to create a new API key."""
    name: str = Field(default="Default Key", max_length=255)
    permissions: Optional[List[str]] = None
    rate_limit_per_minute: int = Field(default=60, ge=1, le=10000)
    expires_in_days: Optional[int] = Field(default=None, ge=1, le=365)


class SetProviderKeyRequest(BaseModel):
    """Request to set a provider API key."""
    provider: str = Field(..., pattern="^(openai|anthropic|google)$")
    api_key: str = Field(..., min_length=1)
    settings: Optional[Dict[str, Any]] = None


class CreateRoutingPolicyRequest(BaseModel):
    """Request to create a routing policy."""
    name: str = Field(..., min_length=1, max_length=255)
    strategy: str = Field(default="cost", pattern="^(cost|latency|quality|custom)$")
    fallback_chain: Optional[List[str]] = None
    is_default: bool = False
    max_latency_ms: Optional[int] = Field(default=None, ge=100, le=300000)
    max_cost_per_request: Optional[float] = Field(default=None, ge=0)


# ============================================================
# Helper Functions
# ============================================================

def _require_prod_mode(auth: AuthContext):
    """Ensure we're in production mode with DB available."""
    if not is_prod_mode():
        raise SemanticError(
            ErrorDetails(
                code="local_mode",
                message="Management APIs require MODE=prod. Set MODE=prod and configure DATABASE_URL.",
                type=ErrorType.SEMANTIC,
                request_id=auth.request_id,
                retryable=False,
            ),
            status_code=400,
        )

    db = get_db_optional()
    if db is None:
        raise SemanticError(
            ErrorDetails(
                code="database_unavailable",
                message="Database not configured. Set DATABASE_URL environment variable.",
                type=ErrorType.SEMANTIC,
                request_id=auth.request_id,
                retryable=False,
            ),
            status_code=503,
        )

    return db


def _require_admin(auth: AuthContext):
    """Ensure the request has admin permission."""
    if not auth.has_permission("admin:*") and not auth.has_permission("*"):
        raise SemanticError(
            ErrorDetails(
                code="permission_denied",
                message="Admin permission required",
                type=ErrorType.SEMANTIC,
                request_id=auth.request_id,
                retryable=False,
            ),
            status_code=403,
        )


# ============================================================
# Tenant Endpoints
# ============================================================

@router.post("/tenants")
async def create_tenant(
    request: CreateTenantRequest,
    auth: AuthContext = Depends(get_auth_context),
):
    """
    Create a new tenant.

    Requires admin permission.
    """
    db = _require_prod_mode(auth)
    _require_admin(auth)

    tenant_service = TenantService(db)

    # Check if email already exists
    existing = await tenant_service.get_tenant_by_email(request.email)
    if existing:
        raise SemanticError(
            ErrorDetails(
                code="tenant_exists",
                message=f"Tenant with email {request.email} already exists",
                type=ErrorType.SEMANTIC,
                request_id=auth.request_id,
                retryable=False,
            ),
            status_code=409,
        )

    tenant = await tenant_service.create_tenant(
        name=request.name,
        email=request.email,
        plan=request.plan,
        settings=request.settings,
    )

    return JSONResponse(
        content={
            "id": str(tenant.id),
            "name": tenant.name,
            "email": tenant.email,
            "plan": tenant.plan,
            "is_active": tenant.is_active,
            "created_at": tenant.created_at.isoformat() if tenant.created_at else None,
        },
        headers={"X-Request-Id": auth.request_id},
    )


@router.get("/tenants/me")
async def get_current_tenant(
    auth: AuthContext = Depends(get_auth_context),
):
    """Get the current tenant (based on API key)."""
    return JSONResponse(
        content={
            "id": str(auth.tenant.id),
            "name": auth.tenant.name,
            "email": auth.tenant.email,
            "plan": auth.tenant.plan,
            "is_active": auth.tenant.is_active,
            "settings": auth.tenant.settings,
        },
        headers={"X-Request-Id": auth.request_id},
    )


# ============================================================
# API Key Endpoints
# ============================================================

@router.post("/api-keys")
async def create_api_key(
    request: CreateAPIKeyRequest,
    auth: AuthContext = Depends(get_auth_context),
):
    """
    Create a new API key for the current tenant.

    IMPORTANT: The plaintext key is only returned once in this response.
    Store it securely - it cannot be retrieved again.
    """
    db = _require_prod_mode(auth)

    key_service = APIKeyService(db)
    plaintext_key, api_key = await key_service.create_key(
        tenant_id=auth.tenant_id,
        name=request.name,
        permissions=request.permissions,
        rate_limit_per_minute=request.rate_limit_per_minute,
        expires_in_days=request.expires_in_days,
    )

    return JSONResponse(
        content={
            "id": str(api_key.id),
            "key": plaintext_key,  # Only returned once!
            "key_prefix": api_key.key_prefix,
            "name": api_key.name,
            "permissions": api_key.permissions,
            "rate_limit_per_minute": api_key.rate_limit_per_minute,
            "expires_at": api_key.expires_at.isoformat() if api_key.expires_at else None,
            "created_at": api_key.created_at.isoformat() if api_key.created_at else None,
            "_warning": "Store this key securely. It will not be shown again.",
        },
        headers={"X-Request-Id": auth.request_id},
    )


@router.get("/api-keys")
async def list_api_keys(
    auth: AuthContext = Depends(get_auth_context),
):
    """List all API keys for the current tenant (without secrets)."""
    db = _require_prod_mode(auth)

    key_service = APIKeyService(db)
    keys = await key_service.list_keys(auth.tenant_id)

    return JSONResponse(
        content={
            "object": "list",
            "data": [
                {
                    "id": str(k.id),
                    "key_prefix": k.key_prefix,
                    "name": k.name,
                    "permissions": k.permissions,
                    "rate_limit_per_minute": k.rate_limit_per_minute,
                    "is_active": k.is_active,
                    "last_used_at": k.last_used_at.isoformat() if k.last_used_at else None,
                    "expires_at": k.expires_at.isoformat() if k.expires_at else None,
                    "created_at": k.created_at.isoformat() if k.created_at else None,
                }
                for k in keys
            ],
        },
        headers={"X-Request-Id": auth.request_id},
    )


@router.delete("/api-keys/{key_id}")
async def revoke_api_key(
    key_id: UUID,
    auth: AuthContext = Depends(get_auth_context),
):
    """Revoke (deactivate) an API key."""
    db = _require_prod_mode(auth)

    key_service = APIKeyService(db)
    success = await key_service.revoke_key(key_id)

    if not success:
        raise SemanticError(
            ErrorDetails(
                code="key_not_found",
                message=f"API key {key_id} not found",
                type=ErrorType.SEMANTIC,
                request_id=auth.request_id,
                retryable=False,
            ),
            status_code=404,
        )

    return JSONResponse(
        content={"status": "revoked", "id": str(key_id)},
        headers={"X-Request-Id": auth.request_id},
    )


# ============================================================
# Provider Configuration Endpoints
# ============================================================

@router.post("/providers")
async def set_provider_key(
    request: SetProviderKeyRequest,
    auth: AuthContext = Depends(get_auth_context),
):
    """
    Set or update provider API key for the current tenant.

    The API key is encrypted at rest.
    """
    db = _require_prod_mode(auth)

    config_service = ProviderConfigService(db)
    config = await config_service.set_provider_key(
        tenant_id=auth.tenant_id,
        provider=request.provider,
        api_key=request.api_key,
        settings=request.settings,
    )

    return JSONResponse(
        content={
            "id": str(config.id),
            "provider": config.provider,
            "is_active": config.is_active,
            "settings": config.settings,
            "updated_at": config.updated_at.isoformat() if config.updated_at else None,
        },
        headers={"X-Request-Id": auth.request_id},
    )


@router.get("/providers")
async def list_providers(
    auth: AuthContext = Depends(get_auth_context),
):
    """List all provider configurations for the current tenant."""
    db = _require_prod_mode(auth)

    config_service = ProviderConfigService(db)
    configs = await config_service.list_provider_configs(auth.tenant_id)

    return JSONResponse(
        content={
            "object": "list",
            "data": [
                {
                    "id": str(c.id),
                    "provider": c.provider,
                    "is_active": c.is_active,
                    "has_key": c.api_key_encrypted is not None,
                    "settings": c.settings,
                    "updated_at": c.updated_at.isoformat() if c.updated_at else None,
                }
                for c in configs
            ],
        },
        headers={"X-Request-Id": auth.request_id},
    )


@router.delete("/providers/{provider}")
async def delete_provider(
    provider: str,
    auth: AuthContext = Depends(get_auth_context),
):
    """Delete provider configuration."""
    db = _require_prod_mode(auth)

    config_service = ProviderConfigService(db)
    success = await config_service.delete_provider_config(auth.tenant_id, provider)

    if not success:
        raise SemanticError(
            ErrorDetails(
                code="provider_not_found",
                message=f"Provider {provider} not configured",
                type=ErrorType.SEMANTIC,
                request_id=auth.request_id,
                retryable=False,
            ),
            status_code=404,
        )

    return JSONResponse(
        content={"status": "deleted", "provider": provider},
        headers={"X-Request-Id": auth.request_id},
    )


# ============================================================
# Usage Endpoints
# ============================================================

@router.get("/admin/usage")
async def get_usage(
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    auth: AuthContext = Depends(get_auth_context),
):
    """Get usage summary for the current tenant."""
    db = _require_prod_mode(auth)

    # Parse dates
    start_dt = None
    end_dt = None
    if start_date:
        start_dt = datetime.fromisoformat(start_date)
    if end_date:
        end_dt = datetime.fromisoformat(end_date)

    usage_service = UsageService(db)
    summary = await usage_service.get_usage_summary(
        tenant_id=auth.tenant_id,
        start_date=start_dt,
        end_date=end_dt,
    )

    return JSONResponse(
        content={
            "object": "usage_summary",
            "tenant_id": str(auth.tenant_id),
            "period": {
                "start": start_date,
                "end": end_date,
            },
            "summary": summary,
        },
        headers={"X-Request-Id": auth.request_id},
    )


# ============================================================
# Routing Policy Endpoints
# ============================================================

@router.post("/routing-policies")
async def create_routing_policy(
    request: CreateRoutingPolicyRequest,
    auth: AuthContext = Depends(get_auth_context),
):
    """Create a new routing policy."""
    db = _require_prod_mode(auth)

    policy_service = RoutingPolicyService(db)
    policy = await policy_service.create_policy(
        tenant_id=auth.tenant_id,
        name=request.name,
        strategy=request.strategy,
        fallback_chain=request.fallback_chain,
        is_default=request.is_default,
        max_latency_ms=request.max_latency_ms,
        max_cost_per_request=request.max_cost_per_request,
    )

    return JSONResponse(
        content={
            "id": str(policy.id),
            "name": policy.name,
            "strategy": policy.strategy,
            "fallback_chain": policy.fallback_chain,
            "is_default": policy.is_default,
            "max_latency_ms": policy.max_latency_ms,
            "max_cost_per_request": policy.max_cost_per_request,
            "created_at": policy.created_at.isoformat() if policy.created_at else None,
        },
        headers={"X-Request-Id": auth.request_id},
    )


@router.get("/routing-policies")
async def list_routing_policies(
    auth: AuthContext = Depends(get_auth_context),
):
    """List all routing policies for the current tenant."""
    db = _require_prod_mode(auth)

    policy_service = RoutingPolicyService(db)
    policies = await policy_service.list_policies(auth.tenant_id)

    return JSONResponse(
        content={
            "object": "list",
            "data": [
                {
                    "id": str(p.id),
                    "name": p.name,
                    "strategy": p.strategy,
                    "fallback_chain": p.fallback_chain,
                    "is_default": p.is_default,
                    "max_latency_ms": p.max_latency_ms,
                    "max_cost_per_request": p.max_cost_per_request,
                    "created_at": p.created_at.isoformat() if p.created_at else None,
                }
                for p in policies
            ],
        },
        headers={"X-Request-Id": auth.request_id},
    )
