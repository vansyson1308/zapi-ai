"""
2api.ai - Auth System Tests

Tests for EPIC B - Auth + Multi-tenant system.
Verifies:
- Local mode auth (format check only)
- Auth context creation
- API key service functionality
- Tenant service functionality
- Mode detection
"""

import pytest
import os
from unittest.mock import MagicMock, AsyncMock, patch
from uuid import UUID

from src.auth.config import (
    AuthMode,
    get_auth_mode,
    is_local_mode,
    is_prod_mode,
    LOCAL_DEFAULT_TENANT_ID,
    LOCAL_DEFAULT_API_KEY_ID,
)
from src.auth.middleware import (
    RequestContext,
    get_auth_context,
    get_auth_context_optional,
    _create_local_auth_context,
)
from src.db.models import (
    Tenant,
    APIKey,
    ProviderConfig,
    UsageRecord,
    RoutingPolicy,
    AuthContext,
)
from src.db.services import APIKeyService
from src.core.errors import (
    InvalidAPIKeyError,
    MissingAPIKeyError,
)


# ============================================================
# Mode Detection Tests
# ============================================================

class TestModeDetection:
    """Test auth mode detection from environment."""

    def test_default_mode_is_prod(self):
        """Default mode should be prod when MODE env is not set."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove MODE if it exists
            os.environ.pop("MODE", None)
            mode = get_auth_mode()
            assert mode == AuthMode.PROD

    def test_mode_local_explicit(self):
        """MODE=local should return LOCAL mode."""
        with patch.dict(os.environ, {"MODE": "local"}):
            mode = get_auth_mode()
            assert mode == AuthMode.LOCAL

    def test_mode_prod_explicit(self):
        """MODE=prod should return PROD mode."""
        with patch.dict(os.environ, {"MODE": "prod"}):
            mode = get_auth_mode()
            assert mode == AuthMode.PROD

    def test_is_local_mode_true(self):
        """is_local_mode should return True in local mode."""
        with patch.dict(os.environ, {"MODE": "local"}):
            assert is_local_mode() is True
            assert is_prod_mode() is False

    def test_is_prod_mode_true(self):
        """is_prod_mode should return True in prod mode."""
        with patch.dict(os.environ, {"MODE": "prod"}):
            assert is_prod_mode() is True
            assert is_local_mode() is False


# ============================================================
# Local Auth Context Tests
# ============================================================

class TestLocalAuthContext:
    """Test local mode auth context creation."""

    def test_create_local_auth_context_success(self):
        """Valid key format should create auth context."""
        api_key = "2api_test123456789"
        ctx = _create_local_auth_context(api_key, "req_123", "trace_123")

        assert ctx.tenant_id == UUID(LOCAL_DEFAULT_TENANT_ID)
        assert ctx.api_key_id == UUID(LOCAL_DEFAULT_API_KEY_ID)
        assert ctx.tenant.name == "Local Development"
        assert ctx.api_key.name == "Local Dev Key"
        assert ctx.permissions == ["*"]
        assert ctx.rate_limit_per_minute == 1000
        assert ctx.request_id == "req_123"
        assert ctx.trace_id == "trace_123"

    def test_local_auth_context_has_all_permissions(self):
        """Local mode should have wildcard permission."""
        api_key = "2api_abcdef"
        ctx = _create_local_auth_context(api_key, "req_x", "trace_x")

        assert ctx.has_permission("anything")
        assert ctx.has_permission("admin:tenants:create")
        assert ctx.has_permission("chat:completion")


# ============================================================
# Auth Middleware Tests
# ============================================================

class TestAuthMiddleware:
    """Test auth middleware behavior."""

    @pytest.mark.asyncio
    async def test_missing_auth_header_raises_error(self):
        """Missing Authorization header should raise MissingAPIKeyError."""
        with patch.dict(os.environ, {"MODE": "local"}):
            with pytest.raises(MissingAPIKeyError):
                await get_auth_context(authorization=None, request=None)

    @pytest.mark.asyncio
    async def test_invalid_key_format_raises_error(self):
        """Invalid key format should raise InvalidAPIKeyError."""
        with patch.dict(os.environ, {"MODE": "local"}):
            with pytest.raises(InvalidAPIKeyError) as exc_info:
                await get_auth_context(authorization="sk_invalid_key", request=None)

            assert "2api_" in exc_info.value.error.message

    @pytest.mark.asyncio
    async def test_valid_key_in_local_mode(self):
        """Valid key format should succeed in local mode."""
        with patch.dict(os.environ, {"MODE": "local"}):
            ctx = await get_auth_context(authorization="2api_test_key_123", request=None)

            assert ctx is not None
            assert ctx.tenant is not None
            assert ctx.api_key is not None

    @pytest.mark.asyncio
    async def test_bearer_token_extracted(self):
        """Bearer token prefix should be extracted."""
        with patch.dict(os.environ, {"MODE": "local"}):
            ctx = await get_auth_context(
                authorization="Bearer 2api_bearer_test",
                request=None
            )

            assert ctx is not None
            assert ctx.api_key.key_prefix.startswith("2api_")

    @pytest.mark.asyncio
    async def test_optional_auth_returns_none(self):
        """Optional auth should return None when no header."""
        with patch.dict(os.environ, {"MODE": "local"}):
            ctx = await get_auth_context_optional(authorization=None, request=None)
            assert ctx is None

    @pytest.mark.asyncio
    async def test_optional_auth_returns_context(self):
        """Optional auth should return context when valid key."""
        with patch.dict(os.environ, {"MODE": "local"}):
            ctx = await get_auth_context_optional(
                authorization="2api_valid_key",
                request=None
            )
            assert ctx is not None


# ============================================================
# API Key Service Tests
# ============================================================

class TestAPIKeyService:
    """Test API key service functionality."""

    def test_generate_key_format(self):
        """Generated key should have correct format."""
        full_key, key_hash, key_prefix = APIKeyService.generate_key()

        # Key should start with 2api_
        assert full_key.startswith("2api_")

        # Key should be long enough (2api_ + 64 hex chars)
        assert len(full_key) == 5 + 64  # "2api_" + 64 hex chars

        # Hash should be 64 chars (SHA-256 hex)
        assert len(key_hash) == 64

        # Prefix should be first 12 chars
        assert key_prefix == full_key[:12]

    def test_generate_key_unique(self):
        """Each generated key should be unique."""
        keys = set()
        for _ in range(100):
            full_key, _, _ = APIKeyService.generate_key()
            keys.add(full_key)

        assert len(keys) == 100

    def test_hash_key_deterministic(self):
        """Same key should always produce same hash."""
        key = "2api_test_key_123"
        hash1 = APIKeyService.hash_key(key)
        hash2 = APIKeyService.hash_key(key)

        assert hash1 == hash2

    def test_hash_key_different_for_different_keys(self):
        """Different keys should produce different hashes."""
        hash1 = APIKeyService.hash_key("2api_key1")
        hash2 = APIKeyService.hash_key("2api_key2")

        assert hash1 != hash2


# ============================================================
# Data Model Tests
# ============================================================

class TestDataModels:
    """Test data model functionality."""

    def test_tenant_model_defaults(self):
        """Tenant model should have sensible defaults."""
        tenant = Tenant(
            id=UUID("12345678-1234-1234-1234-123456789012"),
            name="Test Tenant",
            email="test@example.com",
        )

        assert tenant.plan == "free"
        assert tenant.settings == {}
        assert tenant.is_active is True

    def test_api_key_has_permission_wildcard(self):
        """Wildcard permission should match anything."""
        key = APIKey(
            id=UUID("12345678-1234-1234-1234-123456789012"),
            tenant_id=UUID("12345678-1234-1234-1234-123456789012"),
            key_hash="abc123",
            key_prefix="2api_abc",
            permissions=["*"],
        )

        assert key.has_permission("anything") is True
        assert key.has_permission("admin") is True

    def test_api_key_has_permission_specific(self):
        """Specific permissions should match exactly."""
        key = APIKey(
            id=UUID("12345678-1234-1234-1234-123456789012"),
            tenant_id=UUID("12345678-1234-1234-1234-123456789012"),
            key_hash="abc123",
            key_prefix="2api_abc",
            permissions=["chat", "embedding"],
        )

        assert key.has_permission("chat") is True
        assert key.has_permission("embedding") is True
        assert key.has_permission("admin") is False

    def test_api_key_is_expired_no_expiry(self):
        """Key with no expiry should not be expired."""
        key = APIKey(
            id=UUID("12345678-1234-1234-1234-123456789012"),
            tenant_id=UUID("12345678-1234-1234-1234-123456789012"),
            key_hash="abc123",
            key_prefix="2api_abc",
            expires_at=None,
        )

        assert key.is_expired() is False

    def test_auth_context_has_permission(self):
        """AuthContext should delegate permission check."""
        tenant = Tenant(
            id=UUID("12345678-1234-1234-1234-123456789012"),
            name="Test",
            email="test@test.com",
        )
        api_key = APIKey(
            id=UUID("12345678-1234-1234-1234-123456789012"),
            tenant_id=tenant.id,
            key_hash="abc",
            key_prefix="2api_",
            permissions=["chat"],
        )
        ctx = AuthContext(
            tenant_id=tenant.id,
            api_key_id=api_key.id,
            tenant=tenant,
            api_key=api_key,
            permissions=["chat"],
            rate_limit_per_minute=60,
        )

        assert ctx.has_permission("chat") is True
        assert ctx.has_permission("admin") is False

    def test_usage_record_defaults(self):
        """UsageRecord should have sensible defaults."""
        record = UsageRecord()

        assert record.operation == "chat"
        assert record.input_tokens == 0
        assert record.output_tokens == 0
        assert record.total_tokens == 0
        assert record.cost_usd == 0.0
        assert record.status == "success"
        assert record.fallback_used is False


# ============================================================
# Request Context Tests
# ============================================================

class TestRequestContext:
    """Test request context for tracing."""

    def test_request_context_has_ids(self):
        """RequestContext should generate unique IDs."""
        ctx = RequestContext()

        assert ctx.request_id.startswith("req_")
        assert ctx.trace_id.startswith("trace_")
        assert len(ctx.request_id) == 28  # req_ + 24 hex chars
        assert len(ctx.trace_id) == 30  # trace_ + 24 hex chars

    def test_request_context_unique(self):
        """Each RequestContext should have unique IDs."""
        contexts = [RequestContext() for _ in range(10)]
        request_ids = {ctx.request_id for ctx in contexts}

        assert len(request_ids) == 10


# ============================================================
# Run Tests
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
