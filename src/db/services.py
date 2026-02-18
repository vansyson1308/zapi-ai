"""
2api.ai - Database Services

Business logic for database operations.
"""

import hashlib
import os
import secrets
import json
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

from ..security import KeyEncryptor, default_encryptor_from_env, OpenSSLEncryptor
from uuid import UUID

from .connection import DatabasePool
from .models import (
    Tenant,
    APIKey,
    ProviderConfig,
    UsageRecord,
    RoutingPolicy,
    AuthContext,
)


class APIKeyService:
    """
    Service for API key operations.

    Handles:
    - Key generation with secure random bytes
    - Key hashing (SHA-256)
    - Key validation and lookup
    """

    KEY_PREFIX = "2api_"
    KEY_LENGTH = 32  # Random bytes (will be 64 hex chars)

    def __init__(self, db: DatabasePool):
        self.db = db

    @staticmethod
    def generate_key() -> Tuple[str, str, str]:
        """
        Generate a new API key.

        Returns:
            Tuple of (full_key, key_hash, key_prefix)
            - full_key: The complete key to give to user (only returned once!)
            - key_hash: SHA-256 hash to store in DB
            - key_prefix: First part for identification
        """
        # Generate secure random bytes
        random_bytes = secrets.token_hex(APIKeyService.KEY_LENGTH)
        full_key = f"{APIKeyService.KEY_PREFIX}{random_bytes}"

        # Hash for storage
        key_hash = hashlib.sha256(full_key.encode()).hexdigest()

        # Prefix for identification (2api_ + first 7 chars)
        key_prefix = full_key[:12]

        return full_key, key_hash, key_prefix

    @staticmethod
    def hash_key(key: str) -> str:
        """Hash an API key for lookup."""
        return hashlib.sha256(key.encode()).hexdigest()

    async def create_key(
        self,
        tenant_id: UUID,
        name: str = "Default Key",
        permissions: Optional[List[str]] = None,
        rate_limit_per_minute: int = 60,
        expires_in_days: Optional[int] = None,
    ) -> Tuple[str, APIKey]:
        """
        Create a new API key for a tenant.

        Args:
            tenant_id: UUID of the tenant
            name: Human-readable name for the key
            permissions: List of permissions (default: ["*"])
            rate_limit_per_minute: Rate limit for this key
            expires_in_days: Optional expiration in days

        Returns:
            Tuple of (plaintext_key, api_key_record)
            IMPORTANT: plaintext_key is only returned once!
        """
        full_key, key_hash, key_prefix = self.generate_key()

        if permissions is None:
            permissions = ["*"]

        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)

        query = """
            INSERT INTO api_keys (
                tenant_id, key_hash, key_prefix, name,
                permissions, rate_limit_per_minute, expires_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING *
        """

        record = await self.db.fetchrow(
            query,
            tenant_id,
            key_hash,
            key_prefix,
            name,
            json.dumps(permissions),
            rate_limit_per_minute,
            expires_at,
        )

        return full_key, APIKey.from_record(record)

    async def validate_key(self, key: str) -> Optional[APIKey]:
        """
        Validate an API key and return the key record if valid.

        Args:
            key: The full API key

        Returns:
            APIKey if valid, None if invalid/expired/inactive
        """
        if not key.startswith(self.KEY_PREFIX):
            return None

        key_hash = self.hash_key(key)

        query = """
            SELECT * FROM api_keys
            WHERE key_hash = $1 AND is_active = TRUE
        """

        record = await self.db.fetchrow(query, key_hash)
        if record is None:
            return None

        api_key = APIKey.from_record(record)

        # Check expiration
        if api_key.is_expired():
            return None

        return api_key

    async def update_last_used(self, key_id: UUID) -> None:
        """Update the last_used_at timestamp for a key."""
        query = """
            UPDATE api_keys SET last_used_at = NOW()
            WHERE id = $1
        """
        await self.db.execute(query, key_id)

    async def list_keys(self, tenant_id: UUID) -> List[APIKey]:
        """List all API keys for a tenant (without hashes)."""
        query = """
            SELECT * FROM api_keys
            WHERE tenant_id = $1
            ORDER BY created_at DESC
        """
        records = await self.db.fetch(query, tenant_id)
        return [APIKey.from_record(r) for r in records]

    async def revoke_key(self, key_id: UUID) -> bool:
        """Revoke (deactivate) an API key."""
        query = """
            UPDATE api_keys SET is_active = FALSE
            WHERE id = $1
            RETURNING id
        """
        result = await self.db.fetchrow(query, key_id)
        return result is not None

    async def delete_key(self, key_id: UUID) -> bool:
        """Permanently delete an API key."""
        query = "DELETE FROM api_keys WHERE id = $1 RETURNING id"
        result = await self.db.fetchrow(query, key_id)
        return result is not None


class TenantService:
    """
    Service for tenant operations.

    Handles:
    - Tenant CRUD
    - Tenant lookup by API key
    """

    def __init__(self, db: DatabasePool):
        self.db = db

    async def create_tenant(
        self,
        name: str,
        email: str,
        plan: str = "free",
        settings: Optional[dict] = None,
    ) -> Tenant:
        """Create a new tenant."""
        query = """
            INSERT INTO tenants (name, email, plan, settings)
            VALUES ($1, $2, $3, $4)
            RETURNING *
        """
        record = await self.db.fetchrow(
            query,
            name,
            email,
            plan,
            json.dumps(settings or {}),
        )
        return Tenant.from_record(record)

    async def get_tenant(self, tenant_id: UUID) -> Optional[Tenant]:
        """Get tenant by ID."""
        query = "SELECT * FROM tenants WHERE id = $1 AND is_active = TRUE"
        record = await self.db.fetchrow(query, tenant_id)
        if record is None:
            return None
        return Tenant.from_record(record)

    async def get_tenant_by_email(self, email: str) -> Optional[Tenant]:
        """Get tenant by email."""
        query = "SELECT * FROM tenants WHERE email = $1 AND is_active = TRUE"
        record = await self.db.fetchrow(query, email)
        if record is None:
            return None
        return Tenant.from_record(record)

    async def update_tenant(
        self,
        tenant_id: UUID,
        name: Optional[str] = None,
        plan: Optional[str] = None,
        settings: Optional[dict] = None,
    ) -> Optional[Tenant]:
        """Update tenant fields."""
        updates = []
        values = []
        param_num = 1

        if name is not None:
            updates.append(f"name = ${param_num}")
            values.append(name)
            param_num += 1

        if plan is not None:
            updates.append(f"plan = ${param_num}")
            values.append(plan)
            param_num += 1

        if settings is not None:
            updates.append(f"settings = ${param_num}")
            values.append(json.dumps(settings))
            param_num += 1

        if not updates:
            return await self.get_tenant(tenant_id)

        values.append(tenant_id)
        query = f"""
            UPDATE tenants
            SET {', '.join(updates)}
            WHERE id = ${param_num}
            RETURNING *
        """

        record = await self.db.fetchrow(query, *values)
        if record is None:
            return None
        return Tenant.from_record(record)

    async def deactivate_tenant(self, tenant_id: UUID) -> bool:
        """Deactivate a tenant (soft delete)."""
        query = """
            UPDATE tenants SET is_active = FALSE
            WHERE id = $1
            RETURNING id
        """
        result = await self.db.fetchrow(query, tenant_id)
        return result is not None


class ProviderConfigService:
    """
    Service for provider configuration operations.

    Handles:
    - Provider API key storage (encrypted)
    - Per-tenant provider settings
    """

    def __init__(
        self,
        db: DatabasePool,
        encryption_key: Optional[str] = None,
        encryptor: Optional[KeyEncryptor] = None,
    ):
        self.db = db
        # Pluggable encryptor interface; OpenSSL encryptor is the default baseline.
        self._encryptor: KeyEncryptor = encryptor or (
            OpenSSLEncryptor(encryption_key) if encryption_key else default_encryptor_from_env()
        )

    def _encrypt(self, plaintext: str) -> str:
        """Encrypt sensitive provider secrets before storage."""
        return self._encryptor.encrypt(plaintext)

    def _decrypt(self, ciphertext: str) -> str:
        """Decrypt sensitive provider secrets after retrieval."""
        return self._encryptor.decrypt(ciphertext)

    async def set_provider_key(
        self,
        tenant_id: UUID,
        provider: str,
        api_key: str,
        settings: Optional[dict] = None,
    ) -> ProviderConfig:
        """
        Set or update provider API key for a tenant.

        Args:
            tenant_id: UUID of the tenant
            provider: Provider name (openai, anthropic, google)
            api_key: Provider's API key (will be encrypted)
            settings: Optional provider-specific settings
        """
        encrypted_key = self._encrypt(api_key)

        query = """
            INSERT INTO provider_configs (tenant_id, provider, api_key_encrypted, settings)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (tenant_id, provider)
            DO UPDATE SET
                api_key_encrypted = EXCLUDED.api_key_encrypted,
                settings = EXCLUDED.settings,
                updated_at = NOW()
            RETURNING *
        """

        record = await self.db.fetchrow(
            query,
            tenant_id,
            provider,
            encrypted_key,
            json.dumps(settings or {}),
        )
        return ProviderConfig.from_record(record)

    async def get_provider_key(
        self,
        tenant_id: UUID,
        provider: str,
    ) -> Optional[str]:
        """
        Get decrypted provider API key for a tenant.

        Returns None if not configured or inactive.
        """
        query = """
            SELECT api_key_encrypted FROM provider_configs
            WHERE tenant_id = $1 AND provider = $2 AND is_active = TRUE
        """
        result = await self.db.fetchval(query, tenant_id, provider)
        if result is None:
            return None
        return self._decrypt(result)

    async def get_provider_config(
        self,
        tenant_id: UUID,
        provider: str,
    ) -> Optional[ProviderConfig]:
        """Get full provider config (without decrypted key)."""
        query = """
            SELECT * FROM provider_configs
            WHERE tenant_id = $1 AND provider = $2 AND is_active = TRUE
        """
        record = await self.db.fetchrow(query, tenant_id, provider)
        if record is None:
            return None
        return ProviderConfig.from_record(record)

    async def list_provider_configs(self, tenant_id: UUID) -> List[ProviderConfig]:
        """List all provider configs for a tenant."""
        query = """
            SELECT * FROM provider_configs
            WHERE tenant_id = $1
            ORDER BY provider
        """
        records = await self.db.fetch(query, tenant_id)
        return [ProviderConfig.from_record(r) for r in records]

    async def delete_provider_config(self, tenant_id: UUID, provider: str) -> bool:
        """Delete provider configuration."""
        query = """
            DELETE FROM provider_configs
            WHERE tenant_id = $1 AND provider = $2
            RETURNING id
        """
        result = await self.db.fetchrow(query, tenant_id, provider)
        return result is not None


class UsageService:
    """Service for usage tracking."""

    def __init__(self, db: DatabasePool):
        self.db = db

    async def record_usage(self, usage: UsageRecord) -> UsageRecord:
        """Record a usage event."""
        query = """
            INSERT INTO usage_records (
                tenant_id, api_key_id, request_id, provider, model, operation,
                input_tokens, output_tokens, total_tokens, cost_usd,
                latency_ms, status, error_code, error_message,
                routing_strategy, fallback_used, metadata
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
            RETURNING *
        """

        record = await self.db.fetchrow(
            query,
            usage.tenant_id,
            usage.api_key_id,
            usage.request_id,
            usage.provider,
            usage.model,
            usage.operation,
            usage.input_tokens,
            usage.output_tokens,
            usage.total_tokens,
            usage.cost_usd,
            usage.latency_ms,
            usage.status,
            usage.error_code,
            usage.error_message,
            usage.routing_strategy,
            usage.fallback_used,
            json.dumps(usage.metadata),
        )
        return UsageRecord.from_record(record)

    async def get_usage_summary(
        self,
        tenant_id: UUID,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> dict:
        """Get usage summary for a tenant."""
        where_clauses = ["tenant_id = $1"]
        params = [tenant_id]
        param_num = 2

        if start_date:
            where_clauses.append(f"created_at >= ${param_num}")
            params.append(start_date)
            param_num += 1

        if end_date:
            where_clauses.append(f"created_at <= ${param_num}")
            params.append(end_date)
            param_num += 1

        query = f"""
            SELECT
                COUNT(*) as total_requests,
                SUM(input_tokens) as total_input_tokens,
                SUM(output_tokens) as total_output_tokens,
                SUM(total_tokens) as total_tokens,
                SUM(cost_usd) as total_cost,
                AVG(latency_ms)::INTEGER as avg_latency_ms,
                COUNT(*) FILTER (WHERE status = 'error') as error_count
            FROM usage_records
            WHERE {' AND '.join(where_clauses)}
        """

        row = await self.db.fetchrow(query, *params)
        return {
            "total_requests": row["total_requests"] or 0,
            "total_input_tokens": row["total_input_tokens"] or 0,
            "total_output_tokens": row["total_output_tokens"] or 0,
            "total_tokens": row["total_tokens"] or 0,
            "total_cost_usd": float(row["total_cost"] or 0),
            "avg_latency_ms": row["avg_latency_ms"] or 0,
            "error_count": row["error_count"] or 0,
        }


class RoutingPolicyService:
    """Service for routing policy operations."""

    def __init__(self, db: DatabasePool):
        self.db = db

    async def get_default_policy(self, tenant_id: UUID) -> Optional[RoutingPolicy]:
        """Get the default routing policy for a tenant."""
        query = """
            SELECT * FROM routing_policies
            WHERE tenant_id = $1 AND is_default = TRUE
            LIMIT 1
        """
        record = await self.db.fetchrow(query, tenant_id)
        if record is None:
            return None
        return RoutingPolicy.from_record(record)

    async def create_policy(
        self,
        tenant_id: UUID,
        name: str,
        strategy: str = "cost",
        fallback_chain: Optional[List[str]] = None,
        is_default: bool = False,
        max_latency_ms: Optional[int] = None,
        max_cost_per_request: Optional[float] = None,
    ) -> RoutingPolicy:
        """Create a routing policy."""
        # If this is default, unset other defaults first
        if is_default:
            await self.db.execute(
                "UPDATE routing_policies SET is_default = FALSE WHERE tenant_id = $1",
                tenant_id,
            )

        query = """
            INSERT INTO routing_policies (
                tenant_id, name, strategy, fallback_chain,
                is_default, max_latency_ms, max_cost_per_request
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING *
        """

        record = await self.db.fetchrow(
            query,
            tenant_id,
            name,
            strategy,
            fallback_chain or [],
            is_default,
            max_latency_ms,
            max_cost_per_request,
        )
        return RoutingPolicy.from_record(record)

    async def list_policies(self, tenant_id: UUID) -> List[RoutingPolicy]:
        """List all routing policies for a tenant."""
        query = """
            SELECT * FROM routing_policies
            WHERE tenant_id = $1
            ORDER BY is_default DESC, name
        """
        records = await self.db.fetch(query, tenant_id)
        return [RoutingPolicy.from_record(r) for r in records]
