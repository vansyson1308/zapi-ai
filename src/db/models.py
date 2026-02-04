"""
2api.ai - Database Models

Dataclass models for database entities.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID


@dataclass
class Tenant:
    """Tenant (customer/organization) model."""

    id: UUID
    name: str
    email: str
    plan: str = "free"  # free, starter, pro, enterprise
    settings: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @classmethod
    def from_record(cls, record) -> "Tenant":
        """Create Tenant from database record."""
        return cls(
            id=record["id"],
            name=record["name"],
            email=record["email"],
            plan=record["plan"],
            settings=record["settings"] or {},
            is_active=record["is_active"],
            created_at=record["created_at"],
            updated_at=record["updated_at"],
        )


@dataclass
class APIKey:
    """API Key model."""

    id: UUID
    tenant_id: UUID
    key_hash: str  # SHA-256 hash
    key_prefix: str  # "2api_" + first 7 chars
    name: str = "Default Key"
    permissions: List[str] = field(default_factory=lambda: ["*"])
    rate_limit_per_minute: int = 60
    is_active: bool = True
    last_used_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    created_at: Optional[datetime] = None

    @classmethod
    def from_record(cls, record) -> "APIKey":
        """Create APIKey from database record."""
        permissions = record["permissions"]
        if isinstance(permissions, str):
            import json
            permissions = json.loads(permissions)

        return cls(
            id=record["id"],
            tenant_id=record["tenant_id"],
            key_hash=record["key_hash"],
            key_prefix=record["key_prefix"],
            name=record["name"],
            permissions=permissions or ["*"],
            rate_limit_per_minute=record["rate_limit_per_minute"],
            is_active=record["is_active"],
            last_used_at=record["last_used_at"],
            expires_at=record["expires_at"],
            created_at=record["created_at"],
        )

    def is_expired(self) -> bool:
        """Check if key has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(self.expires_at.tzinfo) > self.expires_at

    def has_permission(self, permission: str) -> bool:
        """Check if key has specific permission."""
        if "*" in self.permissions:
            return True
        return permission in self.permissions


@dataclass
class ProviderConfig:
    """Provider configuration per tenant."""

    id: UUID
    tenant_id: UUID
    provider: str  # openai, anthropic, google
    api_key_encrypted: Optional[str] = None
    settings: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @classmethod
    def from_record(cls, record) -> "ProviderConfig":
        """Create ProviderConfig from database record."""
        return cls(
            id=record["id"],
            tenant_id=record["tenant_id"],
            provider=record["provider"],
            api_key_encrypted=record["api_key_encrypted"],
            settings=record["settings"] or {},
            is_active=record["is_active"],
            created_at=record["created_at"],
            updated_at=record["updated_at"],
        )


@dataclass
class UsageRecord:
    """Usage record for billing and analytics."""

    id: Optional[UUID] = None
    tenant_id: Optional[UUID] = None
    api_key_id: Optional[UUID] = None
    request_id: str = ""
    provider: str = ""
    model: str = ""
    operation: str = "chat"  # chat, embedding, image
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: Optional[int] = None
    status: str = "success"  # success, error, timeout, rate_limited
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    routing_strategy: Optional[str] = None
    fallback_used: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None

    @classmethod
    def from_record(cls, record) -> "UsageRecord":
        """Create UsageRecord from database record."""
        return cls(
            id=record["id"],
            tenant_id=record["tenant_id"],
            api_key_id=record["api_key_id"],
            request_id=record["request_id"],
            provider=record["provider"],
            model=record["model"],
            operation=record["operation"],
            input_tokens=record["input_tokens"],
            output_tokens=record["output_tokens"],
            total_tokens=record["total_tokens"],
            cost_usd=float(record["cost_usd"]),
            latency_ms=record["latency_ms"],
            status=record["status"],
            error_code=record["error_code"],
            error_message=record["error_message"],
            routing_strategy=record["routing_strategy"],
            fallback_used=record["fallback_used"],
            metadata=record["metadata"] or {},
            created_at=record["created_at"],
        )


@dataclass
class RoutingPolicy:
    """Routing policy for a tenant."""

    id: UUID
    tenant_id: UUID
    name: str
    is_default: bool = False
    strategy: str = "cost"  # cost, latency, quality, custom
    fallback_chain: List[str] = field(default_factory=list)
    max_latency_ms: Optional[int] = None
    max_cost_per_request: Optional[float] = None
    rules: List[Dict[str, Any]] = field(default_factory=list)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @classmethod
    def from_record(cls, record) -> "RoutingPolicy":
        """Create RoutingPolicy from database record."""
        rules = record["rules"]
        if isinstance(rules, str):
            import json
            rules = json.loads(rules)

        return cls(
            id=record["id"],
            tenant_id=record["tenant_id"],
            name=record["name"],
            is_default=record["is_default"],
            strategy=record["strategy"],
            fallback_chain=record["fallback_chain"] or [],
            max_latency_ms=record["max_latency_ms"],
            max_cost_per_request=float(record["max_cost_per_request"]) if record["max_cost_per_request"] else None,
            rules=rules or [],
            created_at=record["created_at"],
            updated_at=record["updated_at"],
        )


@dataclass
class AuthContext:
    """
    Authentication context for a request.

    Populated by auth middleware after validating API key.
    """

    tenant_id: UUID
    api_key_id: UUID
    tenant: Tenant
    api_key: APIKey
    permissions: List[str]
    rate_limit_per_minute: int

    # Request tracing
    request_id: str = ""
    trace_id: str = ""

    def has_permission(self, permission: str) -> bool:
        """Check if request has specific permission."""
        if "*" in self.permissions:
            return True
        return permission in self.permissions
