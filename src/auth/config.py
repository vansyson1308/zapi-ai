"""
2api.ai - Auth Configuration

Handles local vs production mode for authentication and startup safety checks.
"""

import os
from enum import Enum
from typing import List


class AuthMode(str, Enum):
    """Authentication mode."""

    LOCAL = "local"  # Relaxed auth for development (format-check only)
    PROD = "prod"    # Full DB-backed auth
    TEST = "test"    # Deterministic test mode (in-memory quotas)


def get_auth_mode() -> AuthMode:
    """
    Get the current authentication mode.

    MODE must be one of: local, prod/production, test.

    Default: prod (fail-closed default for safer deployments).
    """
    mode = os.getenv("MODE", "prod").lower().strip()
    if mode in {"prod", "production"}:
        return AuthMode.PROD
    if mode == "local":
        return AuthMode.LOCAL
    if mode == "test":
        return AuthMode.TEST
    raise ValueError("Invalid MODE. Use one of: local, prod, production, test")


def is_local_mode() -> bool:
    """Check if running in local mode."""
    return get_auth_mode() == AuthMode.LOCAL


def is_prod_mode() -> bool:
    """Check if running in production mode."""
    return get_auth_mode() == AuthMode.PROD


# Local mode configuration
LOCAL_DEFAULT_TENANT_ID = "00000000-0000-0000-0000-000000000001"
LOCAL_DEFAULT_API_KEY_ID = "00000000-0000-0000-0000-000000000002"


def get_local_env_keys() -> dict:
    """
    Get provider API keys from environment for local mode.

    In local mode, we use environment variables directly
    instead of per-tenant encrypted credentials.
    """
    return {
        "openai": os.getenv("OPENAI_API_KEY"),
        "anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "google": os.getenv("GOOGLE_API_KEY"),
    }


def is_test_mode() -> bool:
    """Check if running in test mode."""
    return get_auth_mode() == AuthMode.TEST


def get_cors_allowed_origins() -> List[str]:
    """Parse CORS_ALLOW_ORIGINS from environment."""
    raw = os.getenv("CORS_ALLOW_ORIGINS", "")
    if not raw.strip():
        return []
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def validate_security_config() -> None:
    """Fail closed for unsafe production startup configuration."""
    mode = get_auth_mode()
    if mode in {AuthMode.LOCAL, AuthMode.TEST}:
        return

    # Production guardrails
    if os.getenv("USE_STUB_ADAPTERS", "false").lower() in {"1", "true", "yes"}:
        raise RuntimeError("USE_STUB_ADAPTERS is not allowed in production mode")

    if not os.getenv("DATABASE_URL"):
        raise RuntimeError("DATABASE_URL is required in production mode")

    if not os.getenv("FERNET_KEY"):
        raise RuntimeError("FERNET_KEY is required in production mode")

    origins = get_cors_allowed_origins()
    if not origins:
        raise RuntimeError("CORS_ALLOW_ORIGINS must be set in production mode")
    if "*" in origins:
        raise RuntimeError("CORS_ALLOW_ORIGINS cannot include '*' in production mode")
