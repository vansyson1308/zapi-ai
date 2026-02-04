"""
2api.ai - Auth Configuration

Handles local vs production mode for authentication.
"""

import os
from enum import Enum
from typing import Optional


class AuthMode(str, Enum):
    """Authentication mode."""

    LOCAL = "local"  # Relaxed auth for development (format-check only)
    PROD = "prod"    # Full DB-backed auth


def get_auth_mode() -> AuthMode:
    """
    Get the current authentication mode.

    Set via MODE environment variable:
    - MODE=local: Development mode, relaxed auth
    - MODE=prod: Production mode, full DB auth

    Default: local (for vibe coding friendliness)
    """
    mode = os.getenv("MODE", "local").lower()
    if mode == "prod" or mode == "production":
        return AuthMode.PROD
    return AuthMode.LOCAL


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
