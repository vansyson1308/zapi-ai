"""
2api.ai - Authentication Module

Provides API key validation and tenant context for requests.
"""

from .middleware import (
    get_auth_context,
    get_auth_context_optional,
    require_permission,
)
from .config import AuthMode, get_auth_mode

__all__ = [
    "get_auth_context",
    "get_auth_context_optional",
    "require_permission",
    "AuthMode",
    "get_auth_mode",
]
