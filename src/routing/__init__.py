"""
2api.ai Routing Module

Intelligent routing between AI providers with support for:
- Fallback handling
- Load balancing
- Cost optimization
- Latency optimization
"""

from .router import Router, ProviderStats, RoutingResult

__all__ = [
    "Router",
    "ProviderStats",
    "RoutingResult",
]
