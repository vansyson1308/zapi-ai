"""
2api.ai - Routing Module

Intelligent request routing with:
- Circuit breaker pattern
- Multiple routing strategies (cost, latency, quality)
- Fallback chains with semantic drift protection
- Real-time health tracking
"""

from .router import Router, ProviderStats, RoutingResult
from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerRegistry,
    CircuitState,
)
from .strategies import (
    BaseStrategy,
    CostStrategy,
    LatencyStrategy,
    QualityStrategy,
    BalancedStrategy,
    ProviderMetrics,
    RoutingConstraints,
    ScoredCandidate,
    get_strategy,
)
from .fallback import (
    FallbackChain,
    FallbackChainConfig,
    FallbackCoordinator,
    FallbackPhase,
    FallbackResult,
    RequestPhaseTracker,
    create_fallback_chain,
    check_fallback_eligibility,
)
from .health import (
    HealthTracker,
    HealthRegistry,
    HealthScore,
    HealthSnapshot,
    LatencyStats,
)

__all__ = [
    # Router
    "Router",
    "ProviderStats",
    "RoutingResult",
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerRegistry",
    "CircuitState",
    # Strategies
    "BaseStrategy",
    "CostStrategy",
    "LatencyStrategy",
    "QualityStrategy",
    "BalancedStrategy",
    "ProviderMetrics",
    "RoutingConstraints",
    "ScoredCandidate",
    "get_strategy",
    # Fallback
    "FallbackChain",
    "FallbackChainConfig",
    "FallbackCoordinator",
    "FallbackPhase",
    "FallbackResult",
    "RequestPhaseTracker",
    "create_fallback_chain",
    "check_fallback_eligibility",
    # Health
    "HealthTracker",
    "HealthRegistry",
    "HealthScore",
    "HealthSnapshot",
    "LatencyStats",
]
