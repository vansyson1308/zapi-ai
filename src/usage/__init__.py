"""
2api.ai - Usage Tracking Module

Complete usage tracking and billing support with:
- Comprehensive pricing catalog for all providers
- Token counting and estimation
- Real-time usage tracking
- Cost calculation
- Usage aggregation and reporting
- Rate limiting and quotas

Key Components:
- Pricing: Model pricing catalog
- Estimator: Token count estimation
- Tracker: Request-level usage tracking
- Aggregator: Usage analytics and reporting
- Limits: Rate limiting and quotas
"""

from .pricing import (
    # Classes
    ModelPrice,
    PricingCatalog,
    PricingTier,
    # Functions
    get_model_price,
    calculate_cost,
    list_models,
    compare_costs,
)
from .estimator import (
    # Classes
    TokenEstimate,
    TokenEstimator,
    # Functions
    estimate_tokens,
    estimate_message_tokens,
    estimate_request_cost,
)
from .tracker import (
    # Classes
    UsageStatus,
    OperationType,
    UsageRecord,
    RequestTracker,
    UsageTracker,
    # Functions
    get_usage_tracker,
    set_usage_tracker,
    start_tracking,
    complete_tracking,
)
from .aggregator import (
    # Classes
    AggregationPeriod,
    UsageAggregate,
    CostBreakdown,
    UsageAggregator,
    # Functions
    aggregate_usage,
    get_cost_breakdown,
    get_top_models,
)
from .limits import (
    # Classes
    LimitType,
    LimitPeriod,
    LimitAction,
    UsageLimit,
    LimitStatus,
    QuotaConfig,
    LimitCheckResult,
    SlidingWindowCounter,
    UsageLimiter,
    # Functions
    get_limiter,
    set_quota,
    check_limits,
    record_usage,
    get_limit_status,
)

__all__ = [
    # Pricing
    "ModelPrice",
    "PricingCatalog",
    "PricingTier",
    "get_model_price",
    "calculate_cost",
    "list_models",
    "compare_costs",
    # Estimator
    "TokenEstimate",
    "TokenEstimator",
    "estimate_tokens",
    "estimate_message_tokens",
    "estimate_request_cost",
    # Tracker
    "UsageStatus",
    "OperationType",
    "UsageRecord",
    "RequestTracker",
    "UsageTracker",
    "get_usage_tracker",
    "set_usage_tracker",
    "start_tracking",
    "complete_tracking",
    # Aggregator
    "AggregationPeriod",
    "UsageAggregate",
    "CostBreakdown",
    "UsageAggregator",
    "aggregate_usage",
    "get_cost_breakdown",
    "get_top_models",
    # Limits
    "LimitType",
    "LimitPeriod",
    "LimitAction",
    "UsageLimit",
    "LimitStatus",
    "QuotaConfig",
    "LimitCheckResult",
    "SlidingWindowCounter",
    "UsageLimiter",
    "get_limiter",
    "set_quota",
    "check_limits",
    "record_usage",
    "get_limit_status",
]
