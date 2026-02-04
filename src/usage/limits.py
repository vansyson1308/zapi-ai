"""
2api.ai - Usage Limits & Quotas

Manages usage limits and quotas for tenants.

Features:
- Rate limiting (requests per minute)
- Token quotas (daily, monthly)
- Cost limits
- Soft and hard limits
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
from enum import Enum


class LimitType(str, Enum):
    """Types of usage limits."""
    RATE = "rate"           # Requests per time window
    TOKENS = "tokens"       # Token count
    COST = "cost"           # Cost in USD
    REQUESTS = "requests"   # Total request count


class LimitPeriod(str, Enum):
    """Time periods for limits."""
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    MONTH = "month"


class LimitAction(str, Enum):
    """Action to take when limit is exceeded."""
    REJECT = "reject"       # Reject the request
    THROTTLE = "throttle"   # Slow down (add delay)
    WARN = "warn"           # Allow but warn
    LOG = "log"             # Just log, allow request


@dataclass
class UsageLimit:
    """
    Defines a single usage limit.

    Can be a rate limit, token quota, or cost limit.
    """
    limit_type: LimitType
    period: LimitPeriod
    limit_value: float
    action: LimitAction = LimitAction.REJECT
    soft_limit_percent: float = 80.0  # Warn at this percentage
    name: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = f"{self.limit_type.value}_{self.period.value}"

    def get_period_seconds(self) -> int:
        """Get period duration in seconds."""
        periods = {
            LimitPeriod.MINUTE: 60,
            LimitPeriod.HOUR: 3600,
            LimitPeriod.DAY: 86400,
            LimitPeriod.MONTH: 2592000  # 30 days
        }
        return periods.get(self.period, 60)


@dataclass
class LimitStatus:
    """
    Current status of a limit.

    Shows current usage against the limit.
    """
    limit: UsageLimit
    current_value: float
    remaining: float
    is_exceeded: bool = False
    is_soft_exceeded: bool = False
    reset_at: Optional[datetime] = None

    @property
    def usage_percent(self) -> float:
        """Get usage as percentage of limit."""
        if self.limit.limit_value == 0:
            return 0.0
        return (self.current_value / self.limit.limit_value) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.limit.name,
            "type": self.limit.limit_type.value,
            "period": self.limit.period.value,
            "limit": self.limit.limit_value,
            "current": self.current_value,
            "remaining": self.remaining,
            "usage_percent": round(self.usage_percent, 2),
            "is_exceeded": self.is_exceeded,
            "is_soft_exceeded": self.is_soft_exceeded,
            "reset_at": self.reset_at.isoformat() if self.reset_at else None,
            "action": self.limit.action.value
        }


@dataclass
class QuotaConfig:
    """
    Complete quota configuration for a tenant.

    Groups multiple limits together.
    """
    tenant_id: str
    limits: List[UsageLimit] = field(default_factory=list)
    enabled: bool = True

    # Plan-based quotas
    plan: str = "free"  # free, starter, pro, enterprise

    @classmethod
    def for_plan(cls, tenant_id: str, plan: str) -> "QuotaConfig":
        """Create quota config based on plan."""
        plans = {
            "free": [
                UsageLimit(LimitType.RATE, LimitPeriod.MINUTE, 10),
                UsageLimit(LimitType.TOKENS, LimitPeriod.DAY, 100_000),
                UsageLimit(LimitType.COST, LimitPeriod.MONTH, 5.0),
            ],
            "starter": [
                UsageLimit(LimitType.RATE, LimitPeriod.MINUTE, 60),
                UsageLimit(LimitType.TOKENS, LimitPeriod.DAY, 1_000_000),
                UsageLimit(LimitType.COST, LimitPeriod.MONTH, 100.0),
            ],
            "pro": [
                UsageLimit(LimitType.RATE, LimitPeriod.MINUTE, 300),
                UsageLimit(LimitType.TOKENS, LimitPeriod.DAY, 10_000_000),
                UsageLimit(LimitType.COST, LimitPeriod.MONTH, 1000.0),
            ],
            "enterprise": [
                UsageLimit(LimitType.RATE, LimitPeriod.MINUTE, 1000),
                UsageLimit(LimitType.TOKENS, LimitPeriod.DAY, 100_000_000),
                UsageLimit(LimitType.COST, LimitPeriod.MONTH, 10000.0),
            ]
        }

        limits = plans.get(plan, plans["free"])
        return cls(tenant_id=tenant_id, limits=limits, plan=plan)


@dataclass
class LimitCheckResult:
    """
    Result of checking limits for a request.
    """
    allowed: bool
    exceeded_limits: List[LimitStatus] = field(default_factory=list)
    warnings: List[LimitStatus] = field(default_factory=list)
    throttle_ms: int = 0  # Delay to apply
    retry_after: Optional[int] = None  # Seconds until limit resets

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "allowed": self.allowed,
            "exceeded_limits": [l.to_dict() for l in self.exceeded_limits],
            "warnings": [w.to_dict() for w in self.warnings],
            "throttle_ms": self.throttle_ms,
            "retry_after": self.retry_after
        }


class SlidingWindowCounter:
    """
    Sliding window rate counter.

    Uses a sliding window algorithm for accurate rate limiting.
    """

    def __init__(self, window_seconds: int):
        """
        Initialize counter.

        Args:
            window_seconds: Size of sliding window
        """
        self.window_seconds = window_seconds
        self._buckets: Dict[int, float] = {}  # timestamp -> count
        self._lock = asyncio.Lock()

    async def increment(self, amount: float = 1.0) -> float:
        """
        Increment counter and return current count.

        Args:
            amount: Amount to add

        Returns:
            Current count in window
        """
        async with self._lock:
            now = int(time.time())
            self._buckets[now] = self._buckets.get(now, 0) + amount
            self._cleanup(now)
            return self._get_count(now)

    async def get_count(self) -> float:
        """Get current count in window."""
        async with self._lock:
            now = int(time.time())
            self._cleanup(now)
            return self._get_count(now)

    def _cleanup(self, now: int):
        """Remove old buckets."""
        cutoff = now - self.window_seconds
        self._buckets = {
            ts: count for ts, count in self._buckets.items()
            if ts > cutoff
        }

    def _get_count(self, now: int) -> float:
        """Get weighted count for sliding window."""
        cutoff = now - self.window_seconds
        total = 0.0

        for ts, count in self._buckets.items():
            if ts > cutoff:
                # Weight by position in window
                age = now - ts
                weight = 1 - (age / self.window_seconds)
                total += count * weight

        return total


class UsageLimiter:
    """
    Manages usage limits for tenants.

    Provides:
    - Rate limiting with sliding windows
    - Token quota tracking
    - Cost limit tracking
    """

    def __init__(self):
        """Initialize limiter."""
        # Quota configs by tenant
        self._quotas: Dict[str, QuotaConfig] = {}

        # Rate counters: tenant_id -> limit_name -> counter
        self._rate_counters: Dict[str, Dict[str, SlidingWindowCounter]] = defaultdict(dict)

        # Usage counters: tenant_id -> limit_name -> {value, period_start}
        self._usage_counters: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)

    def set_quota(self, config: QuotaConfig):
        """
        Set quota configuration for a tenant.

        Args:
            config: Quota configuration
        """
        self._quotas[config.tenant_id] = config

        # Initialize counters for rate limits
        for limit in config.limits:
            if limit.limit_type == LimitType.RATE:
                self._rate_counters[config.tenant_id][limit.name] = SlidingWindowCounter(
                    limit.get_period_seconds()
                )

    def get_quota(self, tenant_id: str) -> Optional[QuotaConfig]:
        """Get quota configuration for a tenant."""
        return self._quotas.get(tenant_id)

    async def check_limits(
        self,
        tenant_id: str,
        tokens: int = 0,
        cost: float = 0.0
    ) -> LimitCheckResult:
        """
        Check if a request is allowed under current limits.

        Args:
            tenant_id: Tenant making request
            tokens: Estimated tokens for request
            cost: Estimated cost for request

        Returns:
            LimitCheckResult indicating if allowed
        """
        config = self._quotas.get(tenant_id)
        if not config or not config.enabled:
            return LimitCheckResult(allowed=True)

        result = LimitCheckResult(allowed=True)

        for limit in config.limits:
            status = await self._check_single_limit(
                tenant_id, limit, tokens, cost
            )

            if status.is_exceeded:
                result.exceeded_limits.append(status)

                if limit.action == LimitAction.REJECT:
                    result.allowed = False
                    if status.reset_at:
                        retry_after = int((status.reset_at - datetime.utcnow()).total_seconds())
                        result.retry_after = max(retry_after, 1)
                elif limit.action == LimitAction.THROTTLE:
                    # Calculate throttle delay
                    overage = status.current_value - limit.limit_value
                    delay = int(overage * 100)  # 100ms per unit over
                    result.throttle_ms = max(result.throttle_ms, delay)

            elif status.is_soft_exceeded:
                result.warnings.append(status)

        return result

    async def record_usage(
        self,
        tenant_id: str,
        tokens: int = 0,
        cost: float = 0.0
    ):
        """
        Record usage against limits.

        Call this after a request completes.

        Args:
            tenant_id: Tenant ID
            tokens: Actual tokens used
            cost: Actual cost
        """
        config = self._quotas.get(tenant_id)
        if not config:
            return

        now = datetime.utcnow()

        for limit in config.limits:
            # Rate limits are already recorded in check_limits
            if limit.limit_type == LimitType.RATE:
                continue

            # Get or create usage counter
            counter_key = limit.name
            if counter_key not in self._usage_counters[tenant_id]:
                self._usage_counters[tenant_id][counter_key] = {
                    "value": 0,
                    "period_start": self._get_period_start(now, limit.period)
                }

            counter = self._usage_counters[tenant_id][counter_key]

            # Reset if period expired
            period_start = self._get_period_start(now, limit.period)
            if counter["period_start"] < period_start:
                counter["value"] = 0
                counter["period_start"] = period_start

            # Add usage
            if limit.limit_type == LimitType.TOKENS:
                counter["value"] += tokens
            elif limit.limit_type == LimitType.COST:
                counter["value"] += cost
            elif limit.limit_type == LimitType.REQUESTS:
                counter["value"] += 1

    async def get_limit_status(
        self,
        tenant_id: str
    ) -> List[LimitStatus]:
        """
        Get current status of all limits for a tenant.

        Args:
            tenant_id: Tenant ID

        Returns:
            List of limit statuses
        """
        config = self._quotas.get(tenant_id)
        if not config:
            return []

        statuses = []
        for limit in config.limits:
            status = await self._check_single_limit(tenant_id, limit, 0, 0)
            statuses.append(status)

        return statuses

    async def _check_single_limit(
        self,
        tenant_id: str,
        limit: UsageLimit,
        tokens: int,
        cost: float
    ) -> LimitStatus:
        """Check a single limit."""
        now = datetime.utcnow()

        if limit.limit_type == LimitType.RATE:
            # Use sliding window counter
            counter = self._rate_counters[tenant_id].get(limit.name)
            if counter:
                current = await counter.increment()
            else:
                current = 0
        else:
            # Use usage counter
            counter = self._usage_counters[tenant_id].get(limit.name, {})

            # Check if period expired
            period_start = self._get_period_start(now, limit.period)
            if counter.get("period_start", period_start) < period_start:
                current = 0
            else:
                current = counter.get("value", 0)

            # Add pending usage
            if limit.limit_type == LimitType.TOKENS:
                current += tokens
            elif limit.limit_type == LimitType.COST:
                current += cost

        remaining = max(0, limit.limit_value - current)
        is_exceeded = current > limit.limit_value
        is_soft_exceeded = current > (limit.limit_value * limit.soft_limit_percent / 100)

        # Calculate reset time
        reset_at = self._get_period_end(now, limit.period)

        return LimitStatus(
            limit=limit,
            current_value=current,
            remaining=remaining,
            is_exceeded=is_exceeded,
            is_soft_exceeded=is_soft_exceeded,
            reset_at=reset_at
        )

    def _get_period_start(self, dt: datetime, period: LimitPeriod) -> datetime:
        """Get start of current period."""
        if period == LimitPeriod.MINUTE:
            return dt.replace(second=0, microsecond=0)
        elif period == LimitPeriod.HOUR:
            return dt.replace(minute=0, second=0, microsecond=0)
        elif period == LimitPeriod.DAY:
            return dt.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == LimitPeriod.MONTH:
            return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        return dt

    def _get_period_end(self, dt: datetime, period: LimitPeriod) -> datetime:
        """Get end of current period."""
        start = self._get_period_start(dt, period)

        if period == LimitPeriod.MINUTE:
            return start + timedelta(minutes=1)
        elif period == LimitPeriod.HOUR:
            return start + timedelta(hours=1)
        elif period == LimitPeriod.DAY:
            return start + timedelta(days=1)
        elif period == LimitPeriod.MONTH:
            # First day of next month
            if start.month == 12:
                return start.replace(year=start.year + 1, month=1)
            return start.replace(month=start.month + 1)
        return start


# Global limiter instance
_limiter = UsageLimiter()


def get_limiter() -> UsageLimiter:
    """Get global limiter instance."""
    return _limiter


def set_quota(config: QuotaConfig):
    """Set quota for a tenant."""
    _limiter.set_quota(config)


async def check_limits(
    tenant_id: str,
    tokens: int = 0,
    cost: float = 0.0
) -> LimitCheckResult:
    """Check if request is allowed under limits."""
    return await _limiter.check_limits(tenant_id, tokens, cost)


async def record_usage(
    tenant_id: str,
    tokens: int = 0,
    cost: float = 0.0
):
    """Record usage against limits."""
    await _limiter.record_usage(tenant_id, tokens, cost)


async def get_limit_status(tenant_id: str) -> List[LimitStatus]:
    """Get current limit status for a tenant."""
    return await _limiter.get_limit_status(tenant_id)
