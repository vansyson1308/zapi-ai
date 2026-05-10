"""AgentOps shared dataclasses.

Kept separate from logic modules so other parts of the codebase can import
the schemas without pulling in Redis/PII/middleware dependencies.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ============================================================
# Budget primitives
# ============================================================


class BudgetScope(str, Enum):
    """The dimension a budget cap applies to."""

    TENANT = "tenant"
    CUSTOMER = "customer"
    FEATURE = "feature"
    AGENT = "agent"
    API_KEY = "api_key"


class BudgetPeriod(str, Enum):
    """Reset cadence for a budget."""

    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"

    def seconds(self) -> int:
        return {
            BudgetPeriod.HOUR: 3600,
            BudgetPeriod.DAY: 86400,
            BudgetPeriod.WEEK: 604800,
            BudgetPeriod.MONTH: 2592000,  # 30d, good enough for our purposes
        }[self]


class HardAction(str, Enum):
    """What to do when a hard limit is breached."""

    BLOCK = "block"  # default: reject pre-request, halt mid-stream
    WARN = "warn"  # let through but emit BUDGET_EXCEEDED event
    LOG = "log"  # quietest: only structured log, no event


@dataclass(frozen=True)
class BudgetCap:
    """A single budget cap rule."""

    scope: BudgetScope
    scope_id: str
    period: BudgetPeriod
    limit_usd: Optional[float] = None
    limit_tokens: Optional[int] = None
    soft_threshold_pct: float = 0.8  # emit warning at 80%
    hard_action: HardAction = HardAction.BLOCK

    def __post_init__(self) -> None:
        if self.limit_usd is None and self.limit_tokens is None:
            raise ValueError("BudgetCap requires at least one of limit_usd or limit_tokens")
        if self.limit_usd is not None and self.limit_usd <= 0:
            raise ValueError("limit_usd must be > 0")
        if self.limit_tokens is not None and self.limit_tokens <= 0:
            raise ValueError("limit_tokens must be > 0")
        if not (0.0 < self.soft_threshold_pct <= 1.0):
            raise ValueError("soft_threshold_pct must be in (0, 1]")
        if not self.scope_id:
            raise ValueError("scope_id must be non-empty")

    @property
    def key(self) -> str:
        """Unique identifier for this cap (used as Redis key suffix)."""
        return f"{self.scope.value}:{self.scope_id}:{self.period.value}"


@dataclass
class BudgetUsageSnapshot:
    """Current usage for one BudgetCap, as of a point in time."""

    cap: BudgetCap
    used_usd: float = 0.0
    used_tokens: int = 0
    period_resets_at: Optional[float] = None  # unix timestamp; None = TTL pending

    @property
    def is_exceeded(self) -> bool:
        if self.cap.limit_usd is not None and self.used_usd >= self.cap.limit_usd:
            return True
        if self.cap.limit_tokens is not None and self.used_tokens >= self.cap.limit_tokens:
            return True
        return False

    @property
    def is_soft_exceeded(self) -> bool:
        if self.cap.limit_usd is not None:
            soft = self.cap.limit_usd * self.cap.soft_threshold_pct
            if self.used_usd >= soft:
                return True
        if self.cap.limit_tokens is not None:
            soft = self.cap.limit_tokens * self.cap.soft_threshold_pct
            if self.used_tokens >= soft:
                return True
        return False

    def usage_pct(self) -> float:
        """0.0–1.0. Picks the higher of cost-pct vs token-pct (most-constrained)."""
        ratios: List[float] = []
        if self.cap.limit_usd is not None and self.cap.limit_usd > 0:
            ratios.append(self.used_usd / self.cap.limit_usd)
        if self.cap.limit_tokens is not None and self.cap.limit_tokens > 0:
            ratios.append(self.used_tokens / self.cap.limit_tokens)
        return max(ratios) if ratios else 0.0


# ============================================================
# Attribution
# ============================================================


@dataclass
class AttributionContext:
    """
    The set of identifiers that let us answer "who ran up this cost".

    Populated from request headers (X-Customer-Id, X-Feature-Id, X-Agent-Id)
    or, in the future, from authenticated session claims.
    """

    tenant_id: Optional[str] = None
    api_key_id: Optional[str] = None
    customer_id: Optional[str] = None
    feature_id: Optional[str] = None
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    extra: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Optional[str]]:
        d: Dict[str, Any] = {
            "tenant_id": self.tenant_id,
            "api_key_id": self.api_key_id,
            "customer_id": self.customer_id,
            "feature_id": self.feature_id,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
        }
        if self.extra:
            d["extra"] = dict(self.extra)
        return d

    def is_empty(self) -> bool:
        return not any(
            [
                self.tenant_id,
                self.customer_id,
                self.feature_id,
                self.agent_id,
                self.session_id,
            ]
        )


# ============================================================
# PII policy
# ============================================================


class PIIPolicyMode(str, Enum):
    """How to react when PII is found."""

    OFF = "off"  # detection disabled
    DETECT = "detect"  # log only, no modification
    REDACT = "redact"  # replace with token like [REDACTED:EMAIL]
    BLOCK = "block"  # raise PIIDetectedError


@dataclass
class PIIPolicy:
    """Per-tenant or per-feature PII handling policy."""

    mode: PIIPolicyMode = PIIPolicyMode.REDACT
    # Labels (e.g. "EMAIL", "PHONE", "CREDIT_CARD") to ENFORCE on. Empty = all.
    enforce_labels: List[str] = field(default_factory=list)
    # Labels to explicitly skip even if detected (e.g. allow EMAIL inside legal team).
    allow_labels: List[str] = field(default_factory=list)


# ============================================================
# Kill event (emitted on EventBus)
# ============================================================


@dataclass
class KillSwitchEvent:
    """Payload for EVENT_KILL_SWITCH_TRIGGERED."""

    cap: BudgetCap
    snapshot: BudgetUsageSnapshot
    request_id: str
    attribution: AttributionContext
    timestamp: float = field(default_factory=time.time)
    reason: str = "budget_exceeded"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cap": {
                "scope": self.cap.scope.value,
                "scope_id": self.cap.scope_id,
                "period": self.cap.period.value,
                "limit_usd": self.cap.limit_usd,
                "limit_tokens": self.cap.limit_tokens,
            },
            "snapshot": {
                "used_usd": self.snapshot.used_usd,
                "used_tokens": self.snapshot.used_tokens,
                "usage_pct": round(self.snapshot.usage_pct(), 4),
            },
            "request_id": self.request_id,
            "attribution": self.attribution.to_dict(),
            "timestamp": self.timestamp,
            "reason": self.reason,
        }
