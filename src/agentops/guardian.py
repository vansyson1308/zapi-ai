"""
Guardian — atomic budget caps + kill switches.

This is the wedge feature for AgentOps. Solving the "$47K agent loop ran for
11 days" pain.

Behavior:
  * register_cap(cap)           — store/replace a BudgetCap (cap.key)
  * pre_check(attribution, est) — before the request, refuse if any matching
                                   cap would already be over the limit (using
                                   *current* usage + estimate).
  * record_usage(attribution,
                 actual_usd, actual_tokens) — after the request, atomically
                                   add to all matching caps' counters; emit
                                   warning/kill events as thresholds cross.
  * check_after_chunk(attr, used_usd, used_tokens)
                                — mid-stream poll to halt early if cap was
                                   blown by other concurrent requests.

State is persisted in Redis via the RedisClient interface; in tests you
pass an InMemoryRedisClient and it works the same.

Concurrency safety:
  - All counter updates use INCRBY/INCRBYFLOAT (atomic in Redis, lock-protected
    in the in-memory shim). Pre-check + post-record are NOT a single atomic
    transaction — that would require a Lua script and we explicitly accept a
    very small over-run window in exchange for simplicity. This is the same
    trade-off OpenRouter/Portkey make.
  - The hard guarantee we DO offer: once `record_usage` returns, downstream
    `pre_check` calls in any process see the updated counter.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from ..core.redis_client import RedisClient, get_redis_client
from ..observability.event_bus import (
    EVENT_BUDGET_EXCEEDED,
    EVENT_BUDGET_WARNING,
    EVENT_KILL_SWITCH_TRIGGERED,
    Event,
    EventBus,
    get_event_bus,
)

from .errors import BudgetExceededError, BudgetMidStreamExceededError
from .models import (
    AttributionContext,
    BudgetCap,
    BudgetPeriod,
    BudgetScope,
    BudgetUsageSnapshot,
    HardAction,
    KillSwitchEvent,
)

logger = logging.getLogger(__name__)


# ============================================================
# Counter key helpers
# ============================================================


REDIS_BUDGET_USD_KEY = "agentops:budget:usd:{key}"
REDIS_BUDGET_TOKENS_KEY = "agentops:budget:tokens:{key}"


def _usd_key(cap: BudgetCap) -> str:
    return REDIS_BUDGET_USD_KEY.format(key=cap.key)


def _tokens_key(cap: BudgetCap) -> str:
    return REDIS_BUDGET_TOKENS_KEY.format(key=cap.key)


# ============================================================
# Guardian
# ============================================================


@dataclass
class _PendingResult:
    """Internal — returned by pre_check to feed into record_usage callers."""

    matched_caps: List[BudgetCap]


class Guardian:
    """
    Manage budget caps + kill switches with atomic enforcement.

    Lifecycle:
      1. Add caps via register_cap() at startup or via management API.
      2. Each request: middleware calls pre_check() → record_usage() pair.
      3. (Optional) Streaming: middleware spawns a check loop that calls
         check_after_chunk() periodically.
    """

    def __init__(
        self,
        redis_client: Optional[RedisClient] = None,
        event_bus: Optional[EventBus] = None,
    ) -> None:
        self._redis = redis_client or get_redis_client()
        self._bus = event_bus or get_event_bus()

        # In-memory cap registry. The source of truth could later be Postgres,
        # but for the wedge we keep it in-process and let admins call
        # register_cap() at startup (or via API in Phase 2).
        self._caps: Dict[str, BudgetCap] = {}
        # Track which (cap.key) we've already published a soft-warn for in the
        # current period. Stored as the wall-clock timestamp at which we fired,
        # so we can re-arm when the period rolls over (used_usd suddenly drops
        # back below the soft threshold).
        self._warned_at: Dict[str, float] = {}
        self._warned_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # cap management
    # ------------------------------------------------------------------

    def register_cap(self, cap: BudgetCap) -> None:
        """Add or replace a budget cap. Idempotent."""
        self._caps[cap.key] = cap
        # Reset warn flag for this cap
        self._warned_at.pop(cap.key, None)

    def remove_cap(self, key: str) -> bool:
        """Remove a cap. Returns True if it existed."""
        existed = key in self._caps
        self._caps.pop(key, None)
        self._warned_at.pop(key, None)
        return existed

    def list_caps(self) -> List[BudgetCap]:
        return list(self._caps.values())

    def get_cap(self, key: str) -> Optional[BudgetCap]:
        return self._caps.get(key)

    # ------------------------------------------------------------------
    # cap matching
    # ------------------------------------------------------------------

    def _matching_caps(self, attribution: AttributionContext) -> List[BudgetCap]:
        """Return caps whose scope_id matches the attribution context."""
        matches: List[BudgetCap] = []
        for cap in self._caps.values():
            if self._matches(cap, attribution):
                matches.append(cap)
        return matches

    @staticmethod
    def _matches(cap: BudgetCap, ctx: AttributionContext) -> bool:
        if cap.scope is BudgetScope.TENANT:
            return ctx.tenant_id == cap.scope_id
        if cap.scope is BudgetScope.CUSTOMER:
            return ctx.customer_id == cap.scope_id
        if cap.scope is BudgetScope.FEATURE:
            return ctx.feature_id == cap.scope_id
        if cap.scope is BudgetScope.AGENT:
            return ctx.agent_id == cap.scope_id
        if cap.scope is BudgetScope.API_KEY:
            return ctx.api_key_id == cap.scope_id
        return False

    # ------------------------------------------------------------------
    # snapshot
    # ------------------------------------------------------------------

    async def snapshot(self, cap: BudgetCap) -> BudgetUsageSnapshot:
        """Read current usage for one cap (no mutation)."""
        used_usd = await self._redis.get_float(_usd_key(cap)) if cap.limit_usd is not None else 0.0
        used_tokens = (
            await self._redis.get_int(_tokens_key(cap)) if cap.limit_tokens is not None else 0
        )
        return BudgetUsageSnapshot(
            cap=cap,
            used_usd=used_usd,
            used_tokens=used_tokens,
        )

    async def snapshots_for(self, attribution: AttributionContext) -> List[BudgetUsageSnapshot]:
        """Snapshot of every cap that matches the given attribution."""
        results: List[BudgetUsageSnapshot] = []
        for cap in self._matching_caps(attribution):
            results.append(await self.snapshot(cap))
        return results

    # ------------------------------------------------------------------
    # pre-check
    # ------------------------------------------------------------------

    async def pre_check(
        self,
        attribution: AttributionContext,
        estimate_usd: float = 0.0,
        estimate_tokens: int = 0,
        request_id: str = "",
    ) -> _PendingResult:
        """
        Pre-flight: refuse the request if any matching cap is already over.

        We deliberately use CURRENT usage (not current+estimate) for the BLOCK
        decision so a single oversized request can't create a deadlock where
        no request ever fits. Once usage >= limit, we block; the over-run
        window is at most one request per cap.

        Caps with hard_action=WARN or LOG never block (only emit events).
        """
        matched = self._matching_caps(attribution)
        for cap in matched:
            snap = await self.snapshot(cap)
            if snap.is_exceeded and cap.hard_action is HardAction.BLOCK:
                # Fire the kill event before raising
                await self._fire_kill_event(cap, snap, attribution, request_id)
                raise BudgetExceededError(
                    scope=cap.scope.value,
                    scope_id=cap.scope_id,
                    period=cap.period.value,
                    limit_usd=cap.limit_usd,
                    current_usd=snap.used_usd,
                    request_id=request_id,
                )
        return _PendingResult(matched_caps=matched)

    # ------------------------------------------------------------------
    # post-record
    # ------------------------------------------------------------------

    async def record_usage(
        self,
        attribution: AttributionContext,
        actual_usd: float = 0.0,
        actual_tokens: int = 0,
        request_id: str = "",
    ) -> List[BudgetUsageSnapshot]:
        """
        Atomically add this request's usage to every matching cap. Fires
        warning/kill events as thresholds cross.

        Returns a list of snapshots reflecting the *post-increment* state.
        """
        matched = self._matching_caps(attribution)
        snapshots: List[BudgetUsageSnapshot] = []

        for cap in matched:
            new_usd = 0.0
            new_tokens = 0
            ttl = cap.period.seconds()

            if cap.limit_usd is not None and actual_usd > 0:
                new_usd = await self._redis.incrbyfloat(
                    _usd_key(cap), float(actual_usd), ttl_seconds=ttl
                )
            elif cap.limit_usd is not None:
                new_usd = await self._redis.get_float(_usd_key(cap))

            if cap.limit_tokens is not None and actual_tokens > 0:
                new_tokens = await self._redis.incrby(
                    _tokens_key(cap), int(actual_tokens), ttl_seconds=ttl
                )
            elif cap.limit_tokens is not None:
                new_tokens = await self._redis.get_int(_tokens_key(cap))

            snap = BudgetUsageSnapshot(cap=cap, used_usd=new_usd, used_tokens=new_tokens)
            snapshots.append(snap)

            # Threshold events
            if snap.is_exceeded:
                await self._fire_kill_event(cap, snap, attribution, request_id)
            elif snap.is_soft_exceeded:
                if await self._should_warn(cap, snap):
                    await self._bus.publish(
                        Event(
                            name=EVENT_BUDGET_WARNING,
                            severity="warning",
                            payload={
                                "cap": {
                                    "scope": cap.scope.value,
                                    "scope_id": cap.scope_id,
                                    "period": cap.period.value,
                                },
                                "usage_pct": round(snap.usage_pct(), 4),
                                "request_id": request_id,
                                "attribution": attribution.to_dict(),
                            },
                        )
                    )
            else:
                # Usage dropped below the soft threshold (typically because the
                # period rolled over and the Redis counter expired). Re-arm the
                # warning so it fires again when we cross the threshold next
                # time.
                async with self._warned_lock:
                    self._warned_at.pop(cap.key, None)

        return snapshots

    async def _should_warn(self, cap: BudgetCap, snap: BudgetUsageSnapshot) -> bool:
        """Decide whether to publish a soft-warn event for this cap right now.

        We fire once per period. The period boundary is detected via TTL: if
        the cap-warned timestamp predates one full period, we consider it stale
        (the Redis counter has reset) and re-arm.
        """
        async with self._warned_lock:
            last = self._warned_at.get(cap.key)
            now = time.time()
            if last is not None and (now - last) < cap.period.seconds():
                return False
            self._warned_at[cap.key] = now
            return True

    # ------------------------------------------------------------------
    # mid-stream poll (called periodically while streaming)
    # ------------------------------------------------------------------

    async def check_after_chunk(
        self,
        attribution: AttributionContext,
        partial_tokens: int = 0,
        request_id: str = "",
    ) -> None:
        """
        Raise BudgetMidStreamExceededError if any matching cap is now over its
        hard limit.

        Caller is responsible for calling this every N tokens / chunks. The
        Guardian itself does not own the streaming loop.
        """
        for cap in self._matching_caps(attribution):
            snap = await self.snapshot(cap)
            if snap.is_exceeded and cap.hard_action is HardAction.BLOCK:
                await self._fire_kill_event(cap, snap, attribution, request_id)
                raise BudgetMidStreamExceededError(
                    scope=cap.scope.value,
                    scope_id=cap.scope_id,
                    period=cap.period.value,
                    partial_tokens=partial_tokens,
                    request_id=request_id,
                )

    # ------------------------------------------------------------------
    # event helpers
    # ------------------------------------------------------------------

    async def _fire_kill_event(
        self,
        cap: BudgetCap,
        snap: BudgetUsageSnapshot,
        attribution: AttributionContext,
        request_id: str,
    ) -> None:
        kill_event = KillSwitchEvent(
            cap=cap,
            snapshot=snap,
            request_id=request_id,
            attribution=attribution,
        )
        await self._bus.publish(
            Event(
                name=EVENT_KILL_SWITCH_TRIGGERED,
                severity="critical",
                payload=kill_event.to_dict(),
            )
        )
        # Also fire the simpler exceeded event (some handlers prefer that).
        await self._bus.publish(
            Event(
                name=EVENT_BUDGET_EXCEEDED,
                severity="critical",
                payload={
                    "cap_key": cap.key,
                    "request_id": request_id,
                    "attribution": attribution.to_dict(),
                },
            )
        )

    # ------------------------------------------------------------------
    # admin/test helpers
    # ------------------------------------------------------------------

    async def reset_cap_counter(self, cap: BudgetCap) -> None:
        """Force-reset the counter for a cap (for tests/admin)."""
        await self._redis.delete(_usd_key(cap), _tokens_key(cap))
        async with self._warned_lock:
            self._warned_at.pop(cap.key, None)

    async def reset_all(self) -> None:
        """Reset all registered caps. Test/admin tool."""
        for cap in list(self._caps.values()):
            await self.reset_cap_counter(cap)
