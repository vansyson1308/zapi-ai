"""
2api.ai - Usage Tracker

Tracks and records API usage for billing and analytics.

Features:
- Real-time usage tracking per request
- Tenant-level aggregation
- Async recording to database
- In-memory caching for performance
- Per-customer/feature/agent attribution dimensions
- Dead-letter queue for failed flushes (no silent data loss)
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple
from collections import defaultdict
from enum import Enum

from .pricing import calculate_cost, get_model_price

_logger = logging.getLogger(__name__)


class UsageStatus(str, Enum):
    """Status of a usage record."""
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    CONTENT_FILTERED = "content_filtered"
    CANCELLED = "cancelled"


class OperationType(str, Enum):
    """Type of API operation."""
    CHAT = "chat"
    CHAT_STREAM = "chat_stream"
    EMBEDDING = "embedding"
    IMAGE = "image"


@dataclass
class UsageRecord:
    """
    A single usage record for billing and analytics.

    This is the core data structure for tracking API usage.
    """
    # Identifiers
    request_id: str
    tenant_id: Optional[str] = None
    api_key_id: Optional[str] = None

    # Attribution dimensions (AgentOps): let us slice cost by customer/feature/agent
    customer_id: Optional[str] = None
    feature_id: Optional[str] = None
    agent_id: Optional[str] = None
    session_id: Optional[str] = None

    # Request details
    provider: str = ""
    model: str = ""
    operation: OperationType = OperationType.CHAT

    # Token usage
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    # Reasoning/thinking tokens: hidden tokens billed by o3, Claude thinking,
    # Gemini deep-think etc. Tracked separately so we can attribute "hidden"
    # spend per feature/customer.
    reasoning_tokens: int = 0

    # Cost
    cost_usd: float = 0.0

    # Performance
    latency_ms: int = 0
    time_to_first_token_ms: Optional[int] = None

    # Status
    status: UsageStatus = UsageStatus.SUCCESS
    error_code: Optional[str] = None
    error_message: Optional[str] = None

    # Routing
    routing_strategy: Optional[str] = None
    fallback_used: bool = False
    fallback_providers: List[str] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        """Calculate derived fields."""
        if self.total_tokens == 0:
            self.total_tokens = self.input_tokens + self.output_tokens

        if self.cost_usd == 0.0 and (self.input_tokens > 0 or self.output_tokens > 0):
            self.cost_usd = calculate_cost(
                model_id=self.model,
                input_tokens=self.input_tokens,
                output_tokens=self.output_tokens,
                cached_tokens=self.cached_tokens
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "request_id": self.request_id,
            "tenant_id": self.tenant_id,
            "api_key_id": self.api_key_id,
            "customer_id": self.customer_id,
            "feature_id": self.feature_id,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "provider": self.provider,
            "model": self.model,
            "operation": self.operation.value,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cached_tokens": self.cached_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cost_usd": self.cost_usd,
            "latency_ms": self.latency_ms,
            "time_to_first_token_ms": self.time_to_first_token_ms,
            "status": self.status.value,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "routing_strategy": self.routing_strategy,
            "fallback_used": self.fallback_used,
            "fallback_providers": self.fallback_providers,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class RequestTracker:
    """
    Tracks a single request through its lifecycle.

    Use this to build up usage data as a request progresses.
    """
    request_id: str
    tenant_id: Optional[str] = None
    api_key_id: Optional[str] = None
    # Attribution dimensions
    customer_id: Optional[str] = None
    feature_id: Optional[str] = None
    agent_id: Optional[str] = None
    session_id: Optional[str] = None

    model: str = ""
    provider: str = ""
    operation: OperationType = OperationType.CHAT

    # Timing
    _start_time: float = field(default_factory=time.time)
    _first_token_time: Optional[float] = None
    _end_time: Optional[float] = None

    # Token counts (accumulated)
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0

    # Status tracking
    status: UsageStatus = UsageStatus.SUCCESS
    error_code: Optional[str] = None
    error_message: Optional[str] = None

    # Routing
    routing_strategy: Optional[str] = None
    fallback_providers: List[str] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def record_first_token(self):
        """Record time of first token (for streaming)."""
        if self._first_token_time is None:
            self._first_token_time = time.time()

    def add_tokens(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_tokens: int = 0,
        reasoning_tokens: int = 0,
    ):
        """Add token counts (incl. reasoning/thinking tokens for o3/Claude)."""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.cached_tokens += cached_tokens
        self.reasoning_tokens += reasoning_tokens

    def set_attribution(
        self,
        customer_id: Optional[str] = None,
        feature_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        """Apply AgentOps attribution dimensions."""
        if customer_id is not None:
            self.customer_id = customer_id
        if feature_id is not None:
            self.feature_id = feature_id
        if agent_id is not None:
            self.agent_id = agent_id
        if session_id is not None:
            self.session_id = session_id

    def set_error(self, code: str, message: str, status: UsageStatus = UsageStatus.ERROR):
        """Record an error."""
        self.status = status
        self.error_code = code
        self.error_message = message

    def add_fallback(self, provider: str):
        """Record a fallback provider used."""
        self.fallback_providers.append(provider)

    def complete(self) -> UsageRecord:
        """
        Complete tracking and return final usage record.

        This calculates final metrics and creates the record.
        """
        self._end_time = time.time()

        latency_ms = int((self._end_time - self._start_time) * 1000)

        ttft_ms = None
        if self._first_token_time:
            ttft_ms = int((self._first_token_time - self._start_time) * 1000)

        return UsageRecord(
            request_id=self.request_id,
            tenant_id=self.tenant_id,
            api_key_id=self.api_key_id,
            customer_id=self.customer_id,
            feature_id=self.feature_id,
            agent_id=self.agent_id,
            session_id=self.session_id,
            provider=self.provider,
            model=self.model,
            operation=self.operation,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cached_tokens=self.cached_tokens,
            reasoning_tokens=self.reasoning_tokens,
            latency_ms=latency_ms,
            time_to_first_token_ms=ttft_ms,
            status=self.status,
            error_code=self.error_code,
            error_message=self.error_message,
            routing_strategy=self.routing_strategy,
            fallback_used=len(self.fallback_providers) > 0,
            fallback_providers=self.fallback_providers,
            metadata=self.metadata
        )


# Type for storage callback
StorageCallback = Callable[[UsageRecord], Coroutine[Any, Any, None]]


class UsageTracker:
    """
    Central usage tracking service.

    Handles:
    - Creating request trackers
    - Recording completed usage
    - In-memory caching
    - Async persistence
    """

    def __init__(
        self,
        storage_callback: Optional[StorageCallback] = None,
        buffer_size: int = 100,
        flush_interval_seconds: float = 5.0,
        dlq_max_size: int = 10_000,
    ):
        """
        Initialize usage tracker.

        Args:
            storage_callback: Async function to persist records
            buffer_size: Max records to buffer before flush
            flush_interval_seconds: Time between automatic flushes
            dlq_max_size: Max records in dead-letter queue before oldest are dropped
        """
        self._storage_callback = storage_callback
        self._buffer_size = buffer_size
        self._flush_interval = flush_interval_seconds
        self._dlq_max_size = dlq_max_size

        # Active trackers
        self._active_trackers: Dict[str, RequestTracker] = {}

        # Buffer for async storage
        self._buffer: List[UsageRecord] = []
        self._buffer_lock = asyncio.Lock()

        # In-memory aggregates (for fast queries) keyed by tenant_id.
        self._tenant_usage: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"tokens": 0, "cost": 0.0, "requests": 0, "reasoning_tokens": 0}
        )
        # Attribution-aware aggregates: keyed by (tenant_id, customer_id, feature_id, agent_id).
        self._attribution_usage: Dict[
            Tuple[Optional[str], Optional[str], Optional[str], Optional[str]],
            Dict[str, float],
        ] = defaultdict(
            lambda: {"tokens": 0, "cost": 0.0, "requests": 0, "reasoning_tokens": 0}
        )

        # Dead-letter queue for failed flushes — kept in-memory; periodic
        # retry attempts run in background.
        self._dlq: List[UsageRecord] = []
        self._dlq_lock = asyncio.Lock()

        # Background task handles
        self._flush_task: Optional[asyncio.Task] = None
        self._dlq_retry_task: Optional[asyncio.Task] = None

    def start_tracking(
        self,
        request_id: str,
        tenant_id: Optional[str] = None,
        api_key_id: Optional[str] = None,
        model: str = "",
        provider: str = "",
        operation: OperationType = OperationType.CHAT,
        customer_id: Optional[str] = None,
        feature_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> RequestTracker:
        """
        Start tracking a new request.

        Args:
            request_id: Unique request identifier
            tenant_id: Tenant making the request
            api_key_id: API key used
            model: Model being used
            provider: Provider handling request
            operation: Type of operation
            customer_id, feature_id, agent_id, session_id: AgentOps attribution

        Returns:
            RequestTracker for this request
        """
        tracker = RequestTracker(
            request_id=request_id,
            tenant_id=tenant_id,
            api_key_id=api_key_id,
            customer_id=customer_id,
            feature_id=feature_id,
            agent_id=agent_id,
            session_id=session_id,
            model=model,
            provider=provider,
            operation=operation,
        )

        self._active_trackers[request_id] = tracker
        return tracker

    def get_tracker(self, request_id: str) -> Optional[RequestTracker]:
        """Get active tracker by request ID."""
        return self._active_trackers.get(request_id)

    async def complete_tracking(self, tracker: RequestTracker) -> UsageRecord:
        """
        Complete tracking and record usage.

        Args:
            tracker: The request tracker to complete

        Returns:
            Final usage record
        """
        # Create final record
        record = tracker.complete()

        # Remove from active
        self._active_trackers.pop(tracker.request_id, None)

        # Update aggregates
        self._update_aggregates(record)

        # Add to buffer
        async with self._buffer_lock:
            self._buffer.append(record)

            # Flush if buffer full
            if len(self._buffer) >= self._buffer_size:
                await self._flush_buffer()

        return record

    async def record_usage(self, record: UsageRecord):
        """
        Directly record a usage record.

        Use this when you have a complete record (e.g., from recovery).
        """
        self._update_aggregates(record)

        async with self._buffer_lock:
            self._buffer.append(record)

            if len(self._buffer) >= self._buffer_size:
                await self._flush_buffer()

    def _update_aggregates(self, record: UsageRecord) -> None:
        """Update tenant + attribution aggregates from a completed record."""
        if record.tenant_id:
            agg = self._tenant_usage[record.tenant_id]
            agg["tokens"] += record.total_tokens
            agg["cost"] += record.cost_usd
            agg["requests"] += 1
            agg["reasoning_tokens"] += record.reasoning_tokens

        # Attribution rollup; only meaningful if at least one extra dimension is set
        if record.customer_id or record.feature_id or record.agent_id:
            key = (
                record.tenant_id,
                record.customer_id,
                record.feature_id,
                record.agent_id,
            )
            agg = self._attribution_usage[key]
            agg["tokens"] += record.total_tokens
            agg["cost"] += record.cost_usd
            agg["requests"] += 1
            agg["reasoning_tokens"] += record.reasoning_tokens

    async def _flush_buffer(self):
        """Flush buffered records to storage. Failures go to the DLQ."""
        if not self._buffer:
            return
        if not self._storage_callback:
            # No storage configured — drop the buffer (in-memory mode)
            self._buffer.clear()
            return

        records = self._buffer.copy()
        self._buffer.clear()

        failed: List[UsageRecord] = []
        for record in records:
            try:
                await self._storage_callback(record)
            except Exception as e:
                _logger.warning(
                    "usage flush failed: request_id=%s error=%s",
                    record.request_id,
                    e,
                )
                failed.append(record)

        if failed:
            await self._enqueue_dlq(failed)

    async def _enqueue_dlq(self, records: List[UsageRecord]) -> None:
        """Add records to the DLQ, evicting oldest if over capacity."""
        async with self._dlq_lock:
            self._dlq.extend(records)
            overflow = len(self._dlq) - self._dlq_max_size
            if overflow > 0:
                dropped = self._dlq[:overflow]
                self._dlq = self._dlq[overflow:]
                _logger.error(
                    "usage DLQ overflow: dropping %d oldest records (limit=%d)",
                    len(dropped),
                    self._dlq_max_size,
                )

    async def retry_dlq(self) -> int:
        """
        Attempt to flush DLQ records to storage_callback.

        Returns the number of records successfully flushed.

        Concurrency: we hold the DLQ lock for the entire retry pass so a
        concurrent `_enqueue_dlq` call cannot have its records silently
        overwritten when we re-queue still-failing records. Storage callbacks
        are invoked while holding the lock — that's deliberate; if they're
        slow, callers can run `retry_dlq` less frequently.
        """
        if not self._storage_callback:
            return 0

        async with self._dlq_lock:
            if not self._dlq:
                return 0
            pending = self._dlq[:]
            self._dlq.clear()

            succeeded = 0
            still_failed: List[UsageRecord] = []
            for record in pending:
                try:
                    await self._storage_callback(record)
                    succeeded += 1
                except Exception as e:
                    _logger.debug(
                        "DLQ retry still failing: request_id=%s error=%s",
                        record.request_id,
                        e,
                    )
                    still_failed.append(record)

            # Merge: still-failed first (they're older), then any records that
            # arrived during the retry (already appended via _enqueue_dlq's lock —
            # but since we held the lock, that path was blocked, so self._dlq is
            # exactly what was added by anyone who took the lock before us).
            # Simpler: prepend still_failed.
            if still_failed:
                # Respect dlq_max_size — drop oldest still-failing records if needed
                combined = still_failed + self._dlq
                if len(combined) > self._dlq_max_size:
                    overflow = len(combined) - self._dlq_max_size
                    combined = combined[overflow:]
                    _logger.error(
                        "usage DLQ overflow during retry: dropped %d records",
                        overflow,
                    )
                self._dlq = combined
            return succeeded

    def dlq_size(self) -> int:
        """Return current DLQ size (test/observability helper)."""
        return len(self._dlq)

    async def flush(self):
        """Manually flush all buffered records."""
        async with self._buffer_lock:
            await self._flush_buffer()

    def get_tenant_usage(self, tenant_id: str) -> Dict[str, Any]:
        """
        Get current in-memory usage for a tenant.

        Note: This is approximate, based on in-memory data.
        For accurate totals, query the database.
        """
        usage = self._tenant_usage.get(tenant_id, {})
        return {
            "tenant_id": tenant_id,
            "total_tokens": usage.get("tokens", 0),
            "total_cost_usd": usage.get("cost", 0.0),
            "request_count": usage.get("requests", 0),
            "reasoning_tokens": usage.get("reasoning_tokens", 0),
        }

    def get_attribution_usage(
        self,
        tenant_id: Optional[str] = None,
        customer_id: Optional[str] = None,
        feature_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Slice attribution-aware usage. Any None argument is a wildcard.

        Returns a list of dicts shaped like:
            {tenant_id, customer_id, feature_id, agent_id,
             total_tokens, total_cost_usd, request_count, reasoning_tokens}
        """
        out: List[Dict[str, Any]] = []
        for (t_id, c_id, f_id, a_id), agg in self._attribution_usage.items():
            if tenant_id is not None and t_id != tenant_id:
                continue
            if customer_id is not None and c_id != customer_id:
                continue
            if feature_id is not None and f_id != feature_id:
                continue
            if agent_id is not None and a_id != agent_id:
                continue
            out.append(
                {
                    "tenant_id": t_id,
                    "customer_id": c_id,
                    "feature_id": f_id,
                    "agent_id": a_id,
                    "total_tokens": agg["tokens"],
                    "total_cost_usd": agg["cost"],
                    "request_count": int(agg["requests"]),
                    "reasoning_tokens": int(agg["reasoning_tokens"]),
                }
            )
        return out

    def get_active_request_count(self) -> int:
        """Get number of active requests being tracked."""
        return len(self._active_trackers)

    async def start_background_flush(self):
        """Start background task to periodically flush buffer + retry DLQ."""
        async def flush_loop():
            while True:
                await asyncio.sleep(self._flush_interval)
                try:
                    await self.flush()
                except Exception:
                    pass

        async def dlq_loop():
            # Retry DLQ at the slower of (3 * flush_interval, 30s) to avoid
            # hammering a flapping storage backend.
            interval = max(self._flush_interval * 3.0, 30.0)
            while True:
                await asyncio.sleep(interval)
                try:
                    await self.retry_dlq()
                except Exception:
                    pass

        self._flush_task = asyncio.create_task(flush_loop())
        self._dlq_retry_task = asyncio.create_task(dlq_loop())

    async def stop_background_flush(self):
        """Stop background tasks and flush remaining records."""
        for task in (self._flush_task, self._dlq_retry_task):
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._flush_task = None
        self._dlq_retry_task = None

        # Final flush + final DLQ retry
        await self.flush()
        await self.retry_dlq()


# Global tracker instance
_tracker: Optional[UsageTracker] = None


def get_usage_tracker() -> UsageTracker:
    """Get or create global usage tracker."""
    global _tracker
    if _tracker is None:
        _tracker = UsageTracker()
    return _tracker


def set_usage_tracker(tracker: UsageTracker):
    """Set global usage tracker (for testing/configuration)."""
    global _tracker
    _tracker = tracker


def start_tracking(
    request_id: str,
    tenant_id: Optional[str] = None,
    api_key_id: Optional[str] = None,
    model: str = "",
    provider: str = "",
    operation: OperationType = OperationType.CHAT,
    customer_id: Optional[str] = None,
    feature_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> RequestTracker:
    """Convenience function to start tracking a request."""
    return get_usage_tracker().start_tracking(
        request_id=request_id,
        tenant_id=tenant_id,
        api_key_id=api_key_id,
        model=model,
        provider=provider,
        operation=operation,
        customer_id=customer_id,
        feature_id=feature_id,
        agent_id=agent_id,
        session_id=session_id,
    )


async def complete_tracking(tracker: RequestTracker) -> UsageRecord:
    """Convenience function to complete tracking."""
    return await get_usage_tracker().complete_tracking(tracker)
