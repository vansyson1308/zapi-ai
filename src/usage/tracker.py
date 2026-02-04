"""
2api.ai - Usage Tracker

Tracks and records API usage for billing and analytics.

Features:
- Real-time usage tracking per request
- Tenant-level aggregation
- Async recording to database
- In-memory caching for performance
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Coroutine, Dict, List, Optional
from collections import defaultdict
from enum import Enum

from .pricing import calculate_cost, get_model_price


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

    # Request details
    provider: str = ""
    model: str = ""
    operation: OperationType = OperationType.CHAT

    # Token usage
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0

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
            "provider": self.provider,
            "model": self.model,
            "operation": self.operation.value,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cached_tokens": self.cached_tokens,
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

    def add_tokens(self, input_tokens: int = 0, output_tokens: int = 0, cached_tokens: int = 0):
        """Add token counts."""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.cached_tokens += cached_tokens

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
            provider=self.provider,
            model=self.model,
            operation=self.operation,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cached_tokens=self.cached_tokens,
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
        flush_interval_seconds: float = 5.0
    ):
        """
        Initialize usage tracker.

        Args:
            storage_callback: Async function to persist records
            buffer_size: Max records to buffer before flush
            flush_interval_seconds: Time between automatic flushes
        """
        self._storage_callback = storage_callback
        self._buffer_size = buffer_size
        self._flush_interval = flush_interval_seconds

        # Active trackers
        self._active_trackers: Dict[str, RequestTracker] = {}

        # Buffer for async storage
        self._buffer: List[UsageRecord] = []
        self._buffer_lock = asyncio.Lock()

        # In-memory aggregates (for fast queries)
        self._tenant_usage: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"tokens": 0, "cost": 0.0, "requests": 0}
        )

        # Background task handle
        self._flush_task: Optional[asyncio.Task] = None

    def start_tracking(
        self,
        request_id: str,
        tenant_id: Optional[str] = None,
        api_key_id: Optional[str] = None,
        model: str = "",
        provider: str = "",
        operation: OperationType = OperationType.CHAT
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

        Returns:
            RequestTracker for this request
        """
        tracker = RequestTracker(
            request_id=request_id,
            tenant_id=tenant_id,
            api_key_id=api_key_id,
            model=model,
            provider=provider,
            operation=operation
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

        # Update in-memory aggregates
        if tracker.tenant_id:
            self._tenant_usage[tracker.tenant_id]["tokens"] += record.total_tokens
            self._tenant_usage[tracker.tenant_id]["cost"] += record.cost_usd
            self._tenant_usage[tracker.tenant_id]["requests"] += 1

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
        # Update in-memory aggregates
        if record.tenant_id:
            self._tenant_usage[record.tenant_id]["tokens"] += record.total_tokens
            self._tenant_usage[record.tenant_id]["cost"] += record.cost_usd
            self._tenant_usage[record.tenant_id]["requests"] += 1

        # Add to buffer
        async with self._buffer_lock:
            self._buffer.append(record)

            if len(self._buffer) >= self._buffer_size:
                await self._flush_buffer()

    async def _flush_buffer(self):
        """Flush buffered records to storage."""
        if not self._buffer or not self._storage_callback:
            return

        # Get records to flush
        records = self._buffer.copy()
        self._buffer.clear()

        # Store each record
        for record in records:
            try:
                await self._storage_callback(record)
            except Exception as e:
                # Log error but don't lose records
                # In production, would use proper logging
                print(f"Error storing usage record: {e}")
                # Could re-add to buffer or dead-letter queue

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
            "request_count": usage.get("requests", 0)
        }

    def get_active_request_count(self) -> int:
        """Get number of active requests being tracked."""
        return len(self._active_trackers)

    async def start_background_flush(self):
        """Start background task to periodically flush buffer."""
        async def flush_loop():
            while True:
                await asyncio.sleep(self._flush_interval)
                try:
                    await self.flush()
                except Exception:
                    pass  # Continue on error

        self._flush_task = asyncio.create_task(flush_loop())

    async def stop_background_flush(self):
        """Stop background flush task and flush remaining records."""
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Final flush
        await self.flush()


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
    operation: OperationType = OperationType.CHAT
) -> RequestTracker:
    """Convenience function to start tracking a request."""
    return get_usage_tracker().start_tracking(
        request_id=request_id,
        tenant_id=tenant_id,
        api_key_id=api_key_id,
        model=model,
        provider=provider,
        operation=operation
    )


async def complete_tracking(tracker: RequestTracker) -> UsageRecord:
    """Convenience function to complete tracking."""
    return await get_usage_tracker().complete_tracking(tracker)
