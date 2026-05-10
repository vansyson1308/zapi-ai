"""
Tests for bugs surfaced by the post-implementation adversarial audit.

Each test names the audit finding it covers.
"""

from __future__ import annotations

import asyncio
from typing import List

import pytest

from src.adapters.base import AdapterConfig, BaseAdapter, ProviderHealth
from src.adapters.stub_adapter import StubAdapter
from src.agentops import (
    AttributionContext,
    BudgetCap,
    BudgetPeriod,
    BudgetScope,
    Guardian,
)
from src.agentops.attribution import attribution_context_from_headers
from src.core.models import ChatCompletionRequest, Message, Provider, Role
from src.core.redis_client import InMemoryRedisClient
from src.observability.event_bus import (
    EVENT_BUDGET_WARNING,
    Event,
    EventBus,
)
from src.routing.circuit_breaker import CircuitBreakerRegistry
from src.routing.health import HealthRegistry
from src.routing.provider_selector import ProviderSelector
from src.routing.streaming_manager import StreamingManager
from src.security.encryption import (
    EncryptionError,
    KeyringEncryptor,
    KeyringEntry,
)
from src.streaming.normalizer import StreamNormalizer
from src.streaming.tool_calls import ToolCallStreamTracker
from src.usage.tracker import UsageRecord, UsageTracker


# ============================================================
# P1: KeyringEncryptor empty payload bypass
# ============================================================


def test_keyring_rejects_empty_v2_payload():
    enc = KeyringEncryptor([KeyringEntry(id="v1", secret="x")])
    with pytest.raises(EncryptionError):
        enc.decrypt("enc:v2:k=v1:")


def test_keyring_rejects_empty_v2_key_id():
    enc = KeyringEncryptor([KeyringEntry(id="v1", secret="x")])
    with pytest.raises(EncryptionError):
        enc.decrypt("enc:v2:k=:somebody")


# ============================================================
# P1: StreamingManager empty choices list IndexError
# ============================================================


def test_streaming_manager_handles_empty_choices_list():
    """`{"choices": []}` should not crash phase tracking."""
    mgr = StreamingManager(
        record_success=lambda *_a, **_kw: None,
        record_failure=lambda *_a, **_kw: None,
    )

    class FakeTracker:
        def can_fallback(self):
            return True

        def mark_content_started(self, c):
            self._got = c

        def append_content(self, c):
            pass

    tracker = FakeTracker()
    tool_tracker = ToolCallStreamTracker()
    # Should not raise
    mgr._inspect_event_for_phase_tracking(
        'data: {"choices": [], "id": "x"}',
        tracker,
        tool_tracker,
    )


def test_streaming_manager_handles_null_choice_entry():
    """Some providers send `{"choices": [null]}` for keep-alive."""
    mgr = StreamingManager(
        record_success=lambda *_a, **_kw: None,
        record_failure=lambda *_a, **_kw: None,
    )

    class FakeTracker:
        def can_fallback(self):
            return True

        def mark_content_started(self, c):
            pass

        def append_content(self, c):
            pass

    mgr._inspect_event_for_phase_tracking(
        'data: {"choices": [null]}',
        FakeTracker(),
        ToolCallStreamTracker(),
    )


# ============================================================
# P1: UsageTracker DLQ race
# ============================================================


@pytest.mark.asyncio
async def test_dlq_retry_is_concurrent_safe():
    """retry_dlq + concurrent flushes don't drop records or double-count."""
    seen: List[str] = []
    fail_first = {"n": 0}

    async def storage(record: UsageRecord):
        # Fail the first 5 records, then succeed
        if fail_first["n"] < 5:
            fail_first["n"] += 1
            raise RuntimeError("transient")
        seen.append(record.request_id)

    tracker_svc = UsageTracker(storage_callback=storage, buffer_size=1)

    async def producer(start: int, n: int):
        for i in range(start, start + n):
            rt = tracker_svc.start_tracking(request_id=f"r-{i}", tenant_id="t")
            rt.add_tokens(input_tokens=1, output_tokens=1)
            await tracker_svc.complete_tracking(rt)

    # 2 producers and 1 retry running concurrently
    await asyncio.gather(
        producer(0, 5),
        producer(100, 5),
        tracker_svc.retry_dlq(),
        tracker_svc.retry_dlq(),
    )

    # Drain remaining DLQ
    await tracker_svc.retry_dlq()

    # We should not lose any records: total flushed + currently in DLQ == 10
    assert len(seen) + tracker_svc.dlq_size() == 10


@pytest.mark.asyncio
async def test_dlq_retry_returns_zero_when_empty():
    async def storage(record: UsageRecord):
        return None

    tracker_svc = UsageTracker(storage_callback=storage)
    assert await tracker_svc.retry_dlq() == 0


# ============================================================
# P1: Guardian warning resets across periods
# ============================================================


@pytest.mark.asyncio
async def test_warning_re_arms_after_period_reset():
    redis = InMemoryRedisClient()
    bus = EventBus(default_fire_and_forget=False)
    guardian = Guardian(redis_client=redis, event_bus=bus)

    cap = BudgetCap(
        scope=BudgetScope.CUSTOMER,
        scope_id="acme",
        period=BudgetPeriod.HOUR,
        limit_usd=10.0,
        soft_threshold_pct=0.8,
    )
    guardian.register_cap(cap)
    attr = AttributionContext(customer_id="acme")

    warnings: List[Event] = []

    async def handler(evt: Event):
        warnings.append(evt)

    bus.subscribe(EVENT_BUDGET_WARNING, handler)

    # Period 1: cross threshold
    await guardian.record_usage(attr, actual_usd=8.5)
    assert len(warnings) == 1

    # Reset the underlying counter (simulating period rollover)
    await guardian.reset_cap_counter(cap)

    # Period 2: cross threshold again — should warn again
    await guardian.record_usage(attr, actual_usd=8.5)
    assert len(warnings) == 2


@pytest.mark.asyncio
async def test_warn_resets_when_usage_drops_below_threshold():
    """Lower the underlying counter; next record_usage should re-arm warn flag."""
    redis = InMemoryRedisClient()
    bus = EventBus(default_fire_and_forget=False)
    guardian = Guardian(redis_client=redis, event_bus=bus)

    cap = BudgetCap(
        scope=BudgetScope.CUSTOMER,
        scope_id="acme",
        period=BudgetPeriod.DAY,
        limit_usd=10.0,
    )
    guardian.register_cap(cap)
    attr = AttributionContext(customer_id="acme")

    warnings: List[Event] = []
    bus.subscribe(EVENT_BUDGET_WARNING, lambda evt: warnings.append(evt))

    await guardian.record_usage(attr, actual_usd=8.5)
    assert len(warnings) == 1

    # Reset underlying counter, then a small spend → no warn (under threshold)
    await guardian.reset_cap_counter(cap)
    await guardian.record_usage(attr, actual_usd=1.0)
    assert len(warnings) == 1  # still 1; we re-armed but haven't crossed yet

    # Now go above threshold again
    await guardian.record_usage(attr, actual_usd=8.0)
    assert len(warnings) == 2


# ============================================================
# P1: EventBus _inflight memory growth
# ============================================================


@pytest.mark.asyncio
async def test_event_bus_inflight_does_not_grow_unbounded():
    bus = EventBus(default_fire_and_forget=True)

    async def fast_handler(_evt: Event):
        return None

    bus.subscribe("topic", fast_handler)

    for _ in range(500):
        await bus.publish(Event(name="topic"))

    # Drain finishes everything; in-flight should be empty after.
    await bus.drain()
    assert len(bus._inflight) == 0


@pytest.mark.asyncio
async def test_event_bus_drain_cancels_stragglers_on_timeout():
    bus = EventBus(default_fire_and_forget=True)

    async def stuck(_evt: Event):
        await asyncio.sleep(60)

    bus.subscribe("topic", stuck)
    await bus.publish(Event(name="topic"))
    await bus.drain(timeout=0.05)
    # All previously in-flight tasks should be cancelled or done
    for t in list(bus._inflight):
        assert t.done() or t.cancelled()


# ============================================================
# P2: AttributionContext NFC normalization
# ============================================================


def test_attribution_normalizes_unicode_nfc():
    # é (precomposed) and "é" (decomposed) should collapse
    ctx_pre = attribution_context_from_headers({"X-Customer-Id": "café"})
    ctx_dec = attribution_context_from_headers({"X-Customer-Id": "café"})
    assert ctx_pre.customer_id == ctx_dec.customer_id


# ============================================================
# P2: ProviderSelector graceful degradation on adapter list_models() failure
# ============================================================


class _FailingListAdapter(BaseAdapter):
    provider = Provider.OPENAI

    def __init__(self):
        super().__init__(AdapterConfig(api_key="x"))

    async def chat_completion(self, request, request_id=""):
        raise NotImplementedError

    async def chat_completion_stream(self, request, request_id=""):
        raise NotImplementedError
        yield  # type: ignore[unreachable]

    async def embedding(self, request, request_id=""):
        raise NotImplementedError

    async def image_generation(self, request, request_id=""):
        raise NotImplementedError

    def list_models(self):
        raise RuntimeError("upstream list_models broke")

    async def health_check(self):
        return ProviderHealth(provider=self.provider, is_healthy=False)


def test_selector_init_does_not_crash_when_one_adapter_throws():
    """A failing list_models() should not break Router init for OTHER adapters."""
    cb = CircuitBreakerRegistry()
    health = HealthRegistry()
    adapters = {
        Provider.OPENAI: _FailingListAdapter(),
        Provider.ANTHROPIC: StubAdapter(AdapterConfig(api_key="stub")),
    }
    # If we don't catch the exception, ProviderSelector init throws.
    selector = ProviderSelector(
        adapters=adapters,
        circuit_breakers=cb,
        health_registry=health,
    )
    # Stub adapter's models should be registered; OpenAI's are skipped.
    ids = list(selector.model_registry.keys())
    assert any(model_id.startswith("openai/") is False for model_id in ids)


# ============================================================
# P2: BudgetCap is frozen but registry replacement works
# ============================================================


@pytest.mark.asyncio
async def test_budget_cap_frozen_dataclass_does_not_break_registry_replace():
    redis = InMemoryRedisClient()
    bus = EventBus(default_fire_and_forget=False)
    guardian = Guardian(redis_client=redis, event_bus=bus)

    cap1 = BudgetCap(
        scope=BudgetScope.CUSTOMER,
        scope_id="x",
        period=BudgetPeriod.DAY,
        limit_usd=5.0,
    )
    cap2 = BudgetCap(
        scope=BudgetScope.CUSTOMER,
        scope_id="x",
        period=BudgetPeriod.DAY,
        limit_usd=10.0,
    )
    guardian.register_cap(cap1)
    guardian.register_cap(cap2)
    assert guardian.get_cap(cap1.key).limit_usd == 10.0
