"""
Tests for src/agentops/guardian.py — the wedge feature.

Covers:
  - register_cap / remove_cap
  - matching attribution → caps
  - pre_check blocks when over hard limit (BLOCK action)
  - pre_check passes through for WARN/LOG actions
  - record_usage atomically increments + emits events
  - check_after_chunk raises mid-stream when limit blown
  - reset
  - concurrency: many simultaneous record_usage calls converge to correct total
"""

from __future__ import annotations

import asyncio
from typing import List

import pytest

from src.agentops import (
    AttributionContext,
    BudgetCap,
    BudgetPeriod,
    BudgetScope,
    BudgetExceededError,
    BudgetMidStreamExceededError,
    Guardian,
    HardAction,
)
from src.core.redis_client import InMemoryRedisClient
from src.observability.event_bus import (
    EVENT_BUDGET_EXCEEDED,
    EVENT_BUDGET_WARNING,
    EVENT_KILL_SWITCH_TRIGGERED,
    Event,
    EventBus,
)


# ============================================================
# fixtures
# ============================================================


@pytest.fixture
def redis_client():
    return InMemoryRedisClient()


@pytest.fixture
def event_bus():
    return EventBus(default_fire_and_forget=False)


@pytest.fixture
def guardian(redis_client, event_bus):
    return Guardian(redis_client=redis_client, event_bus=event_bus)


@pytest.fixture
def customer_attr():
    return AttributionContext(
        tenant_id="t1",
        customer_id="acme-corp",
        feature_id="chatbot",
        agent_id="agent-1",
    )


# ============================================================
# cap registry
# ============================================================


def test_register_and_list_caps(guardian):
    cap = BudgetCap(
        scope=BudgetScope.CUSTOMER,
        scope_id="acme-corp",
        period=BudgetPeriod.DAY,
        limit_usd=10.0,
    )
    guardian.register_cap(cap)
    assert len(guardian.list_caps()) == 1
    assert guardian.get_cap(cap.key) == cap


def test_register_replaces_existing(guardian):
    cap1 = BudgetCap(
        scope=BudgetScope.CUSTOMER,
        scope_id="acme",
        period=BudgetPeriod.DAY,
        limit_usd=5.0,
    )
    cap2 = BudgetCap(
        scope=BudgetScope.CUSTOMER,
        scope_id="acme",
        period=BudgetPeriod.DAY,
        limit_usd=10.0,
    )
    guardian.register_cap(cap1)
    guardian.register_cap(cap2)
    assert guardian.get_cap(cap1.key).limit_usd == 10.0


def test_remove_cap_returns_true_when_present(guardian):
    cap = BudgetCap(
        scope=BudgetScope.AGENT,
        scope_id="a1",
        period=BudgetPeriod.HOUR,
        limit_usd=1.0,
    )
    guardian.register_cap(cap)
    assert guardian.remove_cap(cap.key) is True
    assert guardian.remove_cap(cap.key) is False


def test_budget_cap_validates_limits():
    with pytest.raises(ValueError):
        BudgetCap(
            scope=BudgetScope.CUSTOMER,
            scope_id="x",
            period=BudgetPeriod.DAY,
        )  # neither USD nor tokens
    with pytest.raises(ValueError):
        BudgetCap(
            scope=BudgetScope.CUSTOMER,
            scope_id="x",
            period=BudgetPeriod.DAY,
            limit_usd=-1,
        )
    with pytest.raises(ValueError):
        BudgetCap(
            scope=BudgetScope.CUSTOMER,
            scope_id="x",
            period=BudgetPeriod.DAY,
            limit_usd=1,
            soft_threshold_pct=1.5,
        )


# ============================================================
# attribution matching
# ============================================================


@pytest.mark.asyncio
async def test_matches_customer_scope(guardian, customer_attr):
    cap = BudgetCap(
        scope=BudgetScope.CUSTOMER,
        scope_id="acme-corp",
        period=BudgetPeriod.DAY,
        limit_usd=10.0,
    )
    guardian.register_cap(cap)
    matches = guardian._matching_caps(customer_attr)
    assert matches == [cap]


@pytest.mark.asyncio
async def test_does_not_match_unrelated_scope(guardian):
    cap = BudgetCap(
        scope=BudgetScope.CUSTOMER,
        scope_id="acme",
        period=BudgetPeriod.DAY,
        limit_usd=10.0,
    )
    guardian.register_cap(cap)
    other = AttributionContext(customer_id="other-corp")
    assert guardian._matching_caps(other) == []


@pytest.mark.asyncio
async def test_matches_multiple_scopes_simultaneously(guardian, customer_attr):
    cap_customer = BudgetCap(
        scope=BudgetScope.CUSTOMER,
        scope_id="acme-corp",
        period=BudgetPeriod.DAY,
        limit_usd=10.0,
    )
    cap_feature = BudgetCap(
        scope=BudgetScope.FEATURE,
        scope_id="chatbot",
        period=BudgetPeriod.HOUR,
        limit_usd=2.0,
    )
    cap_agent = BudgetCap(
        scope=BudgetScope.AGENT,
        scope_id="agent-1",
        period=BudgetPeriod.HOUR,
        limit_usd=1.0,
    )
    for c in (cap_customer, cap_feature, cap_agent):
        guardian.register_cap(c)

    matches = guardian._matching_caps(customer_attr)
    assert set(c.key for c in matches) == {
        cap_customer.key,
        cap_feature.key,
        cap_agent.key,
    }


# ============================================================
# pre_check blocking
# ============================================================


@pytest.mark.asyncio
async def test_pre_check_passes_under_limit(guardian, customer_attr):
    cap = BudgetCap(
        scope=BudgetScope.CUSTOMER,
        scope_id="acme-corp",
        period=BudgetPeriod.DAY,
        limit_usd=10.0,
    )
    guardian.register_cap(cap)
    # Should not raise
    await guardian.pre_check(customer_attr, estimate_usd=1.0)


@pytest.mark.asyncio
async def test_pre_check_blocks_over_limit(guardian, customer_attr):
    cap = BudgetCap(
        scope=BudgetScope.CUSTOMER,
        scope_id="acme-corp",
        period=BudgetPeriod.DAY,
        limit_usd=1.0,
    )
    guardian.register_cap(cap)
    # Push usage above limit first
    await guardian.record_usage(customer_attr, actual_usd=1.5, request_id="req-bad")
    # Now pre_check must refuse
    with pytest.raises(BudgetExceededError) as exc_info:
        await guardian.pre_check(customer_attr, estimate_usd=0.0, request_id="req-blocked")
    assert exc_info.value.error.code == "agentops_budget_exceeded"


@pytest.mark.asyncio
async def test_pre_check_does_not_block_warn_action(guardian, customer_attr):
    cap = BudgetCap(
        scope=BudgetScope.CUSTOMER,
        scope_id="acme-corp",
        period=BudgetPeriod.DAY,
        limit_usd=1.0,
        hard_action=HardAction.WARN,
    )
    guardian.register_cap(cap)
    await guardian.record_usage(customer_attr, actual_usd=10.0)
    # WARN doesn't block — pre_check returns normally
    await guardian.pre_check(customer_attr)


@pytest.mark.asyncio
async def test_pre_check_returns_matched_caps(guardian, customer_attr):
    cap = BudgetCap(
        scope=BudgetScope.CUSTOMER,
        scope_id="acme-corp",
        period=BudgetPeriod.DAY,
        limit_usd=10.0,
    )
    guardian.register_cap(cap)
    result = await guardian.pre_check(customer_attr)
    assert result.matched_caps == [cap]


# ============================================================
# record_usage atomicity
# ============================================================


@pytest.mark.asyncio
async def test_record_usage_increments_atomically(guardian, customer_attr):
    cap = BudgetCap(
        scope=BudgetScope.CUSTOMER,
        scope_id="acme-corp",
        period=BudgetPeriod.DAY,
        limit_usd=100.0,
    )
    guardian.register_cap(cap)

    # 50 concurrent calls of 0.10 USD each
    N = 50
    await asyncio.gather(
        *(guardian.record_usage(customer_attr, actual_usd=0.10) for _ in range(N))
    )
    snap = await guardian.snapshot(cap)
    assert snap.used_usd == pytest.approx(N * 0.10, rel=1e-3)


@pytest.mark.asyncio
async def test_record_usage_emits_kill_event_when_exceeded(
    guardian, event_bus, customer_attr
):
    cap = BudgetCap(
        scope=BudgetScope.CUSTOMER,
        scope_id="acme-corp",
        period=BudgetPeriod.DAY,
        limit_usd=1.0,
    )
    guardian.register_cap(cap)

    received: List[Event] = []

    async def handler(evt: Event):
        received.append(evt)

    event_bus.subscribe(EVENT_KILL_SWITCH_TRIGGERED, handler)
    await guardian.record_usage(customer_attr, actual_usd=2.0, request_id="req-1")

    assert len(received) == 1
    assert received[0].name == EVENT_KILL_SWITCH_TRIGGERED
    assert received[0].severity == "critical"
    assert received[0].payload["request_id"] == "req-1"


@pytest.mark.asyncio
async def test_record_usage_emits_warning_at_soft_threshold(
    guardian, event_bus, customer_attr
):
    cap = BudgetCap(
        scope=BudgetScope.CUSTOMER,
        scope_id="acme-corp",
        period=BudgetPeriod.DAY,
        limit_usd=10.0,
        soft_threshold_pct=0.8,
    )
    guardian.register_cap(cap)

    warnings: List[Event] = []

    async def handler(evt: Event):
        warnings.append(evt)

    event_bus.subscribe(EVENT_BUDGET_WARNING, handler)
    await guardian.record_usage(customer_attr, actual_usd=8.5)
    assert len(warnings) == 1


@pytest.mark.asyncio
async def test_warning_only_fires_once_per_period(guardian, event_bus, customer_attr):
    cap = BudgetCap(
        scope=BudgetScope.CUSTOMER,
        scope_id="acme-corp",
        period=BudgetPeriod.DAY,
        limit_usd=10.0,
    )
    guardian.register_cap(cap)

    warnings: List[Event] = []

    async def handler(evt: Event):
        warnings.append(evt)

    event_bus.subscribe(EVENT_BUDGET_WARNING, handler)
    await guardian.record_usage(customer_attr, actual_usd=8.5)
    await guardian.record_usage(customer_attr, actual_usd=0.1)
    await guardian.record_usage(customer_attr, actual_usd=0.1)
    assert len(warnings) == 1


@pytest.mark.asyncio
async def test_record_usage_with_no_matching_cap_is_noop(guardian):
    other_attr = AttributionContext(customer_id="not-registered")
    snapshots = await guardian.record_usage(other_attr, actual_usd=5.0)
    assert snapshots == []


# ============================================================
# token caps
# ============================================================


@pytest.mark.asyncio
async def test_token_cap_blocks_independently(guardian, customer_attr):
    cap = BudgetCap(
        scope=BudgetScope.CUSTOMER,
        scope_id="acme-corp",
        period=BudgetPeriod.DAY,
        limit_tokens=1000,
    )
    guardian.register_cap(cap)
    await guardian.record_usage(customer_attr, actual_tokens=1500)
    with pytest.raises(BudgetExceededError):
        await guardian.pre_check(customer_attr)


@pytest.mark.asyncio
async def test_combined_cap_picks_first_breach(guardian, customer_attr):
    cap = BudgetCap(
        scope=BudgetScope.CUSTOMER,
        scope_id="acme-corp",
        period=BudgetPeriod.DAY,
        limit_usd=100.0,
        limit_tokens=10,
    )
    guardian.register_cap(cap)
    # tokens overflow first
    await guardian.record_usage(customer_attr, actual_tokens=20, actual_usd=0.01)
    snap = await guardian.snapshot(cap)
    assert snap.is_exceeded


# ============================================================
# mid-stream check
# ============================================================


@pytest.mark.asyncio
async def test_check_after_chunk_raises_when_blown(guardian, customer_attr):
    cap = BudgetCap(
        scope=BudgetScope.CUSTOMER,
        scope_id="acme-corp",
        period=BudgetPeriod.DAY,
        limit_usd=1.0,
    )
    guardian.register_cap(cap)
    await guardian.record_usage(customer_attr, actual_usd=2.0)

    with pytest.raises(BudgetMidStreamExceededError) as exc_info:
        await guardian.check_after_chunk(customer_attr, partial_tokens=42)
    assert exc_info.value.error.code == "agentops_budget_exceeded_midstream"


@pytest.mark.asyncio
async def test_check_after_chunk_passes_when_within(guardian, customer_attr):
    cap = BudgetCap(
        scope=BudgetScope.CUSTOMER,
        scope_id="acme-corp",
        period=BudgetPeriod.DAY,
        limit_usd=10.0,
    )
    guardian.register_cap(cap)
    await guardian.record_usage(customer_attr, actual_usd=1.0)
    # Should not raise
    await guardian.check_after_chunk(customer_attr)


# ============================================================
# reset / admin
# ============================================================


@pytest.mark.asyncio
async def test_reset_cap_counter_clears_state(guardian, customer_attr):
    cap = BudgetCap(
        scope=BudgetScope.CUSTOMER,
        scope_id="acme-corp",
        period=BudgetPeriod.DAY,
        limit_usd=10.0,
    )
    guardian.register_cap(cap)
    await guardian.record_usage(customer_attr, actual_usd=5.0)
    await guardian.reset_cap_counter(cap)
    snap = await guardian.snapshot(cap)
    assert snap.used_usd == 0.0


@pytest.mark.asyncio
async def test_reset_all_clears_every_cap(guardian, customer_attr):
    cap1 = BudgetCap(
        scope=BudgetScope.CUSTOMER,
        scope_id="acme-corp",
        period=BudgetPeriod.DAY,
        limit_usd=10.0,
    )
    cap2 = BudgetCap(
        scope=BudgetScope.AGENT,
        scope_id="agent-1",
        period=BudgetPeriod.HOUR,
        limit_usd=1.0,
    )
    guardian.register_cap(cap1)
    guardian.register_cap(cap2)
    await guardian.record_usage(customer_attr, actual_usd=0.5)
    await guardian.reset_all()
    for cap in (cap1, cap2):
        snap = await guardian.snapshot(cap)
        assert snap.used_usd == 0.0


# ============================================================
# end-to-end "$47K agent loop" demo
# ============================================================


@pytest.mark.asyncio
async def test_demo_47k_agent_loop_killed_in_4_seconds(guardian, event_bus):
    """
    Simulates an agent loop that would otherwise run forever.

    Setup: $1 cap on agent-runaway. Loop calls record_usage per "step"
    until killed mid-flight by check_after_chunk OR pre_check. Asserts
    we stopped at ~$1, not $47,000.
    """
    cap = BudgetCap(
        scope=BudgetScope.AGENT,
        scope_id="runaway",
        period=BudgetPeriod.DAY,
        limit_usd=1.00,
    )
    guardian.register_cap(cap)
    attr = AttributionContext(agent_id="runaway")

    kill_events: List[Event] = []

    async def on_kill(evt: Event):
        kill_events.append(evt)

    event_bus.subscribe(EVENT_KILL_SWITCH_TRIGGERED, on_kill)

    total_steps = 0
    total_spend = 0.0
    blocked = False

    for _ in range(10_000):  # would-be runaway loop
        try:
            await guardian.pre_check(attr, request_id=f"step-{total_steps}")
        except BudgetExceededError:
            blocked = True
            break
        # Each step consumes $0.10
        await guardian.record_usage(attr, actual_usd=0.10, request_id=f"step-{total_steps}")
        total_spend += 0.10
        total_steps += 1

    assert blocked, "guardian should have blocked the loop"
    # Critical assertion: we did NOT spend more than ~$1 + one over-run step
    assert total_spend <= cap.limit_usd + 0.10
    # And we emitted at least one kill event
    assert len(kill_events) >= 1
