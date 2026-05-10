"""
Tests for the usage tracker DLQ + attribution dimensions.
"""

from __future__ import annotations

import asyncio
from typing import List

import pytest

from src.usage.tracker import (
    OperationType,
    UsageRecord,
    UsageTracker,
)


# ============================================================
# attribution dimensions
# ============================================================


@pytest.mark.asyncio
async def test_start_tracking_records_attribution():
    tracker_svc = UsageTracker()
    rt = tracker_svc.start_tracking(
        request_id="req-1",
        tenant_id="t-1",
        api_key_id="k-1",
        customer_id="cust",
        feature_id="feat",
        agent_id="agent",
        session_id="sess",
        model="openai/gpt-4o-mini",
    )
    assert rt.customer_id == "cust"
    assert rt.feature_id == "feat"
    assert rt.agent_id == "agent"
    assert rt.session_id == "sess"


@pytest.mark.asyncio
async def test_complete_tracking_propagates_attribution_to_record():
    tracker_svc = UsageTracker()
    rt = tracker_svc.start_tracking(
        request_id="req-2",
        tenant_id="t-1",
        customer_id="cust-a",
        feature_id="feat-b",
        model="openai/gpt-4o-mini",
    )
    rt.add_tokens(input_tokens=100, output_tokens=50, reasoning_tokens=200)
    record = await tracker_svc.complete_tracking(rt)
    assert record.customer_id == "cust-a"
    assert record.feature_id == "feat-b"
    assert record.reasoning_tokens == 200


@pytest.mark.asyncio
async def test_get_attribution_usage_aggregates():
    tracker_svc = UsageTracker()
    for i in range(3):
        rt = tracker_svc.start_tracking(
            request_id=f"r-{i}",
            tenant_id="t-1",
            customer_id="cust-x",
            feature_id="feat-y",
            model="openai/gpt-4o-mini",
        )
        rt.add_tokens(input_tokens=100, output_tokens=50)
        await tracker_svc.complete_tracking(rt)

    rows = tracker_svc.get_attribution_usage(
        tenant_id="t-1", customer_id="cust-x", feature_id="feat-y"
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["customer_id"] == "cust-x"
    assert row["feature_id"] == "feat-y"
    assert row["request_count"] == 3
    assert row["total_tokens"] == 3 * 150


@pytest.mark.asyncio
async def test_get_attribution_usage_filters_by_dimensions():
    tracker_svc = UsageTracker()
    for cust in ("a", "b"):
        rt = tracker_svc.start_tracking(
            request_id=f"r-{cust}",
            tenant_id="t-1",
            customer_id=cust,
            model="openai/gpt-4o-mini",
        )
        rt.add_tokens(input_tokens=10, output_tokens=10)
        await tracker_svc.complete_tracking(rt)

    only_a = tracker_svc.get_attribution_usage(customer_id="a")
    assert len(only_a) == 1
    assert only_a[0]["customer_id"] == "a"


@pytest.mark.asyncio
async def test_attribution_aggregate_skipped_when_no_dimensions():
    tracker_svc = UsageTracker()
    rt = tracker_svc.start_tracking(
        request_id="r-bare",
        tenant_id="t-1",
        model="openai/gpt-4o-mini",
    )
    rt.add_tokens(input_tokens=10, output_tokens=10)
    await tracker_svc.complete_tracking(rt)

    # Tenant aggregate captured ...
    tenant_usage = tracker_svc.get_tenant_usage("t-1")
    assert tenant_usage["request_count"] == 1
    # ... but no per-customer row created
    assert tracker_svc.get_attribution_usage(tenant_id="t-1") == []


# ============================================================
# DLQ
# ============================================================


@pytest.mark.asyncio
async def test_failed_storage_callback_lands_in_dlq():
    failures: List[UsageRecord] = []

    async def failing_storage(record: UsageRecord):
        failures.append(record)
        raise RuntimeError("simulated DB outage")

    tracker_svc = UsageTracker(
        storage_callback=failing_storage,
        buffer_size=1,  # flush after every record
    )
    rt = tracker_svc.start_tracking(
        request_id="r-fail",
        tenant_id="t-1",
        model="openai/gpt-4o-mini",
    )
    rt.add_tokens(input_tokens=5, output_tokens=5)
    await tracker_svc.complete_tracking(rt)

    assert tracker_svc.dlq_size() == 1
    assert len(failures) == 1


@pytest.mark.asyncio
async def test_dlq_retry_succeeds_when_storage_recovers():
    fail_count = {"n": 0}
    stored: List[UsageRecord] = []

    async def flaky_storage(record: UsageRecord):
        if fail_count["n"] < 1:
            fail_count["n"] += 1
            raise RuntimeError("not yet")
        stored.append(record)

    tracker_svc = UsageTracker(storage_callback=flaky_storage, buffer_size=1)
    rt = tracker_svc.start_tracking(request_id="r-retry", tenant_id="t-1")
    rt.add_tokens(input_tokens=5, output_tokens=5)
    await tracker_svc.complete_tracking(rt)

    assert tracker_svc.dlq_size() == 1
    succeeded = await tracker_svc.retry_dlq()
    assert succeeded == 1
    assert tracker_svc.dlq_size() == 0
    assert len(stored) == 1


@pytest.mark.asyncio
async def test_dlq_caps_size_when_overflowing():
    async def always_fail(record: UsageRecord):
        raise RuntimeError("forever")

    tracker_svc = UsageTracker(
        storage_callback=always_fail,
        buffer_size=1,
        dlq_max_size=5,
    )
    for i in range(20):
        rt = tracker_svc.start_tracking(
            request_id=f"r-{i}",
            tenant_id="t-1",
        )
        rt.add_tokens(input_tokens=1, output_tokens=1)
        await tracker_svc.complete_tracking(rt)

    assert tracker_svc.dlq_size() == 5


@pytest.mark.asyncio
async def test_no_storage_callback_drops_buffer_silently():
    tracker_svc = UsageTracker(storage_callback=None, buffer_size=1)
    rt = tracker_svc.start_tracking(request_id="r", tenant_id="t-1")
    rt.add_tokens(input_tokens=1, output_tokens=1)
    await tracker_svc.complete_tracking(rt)
    # No DLQ entries, no exceptions
    assert tracker_svc.dlq_size() == 0


# ============================================================
# UsageRecord serialization includes new fields
# ============================================================


def test_record_to_dict_includes_attribution_and_reasoning():
    record = UsageRecord(
        request_id="r1",
        tenant_id="t",
        customer_id="cust",
        feature_id="feat",
        agent_id="ag",
        session_id="sess",
        reasoning_tokens=42,
    )
    d = record.to_dict()
    assert d["customer_id"] == "cust"
    assert d["feature_id"] == "feat"
    assert d["agent_id"] == "ag"
    assert d["session_id"] == "sess"
    assert d["reasoning_tokens"] == 42
