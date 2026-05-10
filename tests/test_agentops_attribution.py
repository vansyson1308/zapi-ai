"""
Tests for src/agentops/attribution.py — header parsing + safety.
"""

from __future__ import annotations

import pytest

from src.agentops.attribution import (
    MAX_ID_LENGTH,
    attribution_context_from_headers,
    extend_usage_record,
)
from src.agentops.models import AttributionContext


def test_extracts_basic_headers():
    ctx = attribution_context_from_headers(
        {
            "X-Customer-Id": "cust-123",
            "X-Feature-Id": "chatbot",
            "X-Agent-Id": "agent-7",
        }
    )
    assert ctx.customer_id == "cust-123"
    assert ctx.feature_id == "chatbot"
    assert ctx.agent_id == "agent-7"


def test_lowercase_header_keys_work():
    ctx = attribution_context_from_headers({"x-customer-id": "abc"})
    assert ctx.customer_id == "abc"


def test_session_id_extracted():
    ctx = attribution_context_from_headers({"X-Session-Id": "sess-1"})
    assert ctx.session_id == "sess-1"


def test_extra_headers_via_x_attr():
    ctx = attribution_context_from_headers(
        {
            "X-Attr-Region": "us-east-1",
            "X-Attr-Tier": "pro",
        }
    )
    assert ctx.extra == {"region": "us-east-1", "tier": "pro"}


def test_blank_headers_yield_none():
    ctx = attribution_context_from_headers(
        {"X-Customer-Id": "  ", "X-Feature-Id": ""}
    )
    assert ctx.customer_id is None
    assert ctx.feature_id is None


def test_overlong_id_is_truncated():
    long = "a" * (MAX_ID_LENGTH + 100)
    ctx = attribution_context_from_headers({"X-Customer-Id": long})
    assert ctx.customer_id is not None
    assert len(ctx.customer_id) <= MAX_ID_LENGTH


def test_unsafe_chars_are_sanitized():
    ctx = attribution_context_from_headers({"X-Customer-Id": "abc;DROP TABLE"})
    # Spaces and semicolons replaced
    assert ctx.customer_id is not None
    assert ";" not in ctx.customer_id
    assert " " not in ctx.customer_id


def test_safe_chars_preserved():
    safe = "tenant_123.acme:prod-1"
    ctx = attribution_context_from_headers({"X-Customer-Id": safe})
    assert ctx.customer_id == safe


def test_passes_tenant_and_api_key_through():
    ctx = attribution_context_from_headers(
        {},
        tenant_id="t-1",
        api_key_id="k-1",
    )
    assert ctx.tenant_id == "t-1"
    assert ctx.api_key_id == "k-1"


def test_empty_headers_returns_empty_context():
    ctx = attribution_context_from_headers({})
    assert ctx.is_empty()


def test_attribution_context_is_empty():
    assert AttributionContext().is_empty()
    assert not AttributionContext(customer_id="x").is_empty()
    assert not AttributionContext(tenant_id="t").is_empty()


def test_extend_usage_record_attaches_attribution():
    metadata = {"existing": True}
    ctx = AttributionContext(customer_id="c1", feature_id="f1")
    out = extend_usage_record(metadata, ctx)
    assert out is metadata
    assert metadata["attribution"]["customer_id"] == "c1"
    assert metadata["attribution"]["feature_id"] == "f1"


def test_extend_usage_record_skips_empty_attribution():
    metadata = {"existing": True}
    out = extend_usage_record(metadata, AttributionContext())
    assert "attribution" not in out
