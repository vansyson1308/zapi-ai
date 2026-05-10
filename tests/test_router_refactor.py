"""
Tests covering the router.py refactor.

Goal: prove that splitting the god class into ProviderSelector,
StreamingManager, FallbackOrchestrator preserved the original behavior.

We test the EXTRACTED components in isolation here. The existing
test_routing_system.py + test_streaming_integration.py + test_smoke_api.py
tests still cover the Router facade end-to-end.
"""

from __future__ import annotations

from typing import List

import pytest

from src.adapters.base import AdapterConfig
from src.adapters.stub_adapter import StubAdapter
from src.core.errors import AllProvidersFailedError
from src.core.models import (
    ChatCompletionRequest,
    Message,
    Provider,
    Role,
    RoutingConfig,
    RoutingStrategy,
)
from src.routing.circuit_breaker import CircuitBreakerRegistry
from src.routing.fallback import RequestPhaseTracker
from src.routing.health import HealthRegistry
from src.routing.provider_selector import ProviderSelector, RoutingResult
from src.routing.streaming_manager import StreamingManager
from src.routing.router import Router


# ============================================================
# fixtures
# ============================================================


def _make_chat_request(model: str = "auto") -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model=model,
        messages=[Message(role=Role.USER, content="hi")],
    )


@pytest.fixture
def stub_adapters():
    """One stub adapter for the OPENAI provider — enough for selection tests."""
    return {
        Provider.OPENAI: StubAdapter(AdapterConfig(api_key="stub")),
    }


@pytest.fixture
def selector(stub_adapters):
    cb = CircuitBreakerRegistry()
    health = HealthRegistry()
    return ProviderSelector(
        adapters=stub_adapters,
        circuit_breakers=cb,
        health_registry=health,
    )


# ============================================================
# ProviderSelector
# ============================================================


def test_selector_picks_explicit_provider_when_specified(selector):
    request = _make_chat_request(model="openai/gpt-4o-mini")
    result = selector.select(request=request, capability="chat")
    assert isinstance(result, RoutingResult)
    assert result.selected_provider == Provider.OPENAI
    assert result.decision.strategy_used == "explicit"
    assert result.decision.fallback_used is False


def test_selector_falls_back_to_strategy_for_auto(selector):
    request = _make_chat_request(model="auto")
    result = selector.select(request=request, capability="chat")
    assert result.selected_provider == Provider.OPENAI
    assert result.decision.strategy_used != "explicit"
    assert not result.decision.fallback_used


def test_selector_uses_strategy_when_routing_specified(selector):
    request = ChatCompletionRequest(
        model="auto",
        messages=[Message(role=Role.USER, content="hi")],
        routing=RoutingConfig(strategy=RoutingStrategy.LATENCY),
    )
    result = selector.select(request=request, capability="chat")
    assert result.decision.strategy_used == RoutingStrategy.LATENCY.value


def test_selector_raises_when_no_candidates(stub_adapters):
    """Empty adapter dict + capability filter → AllProvidersFailedError."""
    cb = CircuitBreakerRegistry()
    health = HealthRegistry()
    sel = ProviderSelector(
        adapters={},
        circuit_breakers=cb,
        health_registry=health,
    )
    with pytest.raises(AllProvidersFailedError):
        sel.select(request=_make_chat_request(), capability="chat")


def test_selector_lists_all_models_dedup(selector):
    models = selector.list_all_models()
    seen_ids = [m.id for m in models]
    assert len(seen_ids) == len(set(seen_ids))


# ============================================================
# Router (facade) — public API still works
# ============================================================


@pytest.mark.asyncio
async def test_router_facade_routes_chat_via_stub(stub_adapters):
    router = Router(stub_adapters)
    request = _make_chat_request("openai/gpt-4o-mini")
    response, decision = await router.route_chat(request)
    assert response.choices[0].message.content
    assert decision.strategy_used == "explicit"


@pytest.mark.asyncio
async def test_router_facade_get_stats_shape(stub_adapters):
    router = Router(stub_adapters)
    request = _make_chat_request("openai/gpt-4o-mini")
    await router.route_chat(request)
    stats = router.get_stats()
    assert "openai" in stats
    assert "total_requests" in stats["openai"]
    assert "circuit_state" in stats["openai"]


@pytest.mark.asyncio
async def test_router_legacy_select_provider_method_still_works(stub_adapters):
    """We exposed `_select_provider` as a shim on Router for old tests."""
    router = Router(stub_adapters)
    request = _make_chat_request("openai/gpt-4o-mini")
    result = router._select_provider(request=request, capability="chat")
    assert isinstance(result, RoutingResult)
    assert result.selected_provider == Provider.OPENAI


# ============================================================
# StreamingManager — record_success/record_failure callbacks
# ============================================================


def test_streaming_manager_normalize_chunk_handles_done():
    """Sanity: [DONE] is propagated as a single SSE event."""
    mgr = StreamingManager(
        record_success=lambda *_args, **_kw: None,
        record_failure=lambda *_args, **_kw: None,
    )
    from src.streaming.normalizer import StreamNormalizer
    from src.streaming.tool_calls import ToolCallStreamTracker

    normalizer = StreamNormalizer(model="m", provider="openai", request_id="req")
    events = mgr._normalize_chunk(
        "data: [DONE]\n",
        normalizer,
        "openai",
        ToolCallStreamTracker(),
    )
    assert events == ["data: [DONE]\n\n"]


def test_streaming_manager_normalize_chunk_skips_blank_lines():
    mgr = StreamingManager(
        record_success=lambda *_args, **_kw: None,
        record_failure=lambda *_args, **_kw: None,
    )
    from src.streaming.normalizer import StreamNormalizer
    from src.streaming.tool_calls import ToolCallStreamTracker

    normalizer = StreamNormalizer(model="m", provider="openai", request_id="r")
    events = mgr._normalize_chunk(
        "\n\n",
        normalizer,
        "openai",
        ToolCallStreamTracker(),
    )
    assert events == []


def test_streaming_manager_normalize_chunk_ignores_malformed_json():
    mgr = StreamingManager(
        record_success=lambda *_args, **_kw: None,
        record_failure=lambda *_args, **_kw: None,
    )
    from src.streaming.normalizer import StreamNormalizer
    from src.streaming.tool_calls import ToolCallStreamTracker

    normalizer = StreamNormalizer(model="m", provider="openai", request_id="r")
    events = mgr._normalize_chunk(
        "data: not valid json",
        normalizer,
        "openai",
        ToolCallStreamTracker(),
    )
    assert events == []


# ============================================================
# Router exports backward compat
# ============================================================


def test_routing_module_still_exports_routing_result():
    from src.routing import RoutingResult as ImportedFromInit
    from src.routing.router import RoutingResult as ImportedFromRouter

    assert ImportedFromInit is ImportedFromRouter
