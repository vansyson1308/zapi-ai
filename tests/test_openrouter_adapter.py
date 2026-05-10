"""
Tests for src/adapters/openrouter_adapter.py — passthrough behavior.

We don't hit the network; instead we exercise the model rewriting and
catalog logic.
"""

from __future__ import annotations

import pytest

from src.adapters.base import AdapterConfig
from src.adapters.openrouter_adapter import OpenRouterAdapter
from src.core.models import Provider


@pytest.fixture
def adapter():
    return OpenRouterAdapter(AdapterConfig(api_key="sk-test"))


def test_provider_is_openrouter(adapter):
    assert adapter.provider == Provider.OPENROUTER


def test_default_base_url_points_to_openrouter():
    assert OpenRouterAdapter.DEFAULT_BASE_URL == "https://openrouter.ai/api/v1"


def test_strip_provider_prefix_drops_openrouter_namespace():
    assert (
        OpenRouterAdapter._strip_provider_prefix("openrouter/anthropic/claude-3.5-sonnet")
        == "anthropic/claude-3.5-sonnet"
    )


def test_strip_provider_prefix_passthrough_when_no_prefix():
    assert (
        OpenRouterAdapter._strip_provider_prefix("anthropic/claude-3.5-sonnet")
        == "anthropic/claude-3.5-sonnet"
    )


def test_models_catalog_uses_openrouter_provider(adapter):
    for model in adapter.list_models():
        assert model.provider == Provider.OPENROUTER
        assert model.id.startswith("openrouter/")


def test_models_catalog_has_claude_and_gpt4(adapter):
    ids = {m.id for m in adapter.list_models()}
    assert "openrouter/anthropic/claude-3.5-sonnet" in ids
    assert "openrouter/openai/gpt-4o" in ids


def test_factory_returns_openrouter_adapter():
    from src.adapters import get_adapter

    a = get_adapter("openrouter", AdapterConfig(api_key="sk"))
    assert isinstance(a, OpenRouterAdapter)


def test_factory_unknown_provider_raises():
    from src.adapters import get_adapter

    with pytest.raises(ValueError):
        get_adapter("not-a-real-provider", AdapterConfig(api_key="x"))


def test_build_chat_payload_rewrites_model(adapter):
    from src.core.models import ChatCompletionRequest, Message, Role

    request = ChatCompletionRequest(
        model="openrouter/anthropic/claude-3.5-sonnet",
        messages=[Message(role=Role.USER, content="hi")],
    )
    payload = adapter._build_chat_payload(request)
    # The 'openrouter/' prefix must be stripped before sending upstream
    assert payload["model"] == "anthropic/claude-3.5-sonnet"
