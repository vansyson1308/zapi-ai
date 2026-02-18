"""Integration tests for unified streaming orchestration and normalization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import AsyncIterator, Dict, List
from uuid import UUID

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.adapters.base import AdapterConfig, BaseAdapter, ProviderHealth
from src.api.routes.chat import router as chat_router
from src.auth.middleware import get_auth_context
from src.core.errors import ErrorDetails, ErrorType, InfraError
from src.core.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ModelInfo,
    ModelPricing,
    Provider,
)
from src.db.models import APIKey, AuthContext, Tenant
from src.routing.router import Router
from src.server import twoapi_exception_handler


@dataclass
class _StreamSpec:
    mode: str  # openai | google
    fail_before_content: bool = False
    fail_after_first_chunk: bool = False


class _MockStreamAdapter(BaseAdapter):
    def __init__(self, provider: Provider, spec: _StreamSpec):
        super().__init__(AdapterConfig(api_key="stub"))
        self.provider = provider
        self.spec = spec

    async def chat_completion(self, request: ChatCompletionRequest, request_id: str = "") -> ChatCompletionResponse:
        raise NotImplementedError

    async def chat_completion_stream(self, request: ChatCompletionRequest, request_id: str = "") -> AsyncIterator[str]:
        if self.spec.fail_before_content:
            raise InfraError(
                ErrorDetails(
                    code="upstream_503",
                    message=f"{self.provider.value} unavailable",
                    type=ErrorType.INFRA,
                    provider=self.provider.value,
                    request_id=request_id,
                    retryable=True,
                ),
                status_code=503,
            )

        if self.spec.mode == "openai":
            chunk = {
                "id": "chatcmpl-openai",
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": "hello"}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
        elif self.spec.mode == "google":
            google_chunk = {
                "candidates": [
                    {
                        "content": {"parts": [{"text": "hello"}]},
                        "finishReason": "STOP",
                    }
                ]
            }
            yield f"data: {json.dumps(google_chunk)}\n\n"
        else:
            raise RuntimeError("invalid mode")

        if self.spec.fail_after_first_chunk:
            raise InfraError(
                ErrorDetails(
                    code="stream_broken",
                    message="broken after content",
                    type=ErrorType.INFRA,
                    provider=self.provider.value,
                    request_id=request_id,
                    retryable=False,
                ),
                status_code=502,
            )

        yield "data: [DONE]\n\n"

    async def embedding(self, request: EmbeddingRequest, request_id: str = "") -> EmbeddingResponse:
        raise NotImplementedError

    async def image_generation(self, request: ImageGenerationRequest, request_id: str = "") -> ImageGenerationResponse:
        raise NotImplementedError

    def list_models(self) -> List[ModelInfo]:
        return [
            ModelInfo(
                id=f"{self.provider.value}/mock-chat",
                provider=self.provider,
                name="mock-chat",
                capabilities=["chat"],
                context_window=8000,
                max_output_tokens=1024,
                pricing=ModelPricing(input_per_1m_tokens=0.1, output_per_1m_tokens=0.2),
            )
        ]

    async def health_check(self) -> ProviderHealth:
        return ProviderHealth(provider=self.provider, is_healthy=True)



def _auth() -> AuthContext:
    tenant_id = UUID("00000000-0000-0000-0000-000000000101")
    key_id = UUID("00000000-0000-0000-0000-000000000102")
    tenant = Tenant(id=tenant_id, name="stream", email="s@test", plan="pro", is_active=True, created_at=datetime.utcnow())
    key = APIKey(id=key_id, tenant_id=tenant_id, key_hash="h", key_prefix="2api_", name="k", permissions=["*"], rate_limit_per_minute=999, is_active=True, created_at=datetime.utcnow())
    return AuthContext(tenant_id=tenant_id, api_key_id=key_id, tenant=tenant, api_key=key, permissions=["*"], rate_limit_per_minute=999, request_id="req_stream", trace_id="trace_stream")



def _collect_data_lines(lines: List[str]) -> List[str]:
    out: List[str] = []
    for item in lines:
        for ln in item.splitlines():
            if ln.startswith("data: "):
                out.append(ln[6:])
    return out



def test_api_streaming_uses_router_orchestration_not_direct_adapter():
    class _RouterOnly:
        adapters: Dict[Provider, BaseAdapter] = {}

        async def route_chat_stream(self, request, request_id):
            yield "data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"via-router\"},\"finish_reason\":null}]}\n\n"
            yield "data: [DONE]\n\n"

    app = FastAPI()
    app.add_exception_handler(Exception, twoapi_exception_handler)
    app.include_router(chat_router)
    from src.api.dependencies import get_router
    app.dependency_overrides[get_auth_context] = _auth
    app.dependency_overrides[get_router] = lambda: _RouterOnly()

    client = TestClient(app)
    with client.stream("POST", "/v1/chat/completions", headers={"Authorization": "Bearer 2api_stream"}, json={"model": "auto", "messages": [{"role": "user", "content": "hi"}], "stream": True}) as resp:
        assert resp.status_code == 200
        payloads = _collect_data_lines([ln for ln in resp.iter_lines() if ln])

    assert payloads[-1] == "[DONE]"
    assert any("via-router" in p for p in payloads)


@pytest.mark.asyncio
async def test_router_stream_normalizes_two_providers_to_same_shape():
    router_openai = Router({Provider.OPENAI: _MockStreamAdapter(Provider.OPENAI, _StreamSpec(mode="openai"))})
    router_google = Router({Provider.GOOGLE: _MockStreamAdapter(Provider.GOOGLE, _StreamSpec(mode="google"))})

    req_openai = ChatCompletionRequest(model="openai/mock-chat", messages=[] , stream=True)
    req_google = ChatCompletionRequest(model="google/mock-chat", messages=[] , stream=True)

    openai_events = [c async for c in router_openai.route_chat_stream(req_openai, "req_a")]
    google_events = [c async for c in router_google.route_chat_stream(req_google, "req_b")]

    openai_payloads = _collect_data_lines([e.strip() for e in openai_events if e.strip()])
    google_payloads = _collect_data_lines([e.strip() for e in google_events if e.strip()])

    assert openai_payloads[-1] == "[DONE]"
    assert google_payloads[-1] == "[DONE]"

    o_first = json.loads(openai_payloads[0])
    g_first = json.loads(google_payloads[0])
    assert o_first["object"] == "chat.completion.chunk"
    assert g_first["object"] == "chat.completion.chunk"
    assert o_first["choices"][0]["delta"]["content"] == "hello"
    assert g_first["choices"][0]["delta"]["content"] == "hello"


@pytest.mark.asyncio
async def test_router_stream_fallback_before_first_token():
    router = Router(
        {
            Provider.OPENAI: _MockStreamAdapter(Provider.OPENAI, _StreamSpec(mode="openai", fail_before_content=True)),
            Provider.GOOGLE: _MockStreamAdapter(Provider.GOOGLE, _StreamSpec(mode="google")),
        }
    )

    req = ChatCompletionRequest(
        model="openai/mock-chat",
        messages=[],
        stream=True,
        routing=type("Routing", (), {"strategy": None, "fallback": ["google/mock-chat"], "max_latency_ms": None, "max_cost": None})(),
    )

    events = [c async for c in router.route_chat_stream(req, "req_fallback")]
    payloads = _collect_data_lines([e.strip() for e in events if e.strip()])

    assert payloads[-1] == "[DONE]"
    assert any("hello" in p for p in payloads)


@pytest.mark.asyncio
async def test_router_stream_error_chunk_semantics_after_content():
    router = Router(
        {Provider.OPENAI: _MockStreamAdapter(Provider.OPENAI, _StreamSpec(mode="openai", fail_after_first_chunk=True))}
    )
    req = ChatCompletionRequest(model="openai/mock-chat", messages=[], stream=True)

    events = [c async for c in router.route_chat_stream(req, "req_err")]
    payloads = _collect_data_lines([e.strip() for e in events if e.strip()])

    assert payloads[-1] == "[DONE]"
    error_payload = json.loads(payloads[-2])
    assert "error" in error_payload
    assert error_payload["choices"][0]["finish_reason"] == "error"
