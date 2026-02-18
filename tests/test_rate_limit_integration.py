"""Integration tests for end-to-end rate limiting on costly endpoints."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

import pytest
from fastapi import FastAPI, Header
from fastapi.testclient import TestClient

from src.api.dependencies import get_router
from src.api.routes.chat import router as chat_router
from src.api.routes.embeddings import router as embeddings_router
from src.api.routes.images import router as images_router
from src.auth.middleware import get_auth_context
from src.core.errors import TwoApiException
from src.server import twoapi_exception_handler
from src.core.models import (
    ChatCompletionResponse,
    EmbeddingData,
    EmbeddingResponse,
    FinishReason,
    ImageData,
    ImageGenerationResponse,
    Usage,
)
from src.db.models import APIKey, AuthContext, Tenant
from src.usage import limits as limits_module
from src.usage.limits import UsageLimiter
from src.usage import QuotaConfig, UsageLimit, LimitType, LimitPeriod, set_quota


class _FakeRouter:
    async def route_chat(self, request):
        response = ChatCompletionResponse.create(
            content="ok",
            model="gpt-4o-mini",
            provider="openai",
            usage=Usage(prompt_tokens=5, completion_tokens=2),
            finish_reason=FinishReason.STOP,
        )
        decision = type("Decision", (), {"strategy_used": "cost", "fallback_used": False, "candidates_evaluated": []})
        return response, decision

    async def route_embedding(self, request):
        response = EmbeddingResponse(
            data=[EmbeddingData(embedding=[0.1, 0.2, 0.3], index=0)],
            model=request.model,
            usage=Usage(prompt_tokens=3, completion_tokens=0),
        )
        decision = type("Decision", (), {"strategy_used": "cost", "fallback_used": False, "candidates_evaluated": []})
        return response, decision

    async def route_image(self, request):
        response = ImageGenerationResponse(data=[ImageData(url="https://example.test/i.png")])
        decision = type("Decision", (), {"strategy_used": "cost", "fallback_used": False, "candidates_evaluated": []})
        return response, decision



def _auth_context_for_tenant(tenant_suffix: str, plan: str = "free") -> AuthContext:
    tenant_id = UUID(f"00000000-0000-0000-0000-0000000000{tenant_suffix}")
    api_key_id = UUID(f"00000000-0000-0000-0000-0000000001{tenant_suffix}")

    tenant = Tenant(
        id=tenant_id,
        name=f"Tenant-{tenant_suffix}",
        email=f"t{tenant_suffix}@example.com",
        plan=plan,
        is_active=True,
        created_at=datetime.utcnow(),
    )
    api_key = APIKey(
        id=api_key_id,
        tenant_id=tenant_id,
        key_hash="hash",
        key_prefix="2api_test",
        name="test",
        permissions=["*"],
        rate_limit_per_minute=100,
        is_active=True,
        created_at=datetime.utcnow(),
    )
    return AuthContext(
        tenant_id=tenant_id,
        api_key_id=api_key_id,
        tenant=tenant,
        api_key=api_key,
        permissions=["*"],
        rate_limit_per_minute=100,
        request_id=f"req_{tenant_suffix}",
        trace_id=f"trace_{tenant_suffix}",
    )


@pytest.fixture(autouse=True)
def _reset_limiter_and_mode(monkeypatch):
    monkeypatch.setenv("MODE", "test")
    monkeypatch.setenv("TEST_RATE_LIMIT_RPM", "2")
    limits_module._limiter = UsageLimiter()
    yield
    limits_module._limiter = UsageLimiter()




def _build_test_app(router, auth_override):
    app = FastAPI()
    app.add_exception_handler(TwoApiException, twoapi_exception_handler)
    app.dependency_overrides[get_router] = lambda: router
    app.dependency_overrides[get_auth_context] = auth_override
    return app


def test_chat_rate_limit_returns_429_with_retry_after():
    app = _build_test_app(_FakeRouter(), lambda: _auth_context_for_tenant("11", plan="free"))
    app.include_router(chat_router)

    client = TestClient(app)
    payload = {"model": "auto", "messages": [{"role": "user", "content": "hello"}], "stream": False}

    r1 = client.post("/v1/chat/completions", json=payload)
    r2 = client.post("/v1/chat/completions", json=payload)
    r3 = client.post("/v1/chat/completions", json=payload)

    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r3.status_code == 429
    assert "Retry-After" in r3.headers


def test_embeddings_rate_limit_enforced():
    app = _build_test_app(_FakeRouter(), lambda: _auth_context_for_tenant("12", plan="free"))
    app.include_router(embeddings_router)

    client = TestClient(app)
    payload = {"model": "openai/text-embedding-3-small", "input": "hello"}

    assert client.post("/v1/embeddings", json=payload).status_code == 200
    assert client.post("/v1/embeddings", json=payload).status_code == 200
    third = client.post("/v1/embeddings", json=payload)
    assert third.status_code == 429
    assert "Retry-After" in third.headers


def test_images_rate_limit_enforced():
    app = _build_test_app(_FakeRouter(), lambda: _auth_context_for_tenant("13", plan="free"))
    app.include_router(images_router)

    client = TestClient(app)
    payload = {"prompt": "cat", "model": "openai/dall-e-3"}

    assert client.post("/v1/images/generations", json=payload).status_code == 200
    assert client.post("/v1/images/generations", json=payload).status_code == 200
    third = client.post("/v1/images/generations", json=payload)
    assert third.status_code == 429
    assert "Retry-After" in third.headers


def test_rate_limits_are_tenant_aware():
    def auth_by_token(authorization: str = Header(None, alias="Authorization")):
        token = (authorization or "Bearer 2api_tenant1").replace("Bearer ", "")
        if token.endswith("tenant2"):
            return _auth_context_for_tenant("22", plan="free")
        return _auth_context_for_tenant("21", plan="free")

    app = _build_test_app(_FakeRouter(), auth_by_token)
    app.include_router(chat_router)

    client = TestClient(app)
    payload = {"model": "auto", "messages": [{"role": "user", "content": "hello"}], "stream": False}

    # Exhaust tenant1
    assert client.post("/v1/chat/completions", json=payload, headers={"Authorization": "Bearer 2api_tenant1"}).status_code == 200
    assert client.post("/v1/chat/completions", json=payload, headers={"Authorization": "Bearer 2api_tenant1"}).status_code == 200
    assert client.post("/v1/chat/completions", json=payload, headers={"Authorization": "Bearer 2api_tenant1"}).status_code == 429

    # Tenant2 remains unaffected
    t2 = client.post("/v1/chat/completions", json=payload, headers={"Authorization": "Bearer 2api_tenant2"})
    assert t2.status_code == 200


def _set_rpm_quota_for_auth(auth: AuthContext, rpm: int = 2) -> None:
    set_quota(
        QuotaConfig(
            tenant_id=str(auth.tenant_id),
            plan=getattr(auth.tenant, "plan", "free"),
            limits=[
                UsageLimit(LimitType.RATE, LimitPeriod.MINUTE, rpm),
                UsageLimit(LimitType.TOKENS, LimitPeriod.DAY, 1_000_000),
                UsageLimit(LimitType.COST, LimitPeriod.MONTH, 1_000.0),
            ],
        )
    )


def test_local_mode_with_strict_local_guards_enforces_rate_limits(monkeypatch):
    monkeypatch.setenv("MODE", "local")
    monkeypatch.setenv("STRICT_LOCAL_GUARDS", "true")

    auth = _auth_context_for_tenant("31", plan="free")
    _set_rpm_quota_for_auth(auth, rpm=2)

    app = _build_test_app(_FakeRouter(), lambda: auth)
    app.include_router(chat_router)

    client = TestClient(app)
    payload = {"model": "auto", "messages": [{"role": "user", "content": "hello"}], "stream": False}

    assert client.post("/v1/chat/completions", json=payload).status_code == 200
    assert client.post("/v1/chat/completions", json=payload).status_code == 200

    third = client.post("/v1/chat/completions", json=payload)
    assert third.status_code == 429
    assert "Retry-After" in third.headers


def test_local_mode_without_strict_local_guards_keeps_rate_limit_bypass(monkeypatch):
    monkeypatch.setenv("MODE", "local")
    monkeypatch.delenv("STRICT_LOCAL_GUARDS", raising=False)

    auth = _auth_context_for_tenant("32", plan="free")
    _set_rpm_quota_for_auth(auth, rpm=1)

    app = _build_test_app(_FakeRouter(), lambda: auth)
    app.include_router(chat_router)

    client = TestClient(app)
    payload = {"model": "auto", "messages": [{"role": "user", "content": "hello"}], "stream": False}

    # Local default remains developer-friendly: no rate-limit enforcement shortcut disabled only when strict flag is on.
    assert client.post("/v1/chat/completions", json=payload).status_code == 200
    assert client.post("/v1/chat/completions", json=payload).status_code == 200
    assert client.post("/v1/chat/completions", json=payload).status_code == 200
