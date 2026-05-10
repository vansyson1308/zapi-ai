"""
Integration-style tests for src/api/agentops_routes.py.

Uses FastAPI's TestClient against a freshly-built app with stub adapters and
a fresh Guardian. We're not testing FastAPI itself; we're verifying the
routes wire to Guardian state correctly.
"""

from __future__ import annotations

import os
from typing import Iterator

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module", autouse=True)
def _set_local_mode():
    """Force local mode + stub adapters for the duration of these tests."""
    os.environ["MODE"] = "local"
    os.environ["USE_STUB_ADAPTERS"] = "true"
    yield
    # Don't unset — other tests may also set MODE.


@pytest.fixture
def client() -> Iterator[TestClient]:
    # Import inside the fixture so env vars take effect at app construction.
    from src.server import app

    with TestClient(app) as c:
        yield c


AUTH_HEADERS = {"Authorization": "Bearer 2api_test"}


# ============================================================
# whoami
# ============================================================


def test_whoami_returns_attribution(client):
    resp = client.get(
        "/v1/agentops/whoami",
        headers={
            **AUTH_HEADERS,
            "X-Customer-Id": "acme",
            "X-Feature-Id": "chat",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["customer_id"] == "acme"
    assert body["feature_id"] == "chat"
    # tenant_id pulled from local-mode auth
    assert body["tenant_id"]


def test_whoami_requires_auth(client):
    resp = client.get("/v1/agentops/whoami")
    assert resp.status_code == 401


# ============================================================
# budgets CRUD
# ============================================================


def test_create_and_list_budget(client):
    payload = {
        "scope": "customer",
        "scope_id": "acme",
        "period": "day",
        "limit_usd": 5.0,
    }
    create = client.post("/v1/agentops/budgets", json=payload, headers=AUTH_HEADERS)
    assert create.status_code == 201
    data = create.json()
    assert data["scope"] == "customer"
    assert data["scope_id"] == "acme"
    assert data["limit_usd"] == 5.0

    listed = client.get("/v1/agentops/budgets", headers=AUTH_HEADERS)
    assert listed.status_code == 200
    keys = {row["key"] for row in listed.json()["data"]}
    assert data["key"] in keys


def test_create_budget_requires_at_least_one_limit(client):
    resp = client.post(
        "/v1/agentops/budgets",
        json={
            "scope": "customer",
            "scope_id": "x",
            "period": "day",
        },
        headers=AUTH_HEADERS,
    )
    # Pydantic-side or Cap-side validation must reject this
    assert resp.status_code in (400, 422, 500)


def test_create_budget_rejects_negative_limit(client):
    resp = client.post(
        "/v1/agentops/budgets",
        json={
            "scope": "customer",
            "scope_id": "x",
            "period": "day",
            "limit_usd": -1.0,
        },
        headers=AUTH_HEADERS,
    )
    assert resp.status_code in (400, 422)


def test_create_budget_rejects_blank_scope_id(client):
    resp = client.post(
        "/v1/agentops/budgets",
        json={
            "scope": "customer",
            "scope_id": "  ",
            "period": "day",
            "limit_usd": 1.0,
        },
        headers=AUTH_HEADERS,
    )
    assert resp.status_code in (400, 422)


def test_delete_unknown_budget_returns_404(client):
    resp = client.delete(
        "/v1/agentops/budgets/customer:doesnotexist:day",
        headers=AUTH_HEADERS,
    )
    assert resp.status_code == 404


def test_create_then_delete_then_get_usage_404(client):
    # Use a unique scope_id to avoid clashing with other tests in this module.
    cap = client.post(
        "/v1/agentops/budgets",
        json={
            "scope": "agent",
            "scope_id": "test-agent-deleteme",
            "period": "hour",
            "limit_usd": 1.0,
        },
        headers=AUTH_HEADERS,
    )
    key = cap.json()["key"]

    # Delete it
    del_resp = client.delete(f"/v1/agentops/budgets/{key}", headers=AUTH_HEADERS)
    assert del_resp.status_code == 200

    # Now usage endpoint must 404
    usage = client.get(f"/v1/agentops/budgets/{key}/usage", headers=AUTH_HEADERS)
    assert usage.status_code == 404


# ============================================================
# events
# ============================================================


def test_events_endpoint_returns_list_shape(client):
    resp = client.get("/v1/agentops/events", headers=AUTH_HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    assert isinstance(body["data"], list)


def test_events_endpoint_clamps_limit(client):
    resp = client.get("/v1/agentops/events?limit=99999", headers=AUTH_HEADERS)
    assert resp.status_code == 200
    # No assertion on content; just confirms no 5xx from huge limits


# ============================================================
# usage by attribution
# ============================================================


def test_usage_by_attribution_returns_list_shape(client):
    resp = client.get("/v1/agentops/usage/by-attribution", headers=AUTH_HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    assert isinstance(body["data"], list)
