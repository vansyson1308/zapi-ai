"""
AgentOps admin/management endpoints.

Endpoints (all require auth):
  GET  /v1/agentops/budgets               — list registered caps
  POST /v1/agentops/budgets               — register/update a cap
  DELETE /v1/agentops/budgets/{key}       — remove a cap
  GET  /v1/agentops/budgets/{key}/usage   — current usage snapshot for one cap
  GET  /v1/agentops/usage/by-attribution  — slice usage by customer/feature/agent
  GET  /v1/agentops/events                — list recent kill events (in-memory, last 100)

These endpoints are intentionally minimal — they're enough to demo the
wedge and let design partners verify behavior before we ship the React UI.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from ..agentops import (
    AttributionContext,
    BudgetCap,
    BudgetPeriod,
    BudgetScope,
    Guardian,
    HardAction,
)
from ..agentops.middleware import get_attribution
from ..auth.middleware import get_auth_context
from ..db.models import AuthContext
from ..observability.event_bus import (
    EVENT_BUDGET_EXCEEDED,
    EVENT_BUDGET_WARNING,
    EVENT_KILL_SWITCH_TRIGGERED,
    Event,
    get_event_bus,
)
from ..usage import get_usage_tracker


router = APIRouter(prefix="/v1/agentops", tags=["agentops"])


# ============================================================
# Internal: in-memory event ring buffer
# ============================================================
#
# Wires the EventBus to a small in-memory log so the dashboard can show
# "last 100 events" without us standing up a real database for it. The buffer
# is process-local; in a HA deployment we'd ship these to Redis Streams or
# an APM tool — out of scope for the wedge.

_EVENT_LOG: Deque[Dict[str, Any]] = deque(maxlen=100)


def _record_event_for_dashboard(event: Event) -> None:
    _EVENT_LOG.appendleft(event.to_dict())


_event_logging_subscribed = False


def _ensure_event_subscriptions() -> None:
    """Idempotently subscribe our buffer handler to the event bus."""
    global _event_logging_subscribed
    if _event_logging_subscribed:
        return
    bus = get_event_bus()
    bus.subscribe(EVENT_KILL_SWITCH_TRIGGERED, _record_event_for_dashboard)
    bus.subscribe(EVENT_BUDGET_WARNING, _record_event_for_dashboard)
    bus.subscribe(EVENT_BUDGET_EXCEEDED, _record_event_for_dashboard)
    _event_logging_subscribed = True


# ============================================================
# Guardian dependency
# ============================================================


def _get_guardian(request: Request) -> Guardian:
    """
    Resolve the Guardian instance attached to the running FastAPI app.

    The middleware (AgentOpsMiddleware) is the canonical owner. In tests we
    expose `app.state.guardian` so routes can find it without going through
    the middleware machinery.
    """
    guardian = getattr(request.app.state, "agentops_guardian", None)
    if guardian is not None:
        return guardian
    # Fallback: bare default Guardian using the global redis client.
    return Guardian()


# ============================================================
# Pydantic models for request bodies
# ============================================================


class BudgetCapCreate(BaseModel):
    scope: BudgetScope
    scope_id: str = Field(min_length=1, max_length=256)
    period: BudgetPeriod
    limit_usd: Optional[float] = Field(default=None, gt=0)
    limit_tokens: Optional[int] = Field(default=None, gt=0)
    soft_threshold_pct: float = Field(default=0.8, gt=0.0, le=1.0)
    hard_action: HardAction = HardAction.BLOCK

    @field_validator("scope_id")
    @classmethod
    def _validate_scope_id(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("scope_id cannot be blank")
        return v

    @model_validator(mode="after")
    def _at_least_one_limit(self) -> "BudgetCapCreate":
        if self.limit_usd is None and self.limit_tokens is None:
            raise ValueError(
                "must provide at least one of limit_usd or limit_tokens"
            )
        return self

    def to_cap(self) -> BudgetCap:
        return BudgetCap(
            scope=self.scope,
            scope_id=self.scope_id.strip(),
            period=self.period,
            limit_usd=self.limit_usd,
            limit_tokens=self.limit_tokens,
            soft_threshold_pct=self.soft_threshold_pct,
            hard_action=self.hard_action,
        )


# ============================================================
# Endpoints
# ============================================================


@router.get("/budgets")
async def list_budgets(
    request: Request,
    auth: AuthContext = Depends(get_auth_context),
) -> JSONResponse:
    guardian = _get_guardian(request)
    caps = guardian.list_caps()
    return JSONResponse(
        {
            "object": "list",
            "data": [
                {
                    "key": cap.key,
                    "scope": cap.scope.value,
                    "scope_id": cap.scope_id,
                    "period": cap.period.value,
                    "limit_usd": cap.limit_usd,
                    "limit_tokens": cap.limit_tokens,
                    "soft_threshold_pct": cap.soft_threshold_pct,
                    "hard_action": cap.hard_action.value,
                }
                for cap in caps
            ],
        },
        headers={"X-Request-Id": auth.request_id, "X-Trace-Id": auth.trace_id},
    )


@router.post("/budgets")
async def create_budget(
    body: BudgetCapCreate,
    request: Request,
    auth: AuthContext = Depends(get_auth_context),
) -> JSONResponse:
    guardian = _get_guardian(request)
    cap = body.to_cap()
    guardian.register_cap(cap)
    return JSONResponse(
        {
            "key": cap.key,
            "scope": cap.scope.value,
            "scope_id": cap.scope_id,
            "period": cap.period.value,
            "limit_usd": cap.limit_usd,
            "limit_tokens": cap.limit_tokens,
            "soft_threshold_pct": cap.soft_threshold_pct,
            "hard_action": cap.hard_action.value,
        },
        status_code=201,
        headers={"X-Request-Id": auth.request_id, "X-Trace-Id": auth.trace_id},
    )


@router.delete("/budgets/{key:path}")
async def delete_budget(
    key: str,
    request: Request,
    auth: AuthContext = Depends(get_auth_context),
) -> JSONResponse:
    guardian = _get_guardian(request)
    if not guardian.remove_cap(key):
        raise HTTPException(status_code=404, detail=f"Budget cap '{key}' not found")
    return JSONResponse(
        {"deleted": key},
        headers={"X-Request-Id": auth.request_id, "X-Trace-Id": auth.trace_id},
    )


@router.get("/budgets/{key:path}/usage")
async def budget_usage(
    key: str,
    request: Request,
    auth: AuthContext = Depends(get_auth_context),
) -> JSONResponse:
    guardian = _get_guardian(request)
    cap = guardian.get_cap(key)
    if cap is None:
        raise HTTPException(status_code=404, detail=f"Budget cap '{key}' not found")
    snap = await guardian.snapshot(cap)
    return JSONResponse(
        {
            "key": cap.key,
            "limit_usd": cap.limit_usd,
            "limit_tokens": cap.limit_tokens,
            "used_usd": snap.used_usd,
            "used_tokens": snap.used_tokens,
            "usage_pct": round(snap.usage_pct(), 4),
            "is_exceeded": snap.is_exceeded,
            "is_soft_exceeded": snap.is_soft_exceeded,
        },
        headers={"X-Request-Id": auth.request_id, "X-Trace-Id": auth.trace_id},
    )


@router.get("/usage/by-attribution")
async def usage_by_attribution(
    request: Request,
    auth: AuthContext = Depends(get_auth_context),
    customer_id: Optional[str] = None,
    feature_id: Optional[str] = None,
    agent_id: Optional[str] = None,
) -> JSONResponse:
    """Slice in-memory usage aggregates by attribution dimensions."""
    tracker = get_usage_tracker()
    rows = tracker.get_attribution_usage(
        tenant_id=str(auth.tenant_id) if auth.tenant_id else None,
        customer_id=customer_id,
        feature_id=feature_id,
        agent_id=agent_id,
    )
    return JSONResponse(
        {"object": "list", "data": rows},
        headers={"X-Request-Id": auth.request_id, "X-Trace-Id": auth.trace_id},
    )


@router.get("/events")
async def list_events(
    request: Request,
    auth: AuthContext = Depends(get_auth_context),
    limit: int = 50,
) -> JSONResponse:
    """Recent kill switch / warning / drift events from the in-memory ring buffer."""
    _ensure_event_subscriptions()
    limit = max(1, min(limit, 100))
    events: List[Dict[str, Any]] = list(_EVENT_LOG)[:limit]
    return JSONResponse(
        {"object": "list", "data": events},
        headers={"X-Request-Id": auth.request_id, "X-Trace-Id": auth.trace_id},
    )


@router.get("/whoami")
async def whoami(
    request: Request,
    auth: AuthContext = Depends(get_auth_context),
) -> JSONResponse:
    """Echo the AgentOps attribution context the gateway sees for this request."""
    attribution = get_attribution(request)
    # Merge auth identifiers (route handler runs after middleware; auth is now resolved).
    attribution.tenant_id = attribution.tenant_id or str(auth.tenant_id)
    attribution.api_key_id = attribution.api_key_id or str(auth.api_key_id)
    return JSONResponse(
        attribution.to_dict(),
        headers={"X-Request-Id": auth.request_id, "X-Trace-Id": auth.trace_id},
    )
