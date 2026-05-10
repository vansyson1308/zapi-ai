"""
AgentOps FastAPI middleware — extracts attribution headers and runs Guardian
pre-checks for chat/embedding/image endpoints.

Design:
- Pure middleware. Doesn't replace the existing auth or rate-limit deps; it
  layers on top.
- For pre-check: only runs against the canonical 2api endpoints
  (`POST /v1/chat/completions`, `POST /v1/embeddings`, `POST /v1/images/generations`).
  Other paths pass through untouched.
- Stores AttributionContext on `request.state.agentops_attribution` so route
  handlers can read it without re-parsing headers.
- Errors raised here surface as proper SemanticError responses via the
  global exception handler in server.py.

We deliberately do NOT do PII redaction here, because that requires reading +
mutating the request body which:
  (a) is expensive for streaming requests,
  (b) the route handler already has access to the parsed body and can call
      PIIRedactor on a per-message basis with the configured policy.
The middleware only validates attribution + budget; PII is the route
handler's responsibility.
"""

from __future__ import annotations

from typing import Awaitable, Callable, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from .attribution import attribution_context_from_headers
from .guardian import Guardian
from .models import AttributionContext


# Endpoints that count as billable AI requests (and therefore deserve a
# pre-check). Add new endpoints here as we expose them.
GUARDED_ENDPOINTS = (
    "/v1/chat/completions",
    "/v1/embeddings",
    "/v1/images/generations",
)


class AgentOpsMiddleware(BaseHTTPMiddleware):
    """Extract attribution + run Guardian pre-checks for billable endpoints."""

    def __init__(self, app, guardian: Optional[Guardian] = None) -> None:
        super().__init__(app)
        self._guardian = guardian or Guardian()

    @property
    def guardian(self) -> Guardian:
        return self._guardian

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        # Always parse + stash attribution so downstream handlers get it cheaply.
        attribution = self._build_attribution(request)
        request.state.agentops_attribution = attribution

        # Pre-check for budget caps on guarded endpoints only.
        if request.url.path in GUARDED_ENDPOINTS and request.method == "POST":
            # We do NOT estimate cost here yet (would require parsing the body).
            # The pre-check still catches the most common case: a cap that's
            # already over the limit from prior requests.
            request_id = request.headers.get("x-request-id", "")
            await self._guardian.pre_check(
                attribution=attribution,
                estimate_usd=0.0,
                estimate_tokens=0,
                request_id=request_id,
            )

        return await call_next(request)

    @staticmethod
    def _build_attribution(request: Request) -> AttributionContext:
        # Tenant/api_key info isn't known here (auth runs after middleware in
        # FastAPI's typical flow), so we only derive customer/feature/agent.
        # Auth-derived identifiers are merged in by the route handler that
        # depends on `auth_context`.
        headers = {k: v for k, v in request.headers.items()}
        return attribution_context_from_headers(headers)


def get_attribution(request: Request) -> AttributionContext:
    """Read the AttributionContext stashed by the middleware."""
    return getattr(request.state, "agentops_attribution", AttributionContext())
