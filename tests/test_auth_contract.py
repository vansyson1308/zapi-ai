"""Contract test enforcing auth on all /v1/* endpoints."""

from __future__ import annotations

from fastapi.routing import APIRoute

from src.auth.middleware import get_auth_context
from src.server import app


# Keep this list explicit and minimal.
PUBLIC_V1_ALLOWLIST = {
    "/v1/health",
}


def _has_auth_dependency(route: APIRoute) -> bool:
    """Return True when route depends (directly or transitively) on get_auth_context."""

    stack = [route.dependant]
    seen = set()

    while stack:
        dependant = stack.pop()
        dep_id = id(dependant)
        if dep_id in seen:
            continue
        seen.add(dep_id)

        if dependant.call is get_auth_context:
            return True

        stack.extend(dependant.dependencies)

    return False


def test_all_v1_endpoints_require_auth_except_explicit_allowlist():
    """All /v1/* routes must require auth unless explicitly allowlisted."""

    unauthenticated_routes: list[str] = []

    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue

        if not route.path.startswith("/v1/"):
            continue

        if route.path in PUBLIC_V1_ALLOWLIST:
            continue

        if not _has_auth_dependency(route):
            methods = ",".join(sorted(route.methods)) if route.methods else "UNKNOWN"
            unauthenticated_routes.append(f"{methods} {route.path}")

    assert not unauthenticated_routes, (
        "Unauthenticated /v1 routes found (only explicit allowlist permitted): "
        + "; ".join(sorted(unauthenticated_routes))
    )
