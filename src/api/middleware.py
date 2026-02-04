"""
2api.ai - API Middleware

Custom middleware for request processing.

Provides:
- Request logging
- Metrics collection
- Request ID injection
- Error tracking
"""

import json
import time
import uuid
from typing import Callable, Dict, Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging requests and responses.

    Logs:
    - Request method, path, duration
    - Response status code
    - Error details (if any)
    """

    def __init__(
        self,
        app: ASGIApp,
        logger: Optional[Callable] = None,
        log_body: bool = False,
        exclude_paths: Optional[list] = None
    ):
        super().__init__(app)
        self.logger = logger or print
        self.log_body = log_body
        self.exclude_paths = exclude_paths or ["/health", "/metrics"]

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint
    ) -> Response:
        # Skip logging for excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)

        # Generate request ID if not present
        request_id = request.headers.get("x-request-id")
        if not request_id:
            request_id = f"req_{uuid.uuid4().hex[:24]}"

        # Record start time
        start_time = time.time()

        # Process request
        try:
            response = await call_next(request)
            duration_ms = int((time.time() - start_time) * 1000)

            # Log request
            log_data = {
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
                "client_ip": request.client.host if request.client else None,
            }

            # Add request ID to response headers
            response.headers["X-Request-Id"] = request_id
            response.headers["X-Duration-Ms"] = str(duration_ms)

            self.logger(f"[API] {json.dumps(log_data)}")

            return response

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)

            # Log error
            log_data = {
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": 500,
                "duration_ms": duration_ms,
                "error": str(e),
            }

            self.logger(f"[API ERROR] {json.dumps(log_data)}")

            raise


class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Middleware for collecting metrics.

    Tracks:
    - Request counts by path and status
    - Latency distribution
    - Error rates
    """

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self._metrics: Dict[str, Dict] = {
            "requests": {},  # path -> {count, success, error}
            "latencies": {},  # path -> [latencies]
        }

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint
    ) -> Response:
        path = request.url.path
        start_time = time.time()

        # Initialize metrics for path
        if path not in self._metrics["requests"]:
            self._metrics["requests"][path] = {
                "count": 0,
                "success": 0,
                "error": 0,
            }
            self._metrics["latencies"][path] = []

        self._metrics["requests"][path]["count"] += 1

        try:
            response = await call_next(request)
            duration_ms = int((time.time() - start_time) * 1000)

            # Track success/error
            if response.status_code < 400:
                self._metrics["requests"][path]["success"] += 1
            else:
                self._metrics["requests"][path]["error"] += 1

            # Track latency (keep last 1000)
            latencies = self._metrics["latencies"][path]
            latencies.append(duration_ms)
            if len(latencies) > 1000:
                self._metrics["latencies"][path] = latencies[-1000:]

            return response

        except Exception:
            self._metrics["requests"][path]["error"] += 1
            raise

    def get_metrics(self) -> Dict:
        """Get current metrics."""
        result = {}

        for path, counts in self._metrics["requests"].items():
            latencies = self._metrics["latencies"].get(path, [])

            result[path] = {
                "total_requests": counts["count"],
                "success_count": counts["success"],
                "error_count": counts["error"],
                "error_rate": (
                    counts["error"] / counts["count"] * 100
                    if counts["count"] > 0 else 0
                ),
            }

            if latencies:
                sorted_lat = sorted(latencies)
                result[path]["latency"] = {
                    "avg_ms": sum(latencies) / len(latencies),
                    "min_ms": sorted_lat[0],
                    "max_ms": sorted_lat[-1],
                    "p50_ms": sorted_lat[len(sorted_lat) // 2],
                    "p95_ms": sorted_lat[int(len(sorted_lat) * 0.95)],
                    "p99_ms": sorted_lat[int(len(sorted_lat) * 0.99)],
                }

        return result


class CORSMiddleware:
    """
    Simple CORS middleware.

    Handles preflight requests and adds CORS headers.
    """

    def __init__(
        self,
        app: ASGIApp,
        allow_origins: list = ["*"],
        allow_methods: list = ["*"],
        allow_headers: list = ["*"],
        allow_credentials: bool = True,
        max_age: int = 600
    ):
        self.app = app
        self.allow_origins = allow_origins
        self.allow_methods = allow_methods
        self.allow_headers = allow_headers
        self.allow_credentials = allow_credentials
        self.max_age = max_age

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "")
        headers = dict(scope.get("headers", []))

        # Handle preflight
        if method == "OPTIONS":
            response_headers = [
                (b"access-control-allow-origin", b"*"),
                (b"access-control-allow-methods", b"GET, POST, PUT, DELETE, OPTIONS"),
                (b"access-control-allow-headers", b"*"),
                (b"access-control-max-age", str(self.max_age).encode()),
            ]

            if self.allow_credentials:
                response_headers.append(
                    (b"access-control-allow-credentials", b"true")
                )

            await send({
                "type": "http.response.start",
                "status": 200,
                "headers": response_headers
            })
            await send({"type": "http.response.body", "body": b""})
            return

        # Process request and add CORS headers to response
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.extend([
                    (b"access-control-allow-origin", b"*"),
                    (b"access-control-allow-credentials", b"true"),
                ])
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_wrapper)


class RequestIdMiddleware(BaseHTTPMiddleware):
    """
    Middleware to ensure every request has an ID.

    Adds X-Request-Id header if not present.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint
    ) -> Response:
        # Get or generate request ID
        request_id = request.headers.get("x-request-id")
        if not request_id:
            request_id = f"req_{uuid.uuid4().hex[:24]}"

        # Store in request state for access in route handlers
        request.state.request_id = request_id
        request.state.trace_id = request_id.replace("req_", "trace_")

        # Process request
        response = await call_next(request)

        # Ensure headers are set
        if "x-request-id" not in response.headers:
            response.headers["X-Request-Id"] = request_id
        if "x-trace-id" not in response.headers:
            response.headers["X-Trace-Id"] = request.state.trace_id

        return response


class ErrorTrackingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for tracking and reporting errors.

    Can be extended to send errors to external services
    (Sentry, Datadog, etc.)
    """

    def __init__(
        self,
        app: ASGIApp,
        error_callback: Optional[Callable] = None
    ):
        super().__init__(app)
        self.error_callback = error_callback
        self._error_counts: Dict[str, int] = {}

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint
    ) -> Response:
        try:
            response = await call_next(request)

            # Track 4xx/5xx responses
            if response.status_code >= 400:
                error_key = f"{response.status_code}_{request.url.path}"
                self._error_counts[error_key] = (
                    self._error_counts.get(error_key, 0) + 1
                )

            return response

        except Exception as e:
            # Track exception
            error_key = f"exception_{type(e).__name__}"
            self._error_counts[error_key] = (
                self._error_counts.get(error_key, 0) + 1
            )

            # Call error callback if provided
            if self.error_callback:
                try:
                    self.error_callback(e, request)
                except Exception:
                    pass  # Don't let callback errors break the response

            raise

    def get_error_counts(self) -> Dict[str, int]:
        """Get error counts."""
        return self._error_counts.copy()
