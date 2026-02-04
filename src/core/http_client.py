"""
2api.ai - Robust HTTP Client

HTTP client wrapper with:
- Exponential backoff retry (1-2-4-8-16s with jitter)
- Request correlation (request_id logging)
- Step-based logging for debugging
- Automatic fallback support
"""

import asyncio
import random
import time
import logging
import uuid
from typing import Any, Dict, List, Optional, Callable, TypeVar
from dataclasses import dataclass, field
from enum import Enum

import httpx


# Configure logging
logger = logging.getLogger("2api.http")
logger.setLevel(logging.DEBUG)

# Add handler if none exists
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] %(message)s'
    ))
    logger.addHandler(handler)


T = TypeVar("T")


class RetryableStatusCodes:
    """HTTP status codes that should trigger retry."""
    CODES = {500, 502, 503, 504, 429}

    @classmethod
    def is_retryable(cls, status_code: int) -> bool:
        return status_code in cls.CODES


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 5
    base_delay: float = 1.0  # seconds
    max_delay: float = 16.0  # seconds
    exponential_base: float = 2.0
    jitter_factor: float = 0.25  # 25% jitter
    retryable_status_codes: List[int] = field(
        default_factory=lambda: [500, 502, 503, 504, 429]
    )


@dataclass
class RequestContext:
    """Context for tracking requests through the system."""
    request_id: str = field(default_factory=lambda: f"req_{uuid.uuid4().hex[:12]}")
    step_name: str = ""
    provider: str = ""
    model: str = ""
    base_url: str = ""

    def to_log_extra(self) -> Dict[str, str]:
        return {"request_id": self.request_id}


@dataclass
class HttpResponse:
    """Wrapped HTTP response with metadata."""
    status_code: int
    data: Any
    headers: Dict[str, str]
    request_id: str
    latency_ms: float
    retries_attempted: int = 0
    fallback_used: bool = False


class HttpClientError(Exception):
    """Base error for HTTP client issues."""

    def __init__(
        self,
        message: str,
        status_code: int = 0,
        request_id: str = "",
        step_name: str = "",
        retryable: bool = False,
        response_body: Optional[str] = None
    ):
        self.message = message
        self.status_code = status_code
        self.request_id = request_id
        self.step_name = step_name
        self.retryable = retryable
        self.response_body = response_body
        super().__init__(
            f"[{request_id}] {step_name}: {message} (status={status_code}, retryable={retryable})"
        )


class InfraError(HttpClientError):
    """Infrastructure error (500, 502, 503, 504, timeout)."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, retryable=True, **kwargs)


class RateLimitError(HttpClientError):
    """Rate limit exceeded (429)."""

    def __init__(self, message: str, retry_after: int = 60, **kwargs):
        super().__init__(message, retryable=True, **kwargs)
        self.retry_after = retry_after


def calculate_backoff(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 16.0,
    exponential_base: float = 2.0,
    jitter_factor: float = 0.25
) -> float:
    """
    Calculate delay with exponential backoff and jitter.

    Sequence: 1s, 2s, 4s, 8s, 16s (with ±25% jitter)
    """
    delay = min(base_delay * (exponential_base ** attempt), max_delay)

    # Add jitter
    jitter_range = delay * jitter_factor
    delay = delay + random.uniform(-jitter_range, jitter_range)

    return max(0.1, delay)  # Minimum 100ms


class RobustHttpClient:
    """
    HTTP client with automatic retry, logging, and fallback support.

    Features:
    - Exponential backoff retry (1-2-4-8-16s)
    - Request ID correlation
    - Step-based logging
    - Automatic fallback to alternate providers
    """

    def __init__(
        self,
        base_url: str = "",
        timeout: float = 60.0,
        retry_config: Optional[RetryConfig] = None,
        headers: Optional[Dict[str, str]] = None
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retry_config = retry_config or RetryConfig()
        self.default_headers = headers or {}
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                headers=self.default_headers
            )
        return self._client

    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    def _log_request_start(self, ctx: RequestContext, method: str, url: str, payload_summary: str):
        """Log request start with step name."""
        logger.info(
            f"STEP [{ctx.step_name}] Starting {method} {url} "
            f"(provider={ctx.provider}, model={ctx.model})",
            extra=ctx.to_log_extra()
        )
        logger.debug(
            f"Payload summary: {payload_summary}",
            extra=ctx.to_log_extra()
        )

    def _log_retry(self, ctx: RequestContext, attempt: int, delay: float, error: str):
        """Log retry attempt."""
        logger.warning(
            f"STEP [{ctx.step_name}] Retry {attempt}/{self.retry_config.max_retries} "
            f"after {delay:.2f}s - Error: {error}",
            extra=ctx.to_log_extra()
        )

    def _log_response(self, ctx: RequestContext, status: int, latency_ms: float, retries: int):
        """Log response received."""
        level = logging.INFO if status < 400 else logging.WARNING
        logger.log(
            level,
            f"STEP [{ctx.step_name}] Response: status={status}, "
            f"latency={latency_ms:.0f}ms, retries={retries}",
            extra=ctx.to_log_extra()
        )

    def _log_fallback(self, ctx: RequestContext, from_provider: str, to_provider: str):
        """Log fallback activation."""
        logger.warning(
            f"STEP [{ctx.step_name}] Fallback: {from_provider} -> {to_provider}",
            extra=ctx.to_log_extra()
        )

    def _summarize_payload(self, payload: Dict[str, Any]) -> str:
        """Create safe payload summary (no secrets)."""
        summary = {}
        for key, value in payload.items():
            if key in ("api_key", "key", "token", "secret", "password", "authorization"):
                summary[key] = "***REDACTED***"
            elif key == "messages" and isinstance(value, list):
                summary[key] = f"[{len(value)} messages]"
            elif isinstance(value, str) and len(value) > 100:
                summary[key] = f"{value[:50]}...({len(value)} chars)"
            else:
                summary[key] = value
        return str(summary)

    async def request(
        self,
        method: str,
        path: str,
        step_name: str = "http_request",
        provider: str = "",
        model: str = "",
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        fallback_urls: Optional[List[str]] = None
    ) -> HttpResponse:
        """
        Make HTTP request with retry and fallback support.

        Args:
            method: HTTP method
            path: URL path (appended to base_url)
            step_name: Name of current step for logging
            provider: Provider name for logging
            model: Model name for logging
            json: JSON body
            headers: Additional headers
            params: Query parameters
            request_id: Request ID for correlation (auto-generated if not provided)
            fallback_urls: List of fallback base URLs to try

        Returns:
            HttpResponse with data and metadata

        Raises:
            InfraError: If all retries and fallbacks fail
            RateLimitError: If rate limited (after retries)
        """
        ctx = RequestContext(
            request_id=request_id or f"req_{uuid.uuid4().hex[:12]}",
            step_name=step_name,
            provider=provider,
            model=model,
            base_url=self.base_url
        )

        # Build list of URLs to try (primary + fallbacks)
        urls_to_try = [self.base_url]
        if fallback_urls:
            urls_to_try.extend(fallback_urls)

        last_error: Optional[Exception] = None

        for url_index, base_url in enumerate(urls_to_try):
            is_fallback = url_index > 0
            if is_fallback:
                self._log_fallback(ctx, urls_to_try[url_index - 1], base_url)

            try:
                return await self._request_with_retry(
                    method=method,
                    base_url=base_url,
                    path=path,
                    ctx=ctx,
                    json=json,
                    headers=headers,
                    params=params,
                    is_fallback=is_fallback
                )
            except (InfraError, RateLimitError) as e:
                last_error = e
                logger.warning(
                    f"STEP [{ctx.step_name}] All retries failed for {base_url}, "
                    f"trying fallback..." if url_index < len(urls_to_try) - 1
                    else f"STEP [{ctx.step_name}] All retries and fallbacks exhausted",
                    extra=ctx.to_log_extra()
                )
                continue

        # All attempts failed
        raise last_error or InfraError(
            "All retry attempts failed",
            request_id=ctx.request_id,
            step_name=ctx.step_name
        )

    async def _request_with_retry(
        self,
        method: str,
        base_url: str,
        path: str,
        ctx: RequestContext,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        is_fallback: bool = False
    ) -> HttpResponse:
        """Execute request with retry logic."""
        url = f"{base_url}{path}"
        merged_headers = {**self.default_headers, **(headers or {})}
        merged_headers["X-Request-ID"] = ctx.request_id

        # Log request start
        payload_summary = self._summarize_payload(json or {})
        self._log_request_start(ctx, method, url, payload_summary)

        client = await self._get_client()
        last_error: Optional[Exception] = None
        retries = 0

        for attempt in range(self.retry_config.max_retries + 1):
            start_time = time.time()

            try:
                response = await client.request(
                    method=method,
                    url=url,
                    json=json,
                    headers=merged_headers,
                    params=params
                )

                latency_ms = (time.time() - start_time) * 1000

                # Check if status is retryable
                if response.status_code in self.retry_config.retryable_status_codes:
                    retries = attempt

                    # Special handling for 429
                    if response.status_code == 429:
                        retry_after = int(response.headers.get("Retry-After", 60))
                        if attempt < self.retry_config.max_retries:
                            delay = min(retry_after, self.retry_config.max_delay)
                            self._log_retry(
                                ctx, attempt + 1, delay,
                                f"Rate limited (429), retry-after={retry_after}s"
                            )
                            await asyncio.sleep(delay)
                            continue
                        else:
                            raise RateLimitError(
                                f"Rate limit exceeded after {attempt + 1} attempts",
                                retry_after=retry_after,
                                status_code=429,
                                request_id=ctx.request_id,
                                step_name=ctx.step_name
                            )

                    # Handle 5xx errors
                    if attempt < self.retry_config.max_retries:
                        delay = calculate_backoff(
                            attempt,
                            self.retry_config.base_delay,
                            self.retry_config.max_delay,
                            self.retry_config.exponential_base,
                            self.retry_config.jitter_factor
                        )

                        response_preview = ""
                        try:
                            response_preview = response.text[:200]
                        except Exception:
                            pass

                        self._log_retry(
                            ctx, attempt + 1, delay,
                            f"Status {response.status_code}: {response_preview}"
                        )
                        await asyncio.sleep(delay)
                        continue
                    else:
                        raise InfraError(
                            f"Server error {response.status_code} after {attempt + 1} attempts",
                            status_code=response.status_code,
                            request_id=ctx.request_id,
                            step_name=ctx.step_name,
                            response_body=response.text[:500] if response.text else None
                        )

                # Success or non-retryable error
                self._log_response(ctx, response.status_code, latency_ms, retries)

                try:
                    data = response.json()
                except Exception:
                    data = response.text

                return HttpResponse(
                    status_code=response.status_code,
                    data=data,
                    headers=dict(response.headers),
                    request_id=ctx.request_id,
                    latency_ms=latency_ms,
                    retries_attempted=retries,
                    fallback_used=is_fallback
                )

            except httpx.TimeoutException as e:
                retries = attempt
                if attempt < self.retry_config.max_retries:
                    delay = calculate_backoff(
                        attempt,
                        self.retry_config.base_delay,
                        self.retry_config.max_delay
                    )
                    self._log_retry(ctx, attempt + 1, delay, f"Timeout: {e}")
                    await asyncio.sleep(delay)
                    continue
                else:
                    raise InfraError(
                        f"Request timeout after {attempt + 1} attempts",
                        request_id=ctx.request_id,
                        step_name=ctx.step_name
                    )

            except httpx.ConnectError as e:
                retries = attempt
                if attempt < self.retry_config.max_retries:
                    delay = calculate_backoff(
                        attempt,
                        self.retry_config.base_delay,
                        self.retry_config.max_delay
                    )
                    self._log_retry(ctx, attempt + 1, delay, f"Connection error: {e}")
                    await asyncio.sleep(delay)
                    continue
                else:
                    raise InfraError(
                        f"Connection failed after {attempt + 1} attempts: {e}",
                        request_id=ctx.request_id,
                        step_name=ctx.step_name
                    )

        # Should not reach here
        raise InfraError(
            "Unexpected retry loop exit",
            request_id=ctx.request_id,
            step_name=ctx.step_name
        )


# Sync wrapper for the async client
class SyncHttpClient:
    """Synchronous wrapper for RobustHttpClient."""

    def __init__(self, *args, **kwargs):
        self._async_client = RobustHttpClient(*args, **kwargs)

    def request(self, *args, **kwargs) -> HttpResponse:
        """Make synchronous HTTP request."""
        return asyncio.get_event_loop().run_until_complete(
            self._async_client.request(*args, **kwargs)
        )

    def close(self):
        """Close the client."""
        asyncio.get_event_loop().run_until_complete(
            self._async_client.close()
        )
