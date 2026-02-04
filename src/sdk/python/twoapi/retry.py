"""
2api.ai SDK - Retry Logic

Exponential backoff with jitter for reliable API calls.
"""

import asyncio
import random
import time
from typing import Callable, TypeVar, Optional, List, Any

from .errors import TwoAPIError, RateLimitError


T = TypeVar("T")


def calculate_backoff(
    attempt: int,
    initial_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    jitter: bool = True
) -> float:
    """
    Calculate delay for exponential backoff with optional jitter.

    Args:
        attempt: Current retry attempt (0-based)
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential calculation
        jitter: Whether to add random jitter

    Returns:
        Delay in seconds
    """
    # Calculate exponential delay
    delay = initial_delay * (exponential_base ** attempt)

    # Cap at max delay
    delay = min(delay, max_delay)

    # Add jitter (up to 25% variance)
    if jitter:
        jitter_range = delay * 0.25
        delay = delay + random.uniform(-jitter_range, jitter_range)

    return max(0, delay)


def should_retry(
    error: Exception,
    retry_on_status: Optional[List[int]] = None
) -> bool:
    """
    Determine if an error should be retried.

    Args:
        error: The exception that was raised
        retry_on_status: HTTP status codes to retry on

    Returns:
        True if the request should be retried
    """
    if retry_on_status is None:
        retry_on_status = [429, 500, 502, 503, 504]

    if isinstance(error, TwoAPIError):
        # Check if explicitly marked as retryable
        if error.retryable:
            return True

        # Check status code
        if error.status_code in retry_on_status:
            return True

    return False


def with_retry(
    func: Callable[..., T],
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    retry_on_status: Optional[List[int]] = None,
    on_retry: Optional[Callable[[int, Exception, float], None]] = None
) -> Callable[..., T]:
    """
    Decorator for adding retry logic to synchronous functions.

    Args:
        func: Function to wrap
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries
        max_delay: Maximum delay between retries
        exponential_base: Base for exponential backoff
        retry_on_status: HTTP status codes to retry on
        on_retry: Callback called before each retry with (attempt, error, delay)

    Returns:
        Wrapped function with retry logic
    """
    def wrapper(*args: Any, **kwargs: Any) -> T:
        last_error: Optional[Exception] = None

        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e

                # Check if we should retry
                if attempt >= max_retries or not should_retry(e, retry_on_status):
                    raise

                # Calculate delay
                if isinstance(e, RateLimitError):
                    # Use the server's suggested retry-after
                    delay = float(e.retry_after)
                else:
                    delay = calculate_backoff(
                        attempt,
                        initial_delay,
                        max_delay,
                        exponential_base
                    )

                # Call retry callback if provided
                if on_retry:
                    on_retry(attempt, e, delay)

                # Sleep before retry
                time.sleep(delay)

        # Should never reach here, but just in case
        if last_error:
            raise last_error
        raise RuntimeError("Retry logic error")

    return wrapper


def with_retry_async(
    func: Callable[..., T],
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    retry_on_status: Optional[List[int]] = None,
    on_retry: Optional[Callable[[int, Exception, float], None]] = None
) -> Callable[..., T]:
    """
    Decorator for adding retry logic to async functions.

    Same as with_retry but for async functions.
    """
    async def wrapper(*args: Any, **kwargs: Any) -> T:
        last_error: Optional[Exception] = None

        for attempt in range(max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_error = e

                # Check if we should retry
                if attempt >= max_retries or not should_retry(e, retry_on_status):
                    raise

                # Calculate delay
                if isinstance(e, RateLimitError):
                    delay = float(e.retry_after)
                else:
                    delay = calculate_backoff(
                        attempt,
                        initial_delay,
                        max_delay,
                        exponential_base
                    )

                # Call retry callback if provided
                if on_retry:
                    on_retry(attempt, e, delay)

                # Sleep before retry
                await asyncio.sleep(delay)

        if last_error:
            raise last_error
        raise RuntimeError("Retry logic error")

    return wrapper


class RetryHandler:
    """
    Configurable retry handler for API requests.

    Example:
        handler = RetryHandler(max_retries=5, initial_delay=0.5)

        @handler.wrap
        def make_request():
            return client._request("GET", "/models")

        result = make_request()
    """

    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 30.0,
        exponential_base: float = 2.0,
        retry_on_status: Optional[List[int]] = None,
        on_retry: Optional[Callable[[int, Exception, float], None]] = None
    ):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.retry_on_status = retry_on_status or [429, 500, 502, 503, 504]
        self.on_retry = on_retry

    def wrap(self, func: Callable[..., T]) -> Callable[..., T]:
        """Wrap a synchronous function with retry logic."""
        return with_retry(
            func,
            max_retries=self.max_retries,
            initial_delay=self.initial_delay,
            max_delay=self.max_delay,
            exponential_base=self.exponential_base,
            retry_on_status=self.retry_on_status,
            on_retry=self.on_retry
        )

    def wrap_async(self, func: Callable[..., T]) -> Callable[..., T]:
        """Wrap an async function with retry logic."""
        return with_retry_async(
            func,
            max_retries=self.max_retries,
            initial_delay=self.initial_delay,
            max_delay=self.max_delay,
            exponential_base=self.exponential_base,
            retry_on_status=self.retry_on_status,
            on_retry=self.on_retry
        )

    def execute(self, func: Callable[[], T]) -> T:
        """
        Execute a function with retry logic.

        Args:
            func: A callable that takes no arguments

        Returns:
            The result of the function call
        """
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                return func()
            except Exception as e:
                last_error = e

                # Check if we should retry
                if attempt >= self.max_retries or not should_retry(e, self.retry_on_status):
                    raise

                # Calculate delay
                if isinstance(e, RateLimitError):
                    delay = float(e.retry_after)
                else:
                    delay = calculate_backoff(
                        attempt,
                        self.initial_delay,
                        self.max_delay,
                        self.exponential_base
                    )

                # Call retry callback if provided
                if self.on_retry:
                    self.on_retry(attempt, e, delay)

                # Sleep before retry
                time.sleep(delay)

        if last_error:
            raise last_error
        raise RuntimeError("Retry logic error")

    async def execute_async(self, func: Callable[[], T]) -> T:
        """
        Execute an async function with retry logic.

        Args:
            func: An async callable that takes no arguments

        Returns:
            The result of the function call
        """
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                return await func()
            except Exception as e:
                last_error = e

                # Check if we should retry
                if attempt >= self.max_retries or not should_retry(e, self.retry_on_status):
                    raise

                # Calculate delay
                if isinstance(e, RateLimitError):
                    delay = float(e.retry_after)
                else:
                    delay = calculate_backoff(
                        attempt,
                        self.initial_delay,
                        self.max_delay,
                        self.exponential_base
                    )

                # Call retry callback if provided
                if self.on_retry:
                    self.on_retry(attempt, e, delay)

                # Sleep before retry
                await asyncio.sleep(delay)

        if last_error:
            raise last_error
        raise RuntimeError("Retry logic error")
