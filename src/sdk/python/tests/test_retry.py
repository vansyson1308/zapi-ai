"""
2api.ai Python SDK - Retry Tests
"""

import pytest
from unittest.mock import MagicMock, patch
import time

from twoapi.retry import (
    RetryHandler,
    calculate_backoff,
    with_retry,
)
from twoapi.errors import (
    RateLimitError,
    ProviderError,
    InvalidRequestError,
    TimeoutError,
)


class TestCalculateBackoff:
    """Tests for calculate_backoff function."""

    def test_exponential_delay(self):
        """Test exponential delay calculation."""
        delay0 = calculate_backoff(0, initial_delay=1.0, jitter=False)
        delay1 = calculate_backoff(1, initial_delay=1.0, jitter=False)
        delay2 = calculate_backoff(2, initial_delay=1.0, jitter=False)

        assert delay0 == 1.0
        assert delay1 == 2.0
        assert delay2 == 4.0

    def test_max_delay_cap(self):
        """Test that delay is capped at max_delay."""
        delay = calculate_backoff(
            10,
            initial_delay=1.0,
            max_delay=5.0,
            jitter=False
        )

        assert delay == 5.0

    def test_jitter_variation(self):
        """Test that jitter produces varied delays."""
        delays = set()
        for _ in range(20):
            delay = calculate_backoff(1, initial_delay=1.0, jitter=True)
            delays.add(round(delay, 3))

        # With jitter, we should get varied delays
        assert len(delays) > 1

    def test_custom_exponential_base(self):
        """Test custom exponential base."""
        delay = calculate_backoff(
            2,
            initial_delay=1.0,
            exponential_base=3,
            jitter=False
        )

        assert delay == 9.0  # 1.0 * 3^2

    def test_default_values(self):
        """Test default parameter values."""
        delay = calculate_backoff(0)
        assert delay > 0  # Should have some default delay


class TestRetryHandler:
    """Tests for RetryHandler class."""

    def test_success_no_retry(self):
        """Test successful call with no retry needed."""
        handler = RetryHandler()
        fn = MagicMock(return_value="success")

        result = handler.execute(fn)

        assert result == "success"
        assert fn.call_count == 1

    def test_retry_on_retryable_error(self):
        """Test retry on retryable error."""
        handler = RetryHandler(max_retries=3, initial_delay=0.01)
        fn = MagicMock(side_effect=[
            ProviderError("Error", status_code=500),
            ProviderError("Error", status_code=502),
            "success",
        ])

        result = handler.execute(fn)

        assert result == "success"
        assert fn.call_count == 3

    def test_no_retry_on_non_retryable_error(self):
        """Test no retry on non-retryable error."""
        handler = RetryHandler(max_retries=3)
        fn = MagicMock(side_effect=InvalidRequestError("Bad request", status_code=400))

        with pytest.raises(InvalidRequestError):
            handler.execute(fn)

        assert fn.call_count == 1

    def test_max_retries_exceeded(self):
        """Test that error is raised after max retries."""
        handler = RetryHandler(max_retries=2, initial_delay=0.01)
        fn = MagicMock(side_effect=ProviderError("Error", status_code=500))

        with pytest.raises(ProviderError):
            handler.execute(fn)

        assert fn.call_count == 3  # Initial + 2 retries

    def test_rate_limit_uses_retry_after(self):
        """Test that rate limit error uses retry-after header."""
        handler = RetryHandler(max_retries=1, initial_delay=0.01)

        # Use a very short retry_after for testing
        fn = MagicMock(side_effect=[
            RateLimitError("Rate limited", status_code=429, retry_after=0.05),
            "success",
        ])

        start_time = time.time()
        result = handler.execute(fn)
        elapsed = time.time() - start_time

        assert result == "success"
        # Should have waited at least the retry_after duration
        assert elapsed >= 0.05

    def test_on_retry_callback(self):
        """Test on_retry callback is invoked."""
        on_retry = MagicMock()
        handler = RetryHandler(max_retries=2, initial_delay=0.01, on_retry=on_retry)
        fn = MagicMock(side_effect=[
            ProviderError("Error", status_code=500),
            "success",
        ])

        handler.execute(fn)

        on_retry.assert_called_once()
        call_args = on_retry.call_args[0]
        assert call_args[0] == 0  # attempt number
        assert isinstance(call_args[1], ProviderError)  # error
        assert isinstance(call_args[2], float)  # delay

    def test_custom_retry_on_status(self):
        """Test custom retry_on_status codes."""
        handler = RetryHandler(
            max_retries=1,
            initial_delay=0.01,
            retry_on_status=[503],  # Only retry on 503
        )

        # 500 should not retry
        fn500 = MagicMock(side_effect=ProviderError("Error", status_code=500))
        with pytest.raises(ProviderError):
            handler.execute(fn500)
        assert fn500.call_count == 1

        # 503 should retry
        fn503 = MagicMock(side_effect=[
            ProviderError("Error", status_code=503),
            "success",
        ])
        result = handler.execute(fn503)
        assert result == "success"
        assert fn503.call_count == 2


class TestAsyncRetryHandler:
    """Tests for async RetryHandler operations."""

    @pytest.mark.asyncio
    async def test_async_execute_success(self):
        """Test async execute with success."""
        handler = RetryHandler()

        async def async_fn():
            return "async success"

        result = await handler.execute_async(async_fn)
        assert result == "async success"

    @pytest.mark.asyncio
    async def test_async_retry_on_error(self):
        """Test async retry on error."""
        handler = RetryHandler(max_retries=2, initial_delay=0.01)
        call_count = 0

        async def async_fn():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ProviderError("Error", status_code=500)
            return "success"

        result = await handler.execute_async(async_fn)
        assert result == "success"
        assert call_count == 3


class TestWithRetryDecorator:
    """Tests for with_retry decorator."""

    def test_decorator_wraps_function(self):
        """Test that decorator wraps function with retry logic."""
        call_count = 0

        @with_retry(max_retries=2, initial_delay=0.01)
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ProviderError("Error", status_code=500)
            return "success"

        result = flaky_function()
        assert result == "success"
        assert call_count == 2

    def test_decorator_preserves_arguments(self):
        """Test that decorator preserves function arguments."""
        @with_retry()
        def add(a, b):
            return a + b

        result = add(3, 4)
        assert result == 7

    def test_decorator_preserves_kwargs(self):
        """Test that decorator preserves keyword arguments."""
        @with_retry()
        def greet(name, greeting="Hello"):
            return f"{greeting}, {name}!"

        result = greet("World", greeting="Hi")
        assert result == "Hi, World!"

    @pytest.mark.asyncio
    async def test_decorator_async_function(self):
        """Test decorator with async function."""
        call_count = 0

        @with_retry(max_retries=2, initial_delay=0.01)
        async def async_flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ProviderError("Error", status_code=500)
            return "async success"

        result = await async_flaky()
        assert result == "async success"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
