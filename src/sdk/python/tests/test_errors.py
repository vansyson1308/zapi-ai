"""
2api.ai Python SDK - Error Tests
"""

import pytest

from twoapi.errors import (
    TwoAPIError,
    AuthenticationError,
    RateLimitError,
    InvalidRequestError,
    ProviderError,
    TimeoutError,
    ConnectionError,
    StreamError,
    is_retryable_error,
)


class TestTwoAPIError:
    """Tests for TwoAPIError base class."""

    def test_error_creation(self):
        """Test creating TwoAPIError."""
        error = TwoAPIError("Something went wrong")
        assert str(error) == "Something went wrong"
        assert error.message == "Something went wrong"

    def test_error_with_status_code(self):
        """Test error with status code."""
        error = TwoAPIError("Error", status_code=500)
        assert error.status_code == 500

    def test_error_with_code(self):
        """Test error with error code."""
        error = TwoAPIError("Error", status_code=400, code="invalid_request")
        assert error.code == "invalid_request"

    def test_error_inheritance(self):
        """Test that TwoAPIError inherits from Exception."""
        error = TwoAPIError("Error")
        assert isinstance(error, Exception)

    def test_error_repr(self):
        """Test error string representation."""
        error = TwoAPIError("Test error", status_code=400, code="test_code")
        repr_str = repr(error)
        assert "TwoAPIError" in repr_str
        assert "Test error" in repr_str


class TestAuthenticationError:
    """Tests for AuthenticationError."""

    def test_authentication_error(self):
        """Test creating authentication error."""
        error = AuthenticationError("Invalid API key")
        assert str(error) == "Invalid API key"
        assert isinstance(error, TwoAPIError)

    def test_default_status_code(self):
        """Test default status code for authentication error."""
        error = AuthenticationError("Invalid key")
        # AuthenticationError typically has 401 status
        assert error.status_code == 401 or error.status_code == 0


class TestRateLimitError:
    """Tests for RateLimitError."""

    def test_rate_limit_error(self):
        """Test creating rate limit error."""
        error = RateLimitError("Rate limit exceeded", status_code=429)
        assert error.status_code == 429
        assert isinstance(error, TwoAPIError)

    def test_retry_after(self):
        """Test retry_after property."""
        error = RateLimitError("Rate limit", status_code=429, retry_after=60)
        assert error.retry_after == 60

    def test_default_retry_after(self):
        """Test default retry_after value."""
        error = RateLimitError("Rate limit", status_code=429)
        assert error.retry_after == 60  # Default should be 60 seconds


class TestInvalidRequestError:
    """Tests for InvalidRequestError."""

    def test_invalid_request_error(self):
        """Test creating invalid request error."""
        error = InvalidRequestError("Invalid parameters", status_code=400)
        assert error.status_code == 400
        assert isinstance(error, TwoAPIError)

    def test_with_error_code(self):
        """Test with specific error code."""
        error = InvalidRequestError(
            "Missing required field",
            status_code=400,
            code="missing_field"
        )
        assert error.code == "missing_field"


class TestProviderError:
    """Tests for ProviderError."""

    def test_provider_error(self):
        """Test creating provider error."""
        error = ProviderError("OpenAI unavailable", status_code=503)
        assert error.status_code == 503
        assert isinstance(error, TwoAPIError)

    def test_provider_attribute(self):
        """Test provider attribute."""
        error = ProviderError("Error", status_code=500, provider="openai")
        assert error.provider == "openai"

    def test_default_provider(self):
        """Test default provider value."""
        error = ProviderError("Error", status_code=500)
        assert error.provider == "unknown"


class TestTimeoutError:
    """Tests for TimeoutError."""

    def test_timeout_error(self):
        """Test creating timeout error."""
        error = TimeoutError("Request timed out")
        assert "timed out" in str(error).lower()
        assert isinstance(error, TwoAPIError)


class TestConnectionError:
    """Tests for ConnectionError."""

    def test_connection_error(self):
        """Test creating connection error."""
        error = ConnectionError("Failed to connect")
        assert "connect" in str(error).lower()
        assert isinstance(error, TwoAPIError)


class TestStreamError:
    """Tests for StreamError."""

    def test_stream_error(self):
        """Test creating stream error."""
        error = StreamError("Stream interrupted")
        assert "stream" in str(error).lower()
        assert isinstance(error, TwoAPIError)


class TestIsRetryableError:
    """Tests for is_retryable_error function."""

    def test_rate_limit_error_retryable(self):
        """Test that RateLimitError is retryable."""
        error = RateLimitError("Rate limit", status_code=429)
        assert is_retryable_error(error) is True

    def test_provider_error_retryable(self):
        """Test that ProviderError is retryable."""
        error = ProviderError("Provider down", status_code=503)
        assert is_retryable_error(error) is True

    def test_timeout_error_retryable(self):
        """Test that TimeoutError is retryable."""
        error = TimeoutError("Timeout")
        assert is_retryable_error(error) is True

    def test_connection_error_retryable(self):
        """Test that ConnectionError is retryable."""
        error = ConnectionError("Connection failed")
        assert is_retryable_error(error) is True

    def test_authentication_error_not_retryable(self):
        """Test that AuthenticationError is not retryable."""
        error = AuthenticationError("Invalid key")
        assert is_retryable_error(error) is False

    def test_invalid_request_error_not_retryable(self):
        """Test that InvalidRequestError is not retryable."""
        error = InvalidRequestError("Bad request", status_code=400)
        assert is_retryable_error(error) is False

    def test_stream_error_not_retryable(self):
        """Test that StreamError is not retryable."""
        error = StreamError("Stream error")
        assert is_retryable_error(error) is False

    def test_generic_exception_not_retryable(self):
        """Test that generic Exception is not retryable."""
        error = Exception("Generic error")
        assert is_retryable_error(error) is False

    def test_non_error_values(self):
        """Test non-error values return False."""
        assert is_retryable_error("string") is False
        assert is_retryable_error(123) is False
        assert is_retryable_error(None) is False
        assert is_retryable_error({}) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
