"""
2api.ai - Error System Tests

Tests for the unified error handling system (EPIC A).
Verifies:
- Error factory functions work correctly
- Provider error handlers map HTTP errors to canonical errors
- Streaming error chunks include partial_content
- Infra vs semantic classification is correct
"""

import pytest
import json
from unittest.mock import MagicMock, AsyncMock

from src.core.errors import (
    # Base classes
    TwoApiException,
    InfraError,
    SemanticError,
    ErrorType,
    ErrorDetails,
    # Specific errors
    ConnectionTimeoutError,
    ReadTimeoutError,
    UpstreamError,
    RateLimitedError,
    StreamInterruptedError,
    ProviderDownError,
    AllProvidersFailedError,
    InvalidAPIKeyError,
    MissingAPIKeyError,
    InvalidRequestError,
    ModelNotFoundError,
    ContentFilteredError,
    ContextLengthExceededError,
    # Factory functions
    create_error_from_provider,
    handle_openai_error,
    handle_anthropic_error,
    handle_google_error,
    create_stream_error_chunk,
    is_retryable_before_content,
)


# ============================================================
# Error Classification Tests
# ============================================================

class TestErrorClassification:
    """Test infra vs semantic error classification."""

    def test_connection_timeout_is_infra(self):
        """ConnectionTimeoutError is infra and retryable."""
        error = ConnectionTimeoutError("openai", "req_123")
        assert isinstance(error, InfraError)
        assert error.error.type == ErrorType.INFRA
        assert error.error.retryable is True

    def test_rate_limited_is_infra(self):
        """RateLimitedError is infra and retryable."""
        error = RateLimitedError("openai", 60, request_id="req_123")
        assert isinstance(error, InfraError)
        assert error.error.type == ErrorType.INFRA
        assert error.error.retryable is True
        assert error.error.retry_after == 60

    def test_upstream_500_is_infra(self):
        """UpstreamError (5xx) is infra and retryable."""
        error = UpstreamError("anthropic", 503, "Service unavailable")
        assert isinstance(error, InfraError)
        assert error.error.type == ErrorType.INFRA
        assert error.error.retryable is True

    def test_stream_interrupted_is_infra_but_not_retryable(self):
        """StreamInterruptedError is infra but NOT retryable (semantic drift)."""
        error = StreamInterruptedError("openai", "Hello, I", "req_123")
        assert isinstance(error, InfraError)
        assert error.error.type == ErrorType.INFRA
        assert error.error.retryable is False  # Critical: no retry after content
        assert error.error.partial_content == "Hello, I"

    def test_invalid_api_key_is_semantic(self):
        """InvalidAPIKeyError is semantic and not retryable."""
        error = InvalidAPIKeyError("Bad key", "req_123")
        assert isinstance(error, SemanticError)
        assert error.error.type == ErrorType.SEMANTIC
        assert error.error.retryable is False

    def test_invalid_request_is_semantic(self):
        """InvalidRequestError is semantic and not retryable."""
        error = InvalidRequestError("messages required", "messages", "req_123")
        assert isinstance(error, SemanticError)
        assert error.error.type == ErrorType.SEMANTIC
        assert error.error.retryable is False
        assert error.error.param == "messages"

    def test_model_not_found_is_semantic(self):
        """ModelNotFoundError is semantic and not retryable."""
        error = ModelNotFoundError("gpt-5", "req_123")
        assert isinstance(error, SemanticError)
        assert error.error.type == ErrorType.SEMANTIC
        assert error.error.retryable is False

    def test_content_filtered_is_semantic(self):
        """ContentFilteredError is semantic and not retryable."""
        error = ContentFilteredError("openai", "safety violation", "req_123")
        assert isinstance(error, SemanticError)
        assert error.error.type == ErrorType.SEMANTIC
        assert error.error.retryable is False


# ============================================================
# Error Details Tests
# ============================================================

class TestErrorDetails:
    """Test ErrorDetails serialization."""

    def test_to_dict_includes_required_fields(self):
        """to_dict must include all required fields."""
        details = ErrorDetails(
            code="rate_limited",
            message="Rate limit exceeded",
            type=ErrorType.INFRA,
            request_id="req_123",
            retryable=True,
            retry_after=60,
            provider="openai"
        )
        result = details.to_dict()

        assert "error" in result
        error = result["error"]
        assert error["code"] == "rate_limited"
        assert error["message"] == "Rate limit exceeded"
        assert error["type"] == "infra_error"
        assert error["request_id"] == "req_123"
        assert error["retryable"] is True
        assert error["retry_after"] == 60
        assert error["provider"] == "openai"

    def test_to_dict_excludes_none_fields(self):
        """to_dict should exclude None optional fields."""
        details = ErrorDetails(
            code="invalid_request",
            message="Bad request",
            type=ErrorType.SEMANTIC,
            request_id="req_123",
            retryable=False
        )
        result = details.to_dict()
        error = result["error"]

        # These should not be present since they're None
        assert "provider" not in error
        assert "retry_after" not in error
        assert "partial_content" not in error


# ============================================================
# Provider Error Handler Tests
# ============================================================

class TestOpenAIErrorHandler:
    """Test OpenAI error handler."""

    def test_handles_timeout(self):
        """Timeout should map to ConnectionTimeoutError."""
        import httpx
        error = httpx.ConnectTimeout("Connection timed out")
        result = handle_openai_error(error, "req_123")

        assert isinstance(result, ConnectionTimeoutError)
        assert result.error.code == "connection_timeout"
        assert result.error.provider == "openai"

    def test_handles_rate_limit(self):
        """429 should map to RateLimitedError."""
        import httpx

        response = MagicMock()
        response.status_code = 429
        response.headers = {"retry-after": "30"}
        response.json.return_value = {
            "error": {
                "message": "Rate limit exceeded",
                "type": "rate_limit_error"
            }
        }

        error = httpx.HTTPStatusError("Rate limited", request=MagicMock(), response=response)
        result = handle_openai_error(error, "req_123")

        assert isinstance(result, RateLimitedError)
        assert result.error.code == "rate_limited"
        assert result.error.retry_after == 30

    def test_handles_auth_error(self):
        """401 should map to SemanticError with provider_auth_error."""
        import httpx

        response = MagicMock()
        response.status_code = 401
        response.headers = {}
        response.json.return_value = {
            "error": {
                "message": "Invalid API key",
                "type": "authentication_error"
            }
        }

        error = httpx.HTTPStatusError("Auth failed", request=MagicMock(), response=response)
        result = handle_openai_error(error, "req_123")

        assert isinstance(result, SemanticError)
        assert result.error.code == "provider_auth_error"
        assert result.error.retryable is False

    def test_handles_server_error(self):
        """5xx should map to UpstreamError."""
        import httpx

        response = MagicMock()
        response.status_code = 503
        response.headers = {}
        response.json.return_value = {
            "error": {
                "message": "Service unavailable"
            }
        }

        error = httpx.HTTPStatusError("Server error", request=MagicMock(), response=response)
        result = handle_openai_error(error, "req_123")

        assert isinstance(result, UpstreamError)
        assert result.error.type == ErrorType.INFRA
        assert result.error.retryable is True


class TestAnthropicErrorHandler:
    """Test Anthropic error handler."""

    def test_handles_overload(self):
        """529 should map to UpstreamError."""
        import httpx

        response = MagicMock()
        response.status_code = 529
        response.headers = {}
        response.json.return_value = {
            "error": {
                "type": "overloaded_error",
                "message": "Anthropic is overloaded"
            }
        }

        error = httpx.HTTPStatusError("Overloaded", request=MagicMock(), response=response)
        result = handle_anthropic_error(error, "req_123")

        assert isinstance(result, UpstreamError)
        assert result.error.type == ErrorType.INFRA


class TestGoogleErrorHandler:
    """Test Google error handler."""

    def test_handles_resource_exhausted(self):
        """RESOURCE_EXHAUSTED should map to RateLimitedError."""
        import httpx

        response = MagicMock()
        response.status_code = 429
        response.headers = {}
        response.json.return_value = {
            "error": {
                "code": 429,
                "status": "RESOURCE_EXHAUSTED",
                "message": "Quota exceeded"
            }
        }

        error = httpx.HTTPStatusError("Quota", request=MagicMock(), response=response)
        result = handle_google_error(error, "req_123")

        assert isinstance(result, RateLimitedError)
        assert result.error.provider == "google"


# ============================================================
# Streaming Error Tests
# ============================================================

class TestStreamingErrors:
    """Test streaming error handling."""

    def test_create_stream_error_chunk_format(self):
        """Error chunk must have correct SSE format."""
        error = StreamInterruptedError("openai", "Hello, I can", "req_123")
        chunk = create_stream_error_chunk(error, "Hello, I can")

        # Should be SSE formatted
        assert chunk.startswith("data: ")
        assert chunk.endswith("data: [DONE]\n\n")

        # Parse the error part
        lines = chunk.strip().split("\n\n")
        error_line = lines[0]
        assert error_line.startswith("data: ")
        error_data = json.loads(error_line[6:])

        assert "error" in error_data
        assert error_data["error"]["code"] == "stream_interrupted"
        assert error_data["error"]["partial_content"] == "Hello, I can"
        assert error_data["choices"][0]["finish_reason"] == "error"

    def test_is_retryable_before_content_true_for_infra(self):
        """Infra error without partial_content is retryable."""
        error = ConnectionTimeoutError("openai", "req_123")
        assert is_retryable_before_content(error) is True

    def test_is_retryable_before_content_false_for_stream_interrupted(self):
        """StreamInterruptedError is never retryable (has partial_content)."""
        error = StreamInterruptedError("openai", "Hello", "req_123")
        assert is_retryable_before_content(error) is False

    def test_is_retryable_before_content_false_for_semantic(self):
        """Semantic errors are never retryable."""
        error = InvalidRequestError("bad request", "", "req_123")
        assert is_retryable_before_content(error) is False


# ============================================================
# HTTP Status Code Mapping Tests
# ============================================================

class TestHTTPStatusMapping:
    """Test HTTP status code to error type mapping."""

    @pytest.mark.parametrize("status,expected_type", [
        (400, ErrorType.SEMANTIC),
        (401, ErrorType.SEMANTIC),
        (403, ErrorType.SEMANTIC),
        (404, ErrorType.SEMANTIC),
        (422, ErrorType.SEMANTIC),
        (429, ErrorType.INFRA),  # Rate limit is retryable
        (500, ErrorType.INFRA),
        (502, ErrorType.INFRA),
        (503, ErrorType.INFRA),
        (504, ErrorType.INFRA),
    ])
    def test_status_code_to_error_type(self, status, expected_type):
        """HTTP status codes should map to correct error types."""
        error = create_error_from_provider(
            provider="openai",
            status_code=status,
            error_body={"message": "Test error"},
            request_id="req_123"
        )
        assert error.error.type == expected_type


# ============================================================
# Run Tests
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
