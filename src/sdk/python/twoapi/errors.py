"""
2api.ai SDK - Error Classes

Comprehensive error handling for API interactions.
"""

from typing import Optional, Dict, Any, Union


class TwoAPIError(Exception):
    """
    Base exception for 2api.ai SDK.

    All SDK errors inherit from this class.

    Attributes:
        message: Human-readable error message
        code: Error code for programmatic handling
        status_code: HTTP status code if applicable
        request_id: Request ID for support/debugging
        retryable: Whether the request can be retried
        details: Additional error details
    """

    def __init__(
        self,
        message: str,
        code: str = "unknown",
        status_code: int = 500,
        request_id: Optional[str] = None,
        retryable: bool = False,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.code = code
        self.status_code = status_code
        self.request_id = request_id
        self.retryable = retryable
        self.details = details or {}
        super().__init__(message)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"code={self.code!r}, "
            f"status_code={self.status_code})"
        )

    @classmethod
    def from_response(cls, response_data: Dict[str, Any], status_code: int) -> "TwoAPIError":
        """Create an error from API response data."""
        error = response_data.get("error", {})

        error_classes = {
            "authentication_error": AuthenticationError,
            "invalid_api_key": AuthenticationError,
            "missing_api_key": AuthenticationError,
            "rate_limit_exceeded": RateLimitError,
            "invalid_request": InvalidRequestError,
            "missing_required_field": InvalidRequestError,
            "provider_error": ProviderError,
            "provider_down": ProviderError,
            "timeout": TimeoutError,
            "connection_error": ConnectionError,
        }

        code = error.get("code", "unknown")
        error_class = error_classes.get(code, cls)

        return error_class(
            message=error.get("message", "Unknown error"),
            code=code,
            status_code=status_code,
            request_id=error.get("request_id"),
            retryable=error.get("retryable", False),
            details={
                "type": error.get("type"),
                "param": error.get("param"),
                "provider": error.get("provider"),
            }
        )


class AuthenticationError(TwoAPIError):
    """
    API key is invalid or missing.

    This error occurs when:
    - API key is not provided
    - API key format is invalid
    - API key has been revoked
    - API key doesn't have required permissions
    """

    def __init__(
        self,
        message: str = "Invalid or missing API key",
        code: str = "authentication_error",
        **kwargs
    ):
        kwargs.pop("retryable", None)  # Remove if passed, we force it
        super().__init__(
            message=message,
            code=code,
            status_code=kwargs.pop("status_code", 401),
            retryable=False,
            **kwargs
        )


class RateLimitError(TwoAPIError):
    """
    Rate limit exceeded.

    This error occurs when you've made too many requests.
    Check the retry_after attribute for when to retry.

    Attributes:
        retry_after: Seconds to wait before retrying
    """

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: int = 60,
        **kwargs
    ):
        kwargs.pop("retryable", None)  # Remove if passed, we force it
        kwargs.pop("code", None)  # Remove if passed, we force it
        super().__init__(
            message=message,
            code="rate_limit_exceeded",
            status_code=kwargs.pop("status_code", 429),
            retryable=True,
            **kwargs
        )
        self.retry_after = retry_after


class InvalidRequestError(TwoAPIError):
    """
    Request parameters are invalid.

    This error occurs when:
    - Required parameters are missing
    - Parameter values are out of range
    - Parameter formats are incorrect

    Attributes:
        param: The parameter that caused the error
    """

    def __init__(
        self,
        message: str,
        param: Optional[str] = None,
        **kwargs
    ):
        kwargs.pop("retryable", None)  # Remove if passed, we force it
        super().__init__(
            message=message,
            code=kwargs.pop("code", "invalid_request"),
            status_code=kwargs.pop("status_code", 400),
            retryable=False,
            **kwargs
        )
        self.param = param


class ProviderError(TwoAPIError):
    """
    Error from the AI provider.

    This error occurs when the underlying AI provider
    (OpenAI, Anthropic, Google) returns an error.

    Attributes:
        provider: The provider that caused the error
    """

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            code=kwargs.pop("code", "provider_error"),
            status_code=kwargs.pop("status_code", 502),
            retryable=kwargs.pop("retryable", True),
            **kwargs
        )
        self.provider = provider


class TimeoutError(TwoAPIError):
    """
    Request timed out.

    This error occurs when a request takes longer than
    the configured timeout.
    """

    def __init__(
        self,
        message: str = "Request timed out",
        **kwargs
    ):
        kwargs.pop("retryable", None)  # Remove if passed
        kwargs.pop("code", None)  # Remove if passed
        super().__init__(
            message=message,
            code="timeout",
            status_code=kwargs.pop("status_code", 408),
            retryable=True,
            **kwargs
        )


class ConnectionError(TwoAPIError):
    """
    Failed to connect to the API.

    This error occurs when:
    - Network is unavailable
    - DNS resolution fails
    - Connection is refused
    """

    def __init__(
        self,
        message: str = "Failed to connect to API",
        **kwargs
    ):
        kwargs.pop("retryable", None)  # Remove if passed
        kwargs.pop("code", None)  # Remove if passed
        super().__init__(
            message=message,
            code="connection_error",
            status_code=kwargs.pop("status_code", 503),
            retryable=True,
            **kwargs
        )


class StreamError(TwoAPIError):
    """
    Error during streaming response.

    This error occurs when the stream is interrupted
    before completion.

    Attributes:
        partial_content: Content received before error
    """

    def __init__(
        self,
        message: str = "Stream interrupted",
        partial_content: str = "",
        **kwargs
    ):
        kwargs.pop("retryable", None)  # Remove if passed
        kwargs.pop("code", None)  # Remove if passed
        super().__init__(
            message=message,
            code="stream_error",
            status_code=kwargs.pop("status_code", 500),
            retryable=False,  # Can't retry mid-stream
            **kwargs
        )
        self.partial_content = partial_content


def is_retryable_error(error: Any) -> bool:
    """
    Check if an error is retryable.

    Infrastructure errors (rate limits, timeouts, connection issues,
    provider errors) are retryable. Semantic errors (invalid request,
    authentication) are not.

    Args:
        error: The error to check

    Returns:
        True if the error is retryable, False otherwise
    """
    if isinstance(error, TwoAPIError):
        return error.retryable

    return False
