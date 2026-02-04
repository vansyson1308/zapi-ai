"""
2api.ai - Streaming Error Handling

Handles errors that occur during streaming with proper
"no semantic drift" protection.

Key principle:
- BEFORE content: raise exception (caller can retry/fallback)
- AFTER content started: yield error chunk with partial_content, then stop

This module provides:
- Error chunk formatting
- Partial content tracking
- Error recovery coordination
"""

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class StreamErrorType(str, Enum):
    """Types of streaming errors."""
    # Pre-content errors (retryable)
    CONNECTION_FAILED = "connection_failed"
    AUTHENTICATION_FAILED = "authentication_failed"
    RATE_LIMITED = "rate_limited"
    PROVIDER_UNAVAILABLE = "provider_unavailable"

    # Post-content errors (NOT retryable - semantic drift protection)
    STREAM_INTERRUPTED = "stream_interrupted"
    CONTENT_FILTER_TRIGGERED = "content_filter_triggered"
    MAX_TOKENS_REACHED = "max_tokens_reached"
    TIMEOUT_DURING_STREAM = "timeout_during_stream"


@dataclass
class StreamError:
    """
    Represents an error that occurred during streaming.

    Contains all information needed to properly communicate
    the error to the client.
    """
    type: StreamErrorType
    code: str
    message: str
    provider: str
    request_id: str

    # Timing
    occurred_at: float = field(default_factory=time.time)

    # Content state at error time
    content_started: bool = False
    partial_content: Optional[str] = None
    chunks_delivered: int = 0

    # Error details
    original_error: Optional[str] = None
    http_status: Optional[int] = None
    retry_after: Optional[int] = None

    @property
    def is_retryable(self) -> bool:
        """
        Check if error is retryable.

        CRITICAL: NOT retryable if content has started (semantic drift protection).
        """
        if self.content_started:
            return False

        retryable_types = {
            StreamErrorType.CONNECTION_FAILED,
            StreamErrorType.RATE_LIMITED,
            StreamErrorType.PROVIDER_UNAVAILABLE,
            StreamErrorType.TIMEOUT_DURING_STREAM,
        }
        return self.type in retryable_types

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for SSE transmission."""
        result = {
            "error": {
                "type": self.type.value,
                "code": self.code,
                "message": self.message,
                "provider": self.provider,
                "request_id": self.request_id,
                "retryable": self.is_retryable,
            }
        }

        # Include partial content if any was delivered
        if self.partial_content:
            result["partial_content"] = self.partial_content
            result["chunks_delivered"] = self.chunks_delivered

        # Include retry-after for rate limits
        if self.retry_after:
            result["error"]["retry_after"] = self.retry_after

        return result

    def to_sse(self) -> str:
        """Convert to SSE format."""
        return f"data: {json.dumps(self.to_dict())}\n\n"


class StreamErrorBuilder:
    """Builder for creating stream errors with context."""

    def __init__(self, provider: str, request_id: str):
        self.provider = provider
        self.request_id = request_id
        self._content_started = False
        self._partial_content = ""
        self._chunks_delivered = 0

    def set_content_state(
        self,
        content_started: bool,
        partial_content: str = "",
        chunks_delivered: int = 0
    ):
        """Set the content state at error time."""
        self._content_started = content_started
        self._partial_content = partial_content
        self._chunks_delivered = chunks_delivered

    def connection_failed(
        self,
        message: str = "Failed to connect to provider",
        original_error: Optional[str] = None
    ) -> StreamError:
        """Create a connection failed error."""
        return StreamError(
            type=StreamErrorType.CONNECTION_FAILED,
            code="connection_failed",
            message=message,
            provider=self.provider,
            request_id=self.request_id,
            content_started=self._content_started,
            partial_content=self._partial_content if self._partial_content else None,
            chunks_delivered=self._chunks_delivered,
            original_error=original_error
        )

    def authentication_failed(
        self,
        message: str = "Authentication with provider failed"
    ) -> StreamError:
        """Create an authentication failed error."""
        return StreamError(
            type=StreamErrorType.AUTHENTICATION_FAILED,
            code="authentication_failed",
            message=message,
            provider=self.provider,
            request_id=self.request_id,
            content_started=self._content_started,
            partial_content=self._partial_content if self._partial_content else None,
            chunks_delivered=self._chunks_delivered,
            http_status=401
        )

    def rate_limited(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[int] = None
    ) -> StreamError:
        """Create a rate limited error."""
        return StreamError(
            type=StreamErrorType.RATE_LIMITED,
            code="rate_limited",
            message=message,
            provider=self.provider,
            request_id=self.request_id,
            content_started=self._content_started,
            partial_content=self._partial_content if self._partial_content else None,
            chunks_delivered=self._chunks_delivered,
            http_status=429,
            retry_after=retry_after or 60
        )

    def stream_interrupted(
        self,
        message: str = "Stream was interrupted"
    ) -> StreamError:
        """
        Create a stream interrupted error.

        This is the most common post-content error type.
        """
        return StreamError(
            type=StreamErrorType.STREAM_INTERRUPTED,
            code="stream_interrupted",
            message=message,
            provider=self.provider,
            request_id=self.request_id,
            content_started=self._content_started,
            partial_content=self._partial_content if self._partial_content else None,
            chunks_delivered=self._chunks_delivered
        )

    def content_filtered(
        self,
        message: str = "Content was filtered by safety system"
    ) -> StreamError:
        """Create a content filter error."""
        return StreamError(
            type=StreamErrorType.CONTENT_FILTER_TRIGGERED,
            code="content_filter",
            message=message,
            provider=self.provider,
            request_id=self.request_id,
            content_started=self._content_started,
            partial_content=self._partial_content if self._partial_content else None,
            chunks_delivered=self._chunks_delivered
        )

    def timeout(
        self,
        message: str = "Request timed out during streaming"
    ) -> StreamError:
        """Create a timeout error."""
        return StreamError(
            type=StreamErrorType.TIMEOUT_DURING_STREAM,
            code="timeout",
            message=message,
            provider=self.provider,
            request_id=self.request_id,
            content_started=self._content_started,
            partial_content=self._partial_content if self._partial_content else None,
            chunks_delivered=self._chunks_delivered
        )

    def from_exception(
        self,
        exception: Exception,
        http_status: Optional[int] = None
    ) -> StreamError:
        """
        Create a stream error from an exception.

        Attempts to classify the exception type and create
        an appropriate error.
        """
        error_message = str(exception)
        error_type = StreamErrorType.STREAM_INTERRUPTED
        code = "internal_error"

        # Classify based on exception type/message
        error_lower = error_message.lower()

        if "timeout" in error_lower or "timed out" in error_lower:
            error_type = StreamErrorType.TIMEOUT_DURING_STREAM
            code = "timeout"
        elif "rate" in error_lower or "429" in error_lower:
            error_type = StreamErrorType.RATE_LIMITED
            code = "rate_limited"
        elif "auth" in error_lower or "401" in error_lower:
            error_type = StreamErrorType.AUTHENTICATION_FAILED
            code = "authentication_failed"
        elif "connection" in error_lower or "connect" in error_lower:
            error_type = StreamErrorType.CONNECTION_FAILED
            code = "connection_failed"

        return StreamError(
            type=error_type,
            code=code,
            message=error_message,
            provider=self.provider,
            request_id=self.request_id,
            content_started=self._content_started,
            partial_content=self._partial_content if self._partial_content else None,
            chunks_delivered=self._chunks_delivered,
            original_error=error_message,
            http_status=http_status
        )


def create_error_sse_chunk(
    error_code: str,
    error_message: str,
    provider: str,
    request_id: str,
    partial_content: Optional[str] = None,
    retryable: bool = False,
    retry_after: Optional[int] = None
) -> str:
    """
    Create an SSE error chunk.

    This is the standard format for communicating errors
    during streaming.
    """
    error_data: Dict[str, Any] = {
        "error": {
            "code": error_code,
            "message": error_message,
            "type": "stream_error",
            "provider": provider,
            "request_id": request_id,
            "retryable": retryable
        }
    }

    if partial_content:
        error_data["partial_content"] = partial_content
        error_data["partial_content_length"] = len(partial_content)

    if retry_after:
        error_data["error"]["retry_after"] = retry_after

    return f"data: {json.dumps(error_data)}\n\n"


def should_retry_stream_error(
    error: StreamError,
    content_started: bool,
    attempt_count: int,
    max_attempts: int = 3
) -> bool:
    """
    Determine if a stream error should be retried.

    CRITICAL: Never retry if content has started (semantic drift protection).
    """
    # Absolute rule: no retry after content
    if content_started:
        return False

    # Check attempt count
    if attempt_count >= max_attempts:
        return False

    # Check if error type is retryable
    return error.is_retryable
