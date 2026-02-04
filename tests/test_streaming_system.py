"""
2api.ai - Streaming System Tests

Comprehensive tests for EPIC D - Streaming + SSE Normalization.
Verifies:
- Stream normalizer produces consistent output
- Tool call streaming accumulation
- Error handling with semantic drift protection
- SSE format compliance
"""

import pytest
import json
import time

from src.streaming.normalizer import (
    StreamNormalizer,
    StreamChunk,
    StreamState,
    StreamEventType,
    ToolCallDelta,
    create_normalized_stream,
)
from src.streaming.tool_calls import (
    ToolCallAccumulator,
    ToolCallStreamTracker,
    StreamingToolCallChunk,
    merge_tool_call_deltas,
    create_tool_call_delta_chunk,
)
from src.streaming.errors import (
    StreamError,
    StreamErrorType,
    StreamErrorBuilder,
    create_error_sse_chunk,
    should_retry_stream_error,
)


# ============================================================
# Stream Normalizer Tests
# ============================================================

class TestStreamChunk:
    """Test StreamChunk data structure."""

    def test_to_sse_format(self):
        """Chunk should serialize to valid SSE format."""
        chunk = StreamChunk(
            id="chatcmpl-123",
            model="gpt-4o",
            provider="openai",
            content_delta="Hello",
        )

        sse = chunk.to_sse()

        assert sse.startswith("data: ")
        assert sse.endswith("\n\n")

        # Parse the JSON part
        json_str = sse[6:-2]  # Remove "data: " and "\n\n"
        data = json.loads(json_str)

        assert data["id"] == "chatcmpl-123"
        assert data["model"] == "gpt-4o"
        assert data["provider"] == "openai"
        assert data["choices"][0]["delta"]["content"] == "Hello"

    def test_to_dict_includes_required_fields(self):
        """Chunk dict should have all required OpenAI fields."""
        chunk = StreamChunk(
            id="chatcmpl-123",
            model="gpt-4o",
            provider="openai",
        )

        data = chunk.to_dict()

        assert "id" in data
        assert "object" in data
        assert "created" in data
        assert "model" in data
        assert "choices" in data
        assert data["object"] == "chat.completion.chunk"

    def test_tool_call_serialization(self):
        """Tool calls should serialize correctly."""
        tc = ToolCallDelta(
            index=0,
            id="call_123",
            function_name="get_weather",
            function_arguments_delta='{"location": "NYC"}'
        )

        chunk = StreamChunk(
            id="chatcmpl-123",
            model="gpt-4o",
            provider="openai",
            tool_calls=[tc]
        )

        data = chunk.to_dict()
        tool_calls = data["choices"][0]["delta"]["tool_calls"]

        assert len(tool_calls) == 1
        assert tool_calls[0]["index"] == 0
        assert tool_calls[0]["id"] == "call_123"
        assert tool_calls[0]["function"]["name"] == "get_weather"


class TestStreamState:
    """Test StreamState tracking."""

    def test_initial_state(self):
        """Initial state should allow fallback."""
        state = StreamState(request_id="req_123")

        assert state.content_started is False
        assert state.can_fallback() is True
        assert state.accumulated_content == ""

    def test_mark_content_started(self):
        """Marking content started should block fallback."""
        state = StreamState(request_id="req_123")

        state.mark_content_started("Hello")

        assert state.content_started is True
        assert state.can_fallback() is False
        assert state.accumulated_content == "Hello"

    def test_append_content(self):
        """Content should accumulate correctly."""
        state = StreamState(request_id="req_123")

        state.mark_content_started("Hello")
        state.append_content(" world")
        state.append_content("!")

        assert state.accumulated_content == "Hello world!"
        assert state.chunks_sent == 2  # append_content increments

    def test_tool_call_tracking(self):
        """Tool calls should accumulate correctly."""
        state = StreamState(request_id="req_123")

        # First delta - ID and name
        state.update_tool_call(0, ToolCallDelta(
            index=0,
            id="call_123",
            function_name="get_weather"
        ))

        # Second delta - arguments
        state.update_tool_call(0, ToolCallDelta(
            index=0,
            function_arguments_delta='{"location":'
        ))

        # Third delta - more arguments
        state.update_tool_call(0, ToolCallDelta(
            index=0,
            function_arguments_delta=' "NYC"}'
        ))

        calls = state.finalize_tool_calls()

        assert len(calls) == 1
        assert calls[0]["id"] == "call_123"
        assert calls[0]["function"]["name"] == "get_weather"
        assert calls[0]["function"]["arguments"] == '{"location": "NYC"}'


class TestStreamNormalizer:
    """Test StreamNormalizer behavior."""

    def test_create_content_chunk(self):
        """Should create valid content chunks."""
        normalizer = StreamNormalizer(
            model="gpt-4o",
            provider="openai",
            request_id="req_123"
        )

        chunk = normalizer.create_content_chunk("Hello")

        assert chunk.content_delta == "Hello"
        assert normalizer.has_content_started() is True

    def test_create_first_chunk_with_role(self):
        """First chunk should have role."""
        normalizer = StreamNormalizer(
            model="gpt-4o",
            provider="openai",
            request_id="req_123"
        )

        chunk = normalizer.create_first_chunk()

        assert chunk.role == "assistant"

    def test_create_final_chunk(self):
        """Final chunk should have finish reason."""
        normalizer = StreamNormalizer(
            model="gpt-4o",
            provider="openai",
            request_id="req_123"
        )

        chunk = normalizer.create_final_chunk("stop")

        assert chunk.finish_reason == "stop"

    def test_create_done_event(self):
        """Done event should be [DONE]."""
        normalizer = StreamNormalizer(
            model="gpt-4o",
            provider="openai",
            request_id="req_123"
        )

        done = normalizer.create_done_event()

        assert done == "data: [DONE]\n\n"

    def test_accumulates_content(self):
        """Should accumulate content for partial tracking."""
        normalizer = StreamNormalizer(
            model="gpt-4o",
            provider="openai",
            request_id="req_123"
        )

        normalizer.create_content_chunk("Hello")
        normalizer.create_content_chunk(" world")

        assert normalizer.get_accumulated_content() == "Hello world"

    def test_normalize_openai_chunk(self):
        """Should normalize OpenAI chunks."""
        normalizer = StreamNormalizer(
            model="gpt-4o",
            provider="openai",
            request_id="req_123"
        )

        openai_chunk = {
            "id": "chatcmpl-123",
            "choices": [{
                "index": 0,
                "delta": {"content": "Hello"},
                "finish_reason": None
            }]
        }

        chunk = normalizer.normalize_openai_chunk(openai_chunk)

        assert chunk is not None
        assert chunk.content_delta == "Hello"

    def test_normalize_anthropic_text_delta(self):
        """Should normalize Anthropic text deltas."""
        normalizer = StreamNormalizer(
            model="claude-3-opus",
            provider="anthropic",
            request_id="req_123"
        )

        chunk = normalizer.normalize_anthropic_event(
            "content_block_delta",
            {"delta": {"type": "text_delta", "text": "Hello"}}
        )

        assert chunk is not None
        assert chunk.content_delta == "Hello"

    def test_normalize_google_chunk(self):
        """Should normalize Google Gemini chunks."""
        normalizer = StreamNormalizer(
            model="gemini-1.5-pro",
            provider="google",
            request_id="req_123"
        )

        google_chunk = {
            "candidates": [{
                "content": {
                    "parts": [{"text": "Hello"}]
                }
            }]
        }

        chunk = normalizer.normalize_google_chunk(google_chunk)

        assert chunk is not None
        assert chunk.content_delta == "Hello"


class TestCreateNormalizedStream:
    """Test factory function."""

    def test_creates_normalizer(self):
        """Should create normalizer with correct settings."""
        normalizer = create_normalized_stream(
            provider="openai",
            model="gpt-4o",
            request_id="req_123"
        )

        assert normalizer.state.model == "gpt-4o"
        assert normalizer.state.provider == "openai"
        assert normalizer.state.request_id == "req_123"


# ============================================================
# Tool Call Streaming Tests
# ============================================================

class TestToolCallAccumulator:
    """Test tool call accumulation."""

    def test_update_accumulates(self):
        """Updates should accumulate correctly."""
        acc = ToolCallAccumulator(index=0)

        acc.update(id="call_123")
        acc.update(function_name="get_weather")
        acc.update(arguments_delta='{"loc":')
        acc.update(arguments_delta=' "NYC"}')

        assert acc.id == "call_123"
        assert acc.function_name == "get_weather"
        assert acc.arguments_buffer == '{"loc": "NYC"}'

    def test_to_dict(self):
        """Should convert to OpenAI format."""
        acc = ToolCallAccumulator(index=0)
        acc.update(id="call_123", function_name="test", arguments_delta='{}')

        data = acc.to_dict()

        assert data["id"] == "call_123"
        assert data["type"] == "function"
        assert data["function"]["name"] == "test"
        assert data["function"]["arguments"] == '{}'

    def test_validate_valid(self):
        """Valid tool call should pass validation."""
        acc = ToolCallAccumulator(index=0)
        acc.update(id="call_123", function_name="test", arguments_delta='{"key": "value"}')

        is_valid, error = acc.validate()

        assert is_valid is True
        assert error is None

    def test_validate_missing_id(self):
        """Missing ID should fail validation."""
        acc = ToolCallAccumulator(index=0)
        acc.update(function_name="test", arguments_delta='{}')

        is_valid, error = acc.validate()

        assert is_valid is False
        assert "ID" in error

    def test_validate_invalid_json(self):
        """Invalid JSON should fail validation."""
        acc = ToolCallAccumulator(index=0)
        acc.update(id="call_123", function_name="test", arguments_delta='not json')

        is_valid, error = acc.validate()

        assert is_valid is False
        assert "JSON" in error


class TestToolCallStreamTracker:
    """Test tracking multiple tool calls."""

    def test_track_multiple_calls(self):
        """Should track multiple parallel tool calls."""
        tracker = ToolCallStreamTracker()

        # First tool call
        tracker.update_call(0, id="call_1", function_name="func1")
        tracker.update_call(0, arguments_delta='{"a": 1}')

        # Second tool call
        tracker.update_call(1, id="call_2", function_name="func2")
        tracker.update_call(1, arguments_delta='{"b": 2}')

        assert tracker.call_count() == 2

        calls = tracker.to_list()
        assert calls[0]["id"] == "call_1"
        assert calls[1]["id"] == "call_2"

    def test_finalize(self):
        """Finalize should mark all calls complete."""
        tracker = ToolCallStreamTracker()

        tracker.update_call(0, id="call_1", function_name="func1")
        tracker.finalize()

        assert tracker.is_finalized() is True
        assert tracker.get_call(0).is_complete is True


class TestStreamingToolCallChunk:
    """Test tool call chunk parsing."""

    def test_from_openai(self):
        """Should parse OpenAI format."""
        data = {
            "index": 0,
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"location": "NYC"}'
            }
        }

        chunk = StreamingToolCallChunk.from_openai(data)

        assert chunk.index == 0
        assert chunk.id == "call_123"
        assert chunk.function_name == "get_weather"

    def test_to_delta_dict(self):
        """Should convert to delta format."""
        chunk = StreamingToolCallChunk(
            index=0,
            id="call_123",
            function_name="test",
            function_arguments='{"key": "value"}'
        )

        data = chunk.to_delta_dict()

        assert data["index"] == 0
        assert data["id"] == "call_123"
        assert data["function"]["name"] == "test"


class TestMergeToolCallDeltas:
    """Test merging tool call deltas."""

    def test_merge_multiple_deltas(self):
        """Should merge deltas into complete call."""
        deltas = [
            StreamingToolCallChunk(index=0, id="call_123", type="function", function_name="test"),
            StreamingToolCallChunk(index=0, function_arguments='{"key":'),
            StreamingToolCallChunk(index=0, function_arguments=' "value"}'),
        ]

        merged = merge_tool_call_deltas(deltas)

        assert merged["id"] == "call_123"
        assert merged["function"]["name"] == "test"
        assert merged["function"]["arguments"] == '{"key": "value"}'


# ============================================================
# Streaming Error Tests
# ============================================================

class TestStreamError:
    """Test stream error handling."""

    def test_is_retryable_before_content(self):
        """Should be retryable if content not started."""
        error = StreamError(
            type=StreamErrorType.CONNECTION_FAILED,
            code="connection_failed",
            message="Connection lost",
            provider="openai",
            request_id="req_123",
            content_started=False
        )

        assert error.is_retryable is True

    def test_not_retryable_after_content(self):
        """Should NOT be retryable if content started (semantic drift)."""
        error = StreamError(
            type=StreamErrorType.CONNECTION_FAILED,
            code="connection_failed",
            message="Connection lost",
            provider="openai",
            request_id="req_123",
            content_started=True,
            partial_content="Hello world"
        )

        assert error.is_retryable is False

    def test_to_sse(self):
        """Should serialize to valid SSE."""
        error = StreamError(
            type=StreamErrorType.STREAM_INTERRUPTED,
            code="stream_interrupted",
            message="Stream failed",
            provider="openai",
            request_id="req_123",
            content_started=True,
            partial_content="Hello"
        )

        sse = error.to_sse()

        assert sse.startswith("data: ")
        assert sse.endswith("\n\n")

        data = json.loads(sse[6:-2])
        assert "error" in data
        assert data["partial_content"] == "Hello"


class TestStreamErrorBuilder:
    """Test error builder."""

    def test_connection_failed(self):
        """Should create connection failed error."""
        builder = StreamErrorBuilder("openai", "req_123")
        error = builder.connection_failed("Network error")

        assert error.type == StreamErrorType.CONNECTION_FAILED
        assert error.code == "connection_failed"

    def test_rate_limited_with_retry(self):
        """Should include retry-after for rate limit."""
        builder = StreamErrorBuilder("openai", "req_123")
        error = builder.rate_limited(retry_after=30)

        assert error.type == StreamErrorType.RATE_LIMITED
        assert error.retry_after == 30

    def test_from_exception(self):
        """Should classify exception correctly."""
        builder = StreamErrorBuilder("openai", "req_123")

        # Timeout exception
        error = builder.from_exception(Exception("Connection timed out"))
        assert error.type == StreamErrorType.TIMEOUT_DURING_STREAM

        # Rate limit exception
        error = builder.from_exception(Exception("Rate limit exceeded (429)"))
        assert error.type == StreamErrorType.RATE_LIMITED

    def test_tracks_content_state(self):
        """Should include content state in error."""
        builder = StreamErrorBuilder("openai", "req_123")
        builder.set_content_state(
            content_started=True,
            partial_content="Hello world",
            chunks_delivered=5
        )

        error = builder.stream_interrupted()

        assert error.content_started is True
        assert error.partial_content == "Hello world"
        assert error.chunks_delivered == 5


class TestCreateErrorSseChunk:
    """Test error SSE chunk creation."""

    def test_basic_error_chunk(self):
        """Should create valid error chunk."""
        sse = create_error_sse_chunk(
            error_code="test_error",
            error_message="Test error message",
            provider="openai",
            request_id="req_123"
        )

        assert sse.startswith("data: ")
        data = json.loads(sse[6:-2])

        assert data["error"]["code"] == "test_error"
        assert data["error"]["message"] == "Test error message"

    def test_with_partial_content(self):
        """Should include partial content."""
        sse = create_error_sse_chunk(
            error_code="stream_error",
            error_message="Stream failed",
            provider="openai",
            request_id="req_123",
            partial_content="Hello world"
        )

        data = json.loads(sse[6:-2])
        assert data["partial_content"] == "Hello world"


class TestShouldRetryStreamError:
    """Test retry decision logic."""

    def test_no_retry_after_content(self):
        """Should never retry after content started."""
        error = StreamError(
            type=StreamErrorType.CONNECTION_FAILED,
            code="connection_failed",
            message="Error",
            provider="openai",
            request_id="req_123"
        )

        result = should_retry_stream_error(
            error=error,
            content_started=True,
            attempt_count=0
        )

        assert result is False

    def test_retry_before_content(self):
        """Should allow retry before content."""
        error = StreamError(
            type=StreamErrorType.CONNECTION_FAILED,
            code="connection_failed",
            message="Error",
            provider="openai",
            request_id="req_123"
        )

        result = should_retry_stream_error(
            error=error,
            content_started=False,
            attempt_count=0
        )

        assert result is True

    def test_no_retry_after_max_attempts(self):
        """Should not retry after max attempts."""
        error = StreamError(
            type=StreamErrorType.CONNECTION_FAILED,
            code="connection_failed",
            message="Error",
            provider="openai",
            request_id="req_123"
        )

        result = should_retry_stream_error(
            error=error,
            content_started=False,
            attempt_count=3,
            max_attempts=3
        )

        assert result is False


# ============================================================
# SSE Format Compliance Tests
# ============================================================

class TestSSEFormatCompliance:
    """Test SSE format compliance."""

    def test_chunk_format(self):
        """Chunk should follow SSE specification."""
        chunk = StreamChunk(
            id="chatcmpl-123",
            model="gpt-4o",
            provider="openai",
            content_delta="Hello"
        )

        sse = chunk.to_sse()

        # SSE format: "data: {json}\n\n"
        assert sse.startswith("data: ")
        assert sse.endswith("\n\n")

        # Should be valid JSON
        json_part = sse[6:-2]
        data = json.loads(json_part)
        assert isinstance(data, dict)

    def test_done_event_format(self):
        """[DONE] event should follow SSE spec."""
        normalizer = StreamNormalizer("gpt-4o", "openai", "req_123")
        done = normalizer.create_done_event()

        assert done == "data: [DONE]\n\n"

    def test_no_newlines_in_data(self):
        """Data should not contain raw newlines."""
        chunk = StreamChunk(
            id="chatcmpl-123",
            model="gpt-4o",
            provider="openai",
            content_delta="Hello\nWorld"  # Content with newline
        )

        sse = chunk.to_sse()
        json_part = sse[6:-2]

        # Newline should be escaped in JSON
        assert "\n" not in json_part or json_part.count("\\n") > 0


# ============================================================
# Integration Tests
# ============================================================

class TestStreamingIntegration:
    """Integration tests for streaming components."""

    def test_full_streaming_flow(self):
        """Test complete streaming flow."""
        normalizer = StreamNormalizer(
            model="gpt-4o",
            provider="openai",
            request_id="req_123"
        )

        chunks = []

        # First chunk with role
        chunks.append(normalizer.create_first_chunk().to_sse())

        # Content chunks
        chunks.append(normalizer.create_content_chunk("Hello").to_sse())
        chunks.append(normalizer.create_content_chunk(" world").to_sse())
        chunks.append(normalizer.create_content_chunk("!").to_sse())

        # Final chunk
        chunks.append(normalizer.create_final_chunk("stop").to_sse())
        chunks.append(normalizer.create_done_event())

        # Verify
        assert len(chunks) == 6
        assert normalizer.get_accumulated_content() == "Hello world!"
        assert chunks[-1] == "data: [DONE]\n\n"

    def test_tool_call_streaming_flow(self):
        """Test tool call streaming flow."""
        normalizer = StreamNormalizer(
            model="gpt-4o",
            provider="openai",
            request_id="req_123"
        )

        tracker = ToolCallStreamTracker()

        # Simulate tool call streaming
        tracker.update_call(0, id="call_abc", function_name="get_weather")
        tracker.update_call(0, arguments_delta='{"location":')
        tracker.update_call(0, arguments_delta=' "NYC"}')
        tracker.finalize()

        # Verify
        calls = tracker.to_list()
        assert len(calls) == 1
        assert calls[0]["function"]["arguments"] == '{"location": "NYC"}'

        # Validate
        is_valid, errors = tracker.validate_all()
        assert is_valid is True

    def test_error_during_streaming(self):
        """Test error handling during streaming."""
        normalizer = StreamNormalizer(
            model="gpt-4o",
            provider="openai",
            request_id="req_123"
        )

        # Stream some content
        normalizer.create_content_chunk("Hello")
        normalizer.create_content_chunk(" world")

        # Error occurs
        error_builder = StreamErrorBuilder("openai", "req_123")
        error_builder.set_content_state(
            content_started=True,
            partial_content=normalizer.get_accumulated_content(),
            chunks_delivered=2
        )

        error = error_builder.stream_interrupted("Connection lost")

        # Verify error includes partial content
        assert error.partial_content == "Hello world"
        assert error.is_retryable is False  # Content started


# ============================================================
# Run Tests
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
