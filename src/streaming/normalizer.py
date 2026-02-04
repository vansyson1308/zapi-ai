"""
2api.ai - Stream Normalizer

Normalizes streaming responses from all providers to a unified
OpenAI-compatible SSE format.

Ensures consistent output regardless of source provider:
- Same chunk structure
- Same field names
- Same event format
- Proper tool call delta handling
"""

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, AsyncIterator


class StreamEventType(str, Enum):
    """Types of streaming events."""
    CHUNK = "chunk"           # Content chunk
    TOOL_CALL_START = "tool_call_start"  # Tool call begins
    TOOL_CALL_DELTA = "tool_call_delta"  # Tool call argument delta
    DONE = "done"             # Stream complete
    ERROR = "error"           # Error occurred


@dataclass
class ToolCallDelta:
    """Represents a tool call being streamed."""
    index: int
    id: Optional[str] = None
    type: str = "function"
    function_name: Optional[str] = None
    function_arguments_delta: str = ""


@dataclass
class StreamChunk:
    """
    Normalized stream chunk.

    All providers' chunks are converted to this format before
    being serialized to SSE.
    """
    id: str
    object: str = "chat.completion.chunk"
    created: int = field(default_factory=lambda: int(time.time()))
    model: str = ""
    provider: str = ""

    # Content delta
    content_delta: Optional[str] = None

    # Role (only in first chunk typically)
    role: Optional[str] = None

    # Tool calls (can be streamed incrementally)
    tool_calls: Optional[List[ToolCallDelta]] = None

    # Finish reason (only in final chunk)
    finish_reason: Optional[str] = None

    # Index (for multiple choices)
    index: int = 0

    def to_sse(self) -> str:
        """Convert to SSE format string."""
        data = self.to_dict()
        return f"data: {json.dumps(data)}\n\n"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to OpenAI-compatible dictionary."""
        delta: Dict[str, Any] = {}

        if self.role:
            delta["role"] = self.role

        if self.content_delta is not None:
            delta["content"] = self.content_delta

        if self.tool_calls:
            delta["tool_calls"] = [
                self._tool_call_to_dict(tc)
                for tc in self.tool_calls
            ]

        choice = {
            "index": self.index,
            "delta": delta,
            "finish_reason": self.finish_reason
        }

        result = {
            "id": self.id,
            "object": self.object,
            "created": self.created,
            "model": self.model,
            "choices": [choice]
        }

        # Add 2api extensions
        if self.provider:
            result["provider"] = self.provider

        return result

    def _tool_call_to_dict(self, tc: ToolCallDelta) -> Dict[str, Any]:
        """Convert tool call delta to dictionary."""
        result: Dict[str, Any] = {"index": tc.index}

        if tc.id:
            result["id"] = tc.id
        if tc.type:
            result["type"] = tc.type

        function: Dict[str, Any] = {}
        if tc.function_name:
            function["name"] = tc.function_name
        if tc.function_arguments_delta:
            function["arguments"] = tc.function_arguments_delta

        if function:
            result["function"] = function

        return result


@dataclass
class StreamState:
    """
    Tracks state during streaming.

    Used for:
    - Accumulating partial content
    - Tracking tool calls in progress
    - Semantic drift detection
    """
    request_id: str
    stream_id: str = field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    model: str = ""
    provider: str = ""

    # Content tracking
    content_started: bool = False
    accumulated_content: str = ""
    chunks_sent: int = 0

    # Tool call tracking
    tool_calls_in_progress: Dict[int, ToolCallDelta] = field(default_factory=dict)
    accumulated_tool_calls: List[Dict] = field(default_factory=list)

    # Timing
    started_at: float = field(default_factory=time.time)

    def mark_content_started(self, content: str = ""):
        """Mark that content has started streaming."""
        self.content_started = True
        if content:
            self.accumulated_content += content

    def append_content(self, content: str):
        """Append content to accumulator."""
        self.accumulated_content += content
        self.chunks_sent += 1

    def can_fallback(self) -> bool:
        """Check if fallback is still allowed (no content sent)."""
        return not self.content_started

    def get_partial_content(self) -> str:
        """Get accumulated partial content."""
        return self.accumulated_content

    def update_tool_call(self, index: int, delta: ToolCallDelta):
        """Update a tool call in progress."""
        if index not in self.tool_calls_in_progress:
            self.tool_calls_in_progress[index] = delta
        else:
            existing = self.tool_calls_in_progress[index]
            if delta.id:
                existing.id = delta.id
            if delta.function_name:
                existing.function_name = delta.function_name
            existing.function_arguments_delta += delta.function_arguments_delta

    def finalize_tool_calls(self) -> List[Dict]:
        """Get finalized tool calls."""
        return [
            {
                "id": tc.id,
                "type": tc.type,
                "function": {
                    "name": tc.function_name,
                    "arguments": tc.function_arguments_delta
                }
            }
            for tc in self.tool_calls_in_progress.values()
        ]


class StreamNormalizer:
    """
    Normalizes streaming output from any provider.

    Usage:
        normalizer = StreamNormalizer(
            model="gpt-4o",
            provider="openai",
            request_id="req_123"
        )

        # For each provider-specific chunk:
        normalized = normalizer.normalize_chunk(provider_chunk)
        yield normalized.to_sse()

        # When done:
        yield normalizer.create_done_event()
    """

    def __init__(
        self,
        model: str,
        provider: str,
        request_id: str
    ):
        self.state = StreamState(
            request_id=request_id,
            model=model,
            provider=provider
        )

    def create_chunk(
        self,
        content_delta: Optional[str] = None,
        role: Optional[str] = None,
        tool_calls: Optional[List[ToolCallDelta]] = None,
        finish_reason: Optional[str] = None,
        index: int = 0
    ) -> StreamChunk:
        """
        Create a normalized stream chunk.

        Args:
            content_delta: Text content to stream
            role: Message role (typically only in first chunk)
            tool_calls: Tool call deltas
            finish_reason: Finish reason (only in final chunk)
            index: Choice index

        Returns:
            Normalized StreamChunk
        """
        # Track content
        if content_delta:
            if not self.state.content_started:
                self.state.mark_content_started(content_delta)
            else:
                self.state.append_content(content_delta)

        # Track tool calls
        if tool_calls:
            for tc in tool_calls:
                self.state.update_tool_call(tc.index, tc)

        return StreamChunk(
            id=self.state.stream_id,
            created=int(self.state.started_at),
            model=self.state.model,
            provider=self.state.provider,
            content_delta=content_delta,
            role=role,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            index=index
        )

    def create_first_chunk(self) -> StreamChunk:
        """Create the first chunk with role."""
        return self.create_chunk(role="assistant")

    def create_content_chunk(self, content: str) -> StreamChunk:
        """Create a content chunk."""
        return self.create_chunk(content_delta=content)

    def create_tool_call_chunk(
        self,
        index: int,
        tool_id: Optional[str] = None,
        function_name: Optional[str] = None,
        arguments_delta: str = ""
    ) -> StreamChunk:
        """Create a tool call delta chunk."""
        tc = ToolCallDelta(
            index=index,
            id=tool_id,
            function_name=function_name,
            function_arguments_delta=arguments_delta
        )
        return self.create_chunk(tool_calls=[tc])

    def create_final_chunk(
        self,
        finish_reason: str = "stop"
    ) -> StreamChunk:
        """Create the final chunk with finish reason."""
        return self.create_chunk(finish_reason=finish_reason)

    def create_done_event(self) -> str:
        """Create the [DONE] event."""
        return "data: [DONE]\n\n"

    def create_error_chunk(
        self,
        error_code: str,
        error_message: str,
        partial_content: Optional[str] = None
    ) -> str:
        """
        Create an error chunk for streaming errors.

        This is used when an error occurs AFTER content has started
        (semantic drift protection - we can't retry/fallback).
        """
        error_data = {
            "error": {
                "code": error_code,
                "message": error_message,
                "type": "stream_error",
                "request_id": self.state.request_id,
            }
        }

        if partial_content:
            error_data["partial_content"] = partial_content

        return f"data: {json.dumps(error_data)}\n\n"

    # ============================================================
    # Provider-specific normalization methods
    # ============================================================

    def normalize_openai_chunk(self, chunk_data: Dict) -> Optional[StreamChunk]:
        """
        Normalize an OpenAI streaming chunk.

        OpenAI format is already the target format, so this is mostly passthrough.
        """
        choices = chunk_data.get("choices", [])
        if not choices:
            return None

        choice = choices[0]
        delta = choice.get("delta", {})

        # Extract fields
        content = delta.get("content")
        role = delta.get("role")
        finish_reason = choice.get("finish_reason")

        # Handle tool calls
        tool_calls = None
        if "tool_calls" in delta:
            tool_calls = []
            for tc in delta["tool_calls"]:
                tool_calls.append(ToolCallDelta(
                    index=tc.get("index", 0),
                    id=tc.get("id"),
                    type=tc.get("type", "function"),
                    function_name=tc.get("function", {}).get("name"),
                    function_arguments_delta=tc.get("function", {}).get("arguments", "")
                ))

        return self.create_chunk(
            content_delta=content,
            role=role,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            index=choice.get("index", 0)
        )

    def normalize_anthropic_event(
        self,
        event_type: str,
        event_data: Dict
    ) -> Optional[StreamChunk]:
        """
        Normalize an Anthropic streaming event.

        Anthropic uses a different event structure that needs conversion.
        """
        if event_type == "message_start":
            # First event - extract message ID
            message = event_data.get("message", {})
            self.state.stream_id = message.get("id", self.state.stream_id)
            return self.create_first_chunk()

        elif event_type == "content_block_start":
            # Content block starting
            content_block = event_data.get("content_block", {})
            if content_block.get("type") == "tool_use":
                # Tool call starting
                return self.create_tool_call_chunk(
                    index=event_data.get("index", 0),
                    tool_id=content_block.get("id"),
                    function_name=content_block.get("name")
                )
            return None

        elif event_type == "content_block_delta":
            delta = event_data.get("delta", {})

            if delta.get("type") == "text_delta":
                text = delta.get("text", "")
                return self.create_content_chunk(text)

            elif delta.get("type") == "input_json_delta":
                # Tool call arguments
                partial_json = delta.get("partial_json", "")
                return self.create_tool_call_chunk(
                    index=event_data.get("index", 0),
                    arguments_delta=partial_json
                )

            return None

        elif event_type == "message_delta":
            # Message-level delta (finish reason)
            delta = event_data.get("delta", {})
            stop_reason = delta.get("stop_reason")

            if stop_reason:
                finish_map = {
                    "end_turn": "stop",
                    "max_tokens": "length",
                    "tool_use": "tool_calls",
                    "stop_sequence": "stop"
                }
                return self.create_final_chunk(
                    finish_reason=finish_map.get(stop_reason, "stop")
                )
            return None

        elif event_type == "message_stop":
            return None  # Will be followed by [DONE]

        return None

    def normalize_google_chunk(self, chunk_data: Dict) -> Optional[StreamChunk]:
        """
        Normalize a Google Gemini streaming chunk.

        Gemini has a completely different structure that needs conversion.
        """
        # Check for error
        if "error" in chunk_data:
            return None  # Errors handled separately

        candidates = chunk_data.get("candidates", [])
        if not candidates:
            return None

        candidate = candidates[0]
        content = candidate.get("content", {})
        parts = content.get("parts", [])

        # Extract text content
        for part in parts:
            if "text" in part:
                text = part["text"]
                return self.create_content_chunk(text)

            elif "functionCall" in part:
                # Tool call
                fc = part["functionCall"]
                return self.create_tool_call_chunk(
                    index=0,
                    tool_id=f"call_{uuid.uuid4().hex[:8]}",
                    function_name=fc.get("name"),
                    arguments_delta=json.dumps(fc.get("args", {}))
                )

        # Check for finish
        finish_reason = candidate.get("finishReason")
        if finish_reason:
            finish_map = {
                "STOP": "stop",
                "MAX_TOKENS": "length",
                "SAFETY": "content_filter",
                "RECITATION": "content_filter"
            }
            return self.create_final_chunk(
                finish_reason=finish_map.get(finish_reason, "stop")
            )

        return None

    # ============================================================
    # Convenience methods
    # ============================================================

    def get_state(self) -> StreamState:
        """Get current stream state."""
        return self.state

    def has_content_started(self) -> bool:
        """Check if content has started streaming."""
        return self.state.content_started

    def get_accumulated_content(self) -> str:
        """Get accumulated content so far."""
        return self.state.accumulated_content

    def get_finalized_tool_calls(self) -> List[Dict]:
        """Get finalized tool calls."""
        return self.state.finalize_tool_calls()


def create_normalized_stream(
    provider: str,
    model: str,
    request_id: str
) -> StreamNormalizer:
    """Factory function to create a stream normalizer."""
    return StreamNormalizer(
        model=model,
        provider=provider,
        request_id=request_id
    )
