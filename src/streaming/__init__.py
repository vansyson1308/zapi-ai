"""
2api.ai - Streaming Module

Unified streaming support for all providers with:
- Normalized SSE output format
- Tool call streaming
- Error handling with semantic drift protection
- Partial content tracking
"""

from .normalizer import (
    StreamNormalizer,
    StreamChunk,
    StreamState,
    StreamEventType,
    ToolCallDelta,
    create_normalized_stream,
)
from .tool_calls import (
    ToolCallAccumulator,
    ToolCallStreamTracker,
    StreamingToolCallChunk,
    merge_tool_call_deltas,
    create_tool_call_delta_chunk,
)
from .errors import (
    StreamError,
    StreamErrorType,
    StreamErrorBuilder,
    create_error_sse_chunk,
    should_retry_stream_error,
)

__all__ = [
    # Normalizer
    "StreamNormalizer",
    "StreamChunk",
    "StreamState",
    "StreamEventType",
    "ToolCallDelta",
    "create_normalized_stream",
    # Tool Calls
    "ToolCallAccumulator",
    "ToolCallStreamTracker",
    "StreamingToolCallChunk",
    "merge_tool_call_deltas",
    "create_tool_call_delta_chunk",
    # Errors
    "StreamError",
    "StreamErrorType",
    "StreamErrorBuilder",
    "create_error_sse_chunk",
    "should_retry_stream_error",
]
