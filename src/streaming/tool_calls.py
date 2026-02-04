"""
2api.ai - Tool Call Streaming

Handles streaming of tool/function calls across providers.

Tool calls can be streamed incrementally:
1. Initial chunk with tool call ID and function name
2. Multiple delta chunks with partial arguments JSON
3. Final chunk accumulates all arguments

This module provides utilities for:
- Accumulating tool call deltas
- Validating complete tool calls
- Converting between provider formats
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import uuid


@dataclass
class ToolCallAccumulator:
    """
    Accumulates streaming tool call data.

    Tool calls come in pieces:
    - First: id and function name
    - Then: argument deltas (partial JSON strings)
    - Finally: complete when finish_reason = "tool_calls"
    """
    index: int
    id: Optional[str] = None
    type: str = "function"
    function_name: Optional[str] = None
    arguments_buffer: str = ""
    is_complete: bool = False

    def update(
        self,
        id: Optional[str] = None,
        function_name: Optional[str] = None,
        arguments_delta: str = ""
    ):
        """Update with new delta data."""
        if id:
            self.id = id
        if function_name:
            self.function_name = function_name
        if arguments_delta:
            self.arguments_buffer += arguments_delta

    def mark_complete(self):
        """Mark this tool call as complete."""
        self.is_complete = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to OpenAI tool call format."""
        return {
            "id": self.id or f"call_{uuid.uuid4().hex[:24]}",
            "type": self.type,
            "function": {
                "name": self.function_name or "",
                "arguments": self.arguments_buffer
            }
        }

    def validate(self) -> tuple:
        """
        Validate the accumulated tool call.

        Returns:
            (is_valid, error_message)
        """
        if not self.id:
            return False, "Missing tool call ID"

        if not self.function_name:
            return False, "Missing function name"

        if not self.arguments_buffer:
            return False, "Missing arguments"

        # Try to parse arguments as JSON
        try:
            json.loads(self.arguments_buffer)
        except json.JSONDecodeError as e:
            return False, f"Invalid arguments JSON: {e}"

        return True, None


class ToolCallStreamTracker:
    """
    Tracks multiple tool calls during streaming.

    A single response can contain multiple parallel tool calls.
    Each is tracked by index and accumulated separately.
    """

    def __init__(self):
        self._calls: Dict[int, ToolCallAccumulator] = {}
        self._finalized: bool = False

    def update_call(
        self,
        index: int,
        id: Optional[str] = None,
        function_name: Optional[str] = None,
        arguments_delta: str = ""
    ):
        """
        Update a tool call at the given index.

        Creates the accumulator if it doesn't exist.
        """
        if index not in self._calls:
            self._calls[index] = ToolCallAccumulator(index=index)

        self._calls[index].update(
            id=id,
            function_name=function_name,
            arguments_delta=arguments_delta
        )

    def finalize(self):
        """Mark all tool calls as complete."""
        self._finalized = True
        for call in self._calls.values():
            call.mark_complete()

    def is_finalized(self) -> bool:
        """Check if tracking is finalized."""
        return self._finalized

    def get_call(self, index: int) -> Optional[ToolCallAccumulator]:
        """Get a specific tool call by index."""
        return self._calls.get(index)

    def get_all_calls(self) -> List[ToolCallAccumulator]:
        """Get all tracked tool calls in order."""
        return [
            self._calls[i]
            for i in sorted(self._calls.keys())
        ]

    def to_list(self) -> List[Dict[str, Any]]:
        """Convert all tool calls to list of dicts."""
        return [call.to_dict() for call in self.get_all_calls()]

    def validate_all(self) -> tuple:
        """
        Validate all accumulated tool calls.

        Returns:
            (all_valid, list_of_errors)
        """
        errors = []
        for call in self._calls.values():
            is_valid, error = call.validate()
            if not is_valid:
                errors.append(f"Tool call {call.index}: {error}")

        return len(errors) == 0, errors

    def has_calls(self) -> bool:
        """Check if any tool calls are being tracked."""
        return len(self._calls) > 0

    def call_count(self) -> int:
        """Get number of tracked tool calls."""
        return len(self._calls)


@dataclass
class StreamingToolCallChunk:
    """
    Represents a tool call chunk in streaming format.

    This is the OpenAI-compatible format for tool call deltas.
    """
    index: int
    id: Optional[str] = None
    type: Optional[str] = None
    function_name: Optional[str] = None
    function_arguments: Optional[str] = None

    @classmethod
    def from_openai(cls, data: Dict) -> "StreamingToolCallChunk":
        """Parse from OpenAI format."""
        function = data.get("function", {})
        return cls(
            index=data.get("index", 0),
            id=data.get("id"),
            type=data.get("type"),
            function_name=function.get("name"),
            function_arguments=function.get("arguments")
        )

    @classmethod
    def from_anthropic(cls, data: Dict, index: int = 0) -> "StreamingToolCallChunk":
        """
        Parse from Anthropic format.

        Anthropic sends tool calls as content blocks:
        - content_block_start: {type: "tool_use", id, name}
        - content_block_delta: {delta: {type: "input_json_delta", partial_json}}
        """
        content_block = data.get("content_block", {})
        delta = data.get("delta", {})

        # Content block start
        if content_block.get("type") == "tool_use":
            return cls(
                index=index,
                id=content_block.get("id"),
                type="function",
                function_name=content_block.get("name")
            )

        # Delta with arguments
        if delta.get("type") == "input_json_delta":
            return cls(
                index=index,
                function_arguments=delta.get("partial_json", "")
            )

        return cls(index=index)

    @classmethod
    def from_google(cls, data: Dict, index: int = 0) -> "StreamingToolCallChunk":
        """
        Parse from Google Gemini format.

        Gemini sends function calls as parts:
        {functionCall: {name, args}}
        """
        fc = data.get("functionCall", {})

        return cls(
            index=index,
            id=f"call_{uuid.uuid4().hex[:8]}",
            type="function",
            function_name=fc.get("name"),
            function_arguments=json.dumps(fc.get("args", {}))
        )

    def to_delta_dict(self) -> Dict[str, Any]:
        """Convert to OpenAI delta format."""
        result: Dict[str, Any] = {"index": self.index}

        if self.id:
            result["id"] = self.id
        if self.type:
            result["type"] = self.type

        function: Dict[str, Any] = {}
        if self.function_name:
            function["name"] = self.function_name
        if self.function_arguments:
            function["arguments"] = self.function_arguments

        if function:
            result["function"] = function

        return result


def merge_tool_call_deltas(
    deltas: List[StreamingToolCallChunk]
) -> Dict[str, Any]:
    """
    Merge multiple tool call deltas into a complete tool call.

    Used when reassembling a tool call from streaming deltas.
    """
    if not deltas:
        return {}

    result = {
        "index": deltas[0].index,
        "id": None,
        "type": "function",
        "function": {
            "name": None,
            "arguments": ""
        }
    }

    for delta in deltas:
        if delta.id:
            result["id"] = delta.id
        if delta.type:
            result["type"] = delta.type
        if delta.function_name:
            result["function"]["name"] = delta.function_name
        if delta.function_arguments:
            result["function"]["arguments"] += delta.function_arguments

    return result


def create_tool_call_delta_chunk(
    stream_id: str,
    model: str,
    provider: str,
    index: int,
    tool_id: Optional[str] = None,
    function_name: Optional[str] = None,
    arguments_delta: str = "",
    created: Optional[int] = None
) -> Dict[str, Any]:
    """
    Create a tool call delta chunk in OpenAI format.

    This is the standard format used in SSE streaming.
    """
    import time

    tool_call: Dict[str, Any] = {"index": index}

    if tool_id:
        tool_call["id"] = tool_id
        tool_call["type"] = "function"

    function: Dict[str, Any] = {}
    if function_name:
        function["name"] = function_name
    if arguments_delta:
        function["arguments"] = arguments_delta

    if function:
        tool_call["function"] = function

    return {
        "id": stream_id,
        "object": "chat.completion.chunk",
        "created": created or int(time.time()),
        "model": model,
        "provider": provider,
        "choices": [{
            "index": 0,
            "delta": {
                "tool_calls": [tool_call]
            },
            "finish_reason": None
        }]
    }
