"""
2api.ai - Tool Normalizer

Converts tool definitions and tool calls between different provider formats.

Supported Providers:
- OpenAI: Native format (our target)
- Anthropic: Different schema structure
- Google Gemini: Wrapped in functionDeclarations
"""

import json
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .schema import (
    ToolSchema,
    ToolChoiceSchema,
    ToolChoiceMode,
    ToolCallSchema,
    FunctionSchema,
    FunctionCallSchema,
    ToolResultSchema,
    ToolType,
)


@dataclass
class NormalizationResult:
    """Result of a normalization operation."""
    success: bool
    data: Any
    errors: List[str]

    @classmethod
    def ok(cls, data: Any) -> "NormalizationResult":
        """Create successful result."""
        return cls(success=True, data=data, errors=[])

    @classmethod
    def fail(cls, errors: List[str]) -> "NormalizationResult":
        """Create failed result."""
        return cls(success=False, data=None, errors=errors)


class ToolNormalizer:
    """
    Normalizes tools between OpenAI format and other providers.

    OpenAI format is the canonical format used throughout 2api.ai.
    This normalizer handles:
    - Tool definitions (input)
    - Tool choices (input)
    - Tool calls (output)
    - Tool results (input)
    """

    # ============================================================
    # Tool Definition Normalization (Request → Provider)
    # ============================================================

    def normalize_tools_for_openai(
        self,
        tools: List[ToolSchema]
    ) -> List[Dict[str, Any]]:
        """
        Convert tools to OpenAI format.

        OpenAI is our target format, so this is mostly passthrough.
        """
        return [tool.to_dict() for tool in tools]

    def normalize_tools_for_anthropic(
        self,
        tools: List[ToolSchema]
    ) -> List[Dict[str, Any]]:
        """
        Convert tools to Anthropic format.

        Anthropic uses:
        - input_schema instead of parameters
        - No wrapper type
        """
        result = []

        for tool in tools:
            anthropic_tool = {
                "name": tool.function.name,
            }

            if tool.function.description:
                anthropic_tool["description"] = tool.function.description

            # Anthropic uses input_schema instead of parameters
            if tool.function.parameters:
                anthropic_tool["input_schema"] = tool.function.parameters
            else:
                # Anthropic requires input_schema
                anthropic_tool["input_schema"] = {
                    "type": "object",
                    "properties": {},
                }

            result.append(anthropic_tool)

        return result

    def normalize_tools_for_google(
        self,
        tools: List[ToolSchema]
    ) -> List[Dict[str, Any]]:
        """
        Convert tools to Google Gemini format.

        Gemini wraps function declarations in a tools array:
        [{"functionDeclarations": [...]}]
        """
        function_declarations = []

        for tool in tools:
            gemini_func = {
                "name": tool.function.name,
            }

            if tool.function.description:
                gemini_func["description"] = tool.function.description

            if tool.function.parameters:
                gemini_func["parameters"] = tool.function.parameters
            else:
                # Gemini accepts empty parameters
                gemini_func["parameters"] = {
                    "type": "object",
                    "properties": {},
                }

            function_declarations.append(gemini_func)

        # Return as wrapped structure
        return [{"functionDeclarations": function_declarations}]

    # ============================================================
    # Tool Choice Normalization (Request → Provider)
    # ============================================================

    def normalize_tool_choice_for_openai(
        self,
        choice: ToolChoiceSchema
    ) -> Any:
        """Convert tool choice to OpenAI format."""
        return choice.to_string_or_dict()

    def normalize_tool_choice_for_anthropic(
        self,
        choice: ToolChoiceSchema
    ) -> Dict[str, Any]:
        """
        Convert tool choice to Anthropic format.

        Anthropic uses:
        - {"type": "auto"} for auto
        - {"type": "none"} for none
        - {"type": "any"} for required
        - {"type": "tool", "name": "..."} for specific
        """
        if choice.mode == ToolChoiceMode.AUTO:
            return {"type": "auto"}
        elif choice.mode == ToolChoiceMode.NONE:
            return {"type": "none"}
        elif choice.mode == ToolChoiceMode.REQUIRED:
            return {"type": "any"}
        elif choice.mode == ToolChoiceMode.SPECIFIC and choice.specific_tool:
            return {
                "type": "tool",
                "name": choice.specific_tool.function_name
            }

        return {"type": "auto"}

    def normalize_tool_choice_for_google(
        self,
        choice: ToolChoiceSchema
    ) -> Optional[Dict[str, Any]]:
        """
        Convert tool choice to Google Gemini format.

        Gemini uses:
        - {"mode": "AUTO"} for auto
        - {"mode": "NONE"} for none
        - {"mode": "ANY"} for required
        - {"mode": "TOOL", "allowedFunctionNames": ["..."]} for specific
        """
        if choice.mode == ToolChoiceMode.AUTO:
            return {"mode": "AUTO"}
        elif choice.mode == ToolChoiceMode.NONE:
            return {"mode": "NONE"}
        elif choice.mode == ToolChoiceMode.REQUIRED:
            return {"mode": "ANY"}
        elif choice.mode == ToolChoiceMode.SPECIFIC and choice.specific_tool:
            return {
                "mode": "TOOL",
                "allowedFunctionNames": [choice.specific_tool.function_name]
            }

        return None

    # ============================================================
    # Tool Call Normalization (Provider Response → Unified)
    # ============================================================

    def normalize_tool_calls_from_openai(
        self,
        tool_calls: List[Dict[str, Any]]
    ) -> List[ToolCallSchema]:
        """
        Normalize tool calls from OpenAI response.

        OpenAI format is our target, so this is mostly parsing.
        """
        return [ToolCallSchema.from_dict(tc) for tc in tool_calls]

    def normalize_tool_calls_from_anthropic(
        self,
        content_blocks: List[Dict[str, Any]]
    ) -> List[ToolCallSchema]:
        """
        Normalize tool calls from Anthropic response.

        Anthropic returns tool uses as content blocks:
        [{"type": "tool_use", "id": "...", "name": "...", "input": {...}}]
        """
        result = []

        for block in content_blocks:
            if block.get("type") != "tool_use":
                continue

            tool_call = ToolCallSchema(
                id=block.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                type=ToolType.FUNCTION,
                function=FunctionCallSchema(
                    name=block.get("name", ""),
                    # Anthropic uses 'input' as dict, we need JSON string
                    arguments=json.dumps(block.get("input", {}))
                )
            )
            result.append(tool_call)

        return result

    def normalize_tool_calls_from_google(
        self,
        parts: List[Dict[str, Any]]
    ) -> List[ToolCallSchema]:
        """
        Normalize tool calls from Google Gemini response.

        Gemini returns function calls as parts:
        [{"functionCall": {"name": "...", "args": {...}}}]
        """
        result = []

        for i, part in enumerate(parts):
            if "functionCall" not in part:
                continue

            fc = part["functionCall"]
            tool_call = ToolCallSchema(
                id=f"call_{uuid.uuid4().hex[:8]}_{i}",
                type=ToolType.FUNCTION,
                function=FunctionCallSchema(
                    name=fc.get("name", ""),
                    # Gemini uses 'args' as dict, we need JSON string
                    arguments=json.dumps(fc.get("args", {}))
                )
            )
            result.append(tool_call)

        return result

    # ============================================================
    # Tool Result Normalization (Unified → Provider)
    # ============================================================

    def create_tool_result_message_for_openai(
        self,
        result: ToolResultSchema
    ) -> Dict[str, Any]:
        """
        Create tool result message for OpenAI.

        OpenAI format:
        {"role": "tool", "tool_call_id": "...", "content": "..."}
        """
        return {
            "role": "tool",
            "tool_call_id": result.tool_call_id,
            "content": result.content
        }

    def create_tool_result_message_for_anthropic(
        self,
        result: ToolResultSchema
    ) -> Dict[str, Any]:
        """
        Create tool result message for Anthropic.

        Anthropic format:
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "...", "content": "..."}]}
        """
        tool_result_block = {
            "type": "tool_result",
            "tool_use_id": result.tool_call_id,
            "content": result.content
        }

        if result.is_error:
            tool_result_block["is_error"] = True

        return {
            "role": "user",
            "content": [tool_result_block]
        }

    def create_tool_result_message_for_google(
        self,
        result: ToolResultSchema
    ) -> Dict[str, Any]:
        """
        Create tool result message for Google Gemini.

        Gemini format (uses function name, not ID):
        {"role": "user", "parts": [{"functionResponse": {"name": "...", "response": {"content": "..."}}}]}
        """
        return {
            "role": "user",
            "parts": [{
                "functionResponse": {
                    "name": result.name or result.tool_call_id,
                    "response": {
                        "content": result.content
                    }
                }
            }]
        }

    # ============================================================
    # Batch Operations
    # ============================================================

    def normalize_multiple_tool_results_for_anthropic(
        self,
        results: List[ToolResultSchema]
    ) -> Dict[str, Any]:
        """
        Combine multiple tool results into single Anthropic message.

        Anthropic allows multiple tool results in one user message.
        """
        content_blocks = []

        for result in results:
            block = {
                "type": "tool_result",
                "tool_use_id": result.tool_call_id,
                "content": result.content
            }
            if result.is_error:
                block["is_error"] = True
            content_blocks.append(block)

        return {
            "role": "user",
            "content": content_blocks
        }

    def normalize_multiple_tool_results_for_google(
        self,
        results: List[ToolResultSchema]
    ) -> Dict[str, Any]:
        """
        Combine multiple tool results into single Gemini message.

        Gemini allows multiple function responses in one user message.
        """
        parts = []

        for result in results:
            parts.append({
                "functionResponse": {
                    "name": result.name or result.tool_call_id,
                    "response": {
                        "content": result.content
                    }
                }
            })

        return {
            "role": "user",
            "parts": parts
        }


# Global instance for convenience
_normalizer = ToolNormalizer()


def normalize_tools(tools: List[ToolSchema], provider: str) -> List[Dict[str, Any]]:
    """
    Normalize tools for a specific provider.

    Args:
        tools: List of unified tool schemas
        provider: Target provider ("openai", "anthropic", "google")

    Returns:
        Provider-specific tool format
    """
    if provider == "openai":
        return _normalizer.normalize_tools_for_openai(tools)
    elif provider == "anthropic":
        return _normalizer.normalize_tools_for_anthropic(tools)
    elif provider == "google":
        return _normalizer.normalize_tools_for_google(tools)
    else:
        # Default to OpenAI format
        return _normalizer.normalize_tools_for_openai(tools)


def normalize_tool_choice(choice: ToolChoiceSchema, provider: str) -> Any:
    """
    Normalize tool choice for a specific provider.

    Args:
        choice: Unified tool choice schema
        provider: Target provider

    Returns:
        Provider-specific tool choice format
    """
    if provider == "openai":
        return _normalizer.normalize_tool_choice_for_openai(choice)
    elif provider == "anthropic":
        return _normalizer.normalize_tool_choice_for_anthropic(choice)
    elif provider == "google":
        return _normalizer.normalize_tool_choice_for_google(choice)
    else:
        return _normalizer.normalize_tool_choice_for_openai(choice)


def normalize_tool_calls(data: Any, provider: str) -> List[ToolCallSchema]:
    """
    Normalize tool calls from a provider response.

    Args:
        data: Provider-specific tool call data
        provider: Source provider

    Returns:
        List of unified tool call schemas
    """
    if provider == "openai":
        return _normalizer.normalize_tool_calls_from_openai(data)
    elif provider == "anthropic":
        return _normalizer.normalize_tool_calls_from_anthropic(data)
    elif provider == "google":
        return _normalizer.normalize_tool_calls_from_google(data)
    else:
        return _normalizer.normalize_tool_calls_from_openai(data)
