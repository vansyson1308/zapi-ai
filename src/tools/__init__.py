"""
2api.ai - Tool Calling Module

Unified tool calling support for all providers with:
- Schema normalization across OpenAI/Anthropic/Google
- Validation of tools, tool calls, and results
- Parallel tool execution support
- Provider-specific format conversion

Key Components:
- Schema: Canonical tool representation
- Normalizer: Cross-provider format conversion
- Validator: Tool and argument validation
- Parallel: Concurrent tool execution
"""

from .schema import (
    # Enums
    ToolType,
    ToolChoiceMode,
    JSONSchemaType,
    # Schema classes
    PropertySchema,
    ParametersSchema,
    FunctionSchema,
    ToolSchema,
    ToolChoiceSchema,
    ToolChoiceSpecificSchema,
    FunctionCallSchema,
    ToolCallSchema,
    ToolResultSchema,
    # Type aliases
    ToolList,
    ToolCallList,
)
from .normalizer import (
    ToolNormalizer,
    NormalizationResult,
    # Convenience functions
    normalize_tools,
    normalize_tool_choice,
    normalize_tool_calls,
)
from .validator import (
    ToolValidator,
    ValidationResult,
    ValidationError,
    # Convenience functions
    validate_tool,
    validate_tools,
    validate_tool_call,
    validate_tool_result,
)
from .parallel import (
    ToolCallStatus,
    ToolCallExecution,
    ParallelToolCallTracker,
    ParallelToolExecutor,
    ToolCallBatcher,
    # Convenience functions
    execute_tools_parallel,
    batch_tool_results,
)

__all__ = [
    # Enums
    "ToolType",
    "ToolChoiceMode",
    "JSONSchemaType",
    "ToolCallStatus",
    # Schema
    "PropertySchema",
    "ParametersSchema",
    "FunctionSchema",
    "ToolSchema",
    "ToolChoiceSchema",
    "ToolChoiceSpecificSchema",
    "FunctionCallSchema",
    "ToolCallSchema",
    "ToolResultSchema",
    "ToolList",
    "ToolCallList",
    # Normalizer
    "ToolNormalizer",
    "NormalizationResult",
    "normalize_tools",
    "normalize_tool_choice",
    "normalize_tool_calls",
    # Validator
    "ToolValidator",
    "ValidationResult",
    "ValidationError",
    "validate_tool",
    "validate_tools",
    "validate_tool_call",
    "validate_tool_result",
    # Parallel
    "ToolCallExecution",
    "ParallelToolCallTracker",
    "ParallelToolExecutor",
    "ToolCallBatcher",
    "execute_tools_parallel",
    "batch_tool_results",
]
