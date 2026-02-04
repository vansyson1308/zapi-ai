"""
2api.ai - Tool Schema Definitions

Unified tool schema with validation and conversion utilities.
This is the canonical tool format used internally by 2api.ai.
"""

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union


class ToolType(str, Enum):
    """Types of tools supported."""
    FUNCTION = "function"


class ToolChoiceMode(str, Enum):
    """How the model should use tools."""
    AUTO = "auto"           # Model decides when to use tools
    NONE = "none"           # Model cannot use tools
    REQUIRED = "required"   # Model must use at least one tool
    SPECIFIC = "specific"   # Model must use a specific tool


class JSONSchemaType(str, Enum):
    """JSON Schema types for parameters."""
    STRING = "string"
    NUMBER = "number"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    NULL = "null"


@dataclass
class PropertySchema:
    """Schema for a single property in a function's parameters."""
    type: Union[JSONSchemaType, str]
    description: str = ""
    enum: Optional[List[Any]] = None
    default: Optional[Any] = None
    items: Optional[Dict[str, Any]] = None  # For arrays
    properties: Optional[Dict[str, Any]] = None  # For nested objects
    required: Optional[List[str]] = None  # For nested objects
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON Schema dict."""
        result: Dict[str, Any] = {
            "type": self.type.value if isinstance(self.type, JSONSchemaType) else self.type
        }

        if self.description:
            result["description"] = self.description
        if self.enum is not None:
            result["enum"] = self.enum
        if self.default is not None:
            result["default"] = self.default
        if self.items is not None:
            result["items"] = self.items
        if self.properties is not None:
            result["properties"] = self.properties
        if self.required is not None:
            result["required"] = self.required
        if self.minimum is not None:
            result["minimum"] = self.minimum
        if self.maximum is not None:
            result["maximum"] = self.maximum
        if self.min_length is not None:
            result["minLength"] = self.min_length
        if self.max_length is not None:
            result["maxLength"] = self.max_length
        if self.pattern is not None:
            result["pattern"] = self.pattern

        return result


@dataclass
class ParametersSchema:
    """JSON Schema for function parameters."""
    type: Literal["object"] = "object"
    properties: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    required: List[str] = field(default_factory=list)
    additional_properties: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON Schema dict."""
        result: Dict[str, Any] = {
            "type": self.type,
            "properties": self.properties,
        }

        if self.required:
            result["required"] = self.required

        if not self.additional_properties:
            result["additionalProperties"] = False

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParametersSchema":
        """Create from JSON Schema dict."""
        return cls(
            type=data.get("type", "object"),
            properties=data.get("properties", {}),
            required=data.get("required", []),
            additional_properties=data.get("additionalProperties", False)
        )


@dataclass
class FunctionSchema:
    """
    Schema for a function that can be called by the model.

    This is the unified format used internally.
    """
    name: str
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    strict: bool = False  # OpenAI's strict mode

    def validate_name(self) -> tuple[bool, Optional[str]]:
        """
        Validate function name.

        Rules:
        - Must be 1-64 characters
        - Must match pattern [a-zA-Z0-9_-]+
        """
        if not self.name:
            return False, "Function name is required"

        if len(self.name) > 64:
            return False, f"Function name exceeds 64 characters: {len(self.name)}"

        if not re.match(r'^[a-zA-Z0-9_-]+$', self.name):
            return False, f"Function name contains invalid characters: {self.name}"

        return True, None

    def validate_parameters(self) -> tuple[bool, Optional[str]]:
        """
        Validate parameters schema.

        Basic JSON Schema validation.
        """
        if not self.parameters:
            return True, None

        if not isinstance(self.parameters, dict):
            return False, "Parameters must be an object"

        if "type" in self.parameters and self.parameters["type"] != "object":
            return False, "Parameters type must be 'object'"

        return True, None

    def validate(self) -> tuple[bool, List[str]]:
        """
        Validate the entire function schema.

        Returns:
            (is_valid, list_of_errors)
        """
        errors = []

        valid, error = self.validate_name()
        if not valid:
            errors.append(error)

        valid, error = self.validate_parameters()
        if not valid:
            errors.append(error)

        return len(errors) == 0, errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "name": self.name,
        }

        if self.description:
            result["description"] = self.description

        if self.parameters:
            result["parameters"] = self.parameters

        if self.strict:
            result["strict"] = self.strict

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FunctionSchema":
        """Create from dictionary."""
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            parameters=data.get("parameters", {}),
            strict=data.get("strict", False)
        )


@dataclass
class ToolSchema:
    """
    Unified tool schema.

    This is the canonical representation of a tool in 2api.ai.
    All provider-specific formats are converted to/from this.
    """
    type: ToolType = ToolType.FUNCTION
    function: FunctionSchema = field(default_factory=lambda: FunctionSchema(""))

    def validate(self) -> tuple[bool, List[str]]:
        """Validate the tool schema."""
        errors = []

        if self.type != ToolType.FUNCTION:
            errors.append(f"Unsupported tool type: {self.type}")

        valid, func_errors = self.function.validate()
        errors.extend(func_errors)

        return len(errors) == 0, errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert to OpenAI-compatible format."""
        return {
            "type": self.type.value,
            "function": self.function.to_dict()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolSchema":
        """Create from OpenAI-compatible format."""
        tool_type = ToolType(data.get("type", "function"))
        function_data = data.get("function", {})

        return cls(
            type=tool_type,
            function=FunctionSchema.from_dict(function_data)
        )


@dataclass
class ToolChoiceSpecificSchema:
    """Force a specific tool to be called."""
    type: Literal["function"] = "function"
    function_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type,
            "function": {"name": self.function_name}
        }


@dataclass
class ToolChoiceSchema:
    """
    Unified tool choice schema.

    Determines how the model should use tools.
    """
    mode: ToolChoiceMode = ToolChoiceMode.AUTO
    specific_tool: Optional[ToolChoiceSpecificSchema] = None

    @classmethod
    def auto(cls) -> "ToolChoiceSchema":
        """Create auto tool choice."""
        return cls(mode=ToolChoiceMode.AUTO)

    @classmethod
    def none(cls) -> "ToolChoiceSchema":
        """Create none tool choice."""
        return cls(mode=ToolChoiceMode.NONE)

    @classmethod
    def required(cls) -> "ToolChoiceSchema":
        """Create required tool choice."""
        return cls(mode=ToolChoiceMode.REQUIRED)

    @classmethod
    def specific(cls, function_name: str) -> "ToolChoiceSchema":
        """Create specific tool choice."""
        return cls(
            mode=ToolChoiceMode.SPECIFIC,
            specific_tool=ToolChoiceSpecificSchema(function_name=function_name)
        )

    @classmethod
    def from_any(cls, value: Any) -> "ToolChoiceSchema":
        """
        Create from any supported format.

        Handles:
        - String: "auto", "none", "required"
        - Dict: {"type": "function", "function": {"name": "..."}}
        """
        if value is None:
            return cls.auto()

        if isinstance(value, str):
            if value == "auto":
                return cls.auto()
            elif value == "none":
                return cls.none()
            elif value == "required":
                return cls.required()
            else:
                # Assume it's a function name
                return cls.specific(value)

        if isinstance(value, dict):
            if "function" in value:
                name = value["function"].get("name", "")
                return cls.specific(name)

        return cls.auto()

    def to_string_or_dict(self) -> Union[str, Dict[str, Any]]:
        """
        Convert to OpenAI format (string or dict).

        Returns:
            "auto", "none", "required", or {"type": "function", "function": {"name": "..."}}
        """
        if self.mode == ToolChoiceMode.AUTO:
            return "auto"
        elif self.mode == ToolChoiceMode.NONE:
            return "none"
        elif self.mode == ToolChoiceMode.REQUIRED:
            return "required"
        elif self.mode == ToolChoiceMode.SPECIFIC and self.specific_tool:
            return self.specific_tool.to_dict()

        return "auto"


@dataclass
class FunctionCallSchema:
    """
    Schema for a function call made by the model.

    This represents the function that the model wants to call.
    """
    name: str
    arguments: str  # JSON string of arguments

    def get_arguments_dict(self) -> Dict[str, Any]:
        """Parse arguments JSON to dict."""
        try:
            return json.loads(self.arguments) if self.arguments else {}
        except json.JSONDecodeError:
            return {}

    def validate_arguments(self) -> tuple[bool, Optional[str]]:
        """Validate that arguments is valid JSON."""
        if not self.arguments:
            return True, None

        try:
            json.loads(self.arguments)
            return True, None
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON arguments: {e}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "arguments": self.arguments
        }


@dataclass
class ToolCallSchema:
    """
    Schema for a tool call in a response.

    This is what the model returns when it wants to use a tool.
    """
    id: str
    type: ToolType = ToolType.FUNCTION
    function: FunctionCallSchema = field(default_factory=lambda: FunctionCallSchema("", ""))

    def validate(self) -> tuple[bool, List[str]]:
        """Validate the tool call."""
        errors = []

        if not self.id:
            errors.append("Tool call ID is required")

        if not self.function.name:
            errors.append("Function name is required")

        valid, error = self.function.validate_arguments()
        if not valid:
            errors.append(error)

        return len(errors) == 0, errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert to OpenAI-compatible format."""
        return {
            "id": self.id,
            "type": self.type.value,
            "function": self.function.to_dict()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolCallSchema":
        """Create from OpenAI-compatible format."""
        function_data = data.get("function", {})

        return cls(
            id=data.get("id", ""),
            type=ToolType(data.get("type", "function")),
            function=FunctionCallSchema(
                name=function_data.get("name", ""),
                arguments=function_data.get("arguments", "")
            )
        )


@dataclass
class ToolResultSchema:
    """
    Schema for a tool result message.

    This is what we send back after executing a tool.
    """
    tool_call_id: str
    content: str
    name: Optional[str] = None  # Some providers need function name
    is_error: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "tool_call_id": self.tool_call_id,
            "content": self.content
        }

        if self.name:
            result["name"] = self.name

        if self.is_error:
            result["is_error"] = self.is_error

        return result


# Type aliases for convenience
ToolList = List[ToolSchema]
ToolCallList = List[ToolCallSchema]
