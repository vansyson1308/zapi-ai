"""
2api.ai - Tool Validation

Validates tool definitions, tool calls, and tool results.

Provides:
- Schema validation
- Argument validation against parameter schema
- Cross-provider compatibility checks
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from .schema import (
    ToolSchema,
    ToolCallSchema,
    FunctionSchema,
    ToolResultSchema,
    JSONSchemaType,
)


@dataclass
class ValidationError:
    """Represents a single validation error."""
    path: str  # JSON path to the error
    message: str
    code: str


@dataclass
class ValidationResult:
    """Result of a validation operation."""
    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)

    @classmethod
    def ok(cls) -> "ValidationResult":
        """Create successful validation result."""
        return cls(is_valid=True, errors=[])

    @classmethod
    def fail(cls, errors: List[ValidationError]) -> "ValidationResult":
        """Create failed validation result."""
        return cls(is_valid=False, errors=errors)

    def add_error(self, path: str, message: str, code: str = "validation_error"):
        """Add an error to the result."""
        self.errors.append(ValidationError(path=path, message=message, code=code))
        self.is_valid = False

    def merge(self, other: "ValidationResult", path_prefix: str = ""):
        """Merge another validation result into this one."""
        for error in other.errors:
            new_path = f"{path_prefix}.{error.path}" if path_prefix else error.path
            self.errors.append(ValidationError(
                path=new_path,
                message=error.message,
                code=error.code
            ))
        if not other.is_valid:
            self.is_valid = False


class ToolValidator:
    """
    Validates tools and tool calls.

    Performs:
    - Schema structure validation
    - Name format validation
    - Parameter type validation
    - Argument validation against schema
    """

    # Name patterns
    FUNCTION_NAME_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')
    MAX_FUNCTION_NAME_LENGTH = 64
    MAX_DESCRIPTION_LENGTH = 1024

    # Known JSON Schema types
    VALID_TYPES: Set[str] = {
        "string", "number", "integer", "boolean",
        "array", "object", "null"
    }

    def validate_tool(self, tool: ToolSchema) -> ValidationResult:
        """
        Validate a tool schema.

        Checks:
        - Tool type is valid
        - Function schema is valid
        """
        result = ValidationResult.ok()

        # Validate function schema
        func_result = self.validate_function_schema(tool.function)
        result.merge(func_result, "function")

        return result

    def validate_function_schema(self, func: FunctionSchema) -> ValidationResult:
        """
        Validate a function schema.

        Checks:
        - Name is valid
        - Description is reasonable
        - Parameters schema is valid
        """
        result = ValidationResult.ok()

        # Validate name
        if not func.name:
            result.add_error("name", "Function name is required", "required")
        elif len(func.name) > self.MAX_FUNCTION_NAME_LENGTH:
            result.add_error(
                "name",
                f"Function name exceeds {self.MAX_FUNCTION_NAME_LENGTH} characters",
                "max_length"
            )
        elif not self.FUNCTION_NAME_PATTERN.match(func.name):
            result.add_error(
                "name",
                "Function name must match pattern [a-zA-Z0-9_-]+",
                "pattern"
            )

        # Validate description
        if func.description and len(func.description) > self.MAX_DESCRIPTION_LENGTH:
            result.add_error(
                "description",
                f"Description exceeds {self.MAX_DESCRIPTION_LENGTH} characters",
                "max_length"
            )

        # Validate parameters
        if func.parameters:
            params_result = self.validate_json_schema(func.parameters)
            result.merge(params_result, "parameters")

        return result

    def validate_json_schema(
        self,
        schema: Dict[str, Any],
        path: str = ""
    ) -> ValidationResult:
        """
        Validate a JSON Schema structure.

        Basic validation of common JSON Schema constructs.
        """
        result = ValidationResult.ok()

        if not isinstance(schema, dict):
            result.add_error(path or "schema", "Schema must be an object", "type")
            return result

        # Check type
        schema_type = schema.get("type")
        if schema_type:
            if isinstance(schema_type, str):
                if schema_type not in self.VALID_TYPES:
                    result.add_error(
                        f"{path}.type" if path else "type",
                        f"Invalid type: {schema_type}",
                        "invalid_type"
                    )
            elif isinstance(schema_type, list):
                for t in schema_type:
                    if t not in self.VALID_TYPES:
                        result.add_error(
                            f"{path}.type" if path else "type",
                            f"Invalid type in union: {t}",
                            "invalid_type"
                        )

        # Validate properties (for objects)
        if "properties" in schema:
            properties = schema["properties"]
            if not isinstance(properties, dict):
                result.add_error(
                    f"{path}.properties" if path else "properties",
                    "Properties must be an object",
                    "type"
                )
            else:
                for prop_name, prop_schema in properties.items():
                    prop_path = f"{path}.properties.{prop_name}" if path else f"properties.{prop_name}"
                    prop_result = self.validate_json_schema(prop_schema, prop_path)
                    result.merge(prop_result)

        # Validate items (for arrays)
        if "items" in schema:
            items = schema["items"]
            items_path = f"{path}.items" if path else "items"
            if isinstance(items, dict):
                items_result = self.validate_json_schema(items, items_path)
                result.merge(items_result)
            elif isinstance(items, list):
                for i, item in enumerate(items):
                    item_result = self.validate_json_schema(item, f"{items_path}[{i}]")
                    result.merge(item_result)

        # Validate required
        if "required" in schema:
            required = schema["required"]
            if not isinstance(required, list):
                result.add_error(
                    f"{path}.required" if path else "required",
                    "Required must be an array",
                    "type"
                )
            elif "properties" in schema:
                properties = schema["properties"]
                for req in required:
                    if req not in properties:
                        result.add_error(
                            f"{path}.required" if path else "required",
                            f"Required property '{req}' not in properties",
                            "missing_property"
                        )

        return result

    def validate_tool_call(
        self,
        tool_call: ToolCallSchema,
        available_tools: Optional[List[ToolSchema]] = None
    ) -> ValidationResult:
        """
        Validate a tool call.

        Checks:
        - ID is present
        - Function name is present
        - Arguments are valid JSON
        - If available_tools provided, validates against schema
        """
        result = ValidationResult.ok()

        # Validate ID
        if not tool_call.id:
            result.add_error("id", "Tool call ID is required", "required")

        # Validate function name
        if not tool_call.function.name:
            result.add_error("function.name", "Function name is required", "required")

        # Validate arguments JSON
        if tool_call.function.arguments:
            try:
                args = json.loads(tool_call.function.arguments)
                if not isinstance(args, dict):
                    result.add_error(
                        "function.arguments",
                        "Arguments must be a JSON object",
                        "type"
                    )
            except json.JSONDecodeError as e:
                result.add_error(
                    "function.arguments",
                    f"Invalid JSON: {e}",
                    "json_error"
                )

        # Validate against available tools
        if available_tools and tool_call.function.name:
            tool = next(
                (t for t in available_tools if t.function.name == tool_call.function.name),
                None
            )
            if tool is None:
                result.add_error(
                    "function.name",
                    f"Unknown function: {tool_call.function.name}",
                    "unknown_function"
                )
            elif tool.function.parameters:
                args_result = self.validate_arguments_against_schema(
                    tool_call.function.arguments,
                    tool.function.parameters
                )
                result.merge(args_result, "function.arguments")

        return result

    def validate_arguments_against_schema(
        self,
        arguments: str,
        schema: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate function arguments against a JSON Schema.

        Performs basic type checking and required field validation.
        """
        result = ValidationResult.ok()

        try:
            args = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError:
            result.add_error("", "Invalid JSON arguments", "json_error")
            return result

        if not isinstance(args, dict):
            result.add_error("", "Arguments must be an object", "type")
            return result

        # Check required fields
        required = schema.get("required", [])
        for req in required:
            if req not in args:
                result.add_error(req, f"Required field '{req}' is missing", "required")

        # Check property types
        properties = schema.get("properties", {})
        for prop_name, prop_value in args.items():
            if prop_name in properties:
                prop_schema = properties[prop_name]
                type_result = self._validate_value_type(prop_value, prop_schema, prop_name)
                result.merge(type_result)

        return result

    def _validate_value_type(
        self,
        value: Any,
        schema: Dict[str, Any],
        path: str
    ) -> ValidationResult:
        """Validate a value against a property schema."""
        result = ValidationResult.ok()

        expected_type = schema.get("type")
        if not expected_type:
            return result

        # Handle type unions
        if isinstance(expected_type, list):
            types = expected_type
        else:
            types = [expected_type]

        # Check if value matches any type
        matches = False
        for t in types:
            if self._value_matches_type(value, t):
                matches = True
                break

        if not matches:
            result.add_error(
                path,
                f"Expected type {expected_type}, got {type(value).__name__}",
                "type_mismatch"
            )

        # Validate enum
        if "enum" in schema and value not in schema["enum"]:
            result.add_error(
                path,
                f"Value must be one of: {schema['enum']}",
                "enum"
            )

        return result

    def _value_matches_type(self, value: Any, expected_type: str) -> bool:
        """Check if a value matches an expected JSON Schema type."""
        if expected_type == "null":
            return value is None
        elif expected_type == "boolean":
            return isinstance(value, bool)
        elif expected_type == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        elif expected_type == "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        elif expected_type == "string":
            return isinstance(value, str)
        elif expected_type == "array":
            return isinstance(value, list)
        elif expected_type == "object":
            return isinstance(value, dict)
        else:
            return True  # Unknown type, allow

    def validate_tool_result(
        self,
        result: ToolResultSchema,
        expected_tool_call_ids: Optional[Set[str]] = None
    ) -> ValidationResult:
        """
        Validate a tool result.

        Checks:
        - Tool call ID is present
        - Content is present
        - If expected_tool_call_ids provided, validates ID exists
        """
        validation = ValidationResult.ok()

        if not result.tool_call_id:
            validation.add_error("tool_call_id", "Tool call ID is required", "required")

        if result.content is None:
            validation.add_error("content", "Content is required", "required")

        if expected_tool_call_ids and result.tool_call_id:
            if result.tool_call_id not in expected_tool_call_ids:
                validation.add_error(
                    "tool_call_id",
                    f"Unknown tool call ID: {result.tool_call_id}",
                    "unknown_id"
                )

        return validation

    def validate_tools_list(self, tools: List[ToolSchema]) -> ValidationResult:
        """
        Validate a list of tools.

        Also checks for duplicate function names.
        """
        result = ValidationResult.ok()
        seen_names: Set[str] = set()

        for i, tool in enumerate(tools):
            # Validate each tool
            tool_result = self.validate_tool(tool)
            result.merge(tool_result, f"[{i}]")

            # Check for duplicates
            name = tool.function.name
            if name:
                if name in seen_names:
                    result.add_error(
                        f"[{i}].function.name",
                        f"Duplicate function name: {name}",
                        "duplicate"
                    )
                seen_names.add(name)

        return result


# Global instance
_validator = ToolValidator()


def validate_tool(tool: ToolSchema) -> ValidationResult:
    """Validate a single tool."""
    return _validator.validate_tool(tool)


def validate_tools(tools: List[ToolSchema]) -> ValidationResult:
    """Validate a list of tools."""
    return _validator.validate_tools_list(tools)


def validate_tool_call(
    tool_call: ToolCallSchema,
    available_tools: Optional[List[ToolSchema]] = None
) -> ValidationResult:
    """Validate a tool call."""
    return _validator.validate_tool_call(tool_call, available_tools)


def validate_tool_result(
    result: ToolResultSchema,
    expected_ids: Optional[Set[str]] = None
) -> ValidationResult:
    """Validate a tool result."""
    return _validator.validate_tool_result(result, expected_ids)
