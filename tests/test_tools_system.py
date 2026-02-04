"""
2api.ai - Tool Calling System Tests

Comprehensive tests for the tool calling module covering:
- Schema validation
- Cross-provider normalization
- Tool call validation
- Parallel execution
- Result batching
"""

import asyncio
import json
import pytest
import uuid
from typing import List

from src.tools import (
    # Schema
    ToolType,
    ToolChoiceMode,
    JSONSchemaType,
    PropertySchema,
    ParametersSchema,
    FunctionSchema,
    ToolSchema,
    ToolChoiceSchema,
    ToolChoiceSpecificSchema,
    FunctionCallSchema,
    ToolCallSchema,
    ToolResultSchema,
    # Normalizer
    ToolNormalizer,
    normalize_tools,
    normalize_tool_choice,
    normalize_tool_calls,
    # Validator
    ToolValidator,
    ValidationResult,
    validate_tool,
    validate_tools,
    validate_tool_call,
    validate_tool_result,
    # Parallel
    ToolCallStatus,
    ToolCallExecution,
    ParallelToolCallTracker,
    ParallelToolExecutor,
    ToolCallBatcher,
    execute_tools_parallel,
    batch_tool_results,
)


# ============================================================
# Test Fixtures
# ============================================================

@pytest.fixture
def sample_function_schema():
    """Create a sample function schema."""
    return FunctionSchema(
        name="get_weather",
        description="Get the current weather for a location",
        parameters={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "default": "celsius"
                }
            },
            "required": ["location"]
        }
    )


@pytest.fixture
def sample_tool(sample_function_schema):
    """Create a sample tool."""
    return ToolSchema(
        type=ToolType.FUNCTION,
        function=sample_function_schema
    )


@pytest.fixture
def sample_tools():
    """Create a list of sample tools."""
    return [
        ToolSchema(
            type=ToolType.FUNCTION,
            function=FunctionSchema(
                name="get_weather",
                description="Get current weather",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"]
                }
            )
        ),
        ToolSchema(
            type=ToolType.FUNCTION,
            function=FunctionSchema(
                name="search_web",
                description="Search the web",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "num_results": {"type": "integer", "default": 10}
                    },
                    "required": ["query"]
                }
            )
        ),
        ToolSchema(
            type=ToolType.FUNCTION,
            function=FunctionSchema(
                name="send_email",
                description="Send an email",
                parameters={
                    "type": "object",
                    "properties": {
                        "to": {"type": "string"},
                        "subject": {"type": "string"},
                        "body": {"type": "string"}
                    },
                    "required": ["to", "subject", "body"]
                }
            )
        )
    ]


@pytest.fixture
def sample_tool_call():
    """Create a sample tool call."""
    return ToolCallSchema(
        id="call_abc123",
        type=ToolType.FUNCTION,
        function=FunctionCallSchema(
            name="get_weather",
            arguments='{"location": "San Francisco, CA", "unit": "celsius"}'
        )
    )


@pytest.fixture
def sample_tool_calls():
    """Create multiple sample tool calls."""
    return [
        ToolCallSchema(
            id="call_001",
            type=ToolType.FUNCTION,
            function=FunctionCallSchema(
                name="get_weather",
                arguments='{"location": "New York"}'
            )
        ),
        ToolCallSchema(
            id="call_002",
            type=ToolType.FUNCTION,
            function=FunctionCallSchema(
                name="search_web",
                arguments='{"query": "python programming"}'
            )
        ),
        ToolCallSchema(
            id="call_003",
            type=ToolType.FUNCTION,
            function=FunctionCallSchema(
                name="send_email",
                arguments='{"to": "user@example.com", "subject": "Test", "body": "Hello"}'
            )
        )
    ]


# ============================================================
# Schema Tests
# ============================================================

class TestFunctionSchema:
    """Tests for FunctionSchema."""

    def test_create_basic_function(self):
        """Test creating a basic function schema."""
        func = FunctionSchema(name="test_func")
        assert func.name == "test_func"
        assert func.description == ""
        assert func.parameters == {}

    def test_create_full_function(self, sample_function_schema):
        """Test creating a full function schema."""
        assert sample_function_schema.name == "get_weather"
        assert "weather" in sample_function_schema.description.lower()
        assert "properties" in sample_function_schema.parameters
        assert "location" in sample_function_schema.parameters["properties"]

    def test_validate_valid_name(self):
        """Test validation passes for valid names."""
        valid_names = ["func", "my_function", "get-data", "Function123", "a" * 64]
        for name in valid_names:
            func = FunctionSchema(name=name)
            valid, error = func.validate_name()
            assert valid, f"Name '{name}' should be valid: {error}"

    def test_validate_invalid_names(self):
        """Test validation fails for invalid names."""
        invalid_names = ["", "a" * 65, "func name", "func.name", "func@name"]
        for name in invalid_names:
            func = FunctionSchema(name=name)
            valid, error = func.validate_name()
            assert not valid, f"Name '{name}' should be invalid"

    def test_to_dict(self, sample_function_schema):
        """Test converting to dictionary."""
        data = sample_function_schema.to_dict()
        assert data["name"] == "get_weather"
        assert "description" in data
        assert "parameters" in data

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "name": "test_func",
            "description": "A test function",
            "parameters": {"type": "object", "properties": {}}
        }
        func = FunctionSchema.from_dict(data)
        assert func.name == "test_func"
        assert func.description == "A test function"


class TestToolSchema:
    """Tests for ToolSchema."""

    def test_create_tool(self, sample_tool):
        """Test creating a tool."""
        assert sample_tool.type == ToolType.FUNCTION
        assert sample_tool.function.name == "get_weather"

    def test_validate_valid_tool(self, sample_tool):
        """Test validation passes for valid tool."""
        valid, errors = sample_tool.validate()
        assert valid
        assert len(errors) == 0

    def test_validate_invalid_tool(self):
        """Test validation fails for invalid tool."""
        tool = ToolSchema(
            function=FunctionSchema(name="")  # Empty name
        )
        valid, errors = tool.validate()
        assert not valid
        assert len(errors) > 0

    def test_to_openai_format(self, sample_tool):
        """Test converting to OpenAI format."""
        data = sample_tool.to_dict()
        assert data["type"] == "function"
        assert "function" in data
        assert data["function"]["name"] == "get_weather"

    def test_from_openai_format(self):
        """Test creating from OpenAI format."""
        data = {
            "type": "function",
            "function": {
                "name": "test_func",
                "description": "Test",
                "parameters": {"type": "object"}
            }
        }
        tool = ToolSchema.from_dict(data)
        assert tool.type == ToolType.FUNCTION
        assert tool.function.name == "test_func"


class TestToolChoiceSchema:
    """Tests for ToolChoiceSchema."""

    def test_auto_choice(self):
        """Test auto tool choice."""
        choice = ToolChoiceSchema.auto()
        assert choice.mode == ToolChoiceMode.AUTO
        assert choice.to_string_or_dict() == "auto"

    def test_none_choice(self):
        """Test none tool choice."""
        choice = ToolChoiceSchema.none()
        assert choice.mode == ToolChoiceMode.NONE
        assert choice.to_string_or_dict() == "none"

    def test_required_choice(self):
        """Test required tool choice."""
        choice = ToolChoiceSchema.required()
        assert choice.mode == ToolChoiceMode.REQUIRED
        assert choice.to_string_or_dict() == "required"

    def test_specific_choice(self):
        """Test specific tool choice."""
        choice = ToolChoiceSchema.specific("get_weather")
        assert choice.mode == ToolChoiceMode.SPECIFIC
        result = choice.to_string_or_dict()
        assert isinstance(result, dict)
        assert result["function"]["name"] == "get_weather"

    def test_from_string(self):
        """Test creating from string."""
        for s in ["auto", "none", "required"]:
            choice = ToolChoiceSchema.from_any(s)
            assert choice.mode.value == s

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {"type": "function", "function": {"name": "my_func"}}
        choice = ToolChoiceSchema.from_any(data)
        assert choice.mode == ToolChoiceMode.SPECIFIC
        assert choice.specific_tool.function_name == "my_func"


class TestToolCallSchema:
    """Tests for ToolCallSchema."""

    def test_create_tool_call(self, sample_tool_call):
        """Test creating a tool call."""
        assert sample_tool_call.id == "call_abc123"
        assert sample_tool_call.function.name == "get_weather"

    def test_validate_valid_tool_call(self, sample_tool_call):
        """Test validation passes for valid tool call."""
        valid, errors = sample_tool_call.validate()
        assert valid
        assert len(errors) == 0

    def test_validate_missing_id(self):
        """Test validation fails for missing ID."""
        tc = ToolCallSchema(
            id="",
            function=FunctionCallSchema(name="test", arguments="{}")
        )
        valid, errors = tc.validate()
        assert not valid
        assert any("ID" in e for e in errors)

    def test_validate_invalid_json(self):
        """Test validation fails for invalid JSON arguments."""
        tc = ToolCallSchema(
            id="call_123",
            function=FunctionCallSchema(name="test", arguments="not json")
        )
        valid, errors = tc.validate()
        assert not valid
        assert any("JSON" in e for e in errors)

    def test_get_arguments_dict(self, sample_tool_call):
        """Test getting arguments as dict."""
        args = sample_tool_call.function.get_arguments_dict()
        assert args["location"] == "San Francisco, CA"
        assert args["unit"] == "celsius"

    def test_to_dict(self, sample_tool_call):
        """Test converting to dictionary."""
        data = sample_tool_call.to_dict()
        assert data["id"] == "call_abc123"
        assert data["type"] == "function"
        assert data["function"]["name"] == "get_weather"


# ============================================================
# Normalizer Tests
# ============================================================

class TestToolNormalizer:
    """Tests for ToolNormalizer."""

    @pytest.fixture
    def normalizer(self):
        return ToolNormalizer()

    # Tool normalization tests
    def test_normalize_for_openai(self, normalizer, sample_tools):
        """Test normalizing tools for OpenAI."""
        result = normalizer.normalize_tools_for_openai(sample_tools)
        assert len(result) == 3
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "get_weather"

    def test_normalize_for_anthropic(self, normalizer, sample_tools):
        """Test normalizing tools for Anthropic."""
        result = normalizer.normalize_tools_for_anthropic(sample_tools)
        assert len(result) == 3
        # Anthropic uses input_schema instead of parameters
        assert "input_schema" in result[0]
        assert "parameters" not in result[0]
        assert result[0]["name"] == "get_weather"

    def test_normalize_for_google(self, normalizer, sample_tools):
        """Test normalizing tools for Google Gemini."""
        result = normalizer.normalize_tools_for_google(sample_tools)
        # Google wraps in functionDeclarations
        assert len(result) == 1
        assert "functionDeclarations" in result[0]
        declarations = result[0]["functionDeclarations"]
        assert len(declarations) == 3
        assert declarations[0]["name"] == "get_weather"

    # Tool choice normalization tests
    def test_normalize_choice_for_openai(self, normalizer):
        """Test normalizing tool choice for OpenAI."""
        for mode in ["auto", "none", "required"]:
            choice = ToolChoiceSchema.from_any(mode)
            result = normalizer.normalize_tool_choice_for_openai(choice)
            assert result == mode

        # Specific tool
        choice = ToolChoiceSchema.specific("my_func")
        result = normalizer.normalize_tool_choice_for_openai(choice)
        assert result["function"]["name"] == "my_func"

    def test_normalize_choice_for_anthropic(self, normalizer):
        """Test normalizing tool choice for Anthropic."""
        # Auto
        choice = ToolChoiceSchema.auto()
        result = normalizer.normalize_tool_choice_for_anthropic(choice)
        assert result == {"type": "auto"}

        # None
        choice = ToolChoiceSchema.none()
        result = normalizer.normalize_tool_choice_for_anthropic(choice)
        assert result == {"type": "none"}

        # Required -> any in Anthropic
        choice = ToolChoiceSchema.required()
        result = normalizer.normalize_tool_choice_for_anthropic(choice)
        assert result == {"type": "any"}

        # Specific
        choice = ToolChoiceSchema.specific("my_func")
        result = normalizer.normalize_tool_choice_for_anthropic(choice)
        assert result == {"type": "tool", "name": "my_func"}

    def test_normalize_choice_for_google(self, normalizer):
        """Test normalizing tool choice for Google."""
        # Auto
        choice = ToolChoiceSchema.auto()
        result = normalizer.normalize_tool_choice_for_google(choice)
        assert result == {"mode": "AUTO"}

        # Required
        choice = ToolChoiceSchema.required()
        result = normalizer.normalize_tool_choice_for_google(choice)
        assert result == {"mode": "ANY"}

        # Specific
        choice = ToolChoiceSchema.specific("my_func")
        result = normalizer.normalize_tool_choice_for_google(choice)
        assert result == {"mode": "TOOL", "allowedFunctionNames": ["my_func"]}

    # Tool call normalization tests
    def test_normalize_calls_from_openai(self, normalizer):
        """Test normalizing tool calls from OpenAI."""
        openai_calls = [
            {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "NYC"}'
                }
            }
        ]
        result = normalizer.normalize_tool_calls_from_openai(openai_calls)
        assert len(result) == 1
        assert result[0].id == "call_123"
        assert result[0].function.name == "get_weather"

    def test_normalize_calls_from_anthropic(self, normalizer):
        """Test normalizing tool calls from Anthropic."""
        anthropic_blocks = [
            {
                "type": "tool_use",
                "id": "toolu_123",
                "name": "get_weather",
                "input": {"location": "NYC"}
            },
            {
                "type": "text",
                "text": "Some text"  # Should be ignored
            }
        ]
        result = normalizer.normalize_tool_calls_from_anthropic(anthropic_blocks)
        assert len(result) == 1
        assert result[0].id == "toolu_123"
        assert result[0].function.name == "get_weather"
        # Arguments should be JSON string
        args = json.loads(result[0].function.arguments)
        assert args["location"] == "NYC"

    def test_normalize_calls_from_google(self, normalizer):
        """Test normalizing tool calls from Google Gemini."""
        google_parts = [
            {
                "functionCall": {
                    "name": "get_weather",
                    "args": {"location": "NYC"}
                }
            },
            {
                "text": "Some text"  # Should be ignored
            }
        ]
        result = normalizer.normalize_tool_calls_from_google(google_parts)
        assert len(result) == 1
        assert result[0].function.name == "get_weather"
        # Arguments should be JSON string
        args = json.loads(result[0].function.arguments)
        assert args["location"] == "NYC"

    # Tool result normalization tests
    def test_create_result_for_openai(self, normalizer):
        """Test creating tool result for OpenAI."""
        result = ToolResultSchema(
            tool_call_id="call_123",
            content="Sunny, 72°F"
        )
        msg = normalizer.create_tool_result_message_for_openai(result)
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "call_123"
        assert msg["content"] == "Sunny, 72°F"

    def test_create_result_for_anthropic(self, normalizer):
        """Test creating tool result for Anthropic."""
        result = ToolResultSchema(
            tool_call_id="call_123",
            content="Sunny, 72°F"
        )
        msg = normalizer.create_tool_result_message_for_anthropic(result)
        assert msg["role"] == "user"
        assert len(msg["content"]) == 1
        assert msg["content"][0]["type"] == "tool_result"
        assert msg["content"][0]["tool_use_id"] == "call_123"

    def test_create_result_for_google(self, normalizer):
        """Test creating tool result for Google."""
        result = ToolResultSchema(
            tool_call_id="call_123",
            content="Sunny, 72°F",
            name="get_weather"
        )
        msg = normalizer.create_tool_result_message_for_google(result)
        assert msg["role"] == "user"
        assert "functionResponse" in msg["parts"][0]
        assert msg["parts"][0]["functionResponse"]["name"] == "get_weather"


class TestNormalizerConvenienceFunctions:
    """Tests for normalizer convenience functions."""

    def test_normalize_tools_function(self, sample_tools):
        """Test normalize_tools convenience function."""
        for provider in ["openai", "anthropic", "google"]:
            result = normalize_tools(sample_tools, provider)
            assert result is not None
            if provider == "google":
                assert "functionDeclarations" in result[0]
            else:
                assert len(result) == 3

    def test_normalize_tool_choice_function(self):
        """Test normalize_tool_choice convenience function."""
        choice = ToolChoiceSchema.auto()
        assert normalize_tool_choice(choice, "openai") == "auto"
        assert normalize_tool_choice(choice, "anthropic") == {"type": "auto"}
        assert normalize_tool_choice(choice, "google") == {"mode": "AUTO"}

    def test_normalize_tool_calls_function(self):
        """Test normalize_tool_calls convenience function."""
        openai_data = [
            {"id": "call_1", "type": "function", "function": {"name": "test", "arguments": "{}"}}
        ]
        result = normalize_tool_calls(openai_data, "openai")
        assert len(result) == 1
        assert result[0].function.name == "test"


# ============================================================
# Validator Tests
# ============================================================

class TestToolValidator:
    """Tests for ToolValidator."""

    @pytest.fixture
    def validator(self):
        return ToolValidator()

    def test_validate_valid_tool(self, validator, sample_tool):
        """Test validating a valid tool."""
        result = validator.validate_tool(sample_tool)
        assert result.is_valid
        assert len(result.errors) == 0

    def test_validate_invalid_function_name(self, validator):
        """Test validating tool with invalid function name."""
        tool = ToolSchema(
            function=FunctionSchema(name="invalid name!")
        )
        result = validator.validate_tool(tool)
        assert not result.is_valid
        assert any("pattern" in e.code for e in result.errors)

    def test_validate_json_schema(self, validator):
        """Test JSON Schema validation."""
        valid_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name"]
        }
        result = validator.validate_json_schema(valid_schema)
        assert result.is_valid

    def test_validate_invalid_json_schema_type(self, validator):
        """Test JSON Schema validation with invalid type."""
        invalid_schema = {
            "type": "invalid_type"
        }
        result = validator.validate_json_schema(invalid_schema)
        assert not result.is_valid

    def test_validate_tool_call_valid(self, validator, sample_tool_call, sample_tools):
        """Test validating a valid tool call."""
        result = validator.validate_tool_call(sample_tool_call, sample_tools)
        assert result.is_valid

    def test_validate_tool_call_unknown_function(self, validator, sample_tools):
        """Test validating tool call with unknown function."""
        tc = ToolCallSchema(
            id="call_123",
            function=FunctionCallSchema(name="unknown_func", arguments="{}")
        )
        result = validator.validate_tool_call(tc, sample_tools)
        assert not result.is_valid
        assert any("unknown_function" in e.code for e in result.errors)

    def test_validate_tool_call_missing_required_arg(self, validator, sample_tools):
        """Test validating tool call missing required argument."""
        tc = ToolCallSchema(
            id="call_123",
            function=FunctionCallSchema(
                name="get_weather",
                arguments='{}'  # Missing required 'location'
            )
        )
        result = validator.validate_tool_call(tc, sample_tools)
        assert not result.is_valid
        assert any("required" in e.code for e in result.errors)

    def test_validate_tools_list_duplicates(self, validator):
        """Test validating tool list with duplicates."""
        tools = [
            ToolSchema(function=FunctionSchema(name="func1")),
            ToolSchema(function=FunctionSchema(name="func1")),  # Duplicate
        ]
        result = validator.validate_tools_list(tools)
        assert not result.is_valid
        assert any("duplicate" in e.code for e in result.errors)

    def test_validate_tool_result(self, validator):
        """Test validating a tool result."""
        result = ToolResultSchema(tool_call_id="call_123", content="Result")
        validation = validator.validate_tool_result(result)
        assert validation.is_valid

    def test_validate_tool_result_unknown_id(self, validator):
        """Test validating tool result with unknown ID."""
        result = ToolResultSchema(tool_call_id="call_unknown", content="Result")
        validation = validator.validate_tool_result(
            result,
            expected_tool_call_ids={"call_123", "call_456"}
        )
        assert not validation.is_valid


class TestValidatorConvenienceFunctions:
    """Tests for validator convenience functions."""

    def test_validate_tool_function(self, sample_tool):
        """Test validate_tool convenience function."""
        result = validate_tool(sample_tool)
        assert result.is_valid

    def test_validate_tools_function(self, sample_tools):
        """Test validate_tools convenience function."""
        result = validate_tools(sample_tools)
        assert result.is_valid

    def test_validate_tool_call_function(self, sample_tool_call):
        """Test validate_tool_call convenience function."""
        result = validate_tool_call(sample_tool_call)
        assert result.is_valid


# ============================================================
# Parallel Execution Tests
# ============================================================

class TestParallelToolCallTracker:
    """Tests for ParallelToolCallTracker."""

    def test_create_tracker(self):
        """Test creating a tracker."""
        tracker = ParallelToolCallTracker(request_id="req_123")
        assert tracker.request_id == "req_123"
        assert len(tracker.executions) == 0

    def test_add_tool_calls(self, sample_tool_calls):
        """Test adding tool calls."""
        tracker = ParallelToolCallTracker(request_id="req_123")
        ids = tracker.add_tool_calls(sample_tool_calls)
        assert len(ids) == 3
        assert tracker.pending_count == 3
        assert not tracker.is_all_complete

    def test_execution_lifecycle(self, sample_tool_call):
        """Test execution lifecycle."""
        tracker = ParallelToolCallTracker(request_id="req_123")
        tracker.add_tool_call(sample_tool_call)

        # Start
        tracker.start_execution(sample_tool_call.id)
        assert tracker.running_count == 1
        assert tracker.pending_count == 0

        # Complete
        tracker.complete_execution(sample_tool_call.id, "Weather is sunny")
        assert tracker.completed_count == 1
        assert tracker.is_all_complete

        # Get results
        results = tracker.get_results()
        assert len(results) == 1
        assert results[0].content == "Weather is sunny"

    def test_execution_failure(self, sample_tool_call):
        """Test handling execution failure."""
        tracker = ParallelToolCallTracker(request_id="req_123")
        tracker.add_tool_call(sample_tool_call)
        tracker.start_execution(sample_tool_call.id)

        # Fail
        tracker.fail_execution(sample_tool_call.id, "API error")
        assert tracker.has_failures
        assert tracker.is_all_complete

        # Result should be error
        results = tracker.get_results()
        assert len(results) == 1
        assert results[0].is_error
        assert "API error" in results[0].content

    def test_execution_timeout(self, sample_tool_call):
        """Test handling execution timeout."""
        tracker = ParallelToolCallTracker(request_id="req_123")
        tracker.add_tool_call(sample_tool_call)
        tracker.start_execution(sample_tool_call.id)

        # Timeout
        tracker.timeout_execution(sample_tool_call.id)
        assert tracker.has_failures

        # Result should indicate timeout
        results = tracker.get_results()
        assert len(results) == 1
        assert results[0].is_error
        assert "timed out" in results[0].content.lower()


class TestParallelToolExecutor:
    """Tests for ParallelToolExecutor."""

    @pytest.mark.asyncio
    async def test_execute_single_tool(self, sample_tool_call):
        """Test executing a single tool."""
        executor = ParallelToolExecutor(max_concurrent=5)

        async def mock_executor(tc: ToolCallSchema) -> str:
            return f"Result for {tc.function.name}"

        tracker = await executor.execute_all(
            [sample_tool_call],
            mock_executor
        )

        assert tracker.is_all_complete
        assert tracker.completed_count == 1
        results = tracker.get_results()
        assert "Result for get_weather" in results[0].content

    @pytest.mark.asyncio
    async def test_execute_multiple_tools(self, sample_tool_calls):
        """Test executing multiple tools in parallel."""
        executor = ParallelToolExecutor(max_concurrent=5)

        async def mock_executor(tc: ToolCallSchema) -> str:
            await asyncio.sleep(0.01)  # Simulate some work
            return f"Result for {tc.function.name}"

        tracker = await executor.execute_all(
            sample_tool_calls,
            mock_executor
        )

        assert tracker.is_all_complete
        assert tracker.completed_count == 3
        assert not tracker.has_failures

    @pytest.mark.asyncio
    async def test_execute_with_timeout(self, sample_tool_call):
        """Test execution with timeout."""
        executor = ParallelToolExecutor(max_concurrent=5, default_timeout=0.1)

        async def slow_executor(tc: ToolCallSchema) -> str:
            await asyncio.sleep(1.0)  # Too slow
            return "Result"

        tracker = await executor.execute_all(
            [sample_tool_call],
            slow_executor,
            timeout=0.05
        )

        assert tracker.is_all_complete
        assert tracker.has_failures
        exec_status = tracker.get_execution(sample_tool_call.id)
        assert exec_status.status == ToolCallStatus.TIMEOUT

    @pytest.mark.asyncio
    async def test_execute_with_failure(self, sample_tool_call):
        """Test execution with failure."""
        executor = ParallelToolExecutor(max_concurrent=5)

        async def failing_executor(tc: ToolCallSchema) -> str:
            raise ValueError("Something went wrong")

        tracker = await executor.execute_all(
            [sample_tool_call],
            failing_executor
        )

        assert tracker.is_all_complete
        assert tracker.has_failures
        exec_status = tracker.get_execution(sample_tool_call.id)
        assert exec_status.status == ToolCallStatus.FAILED
        assert "Something went wrong" in exec_status.error

    @pytest.mark.asyncio
    async def test_concurrency_limit(self, sample_tool_calls):
        """Test that concurrency is limited."""
        executor = ParallelToolExecutor(max_concurrent=1)  # Only 1 at a time
        execution_order = []

        async def tracking_executor(tc: ToolCallSchema) -> str:
            execution_order.append(f"start_{tc.id}")
            await asyncio.sleep(0.01)
            execution_order.append(f"end_{tc.id}")
            return "Result"

        await executor.execute_all(
            sample_tool_calls,
            tracking_executor
        )

        # With concurrency=1, executions should be sequential
        # Each "end" should come before the next "start"
        for i in range(len(sample_tool_calls) - 1):
            end_idx = execution_order.index(f"end_call_00{i+1}")
            start_idx = execution_order.index(f"start_call_00{i+2}")
            assert end_idx < start_idx


class TestToolCallBatcher:
    """Tests for ToolCallBatcher."""

    @pytest.fixture
    def batcher(self):
        return ToolCallBatcher()

    @pytest.fixture
    def sample_results(self):
        return [
            ToolResultSchema(
                tool_call_id="call_001",
                content="Weather: Sunny",
                name="get_weather"
            ),
            ToolResultSchema(
                tool_call_id="call_002",
                content="Search results...",
                name="search_web"
            )
        ]

    def test_batch_for_openai(self, batcher, sample_results):
        """Test batching for OpenAI."""
        batch = batcher.batch_for_openai(sample_results)
        # OpenAI wants separate messages
        assert len(batch) == 2
        assert batch[0]["role"] == "tool"
        assert batch[0]["tool_call_id"] == "call_001"
        assert batch[1]["tool_call_id"] == "call_002"

    def test_batch_for_anthropic(self, batcher, sample_results):
        """Test batching for Anthropic."""
        batch = batcher.batch_for_anthropic(sample_results)
        # Anthropic wants single message with multiple blocks
        assert batch["role"] == "user"
        assert len(batch["content"]) == 2
        assert batch["content"][0]["type"] == "tool_result"
        assert batch["content"][0]["tool_use_id"] == "call_001"
        assert batch["content"][1]["tool_use_id"] == "call_002"

    def test_batch_for_google(self, batcher, sample_results):
        """Test batching for Google."""
        batch = batcher.batch_for_google(sample_results)
        # Google wants single message with multiple parts
        assert batch["role"] == "user"
        assert len(batch["parts"]) == 2
        assert "functionResponse" in batch["parts"][0]
        assert batch["parts"][0]["functionResponse"]["name"] == "get_weather"

    def test_batch_for_provider(self, batcher, sample_results):
        """Test batch_for_provider routing."""
        for provider in ["openai", "anthropic", "google"]:
            batch = batcher.batch_for_provider(sample_results, provider)
            assert batch is not None


class TestParallelConvenienceFunctions:
    """Tests for parallel convenience functions."""

    @pytest.mark.asyncio
    async def test_execute_tools_parallel(self, sample_tool_calls):
        """Test execute_tools_parallel convenience function."""
        async def mock_executor(tc: ToolCallSchema) -> str:
            return f"Result for {tc.function.name}"

        tracker = await execute_tools_parallel(
            sample_tool_calls,
            mock_executor,
            timeout=5.0,
            max_concurrent=5
        )

        assert tracker.is_all_complete
        assert len(tracker.get_results()) == 3

    def test_batch_tool_results(self):
        """Test batch_tool_results convenience function."""
        results = [
            ToolResultSchema(tool_call_id="call_1", content="R1", name="f1"),
            ToolResultSchema(tool_call_id="call_2", content="R2", name="f2")
        ]

        for provider in ["openai", "anthropic", "google"]:
            batch = batch_tool_results(results, provider)
            assert batch is not None


# ============================================================
# Integration Tests
# ============================================================

class TestToolCallingIntegration:
    """Integration tests for the complete tool calling flow."""

    @pytest.mark.asyncio
    async def test_full_tool_calling_flow(self, sample_tools):
        """Test complete tool calling flow: define → call → validate → execute → batch."""
        # 1. Define tools
        validation = validate_tools(sample_tools)
        assert validation.is_valid

        # 2. Normalize for provider
        openai_tools = normalize_tools(sample_tools, "openai")
        anthropic_tools = normalize_tools(sample_tools, "anthropic")
        google_tools = normalize_tools(sample_tools, "google")

        assert len(openai_tools) == 3
        assert len(anthropic_tools) == 3
        assert "functionDeclarations" in google_tools[0]

        # 3. Simulate model response with tool calls
        tool_calls = [
            ToolCallSchema(
                id="call_001",
                function=FunctionCallSchema(
                    name="get_weather",
                    arguments='{"location": "NYC"}'
                )
            ),
            ToolCallSchema(
                id="call_002",
                function=FunctionCallSchema(
                    name="search_web",
                    arguments='{"query": "python"}'
                )
            )
        ]

        # 4. Validate tool calls
        for tc in tool_calls:
            result = validate_tool_call(tc, sample_tools)
            assert result.is_valid

        # 5. Execute tools in parallel
        async def mock_executor(tc: ToolCallSchema) -> str:
            args = tc.function.get_arguments_dict()
            if tc.function.name == "get_weather":
                return f"Weather in {args['location']}: Sunny"
            elif tc.function.name == "search_web":
                return f"Results for '{args['query']}': ..."
            return "Unknown function"

        tracker = await execute_tools_parallel(
            tool_calls,
            mock_executor,
            timeout=5.0
        )

        assert tracker.is_all_complete
        assert not tracker.has_failures

        # 6. Batch results for providers
        results = tracker.get_results()
        openai_batch = batch_tool_results(results, "openai")
        anthropic_batch = batch_tool_results(results, "anthropic")
        google_batch = batch_tool_results(results, "google")

        # OpenAI: list of messages
        assert len(openai_batch) == 2

        # Anthropic: single message with content blocks
        assert anthropic_batch["role"] == "user"
        assert len(anthropic_batch["content"]) == 2

        # Google: single message with parts
        assert google_batch["role"] == "user"
        assert len(google_batch["parts"]) == 2

    def test_cross_provider_format_roundtrip(self, sample_tool):
        """Test that tool definitions survive format conversion."""
        normalizer = ToolNormalizer()

        # Convert to each provider format
        openai_format = normalizer.normalize_tools_for_openai([sample_tool])[0]
        anthropic_format = normalizer.normalize_tools_for_anthropic([sample_tool])[0]
        google_format = normalizer.normalize_tools_for_google([sample_tool])[0]["functionDeclarations"][0]

        # All should preserve the function name
        assert openai_format["function"]["name"] == "get_weather"
        assert anthropic_format["name"] == "get_weather"
        assert google_format["name"] == "get_weather"

        # All should preserve the description
        assert "weather" in openai_format["function"]["description"].lower()
        assert "weather" in anthropic_format["description"].lower()
        assert "weather" in google_format["description"].lower()

    def test_error_handling_in_parallel_execution(self):
        """Test that parallel execution handles mixed success/failure."""
        tracker = ParallelToolCallTracker(request_id="test_123")

        # Add 3 tool calls
        for i in range(3):
            tc = ToolCallSchema(
                id=f"call_{i}",
                function=FunctionCallSchema(name=f"func_{i}", arguments="{}")
            )
            tracker.add_tool_call(tc)

        # Succeed first, fail second, timeout third
        tracker.start_execution("call_0")
        tracker.complete_execution("call_0", "Success!")

        tracker.start_execution("call_1")
        tracker.fail_execution("call_1", "Failed!")

        tracker.start_execution("call_2")
        tracker.timeout_execution("call_2")

        # Verify state
        assert tracker.is_all_complete
        assert tracker.has_failures
        assert tracker.completed_count == 3  # All completed (success or failure)

        # Get results
        results = tracker.get_results()
        assert len(results) == 3

        # Check individual results
        success_result = next(r for r in results if r.tool_call_id == "call_0")
        assert not success_result.is_error

        fail_result = next(r for r in results if r.tool_call_id == "call_1")
        assert fail_result.is_error

        timeout_result = next(r for r in results if r.tool_call_id == "call_2")
        assert timeout_result.is_error


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
