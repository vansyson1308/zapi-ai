"""
2api.ai Python SDK - Tool Calling Tests
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio

from twoapi import TwoAPI
from twoapi.tools import tool, create_tool, ToolRunner


class TestToolDecorator:
    """Tests for @tool decorator."""

    def test_tool_decorator_basic(self):
        """Test basic tool decorator usage."""
        @tool(description="Get weather for a location")
        def get_weather(location: str) -> str:
            return f"Weather in {location}: sunny"

        assert get_weather.name == "get_weather"
        assert get_weather.description == "Get weather for a location"
        assert "function" in get_weather.tool

    def test_tool_decorator_with_custom_name(self):
        """Test tool decorator with custom name."""
        @tool(name="custom_name", description="Custom tool")
        def my_function() -> str:
            return "result"

        assert my_function.name == "custom_name"

    def test_tool_decorator_with_parameters(self):
        """Test tool decorator with parameters schema."""
        @tool(
            description="Add two numbers",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"},
                },
                "required": ["a", "b"],
            },
        )
        def add(a: float, b: float) -> str:
            return str(a + b)

        assert add.parameters["properties"]["a"]["type"] == "number"
        assert "required" in add.parameters

    def test_tool_execution(self):
        """Test executing a decorated tool."""
        @tool(description="Multiply")
        def multiply(a: int, b: int) -> str:
            return str(a * b)

        result = multiply(3, 4)
        assert result == "12"

    def test_tool_generates_openai_format(self):
        """Test that tool generates OpenAI-compatible format."""
        @tool(description="Search the web")
        def search(query: str) -> str:
            return f"Results for: {query}"

        tool_def = search.tool
        assert tool_def["type"] == "function"
        assert tool_def["function"]["name"] == "search"
        assert tool_def["function"]["description"] == "Search the web"


class TestCreateTool:
    """Tests for create_tool function."""

    def test_create_tool_basic(self):
        """Test creating tool with positional arguments."""
        my_tool = create_tool(
            name="my_tool",
            description="My tool description",
            parameters={
                "type": "object",
                "properties": {"input": {"type": "string"}},
            },
            execute=lambda args: f"Processed: {args['input']}",
        )

        assert my_tool.name == "my_tool"
        assert my_tool.description == "My tool description"

    def test_create_tool_execution(self):
        """Test executing created tool."""
        my_tool = create_tool(
            name="upper",
            description="Convert to uppercase",
            parameters={"type": "object", "properties": {}},
            execute=lambda args: args.get("text", "").upper(),
        )

        result = my_tool.execute({"text": "hello"})
        assert result == "HELLO"


class TestToolRunner:
    """Tests for ToolRunner class."""

    @pytest.fixture
    def weather_tool(self):
        """Create a weather tool for testing."""
        @tool(
            description="Get weather",
            parameters={
                "type": "object",
                "properties": {"location": {"type": "string"}},
            },
        )
        def get_weather(location: str) -> str:
            return f"Weather in {location}: 22C, sunny"

        return get_weather

    @pytest.fixture
    def search_tool(self):
        """Create a search tool for testing."""
        @tool(description="Search")
        def search(query: str) -> str:
            return f"Results for: {query}"

        return search

    @pytest.fixture
    def mock_client(self):
        """Create a mock TwoAPI client."""
        return MagicMock(spec=TwoAPI)

    def test_runner_creation(self, weather_tool, search_tool):
        """Test creating a ToolRunner."""
        runner = ToolRunner([weather_tool, search_tool])
        assert runner is not None

    def test_runner_with_max_iterations(self, weather_tool):
        """Test ToolRunner with max_iterations."""
        runner = ToolRunner([weather_tool], max_iterations=5)
        assert runner is not None

    def test_execute_tool_call(self, weather_tool):
        """Test executing a single tool call."""
        runner = ToolRunner([weather_tool])

        result = runner.execute_tool_call({
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"location": "Tokyo"}',
            },
        })

        assert result.tool_call_id == "call_123"
        assert result.name == "get_weather"
        assert "Tokyo" in result.result
        assert "22C" in result.result
        assert result.error is None

    def test_execute_unknown_tool(self, weather_tool):
        """Test executing unknown tool returns error."""
        runner = ToolRunner([weather_tool])

        result = runner.execute_tool_call({
            "id": "call_456",
            "type": "function",
            "function": {
                "name": "unknown_tool",
                "arguments": "{}",
            },
        })

        assert result.error is not None
        assert "Unknown tool" in result.error

    def test_execute_invalid_json_arguments(self, weather_tool):
        """Test executing with invalid JSON arguments."""
        runner = ToolRunner([weather_tool])

        result = runner.execute_tool_call({
            "id": "call_789",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": "invalid json",
            },
        })

        assert result.error is not None
        assert "Invalid JSON" in result.error

    def test_execute_tool_that_raises(self):
        """Test executing tool that raises exception."""
        @tool(description="Failing tool")
        def failing_tool() -> str:
            raise ValueError("Execution failed")

        runner = ToolRunner([failing_tool])

        result = runner.execute_tool_call({
            "id": "call_err",
            "type": "function",
            "function": {
                "name": "failing_tool",
                "arguments": "{}",
            },
        })

        assert result.error == "Execution failed"

    def test_run_without_tool_calls(self, mock_client, weather_tool):
        """Test run when model doesn't use tools."""
        mock_client.chat.return_value = MagicMock(
            content="Hello!",
            tool_calls=None,
            model="gpt-4o",
        )

        runner = ToolRunner([weather_tool])
        result = runner.run(mock_client, "Hello")

        assert result.response.content == "Hello!"
        assert len(result.tool_calls_made) == 0
        assert result.total_iterations == 1

    def test_run_with_tool_calls(self, mock_client, weather_tool):
        """Test run with tool calls."""
        # First call returns tool call
        first_response = MagicMock()
        first_response.content = ""
        first_response.tool_calls = [{
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"location": "Tokyo"}',
            },
        }]

        # Second call returns final response
        second_response = MagicMock()
        second_response.content = "The weather in Tokyo is 22C and sunny!"
        second_response.tool_calls = None

        mock_client.chat.side_effect = [first_response, second_response]

        runner = ToolRunner([weather_tool])
        result = runner.run(mock_client, "What is the weather in Tokyo?")

        assert result.response.content == "The weather in Tokyo is 22C and sunny!"
        assert len(result.tool_calls_made) == 1
        assert result.tool_calls_made[0].name == "get_weather"
        assert result.total_iterations == 2

    def test_run_with_multiple_tool_calls(self, mock_client, weather_tool, search_tool):
        """Test run with multiple parallel tool calls."""
        # First call returns multiple tool calls
        first_response = MagicMock()
        first_response.content = ""
        first_response.tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "Tokyo"}',
                },
            },
            {
                "id": "call_2",
                "type": "function",
                "function": {
                    "name": "search",
                    "arguments": '{"query": "Tokyo attractions"}',
                },
            },
        ]

        # Second call returns final response
        second_response = MagicMock()
        second_response.content = "Here is info about Tokyo..."
        second_response.tool_calls = None

        mock_client.chat.side_effect = [first_response, second_response]

        runner = ToolRunner([weather_tool, search_tool])
        result = runner.run(mock_client, "Tell me about Tokyo")

        assert len(result.tool_calls_made) == 2

    def test_run_respects_max_iterations(self, mock_client, weather_tool):
        """Test that run respects max_iterations."""
        # Always return tool calls (infinite loop scenario)
        mock_client.chat.return_value = MagicMock(
            content="",
            tool_calls=[{
                "id": "call_loop",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "Tokyo"}',
                },
            }],
        )

        runner = ToolRunner([weather_tool], max_iterations=3)
        result = runner.run(mock_client, "Loop forever")

        assert result.total_iterations == 3

    def test_run_passes_system_and_model(self, mock_client, weather_tool):
        """Test that run passes system and model options."""
        mock_client.chat.return_value = MagicMock(
            content="Response",
            tool_calls=None,
        )

        runner = ToolRunner([weather_tool])
        runner.run(
            mock_client,
            "Hello",
            model="anthropic/claude-3-opus",
            system="You are a helpful assistant",
        )

        mock_client.chat.assert_called_once()
        call_kwargs = mock_client.chat.call_args[1]
        assert call_kwargs["model"] == "anthropic/claude-3-opus"
        assert call_kwargs["system"] == "You are a helpful assistant"

    def test_run_includes_existing_messages(self, mock_client, weather_tool):
        """Test that run includes existing conversation messages."""
        mock_client.chat.return_value = MagicMock(
            content="Response",
            tool_calls=None,
        )

        runner = ToolRunner([weather_tool])
        runner.run(
            mock_client,
            "New message",
            messages=[
                {"role": "user", "content": "Previous message"},
                {"role": "assistant", "content": "Previous response"},
            ],
        )

        call_args = mock_client.chat.call_args[0][0]
        assert len(call_args) == 3  # Previous + new message

    def test_on_tool_call_callback(self, weather_tool):
        """Test on_tool_call callback is invoked."""
        callback = MagicMock()
        runner = ToolRunner([weather_tool], on_tool_call=callback)

        runner.execute_tool_call({
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"location": "Tokyo"}',
            },
        })

        callback.assert_called_once_with(
            "get_weather",
            '{"location": "Tokyo"}',
            "Weather in Tokyo: 22C, sunny"
        )


class TestAsyncToolRunner:
    """Tests for async ToolRunner operations."""

    @pytest.fixture
    def async_weather_tool(self):
        """Create an async weather tool."""
        @tool(description="Get weather async")
        async def get_weather_async(location: str) -> str:
            return f"Weather in {location}: 22C, sunny"

        return get_weather_async

    @pytest.fixture
    def mock_async_client(self):
        """Create a mock async TwoAPI client."""
        client = MagicMock()
        client.chat = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_async_tool_execution(self, async_weather_tool):
        """Test executing async tool."""
        runner = ToolRunner([async_weather_tool])

        result = await runner.execute_tool_call_async({
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "get_weather_async",
                "arguments": '{"location": "Tokyo"}',
            },
        })

        assert "Tokyo" in result.result

    @pytest.mark.asyncio
    async def test_run_async(self, mock_async_client, async_weather_tool):
        """Test async run method."""
        mock_async_client.chat.return_value = MagicMock(
            content="Response",
            tool_calls=None,
        )

        runner = ToolRunner([async_weather_tool])
        result = await runner.run_async(mock_async_client, "Hello")

        assert result.response.content == "Response"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
