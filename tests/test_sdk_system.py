"""
2api.ai - SDK System Tests

Comprehensive tests for:
- Python SDK client (sync and async)
- Data models
- Error handling
- Retry logic
- Tool calling
"""

import pytest
import json
import time
import asyncio
from typing import Any, Dict, List
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from dataclasses import dataclass


# ============================================================
# Test SDK Models
# ============================================================

class TestMessage:
    """Tests for Message model."""

    def test_create_user_message(self):
        """Test creating a user message."""
        from src.sdk.python.twoapi.models import Message

        msg = Message.user("Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_create_system_message(self):
        """Test creating a system message."""
        from src.sdk.python.twoapi.models import Message

        msg = Message.system("You are helpful")
        assert msg.role == "system"
        assert msg.content == "You are helpful"

    def test_create_assistant_message(self):
        """Test creating an assistant message."""
        from src.sdk.python.twoapi.models import Message

        msg = Message.assistant("Hello!")
        assert msg.role == "assistant"
        assert msg.content == "Hello!"

    def test_create_tool_message(self):
        """Test creating a tool result message."""
        from src.sdk.python.twoapi.models import Message

        msg = Message.tool("call_123", '{"result": 42}')
        assert msg.role == "tool"
        assert msg.tool_call_id == "call_123"
        assert msg.content == '{"result": 42}'

    def test_message_to_dict(self):
        """Test message serialization."""
        from src.sdk.python.twoapi.models import Message

        msg = Message.user("Hello")
        d = msg.to_dict()
        assert d == {"role": "user", "content": "Hello"}

    def test_message_with_name(self):
        """Test message with name field."""
        from src.sdk.python.twoapi.models import Message

        msg = Message(role="user", content="Hello", name="john")
        d = msg.to_dict()
        assert d["name"] == "john"


class TestToolCall:
    """Tests for ToolCall model."""

    def test_create_tool_call(self):
        """Test creating a tool call."""
        from src.sdk.python.twoapi.models import ToolCall, FunctionCall

        tc = ToolCall(
            id="call_123",
            function=FunctionCall(
                name="get_weather",
                arguments='{"location": "NYC"}'
            )
        )
        assert tc.id == "call_123"
        assert tc.function.name == "get_weather"

    def test_tool_call_from_dict(self):
        """Test creating from dictionary."""
        from src.sdk.python.twoapi.models import ToolCall

        data = {
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"location": "NYC"}'
            }
        }
        tc = ToolCall.from_dict(data)
        assert tc.id == "call_123"
        assert tc.function.name == "get_weather"

    def test_tool_call_to_dict(self):
        """Test tool call serialization."""
        from src.sdk.python.twoapi.models import ToolCall, FunctionCall

        tc = ToolCall(
            id="call_123",
            function=FunctionCall(name="test", arguments="{}")
        )
        d = tc.to_dict()
        assert d["id"] == "call_123"
        assert d["function"]["name"] == "test"


class TestTool:
    """Tests for Tool model."""

    def test_create_tool(self):
        """Test creating a tool."""
        from src.sdk.python.twoapi.models import Tool

        tool = Tool.create(
            name="get_weather",
            description="Get weather for a location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        )
        assert tool.function.name == "get_weather"
        assert tool.function.description == "Get weather for a location"

    def test_tool_to_dict(self):
        """Test tool serialization."""
        from src.sdk.python.twoapi.models import Tool

        tool = Tool.create("test_func", "A test function")
        d = tool.to_dict()
        assert d["type"] == "function"
        assert d["function"]["name"] == "test_func"


class TestUsage:
    """Tests for Usage model."""

    def test_create_usage(self):
        """Test creating usage info."""
        from src.sdk.python.twoapi.models import Usage

        usage = Usage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150
        )
        assert usage.prompt_tokens == 100
        assert usage.total_tokens == 150

    def test_usage_from_dict(self):
        """Test creating from dictionary."""
        from src.sdk.python.twoapi.models import Usage

        data = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150
        }
        usage = Usage.from_dict(data)
        assert usage.prompt_tokens == 100


class TestChatResponse:
    """Tests for ChatResponse model."""

    def test_create_from_dict(self):
        """Test creating response from API data."""
        from src.sdk.python.twoapi.models import ChatResponse

        data = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "provider": "openai",
            "choices": [
                {
                    "message": {"content": "Hello!"},
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }
        response = ChatResponse.from_dict(data)
        assert response.id == "chatcmpl-123"
        assert response.content == "Hello!"
        assert response.usage.total_tokens == 15

    def test_response_with_tool_calls(self):
        """Test response with tool calls."""
        from src.sdk.python.twoapi.models import ChatResponse

        data = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": "{}"
                                }
                            }
                        ]
                    },
                    "finish_reason": "tool_calls"
                }
            ],
            "usage": {}
        }
        response = ChatResponse.from_dict(data)
        assert response.has_tool_calls
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].function.name == "get_weather"

    def test_response_with_routing_info(self):
        """Test response with 2api routing info."""
        from src.sdk.python.twoapi.models import ChatResponse

        data = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [
                {"message": {"content": "Hello"}, "finish_reason": "stop"}
            ],
            "usage": {},
            "_2api": {
                "latency_ms": 150,
                "cost_usd": 0.001,
                "provider": "openai",
                "routing_decision": {
                    "strategy_used": "cost",
                    "fallback_used": False
                }
            }
        }
        response = ChatResponse.from_dict(data)
        assert response.routing is not None
        assert response.routing.latency_ms == 150
        assert response.routing.strategy_used == "cost"


class TestEmbeddingResponse:
    """Tests for EmbeddingResponse model."""

    def test_create_from_dict(self):
        """Test creating from API data."""
        from src.sdk.python.twoapi.models import EmbeddingResponse

        data = {
            "data": [
                {"embedding": [0.1, 0.2, 0.3], "index": 0}
            ],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 5, "total_tokens": 5}
        }
        response = EmbeddingResponse.from_dict(data)
        assert len(response.embeddings) == 1
        assert len(response.embeddings[0]) == 3


class TestImageResponse:
    """Tests for ImageResponse model."""

    def test_create_from_dict(self):
        """Test creating from API data."""
        from src.sdk.python.twoapi.models import ImageResponse

        data = {
            "created": 1234567890,
            "data": [
                {"url": "https://example.com/image.png", "revised_prompt": "A cat"}
            ]
        }
        response = ImageResponse.from_dict(data)
        assert len(response.urls) == 1
        assert response.urls[0] == "https://example.com/image.png"
        assert response.revised_prompts[0] == "A cat"


class TestModelInfo:
    """Tests for ModelInfo model."""

    def test_create_from_dict(self):
        """Test creating from API data."""
        from src.sdk.python.twoapi.models import ModelInfo

        data = {
            "id": "openai/gpt-4o",
            "provider": "openai",
            "name": "GPT-4o",
            "capabilities": ["chat", "vision"],
            "context_window": 128000,
            "max_output_tokens": 4096,
            "pricing": {
                "input_per_1m_tokens": 2.5,
                "output_per_1m_tokens": 10.0
            }
        }
        info = ModelInfo.from_dict(data)
        assert info.id == "openai/gpt-4o"
        assert "chat" in info.capabilities
        assert info.pricing.input_per_1m_tokens == 2.5


# ============================================================
# Test SDK Errors
# ============================================================

class TestTwoAPIError:
    """Tests for TwoAPIError."""

    def test_basic_error(self):
        """Test basic error creation."""
        from src.sdk.python.twoapi.errors import TwoAPIError

        error = TwoAPIError("Something went wrong", code="test_error", status_code=500)
        assert error.message == "Something went wrong"
        assert error.code == "test_error"
        assert error.status_code == 500

    def test_error_from_response(self):
        """Test creating error from API response."""
        from src.sdk.python.twoapi.errors import TwoAPIError

        response_data = {
            "error": {
                "code": "invalid_request",
                "message": "Invalid parameter",
                "request_id": "req_123",
                "retryable": False
            }
        }
        error = TwoAPIError.from_response(response_data, 400)
        assert error.code == "invalid_request"
        assert error.request_id == "req_123"

    def test_authentication_error(self):
        """Test authentication error."""
        from src.sdk.python.twoapi.errors import AuthenticationError

        error = AuthenticationError("Invalid API key")
        assert error.status_code == 401
        assert error.retryable is False

    def test_rate_limit_error(self):
        """Test rate limit error."""
        from src.sdk.python.twoapi.errors import RateLimitError

        error = RateLimitError("Rate limit exceeded", retry_after=30)
        assert error.status_code == 429
        assert error.retry_after == 30
        assert error.retryable is True

    def test_invalid_request_error(self):
        """Test invalid request error."""
        from src.sdk.python.twoapi.errors import InvalidRequestError

        error = InvalidRequestError("Missing field", param="model")
        assert error.status_code == 400
        assert error.param == "model"


# ============================================================
# Test Retry Logic
# ============================================================

class TestRetryLogic:
    """Tests for retry logic."""

    def test_calculate_backoff(self):
        """Test backoff calculation."""
        from src.sdk.python.twoapi.retry import calculate_backoff

        delay0 = calculate_backoff(0, initial_delay=1.0, jitter=False)
        delay1 = calculate_backoff(1, initial_delay=1.0, jitter=False)
        delay2 = calculate_backoff(2, initial_delay=1.0, jitter=False)

        assert delay0 == 1.0
        assert delay1 == 2.0
        assert delay2 == 4.0

    def test_calculate_backoff_max_delay(self):
        """Test backoff respects max delay."""
        from src.sdk.python.twoapi.retry import calculate_backoff

        delay = calculate_backoff(10, initial_delay=1.0, max_delay=30.0, jitter=False)
        assert delay == 30.0

    def test_calculate_backoff_jitter(self):
        """Test backoff with jitter varies."""
        from src.sdk.python.twoapi.retry import calculate_backoff

        delays = [calculate_backoff(0, initial_delay=1.0, jitter=True) for _ in range(10)]
        # With jitter, delays should vary
        unique_delays = set(delays)
        assert len(unique_delays) > 1  # Not all the same

    def test_should_retry_retryable_error(self):
        """Test should_retry for retryable errors."""
        from src.sdk.python.twoapi.retry import should_retry
        from src.sdk.python.twoapi.errors import TwoAPIError

        error = TwoAPIError("Error", status_code=500, retryable=True)
        assert should_retry(error) is True

    def test_should_retry_rate_limit(self):
        """Test should_retry for rate limit errors."""
        from src.sdk.python.twoapi.retry import should_retry
        from src.sdk.python.twoapi.errors import RateLimitError

        error = RateLimitError("Rate limit")
        assert should_retry(error) is True

    def test_should_retry_auth_error(self):
        """Test should_retry for auth errors (not retryable)."""
        from src.sdk.python.twoapi.retry import should_retry
        from src.sdk.python.twoapi.errors import AuthenticationError

        error = AuthenticationError("Invalid key")
        assert should_retry(error) is False

    def test_retry_handler_wrap(self):
        """Test RetryHandler.wrap decorator."""
        from src.sdk.python.twoapi.retry import RetryHandler
        from src.sdk.python.twoapi.errors import TwoAPIError

        handler = RetryHandler(max_retries=3, initial_delay=0.01)
        call_count = 0

        @handler.wrap
        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TwoAPIError("Error", status_code=500, retryable=True)
            return "success"

        result = failing_function()
        assert result == "success"
        assert call_count == 3  # Failed twice, succeeded on third


class TestRetryHandlerAsync:
    """Tests for async retry logic."""

    @pytest.mark.asyncio
    async def test_retry_handler_wrap_async(self):
        """Test RetryHandler.wrap_async decorator."""
        from src.sdk.python.twoapi.retry import RetryHandler
        from src.sdk.python.twoapi.errors import TwoAPIError

        handler = RetryHandler(max_retries=3, initial_delay=0.01)
        call_count = 0

        @handler.wrap_async
        async def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TwoAPIError("Error", status_code=500, retryable=True)
            return "success"

        result = await failing_function()
        assert result == "success"
        assert call_count == 3


# ============================================================
# Test Sync Client
# ============================================================

class TestTwoAPIClient:
    """Tests for TwoAPI sync client."""

    def test_client_requires_api_key(self):
        """Test that client requires API key."""
        from src.sdk.python.twoapi.client import TwoAPI
        from src.sdk.python.twoapi.errors import AuthenticationError

        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(AuthenticationError):
                TwoAPI()

    def test_client_with_api_key(self):
        """Test client creation with API key."""
        from src.sdk.python.twoapi.client import TwoAPI

        client = TwoAPI(api_key="2api_test123")
        assert client.api_key == "2api_test123"
        client.close()

    def test_client_default_base_url(self):
        """Test client uses default base URL."""
        from src.sdk.python.twoapi.client import TwoAPI

        client = TwoAPI(api_key="2api_test123")
        assert client.base_url == "https://api.2api.ai/v1"
        client.close()

    def test_client_custom_base_url(self):
        """Test client with custom base URL."""
        from src.sdk.python.twoapi.client import TwoAPI

        client = TwoAPI(api_key="2api_test", base_url="http://localhost:8000/v1")
        assert client.base_url == "http://localhost:8000/v1"
        client.close()

    def test_normalize_messages_string(self):
        """Test normalizing string message."""
        from src.sdk.python.twoapi.client import TwoAPI

        client = TwoAPI(api_key="2api_test")
        messages = client._normalize_messages("Hello")
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"
        client.close()

    def test_normalize_messages_with_system(self):
        """Test normalizing with system message."""
        from src.sdk.python.twoapi.client import TwoAPI

        client = TwoAPI(api_key="2api_test")
        messages = client._normalize_messages("Hello", system="Be helpful")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        client.close()

    def test_normalize_messages_list(self):
        """Test normalizing list of messages."""
        from src.sdk.python.twoapi.client import TwoAPI
        from src.sdk.python.twoapi.models import Message

        client = TwoAPI(api_key="2api_test")
        messages = client._normalize_messages([
            Message.user("Hi"),
            Message.assistant("Hello!")
        ])
        assert len(messages) == 2
        client.close()

    def test_context_manager(self):
        """Test client as context manager."""
        from src.sdk.python.twoapi.client import TwoAPI

        with TwoAPI(api_key="2api_test") as client:
            assert client.api_key == "2api_test"


# ============================================================
# Test Async Client
# ============================================================

class TestAsyncTwoAPIClient:
    """Tests for AsyncTwoAPI client."""

    def test_async_client_requires_api_key(self):
        """Test that async client requires API key."""
        from src.sdk.python.twoapi.async_client import AsyncTwoAPI
        from src.sdk.python.twoapi.errors import AuthenticationError

        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(AuthenticationError):
                AsyncTwoAPI()

    def test_async_client_with_api_key(self):
        """Test async client creation with API key."""
        from src.sdk.python.twoapi.async_client import AsyncTwoAPI

        client = AsyncTwoAPI(api_key="2api_test123")
        assert client.api_key == "2api_test123"

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Test async client as context manager."""
        from src.sdk.python.twoapi.async_client import AsyncTwoAPI

        async with AsyncTwoAPI(api_key="2api_test") as client:
            assert client.api_key == "2api_test"


# ============================================================
# Test Routing Config
# ============================================================

class TestRoutingConfig:
    """Tests for RoutingConfig model."""

    def test_create_routing_config(self):
        """Test creating routing config."""
        from src.sdk.python.twoapi.models import RoutingConfig

        config = RoutingConfig(
            strategy="cost",
            fallback=["anthropic/claude-3-sonnet"],
            max_latency_ms=5000
        )
        assert config.strategy == "cost"
        assert len(config.fallback) == 1

    def test_routing_config_to_dict(self):
        """Test routing config serialization."""
        from src.sdk.python.twoapi.models import RoutingConfig

        config = RoutingConfig(strategy="latency")
        d = config.to_dict()
        assert d == {"strategy": "latency"}

    def test_routing_config_empty(self):
        """Test empty routing config."""
        from src.sdk.python.twoapi.models import RoutingConfig

        config = RoutingConfig()
        d = config.to_dict()
        assert d == {}


class TestRetryConfig:
    """Tests for RetryConfig model."""

    def test_create_retry_config(self):
        """Test creating retry config."""
        from src.sdk.python.twoapi.models import RetryConfig

        config = RetryConfig(
            max_retries=5,
            initial_delay=0.5,
            max_delay=60.0
        )
        assert config.max_retries == 5
        assert config.initial_delay == 0.5

    def test_retry_config_defaults(self):
        """Test retry config defaults."""
        from src.sdk.python.twoapi.models import RetryConfig

        config = RetryConfig()
        assert config.max_retries == 3
        assert config.initial_delay == 1.0
        assert 429 in config.retry_on_status


# ============================================================
# Test Convenience Functions
# ============================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_set_api_key(self):
        """Test setting default API key."""
        from src.sdk.python.twoapi import utils

        utils.set_api_key("2api_test_key")
        assert utils._default_client is not None
        assert utils._default_client.api_key == "2api_test_key"
        utils._default_client.close()
        utils._default_client = None


# ============================================================
# Test HealthStatus
# ============================================================

class TestHealthStatus:
    """Tests for HealthStatus model."""

    def test_create_from_dict(self):
        """Test creating health status from API data."""
        from src.sdk.python.twoapi.models import HealthStatus

        data = {
            "status": "healthy",
            "version": "1.0.0",
            "providers": {
                "openai": {"status": "healthy", "latency_ms": 100},
                "anthropic": {"status": "unhealthy", "error": "timeout"}
            }
        }
        status = HealthStatus.from_dict(data)
        assert status.status == "healthy"
        assert status.version == "1.0.0"
        assert status.providers["openai"].status == "healthy"
        assert status.providers["anthropic"].error == "timeout"


# ============================================================
# Test Integration
# ============================================================

class TestSDKIntegration:
    """Integration tests for SDK."""

    def test_full_message_flow(self):
        """Test complete message flow with tools."""
        from src.sdk.python.twoapi.models import Message, Tool, ToolCall, FunctionCall

        # Create messages
        messages = [
            Message.system("You are helpful"),
            Message.user("What's the weather in NYC?")
        ]

        # Define tools
        tools = [
            Tool.create(
                "get_weather",
                "Get weather for a location",
                {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"]
                }
            )
        ]

        # Simulate tool call response
        tool_calls = [
            ToolCall(
                id="call_123",
                function=FunctionCall(
                    name="get_weather",
                    arguments='{"location": "NYC"}'
                )
            )
        ]

        # Add tool result
        messages.append(Message.assistant(tool_calls=tool_calls))
        messages.append(Message.tool("call_123", '{"temp": 72, "condition": "sunny"}'))

        # Verify message structure
        assert len(messages) == 4
        assert messages[0].role == "system"
        assert messages[3].role == "tool"

    def test_error_hierarchy(self):
        """Test error class hierarchy."""
        from src.sdk.python.twoapi.errors import (
            TwoAPIError,
            AuthenticationError,
            RateLimitError,
            InvalidRequestError,
            ProviderError,
            TimeoutError,
            ConnectionError,
        )

        # All errors should inherit from TwoAPIError
        assert issubclass(AuthenticationError, TwoAPIError)
        assert issubclass(RateLimitError, TwoAPIError)
        assert issubclass(InvalidRequestError, TwoAPIError)
        assert issubclass(ProviderError, TwoAPIError)
        assert issubclass(TimeoutError, TwoAPIError)
        assert issubclass(ConnectionError, TwoAPIError)

    def test_model_serialization_roundtrip(self):
        """Test model serialization and deserialization."""
        from src.sdk.python.twoapi.models import (
            Message,
            Tool,
            ChatResponse,
        )

        # Create a complex message
        msg = Message.user("Hello")
        d = msg.to_dict()

        # Verify structure
        assert d["role"] == "user"
        assert d["content"] == "Hello"

        # Create a tool
        tool = Tool.create("test", "Test function", {"type": "object"})
        tool_dict = tool.to_dict()
        assert tool_dict["type"] == "function"
        assert tool_dict["function"]["name"] == "test"


class TestSDKModuleExports:
    """Test that SDK module exports all expected symbols."""

    def test_main_exports(self):
        """Test main module exports."""
        from src.sdk.python.twoapi import (
            TwoAPI,
            AsyncTwoAPI,
            Message,
            Usage,
            ChatResponse,
            EmbeddingResponse,
            ImageResponse,
            TwoAPIError,
            AuthenticationError,
            RateLimitError,
        )

        assert TwoAPI is not None
        assert AsyncTwoAPI is not None
        assert Message is not None

    def test_convenience_functions_exported(self):
        """Test convenience functions are exported."""
        from src.sdk.python.twoapi import (
            chat,
            chat_stream,
            embed,
            generate_image,
        )

        assert chat is not None
        assert chat_stream is not None
        assert embed is not None
        assert generate_image is not None
