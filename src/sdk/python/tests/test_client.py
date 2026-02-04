"""
2api.ai Python SDK - Client Tests
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import json
import os

from twoapi import TwoAPI
from twoapi.errors import (
    TwoAPIError,
    AuthenticationError,
    RateLimitError,
    InvalidRequestError,
    ProviderError,
    TimeoutError,
    ConnectionError,
)


class TestTwoAPIClient:
    """Tests for TwoAPI client."""

    @pytest.fixture
    def mock_response(self):
        """Create a mock HTTP response."""
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "choices": [{
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop",
            }],
            "model": "gpt-4o",
            "provider": "openai",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
        return response

    @pytest.fixture
    def client(self):
        """Create a TwoAPI client with mock."""
        with patch.dict(os.environ, {"TWOAPI_API_KEY": "test_key"}):
            return TwoAPI()

    def test_client_creation_with_api_key(self):
        """Test creating client with API key."""
        client = TwoAPI(api_key="test_key")
        assert client is not None

    def test_client_creation_from_env(self):
        """Test creating client from environment variable."""
        with patch.dict(os.environ, {"TWOAPI_API_KEY": "env_key"}):
            client = TwoAPI()
            assert client is not None

    def test_client_creation_without_key_raises(self):
        """Test that missing API key raises error."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove all possible key env vars
            for key in ["TWOAPI_API_KEY", "TWOAPI_KEY"]:
                os.environ.pop(key, None)
            with pytest.raises(AuthenticationError):
                TwoAPI()

    def test_custom_base_url(self):
        """Test client with custom base URL."""
        client = TwoAPI(api_key="test", base_url="https://custom.api.com/v1")
        assert client._base_url == "https://custom.api.com/v1"

    def test_custom_timeout(self):
        """Test client with custom timeout."""
        client = TwoAPI(api_key="test", timeout=120.0)
        assert client._timeout == 120.0


class TestSimpleChatAPI:
    """Tests for simple chat API."""

    @pytest.fixture
    def client(self):
        """Create a TwoAPI client."""
        return TwoAPI(api_key="test_key")

    def test_chat_with_string_message(self, client):
        """Test chat with simple string message."""
        with patch.object(client, "_request") as mock_request:
            mock_request.return_value = {
                "choices": [{
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }],
                "model": "gpt-4o",
                "provider": "openai",
            }

            response = client.chat("Hello")

            assert response.content == "Hello!"
            assert response.role == "assistant"
            mock_request.assert_called_once()

    def test_chat_with_message_dict(self, client):
        """Test chat with message dictionary."""
        with patch.object(client, "_request") as mock_request:
            mock_request.return_value = {
                "choices": [{
                    "message": {"role": "assistant", "content": "Response"},
                    "finish_reason": "stop",
                }],
            }

            response = client.chat({"role": "user", "content": "Hello"})

            assert response.content == "Response"

    def test_chat_with_message_list(self, client):
        """Test chat with list of messages."""
        with patch.object(client, "_request") as mock_request:
            mock_request.return_value = {
                "choices": [{
                    "message": {"role": "assistant", "content": "Response"},
                    "finish_reason": "stop",
                }],
            }

            response = client.chat([
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
                {"role": "user", "content": "How are you?"},
            ])

            assert response.content == "Response"

    def test_chat_with_options(self, client):
        """Test chat with optional parameters."""
        with patch.object(client, "_request") as mock_request:
            mock_request.return_value = {
                "choices": [{
                    "message": {"role": "assistant", "content": "Response"},
                    "finish_reason": "stop",
                }],
            }

            client.chat(
                "Hello",
                model="anthropic/claude-3-sonnet",
                temperature=0.7,
                max_tokens=100,
                system="You are helpful.",
            )

            call_kwargs = mock_request.call_args[1]
            assert "json" in call_kwargs
            payload = call_kwargs["json"]
            assert payload["model"] == "anthropic/claude-3-sonnet"
            assert payload["temperature"] == 0.7
            assert payload["max_tokens"] == 100

    def test_chat_returns_tool_calls(self, client):
        """Test chat returns tool calls when present."""
        with patch.object(client, "_request") as mock_request:
            mock_request.return_value = {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location":"Tokyo"}',
                            },
                        }],
                    },
                    "finish_reason": "tool_calls",
                }],
            }

            response = client.chat("What is the weather?", tools=[{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                },
            }])

            assert response.tool_calls is not None
            assert len(response.tool_calls) == 1
            assert response.tool_calls[0]["function"]["name"] == "get_weather"


class TestOpenAICompatibleAPI:
    """Tests for OpenAI-compatible API."""

    @pytest.fixture
    def client(self):
        """Create a TwoAPI client."""
        return TwoAPI(api_key="test_key")

    def test_chat_completions_create(self, client):
        """Test chat.completions.create method."""
        with patch.object(client, "_request_with_retry") as mock_request:
            mock_request.return_value = {
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }

            response = client.chat.completions.create(
                model="openai/gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
            )

            assert response.id == "chatcmpl-123"
            assert response.choices[0].message.content == "Hello!"
            assert response.usage.total_tokens == 15

    def test_chat_completions_with_parameters(self, client):
        """Test chat.completions.create with all parameters."""
        with patch.object(client, "_request_with_retry") as mock_request:
            mock_request.return_value = {
                "choices": [{
                    "message": {"content": "Response"},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }

            client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
                temperature=0.8,
                max_tokens=100,
                top_p=0.9,
                frequency_penalty=0.5,
                presence_penalty=0.5,
                stop=["END"],
                seed=42,
            )

            call_kwargs = mock_request.call_args[1]
            payload = call_kwargs["json"]
            assert payload["temperature"] == 0.8
            assert payload["max_tokens"] == 100
            assert payload["top_p"] == 0.9

    def test_chat_completions_with_tools(self, client):
        """Test chat.completions.create with tools."""
        with patch.object(client, "_request_with_retry") as mock_request:
            mock_request.return_value = {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "test", "arguments": "{}"},
                        }],
                    },
                    "finish_reason": "tool_calls",
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
                tools=[{
                    "type": "function",
                    "function": {"name": "test", "description": "Test tool"},
                }],
            )

            assert response.choices[0].message.tool_calls is not None


class TestEmbeddingsAPI:
    """Tests for embeddings API."""

    @pytest.fixture
    def client(self):
        """Create a TwoAPI client."""
        return TwoAPI(api_key="test_key")

    def test_embed_single_text(self, client):
        """Test embedding single text."""
        with patch.object(client, "_request") as mock_request:
            mock_request.return_value = {
                "data": [{"embedding": [0.1, 0.2, 0.3]}],
                "model": "text-embedding-3-small",
                "provider": "openai",
                "usage": {"prompt_tokens": 5, "total_tokens": 5},
            }

            response = client.embed("Hello world")

            assert len(response.embeddings) == 1
            assert response.embeddings[0] == [0.1, 0.2, 0.3]

    def test_embed_batch_texts(self, client):
        """Test embedding batch of texts."""
        with patch.object(client, "_request") as mock_request:
            mock_request.return_value = {
                "data": [
                    {"embedding": [0.1, 0.2]},
                    {"embedding": [0.3, 0.4]},
                ],
                "model": "text-embedding-3-small",
                "provider": "openai",
            }

            response = client.embed(["Hello", "World"])

            assert len(response.embeddings) == 2


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.fixture
    def client(self):
        """Create a TwoAPI client."""
        return TwoAPI(api_key="test_key", max_retries=0)

    def test_authentication_error(self, client):
        """Test authentication error handling."""
        with patch.object(client._client, "request") as mock_request:
            response = MagicMock()
            response.status_code = 401
            response.json.return_value = {"error": {"message": "Invalid API key"}}
            mock_request.return_value = response

            with pytest.raises(AuthenticationError):
                client.chat("Hello")

    def test_rate_limit_error(self, client):
        """Test rate limit error handling."""
        with patch.object(client._client, "request") as mock_request:
            response = MagicMock()
            response.status_code = 429
            response.headers = {"retry-after": "60"}
            response.json.return_value = {"error": {"message": "Rate limit exceeded"}}
            mock_request.return_value = response

            with pytest.raises(RateLimitError) as exc_info:
                client.chat("Hello")

            assert exc_info.value.retry_after == 60

    def test_invalid_request_error(self, client):
        """Test invalid request error handling."""
        with patch.object(client._client, "request") as mock_request:
            response = MagicMock()
            response.status_code = 400
            response.json.return_value = {"error": {"message": "Invalid parameters"}}
            mock_request.return_value = response

            with pytest.raises(InvalidRequestError):
                client.chat("Hello")

    def test_provider_error(self, client):
        """Test provider error handling."""
        with patch.object(client._client, "request") as mock_request:
            response = MagicMock()
            response.status_code = 500
            response.json.return_value = {
                "error": {"message": "Provider error", "provider": "openai"}
            }
            mock_request.return_value = response

            with pytest.raises(ProviderError):
                client.chat("Hello")


class TestRetryLogic:
    """Tests for retry logic."""

    def test_retry_on_500_error(self):
        """Test retry on server error."""
        client = TwoAPI(api_key="test_key", max_retries=2)

        with patch.object(client._client, "request") as mock_request:
            # First call fails, second succeeds
            error_response = MagicMock()
            error_response.status_code = 500
            error_response.json.return_value = {"error": {"message": "Server error"}}

            success_response = MagicMock()
            success_response.status_code = 200
            success_response.json.return_value = {
                "choices": [{"message": {"content": "Success"}, "finish_reason": "stop"}],
            }

            mock_request.side_effect = [error_response, success_response]

            response = client.chat("Hello")

            assert response.content == "Success"
            assert mock_request.call_count == 2

    def test_no_retry_on_400_error(self):
        """Test no retry on client error."""
        client = TwoAPI(api_key="test_key", max_retries=3)

        with patch.object(client._client, "request") as mock_request:
            response = MagicMock()
            response.status_code = 400
            response.json.return_value = {"error": {"message": "Bad request"}}
            mock_request.return_value = response

            with pytest.raises(InvalidRequestError):
                client.chat("Hello")

            # Should only be called once (no retries)
            assert mock_request.call_count == 1


class TestHealthAPI:
    """Tests for health API."""

    @pytest.fixture
    def client(self):
        """Create a TwoAPI client."""
        return TwoAPI(api_key="test_key")

    def test_health_check(self, client):
        """Test health check."""
        with patch.object(client, "_request") as mock_request:
            mock_request.return_value = {
                "status": "healthy",
                "version": "1.0.0",
                "providers": {
                    "openai": {"status": "healthy", "latency_ms": 50},
                    "anthropic": {"status": "healthy", "latency_ms": 60},
                },
            }

            health = client.health()

            assert health.status == "healthy"
            assert health.version == "1.0.0"
            assert "openai" in health.providers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
