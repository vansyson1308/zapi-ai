"""
2api.ai SDK - Async Client

Async client for non-blocking API interactions.
"""

from __future__ import annotations

import json
import os
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import httpx

from .errors import (
    TwoAPIError,
    AuthenticationError,
    RateLimitError,
    TimeoutError,
    ConnectionError,
)
from .models import (
    Message,
    Tool,
    RoutingConfig,
    RetryConfig,
    ChatResponse,
    EmbeddingResponse,
    ImageResponse,
    ModelInfo,
    HealthStatus,
    Usage,
)
from .retry import RetryHandler


__version__ = "1.0.0"


class AsyncTwoAPI:
    """
    2api.ai Async Python Client.

    A unified async interface to access multiple AI providers through 2api.ai.

    Args:
        api_key: Your 2api.ai API key. If not provided, reads from TWOAPI_API_KEY env var.
        base_url: Base URL for the API. Defaults to https://api.2api.ai/v1
        timeout: Request timeout in seconds. Defaults to 60.
        max_retries: Maximum number of retry attempts. Defaults to 3.
        retry_config: Advanced retry configuration.

    Example:
        >>> client = AsyncTwoAPI(api_key="2api_xxx")
        >>> response = await client.chat("Hello!")
        >>> print(response.content)
        Hello! How can I help you today?
    """

    DEFAULT_BASE_URL = "https://api.2api.ai/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
        retry_config: Optional[RetryConfig] = None
    ):
        self.api_key = api_key or os.getenv("TWOAPI_API_KEY")
        if not self.api_key:
            raise AuthenticationError(
                "API key required. Set TWOAPI_API_KEY environment variable or pass api_key parameter."
            )

        self.base_url = (
            base_url or os.getenv("TWOAPI_BASE_URL") or self.DEFAULT_BASE_URL
        ).rstrip("/")
        self.timeout = timeout

        # Setup retry handler
        if retry_config:
            self._retry_handler = RetryHandler(
                max_retries=retry_config.max_retries,
                initial_delay=retry_config.initial_delay,
                max_delay=retry_config.max_delay,
                exponential_base=retry_config.exponential_base,
                retry_on_status=retry_config.retry_on_status
            )
        else:
            self._retry_handler = RetryHandler(max_retries=max_retries)

        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": f"twoapi-python-async/{__version__}"
                },
                timeout=self.timeout
            )
        return self._client

    async def chat(
        self,
        message: Union[str, List[Message], List[Dict[str, str]]],
        model: str = "auto",
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Tool]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        routing: Optional[RoutingConfig] = None,
        metadata: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> ChatResponse:
        """
        Create a chat completion asynchronously.

        Args:
            message: The message(s) to send.
            model: Model to use. Use "auto" for automatic selection.
            system: Optional system message.
            temperature: Sampling temperature (0-2).
            max_tokens: Maximum tokens to generate.
            tools: List of tools available to the model.
            tool_choice: How the model should use tools.
            routing: Routing configuration.
            metadata: Custom metadata for tracking.

        Returns:
            ChatResponse with the completion.

        Example:
            >>> response = await client.chat("What is 2+2?")
            >>> print(response.content)
        """
        messages = self._normalize_messages(message, system)

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages
        }

        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if tools:
            payload["tools"] = [t.to_dict() for t in tools]
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if metadata:
            payload["metadata"] = metadata

        if routing:
            payload["routing"] = routing.to_dict()

        payload.update(kwargs)

        response = await self._request_with_retry("POST", "/chat/completions", json=payload)
        return ChatResponse.from_dict(response)

    async def chat_stream(
        self,
        message: Union[str, List[Message], List[Dict[str, str]]],
        model: str = "auto",
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Create a streaming chat completion asynchronously.

        Yields content chunks as they arrive.

        Args:
            message: The message(s) to send.
            model: Model to use.
            system: Optional system message.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Yields:
            Content chunks as strings.

        Example:
            >>> async for chunk in client.chat_stream("Tell me a story"):
            ...     print(chunk, end="", flush=True)
        """
        messages = self._normalize_messages(message, system)

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True
        }

        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        payload.update(kwargs)

        client = await self._get_client()

        async with client.stream(
            "POST",
            "/chat/completions",
            json=payload
        ) as response:
            await self._check_response(response)

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        choice = chunk.get("choices", [{}])[0]
                        delta = choice.get("delta", {})
                        content = delta.get("content", "")

                        if content:
                            yield content

                    except json.JSONDecodeError:
                        continue

    async def embed(
        self,
        text: Union[str, List[str]],
        model: str = "openai/text-embedding-3-small",
        dimensions: Optional[int] = None
    ) -> EmbeddingResponse:
        """
        Create embeddings for text asynchronously.

        Args:
            text: Text or list of texts to embed.
            model: Embedding model to use.
            dimensions: Output dimensions (for models that support it).

        Returns:
            EmbeddingResponse with the embeddings.
        """
        payload: Dict[str, Any] = {
            "model": model,
            "input": text
        }

        if dimensions is not None:
            payload["dimensions"] = dimensions

        response = await self._request_with_retry("POST", "/embeddings", json=payload)
        return EmbeddingResponse.from_dict(response)

    async def generate_image(
        self,
        prompt: str,
        model: str = "openai/dall-e-3",
        size: str = "1024x1024",
        n: int = 1,
        quality: str = "standard",
        style: str = "vivid",
        response_format: str = "url"
    ) -> ImageResponse:
        """
        Generate images from a prompt asynchronously.

        Args:
            prompt: Text description of the image to generate.
            model: Image generation model.
            size: Image size.
            n: Number of images to generate.
            quality: "standard" or "hd".
            style: "vivid" or "natural".
            response_format: "url" or "b64_json".

        Returns:
            ImageResponse with URLs or base64 data.
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "n": n,
            "quality": quality,
            "style": style,
            "response_format": response_format
        }

        response = await self._request_with_retry("POST", "/images/generations", json=payload)
        return ImageResponse.from_dict(response)

    async def list_models(
        self,
        provider: Optional[str] = None,
        capability: Optional[str] = None
    ) -> List[ModelInfo]:
        """
        List available models asynchronously.

        Args:
            provider: Filter by provider.
            capability: Filter by capability.

        Returns:
            List of ModelInfo objects.
        """
        params = {}
        if provider:
            params["provider"] = provider
        if capability:
            params["capability"] = capability

        response = await self._request_with_retry("GET", "/models", params=params)
        return [ModelInfo.from_dict(m) for m in response.get("data", [])]

    async def get_model(self, model_id: str) -> ModelInfo:
        """
        Get information about a specific model asynchronously.

        Args:
            model_id: The model ID.

        Returns:
            ModelInfo for the model.
        """
        response = await self._request_with_retry("GET", f"/models/{model_id}")
        return ModelInfo.from_dict(response)

    async def health(self) -> HealthStatus:
        """
        Check API health status asynchronously.

        Returns:
            HealthStatus with provider availability.
        """
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url.replace('/v1', '')}/health")
            return HealthStatus.from_dict(response.json())

    async def get_usage(self) -> Dict[str, Any]:
        """
        Get usage statistics asynchronously.

        Returns:
            Dictionary with usage data.
        """
        return await self._request_with_retry("GET", "/usage")

    # ============================================================
    # Private methods
    # ============================================================

    def _normalize_messages(
        self,
        message: Union[str, List[Message], List[Dict[str, str]]],
        system: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Normalize various message formats to list of dicts."""
        messages = []

        if system:
            messages.append({"role": "system", "content": system})

        if isinstance(message, str):
            messages.append({"role": "user", "content": message})
        elif isinstance(message, list):
            for m in message:
                if isinstance(m, Message):
                    messages.append(m.to_dict())
                elif isinstance(m, dict):
                    messages.append(m)

        return messages

    async def _request(
        self,
        method: str,
        path: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Make an async HTTP request and handle errors."""
        try:
            client = await self._get_client()
            response = await client.request(method, path, **kwargs)
            return self._handle_response(response)

        except httpx.TimeoutException:
            raise TimeoutError("Request timed out")
        except httpx.ConnectError:
            raise ConnectionError("Failed to connect to API")
        except httpx.RequestError as e:
            raise ConnectionError(f"Request failed: {e}")

    async def _request_with_retry(
        self,
        method: str,
        path: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Make an async HTTP request with retry logic."""
        @self._retry_handler.wrap_async
        async def do_request():
            return await self._request(method, path, **kwargs)

        return await do_request()

    def _handle_response(self, response: httpx.Response) -> Dict[str, Any]:
        """Handle HTTP response and raise appropriate errors."""
        if response.status_code == 401:
            raise AuthenticationError("Invalid API key")

        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 60))
            raise RateLimitError("Rate limit exceeded", retry_after=retry_after)

        if response.status_code >= 400:
            try:
                error_data = response.json()
                raise TwoAPIError.from_response(error_data, response.status_code)
            except (json.JSONDecodeError, KeyError):
                raise TwoAPIError(
                    message=response.text or f"HTTP {response.status_code}",
                    status_code=response.status_code
                )

        return response.json()

    async def _check_response(self, response: httpx.Response) -> None:
        """Check streaming response status."""
        if response.status_code == 401:
            raise AuthenticationError("Invalid API key")

        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 60))
            raise RateLimitError("Rate limit exceeded", retry_after=retry_after)

        if response.status_code >= 400:
            raise TwoAPIError(
                message=f"HTTP {response.status_code}",
                status_code=response.status_code
            )

    async def close(self):
        """Close the async HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
