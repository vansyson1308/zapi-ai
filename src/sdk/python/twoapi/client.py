"""
2api.ai SDK - Synchronous Client

Main client for synchronous API interactions.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Generator, List, Optional, Union

import httpx

from .errors import (
    TwoAPIError,
    AuthenticationError,
    RateLimitError,
    InvalidRequestError,
    TimeoutError,
    ConnectionError,
)
from .models import (
    Message,
    Tool,
    ToolCall,
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


class TwoAPI:
    """
    2api.ai Python Client.

    A unified interface to access multiple AI providers through 2api.ai.

    Supports two API styles:
    1. Simple API: client.chat("Hello!")
    2. OpenAI-compatible: client.chat.completions.create(...)

    Args:
        api_key: Your 2api.ai API key. If not provided, reads from TWOAPI_API_KEY env var.
        base_url: Base URL for the API. Defaults to https://api.2api.ai/v1
        timeout: Request timeout in seconds. Defaults to 60.
        max_retries: Maximum number of retry attempts. Defaults to 3.
        retry_config: Advanced retry configuration.
        on_retry: Callback called before each retry.

    Example:
        >>> # Simple usage
        >>> client = TwoAPI(api_key="2api_xxx")
        >>> response = client.chat("Hello!")
        >>> print(response.content)

        >>> # OpenAI-compatible usage
        >>> response = client.chat.completions.create(
        ...     model="openai/gpt-4o",
        ...     messages=[{"role": "user", "content": "Hello!"}]
        ... )
        >>> print(response.choices[0].message.content)
    """

    DEFAULT_BASE_URL = "https://api.2api.ai/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_config: Optional[RetryConfig] = None,
        on_retry: Optional[callable] = None,
    ):
        self.api_key = api_key or os.getenv("TWOAPI_API_KEY") or os.getenv("TWOAPI_KEY")
        if not self.api_key:
            raise AuthenticationError(
                "API key required. Set TWOAPI_API_KEY environment variable or pass api_key parameter."
            )

        self._base_url = (
            base_url or os.getenv("TWOAPI_BASE_URL") or self.DEFAULT_BASE_URL
        ).rstrip("/")
        self._timeout = timeout

        # Setup retry handler
        if retry_config:
            self._retry_handler = RetryHandler(
                max_retries=retry_config.max_retries,
                initial_delay=retry_config.initial_delay,
                max_delay=retry_config.max_delay,
                exponential_base=retry_config.exponential_base,
                retry_on_status=retry_config.retry_on_status,
                on_retry=on_retry,
            )
        else:
            self._retry_handler = RetryHandler(max_retries=max_retries, on_retry=on_retry)

        self._client = httpx.Client(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": f"twoapi-python/{__version__}"
            },
            timeout=timeout
        )

        # OpenAI-compatible interfaces
        # Import here to avoid circular imports
        from .openai_compat import Chat, Embeddings
        self._chat_compat = Chat(self)
        self._embeddings_compat = Embeddings(self)

    @property
    def base_url(self) -> str:
        """Base URL for API requests."""
        return self._base_url

    @property
    def chat(self):
        """
        Chat interface - supports both simple and OpenAI-compatible usage.

        Simple usage (when called):
            response = client.chat("Hello!")

        OpenAI-compatible usage (when accessing .completions):
            response = client.chat.completions.create(...)
        """
        return _ChatProxy(self, self._chat_compat)

    @property
    def embeddings(self):
        """OpenAI-compatible embeddings interface."""
        return self._embeddings_compat

    # ============================================================
    # Simple Chat API
    # ============================================================

    def _chat_impl(
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
        Internal implementation of chat completion.

        Args:
            message: The message(s) to send. Can be:
                - A string (simple user message)
                - A list of Message objects
                - A list of dicts with 'role' and 'content'
            model: Model to use. Use "auto" for automatic selection.
            system: Optional system message.
            temperature: Sampling temperature (0-2).
            max_tokens: Maximum tokens to generate.
            tools: List of tools available to the model.
            tool_choice: How the model should use tools.
            routing: Routing configuration (strategy, fallback, etc.).
            metadata: Custom metadata for tracking.

        Returns:
            ChatResponse with the completion.
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
            payload["tools"] = [t.to_dict() if hasattr(t, 'to_dict') else t for t in tools]
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if metadata:
            payload["metadata"] = metadata

        if routing:
            payload["routing"] = routing.to_dict() if hasattr(routing, 'to_dict') else routing

        payload.update(kwargs)

        response = self._request_with_retry("POST", "/chat/completions", json=payload)
        return ChatResponse.from_dict(response)

    def chat_stream(
        self,
        message: Union[str, List[Message], List[Dict[str, str]]],
        model: str = "auto",
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Generator[str, None, ChatResponse]:
        """
        Create a streaming chat completion.

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
            >>> for chunk in client.chat_stream("Tell me a story"):
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

        accumulated_content = ""
        final_chunk = {}

        with self._client.stream(
            "POST",
            "/chat/completions",
            json=payload
        ) as response:
            self._check_response(response)

            for line in response.iter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        final_chunk = chunk

                        choice = chunk.get("choices", [{}])[0]
                        delta = choice.get("delta", {})
                        content = delta.get("content", "")

                        if content:
                            accumulated_content += content
                            yield content

                    except json.JSONDecodeError:
                        continue

        # Build final response from last chunk
        if final_chunk:
            if "choices" in final_chunk and final_chunk["choices"]:
                final_chunk["choices"][0]["message"] = {
                    "role": "assistant",
                    "content": accumulated_content
                }
            return ChatResponse.from_dict(final_chunk)

        return ChatResponse(
            content=accumulated_content,
            model=model,
            provider="",
            usage=Usage()
        )

    # ============================================================
    # Embeddings API
    # ============================================================

    def embed(
        self,
        text: Union[str, List[str]],
        model: str = "openai/text-embedding-3-small",
        dimensions: Optional[int] = None
    ) -> EmbeddingResponse:
        """
        Create embeddings for text.

        Args:
            text: Text or list of texts to embed.
            model: Embedding model to use.
            dimensions: Output dimensions (for models that support it).

        Returns:
            EmbeddingResponse with the embeddings.

        Example:
            >>> response = client.embed("Hello world")
            >>> print(len(response.embeddings[0]))
        """
        payload: Dict[str, Any] = {
            "model": model,
            "input": text
        }

        if dimensions is not None:
            payload["dimensions"] = dimensions

        response = self._request_with_retry("POST", "/embeddings", json=payload)
        return EmbeddingResponse.from_dict(response)

    # ============================================================
    # Images API
    # ============================================================

    def generate_image(
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
        Generate images from a prompt.

        Args:
            prompt: Text description of the image to generate.
            model: Image generation model.
            size: Image size (e.g., "1024x1024", "1024x1792").
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

        response = self._request_with_retry("POST", "/images/generations", json=payload)
        return ImageResponse.from_dict(response)

    # ============================================================
    # Models API
    # ============================================================

    def list_models(
        self,
        provider: Optional[str] = None,
        capability: Optional[str] = None
    ) -> List[ModelInfo]:
        """
        List available models.

        Args:
            provider: Filter by provider ("openai", "anthropic", "google").
            capability: Filter by capability ("chat", "embedding", "image").

        Returns:
            List of ModelInfo objects.
        """
        params = {}
        if provider:
            params["provider"] = provider
        if capability:
            params["capability"] = capability

        response = self._request_with_retry("GET", "/models", params=params)
        return [ModelInfo.from_dict(m) for m in response.get("data", [])]

    def get_model(self, model_id: str) -> ModelInfo:
        """
        Get information about a specific model.

        Args:
            model_id: The model ID (e.g., "openai/gpt-4o").

        Returns:
            ModelInfo for the model.
        """
        response = self._request_with_retry("GET", f"/models/{model_id}")
        return ModelInfo.from_dict(response)

    # ============================================================
    # Health & Usage API
    # ============================================================

    def health(self) -> HealthStatus:
        """
        Check API health status.

        Returns:
            HealthStatus with provider availability.
        """
        response = httpx.get(f"{self._base_url.replace('/v1', '')}/health")
        return HealthStatus.from_dict(response.json())

    def get_usage(self) -> Dict[str, Any]:
        """
        Get usage statistics for the current API key.

        Returns:
            Dictionary with usage data.
        """
        return self._request_with_retry("GET", "/usage")

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

    def _request(
        self,
        method: str,
        path: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Make an HTTP request and handle errors."""
        try:
            response = self._client.request(method, path, **kwargs)
            return self._handle_response(response)

        except httpx.TimeoutException:
            raise TimeoutError("Request timed out")
        except httpx.ConnectError:
            raise ConnectionError("Failed to connect to API")
        except httpx.RequestError as e:
            raise ConnectionError(f"Request failed: {e}")

    def _request_with_retry(
        self,
        method: str,
        path: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Make an HTTP request with retry logic."""
        def do_request():
            return self._request(method, path, **kwargs)

        return self._retry_handler.execute(do_request)

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

    def _check_response(self, response: httpx.Response) -> None:
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

    # ============================================================
    # Context Manager
    # ============================================================

    def close(self):
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class _ChatProxy:
    """
    Proxy that supports both callable and attribute access.

    Allows both:
    - client.chat("Hello!")
    - client.chat.completions.create(...)
    """

    def __init__(self, client: TwoAPI, chat_compat):
        self._client = client
        self._chat_compat = chat_compat

    def __call__(
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
        """Simple chat API - client.chat("Hello!")"""
        return self._client._chat_impl(
            message=message,
            model=model,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            routing=routing,
            metadata=metadata,
            **kwargs
        )

    @property
    def completions(self):
        """OpenAI-compatible completions interface."""
        return self._chat_compat.completions
