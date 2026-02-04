"""
2api.ai Python SDK

A simple, unified interface to access multiple AI providers.

Quick Start:
    from twoapi import TwoAPI
    
    client = TwoAPI(api_key="2api_xxx")
    
    # Simple chat
    response = client.chat("Hello!")
    print(response.content)
    
    # With specific model
    response = client.chat(
        "Explain quantum computing",
        model="anthropic/claude-3-5-sonnet"
    )
    
    # With streaming
    for chunk in client.chat_stream("Tell me a story"):
        print(chunk, end="", flush=True)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Union

import httpx


__version__ = "1.0.0"
__all__ = ["TwoAPI", "Message", "ChatResponse", "TwoAPIError"]


# ============================================================
# Exceptions
# ============================================================

class TwoAPIError(Exception):
    """Base exception for 2api.ai SDK."""
    
    def __init__(self, message: str, code: str = "unknown", status_code: int = 500):
        self.message = message
        self.code = code
        self.status_code = status_code
        super().__init__(message)


class AuthenticationError(TwoAPIError):
    """API key is invalid or missing."""
    pass


class RateLimitError(TwoAPIError):
    """Rate limit exceeded."""
    
    def __init__(self, message: str, retry_after: int = 60):
        super().__init__(message, code="rate_limit_exceeded", status_code=429)
        self.retry_after = retry_after


class InvalidRequestError(TwoAPIError):
    """Request parameters are invalid."""
    pass


# ============================================================
# Data Classes
# ============================================================

@dataclass
class Message:
    """A chat message."""
    role: str
    content: str
    
    @classmethod
    def system(cls, content: str) -> Message:
        return cls(role="system", content=content)
    
    @classmethod
    def user(cls, content: str) -> Message:
        return cls(role="user", content=content)
    
    @classmethod
    def assistant(cls, content: str) -> Message:
        return cls(role="assistant", content=content)
    
    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class Usage:
    """Token usage information."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class RoutingInfo:
    """Information about how the request was routed."""
    strategy_used: str = ""
    fallback_used: bool = False
    latency_ms: int = 0
    cost_usd: float = 0.0


@dataclass
class ChatResponse:
    """Response from a chat completion."""
    content: str
    model: str
    provider: str
    usage: Usage
    routing: Optional[RoutingInfo] = None
    id: str = ""
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ChatResponse:
        """Create ChatResponse from API response dictionary."""
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        usage_data = data.get("usage", {})
        twoapi_data = data.get("_2api", {})
        
        routing = None
        if twoapi_data:
            routing_decision = twoapi_data.get("routing_decision", {})
            routing = RoutingInfo(
                strategy_used=routing_decision.get("strategy_used", ""),
                fallback_used=routing_decision.get("fallback_used", False),
                latency_ms=twoapi_data.get("latency_ms", 0),
                cost_usd=twoapi_data.get("cost_usd", 0.0)
            )
        
        return cls(
            content=message.get("content", ""),
            model=data.get("model", ""),
            provider=data.get("provider", ""),
            usage=Usage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0)
            ),
            routing=routing,
            id=data.get("id", "")
        )


@dataclass
class EmbeddingResponse:
    """Response from an embedding request."""
    embeddings: List[List[float]]
    model: str
    usage: Usage
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EmbeddingResponse:
        usage_data = data.get("usage", {})
        return cls(
            embeddings=[item["embedding"] for item in data.get("data", [])],
            model=data.get("model", ""),
            usage=Usage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=0,
                total_tokens=usage_data.get("total_tokens", 0)
            )
        )


@dataclass
class ImageResponse:
    """Response from an image generation request."""
    urls: List[str]
    revised_prompts: List[str]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ImageResponse:
        items = data.get("data", [])
        return cls(
            urls=[item.get("url", "") for item in items if item.get("url")],
            revised_prompts=[item.get("revised_prompt", "") for item in items]
        )


# ============================================================
# Main Client
# ============================================================

class TwoAPI:
    """
    2api.ai Python Client.
    
    A unified interface to access multiple AI providers through 2api.ai.
    
    Args:
        api_key: Your 2api.ai API key. If not provided, reads from TWOAPI_API_KEY env var.
        base_url: Base URL for the API. Defaults to https://api.2api.ai/v1
        timeout: Request timeout in seconds. Defaults to 60.
        
    Example:
        >>> client = TwoAPI(api_key="2api_xxx")
        >>> response = client.chat("Hello!")
        >>> print(response.content)
        Hello! How can I help you today?
    """
    
    DEFAULT_BASE_URL = "https://api.2api.ai/v1"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 60
    ):
        self.api_key = api_key or os.getenv("TWOAPI_API_KEY")
        if not self.api_key:
            raise AuthenticationError(
                "API key required. Set TWOAPI_API_KEY environment variable or pass api_key parameter."
            )
        
        self.base_url = (base_url or os.getenv("TWOAPI_BASE_URL") or self.DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout
        
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": f"twoapi-python/{__version__}"
            },
            timeout=timeout
        )
    
    def chat(
        self,
        message: Union[str, List[Message], List[Dict[str, str]]],
        model: str = "auto",
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        routing_strategy: Optional[str] = None,
        fallback: Optional[List[str]] = None,
        **kwargs
    ) -> ChatResponse:
        """
        Create a chat completion.
        
        Args:
            message: The message(s) to send. Can be:
                - A string (simple user message)
                - A list of Message objects
                - A list of dicts with 'role' and 'content'
            model: Model to use. Use "auto" for automatic selection.
                Examples: "openai/gpt-4o", "anthropic/claude-3-5-sonnet", "google/gemini-1.5-pro"
            system: Optional system message.
            temperature: Sampling temperature (0-2).
            max_tokens: Maximum tokens to generate.
            routing_strategy: Routing strategy: "cost", "latency", or "quality".
            fallback: List of fallback models if primary fails.
            
        Returns:
            ChatResponse with the completion.
            
        Example:
            >>> response = client.chat("What is 2+2?")
            >>> print(response.content)
            2+2 equals 4.
            
            >>> response = client.chat(
            ...     "Explain quantum computing",
            ...     model="anthropic/claude-3-5-sonnet",
            ...     max_tokens=500
            ... )
        """
        # Normalize messages
        messages = self._normalize_messages(message, system)
        
        # Build request
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages
        }
        
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        
        # Routing config
        if routing_strategy or fallback:
            payload["routing"] = {}
            if routing_strategy:
                payload["routing"]["strategy"] = routing_strategy
            if fallback:
                payload["routing"]["fallback"] = fallback
        
        # Additional kwargs
        payload.update(kwargs)
        
        # Make request
        response = self._request("POST", "/chat/completions", json=payload)
        return ChatResponse.from_dict(response)
    
    def chat_stream(
        self,
        message: Union[str, List[Message], List[Dict[str, str]]],
        model: str = "auto",
        system: Optional[str] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        Create a streaming chat completion.
        
        Yields content chunks as they arrive.
        
        Args:
            message: The message(s) to send.
            model: Model to use.
            system: Optional system message.
            
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
        payload.update(kwargs)
        
        with self._client.stream(
            "POST",
            "/chat/completions",
            json=payload
        ) as response:
            for line in response.iter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue
    
    def embed(
        self,
        text: Union[str, List[str]],
        model: str = "openai/text-embedding-3-small"
    ) -> EmbeddingResponse:
        """
        Create embeddings for text.
        
        Args:
            text: Text or list of texts to embed.
            model: Embedding model to use.
            
        Returns:
            EmbeddingResponse with the embeddings.
            
        Example:
            >>> response = client.embed("Hello world")
            >>> print(len(response.embeddings[0]))  # Embedding dimension
            1536
        """
        payload = {
            "model": model,
            "input": text
        }
        
        response = self._request("POST", "/embeddings", json=payload)
        return EmbeddingResponse.from_dict(response)
    
    def generate_image(
        self,
        prompt: str,
        model: str = "openai/dall-e-3",
        size: str = "1024x1024",
        n: int = 1,
        quality: str = "standard",
        style: str = "vivid"
    ) -> ImageResponse:
        """
        Generate images from a prompt.
        
        Args:
            prompt: Text description of the image to generate.
            model: Image generation model.
            size: Image size (e.g., "1024x1024").
            n: Number of images to generate.
            quality: "standard" or "hd".
            style: "vivid" or "natural".
            
        Returns:
            ImageResponse with URLs to the generated images.
            
        Example:
            >>> response = client.generate_image("A cat wearing a top hat")
            >>> print(response.urls[0])
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "n": n,
            "quality": quality,
            "style": style
        }
        
        response = self._request("POST", "/images/generations", json=payload)
        return ImageResponse.from_dict(response)
    
    def list_models(
        self,
        provider: Optional[str] = None,
        capability: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List available models.
        
        Args:
            provider: Filter by provider ("openai", "anthropic", "google").
            capability: Filter by capability ("chat", "embedding", "image", "vision", "tools").
            
        Returns:
            List of model information dictionaries.
        """
        params = {}
        if provider:
            params["provider"] = provider
        if capability:
            params["capability"] = capability
        
        response = self._request("GET", "/models", params=params)
        return response.get("data", [])
    
    def health(self) -> Dict[str, Any]:
        """Check API health status."""
        # Health endpoint doesn't require auth
        response = httpx.get(f"{self.base_url.replace('/v1', '')}/health")
        return response.json()
    
    # ============================================================
    # Private methods
    # ============================================================
    
    def _normalize_messages(
        self,
        message: Union[str, List[Message], List[Dict[str, str]]],
        system: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Normalize various message formats to list of dicts."""
        messages = []
        
        # Add system message if provided
        if system:
            messages.append({"role": "system", "content": system})
        
        # Handle different input types
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
            
            if response.status_code == 401:
                raise AuthenticationError("Invalid API key")
            elif response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 60))
                raise RateLimitError("Rate limit exceeded", retry_after=retry_after)
            elif response.status_code >= 400:
                try:
                    error_data = response.json().get("error", {})
                    raise TwoAPIError(
                        message=error_data.get("message", "Unknown error"),
                        code=error_data.get("code", "unknown"),
                        status_code=response.status_code
                    )
                except (json.JSONDecodeError, KeyError):
                    raise TwoAPIError(
                        message=response.text,
                        status_code=response.status_code
                    )
            
            return response.json()
            
        except httpx.TimeoutException:
            raise TwoAPIError("Request timed out", code="timeout")
        except httpx.RequestError as e:
            raise TwoAPIError(f"Request failed: {e}", code="request_error")
    
    def close(self):
        """Close the HTTP client."""
        self._client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


# ============================================================
# Convenience functions
# ============================================================

_default_client: Optional[TwoAPI] = None


def _get_default_client() -> TwoAPI:
    """Get or create the default client."""
    global _default_client
    if _default_client is None:
        _default_client = TwoAPI()
    return _default_client


def chat(
    message: Union[str, List[Message], List[Dict[str, str]]],
    **kwargs
) -> ChatResponse:
    """
    Quick chat function using default client.
    
    Example:
        >>> import twoapi
        >>> response = twoapi.chat("Hello!")
        >>> print(response.content)
    """
    return _get_default_client().chat(message, **kwargs)


def chat_stream(
    message: Union[str, List[Message], List[Dict[str, str]]],
    **kwargs
) -> Generator[str, None, None]:
    """Quick streaming chat function using default client."""
    yield from _get_default_client().chat_stream(message, **kwargs)


def embed(text: Union[str, List[str]], **kwargs) -> EmbeddingResponse:
    """Quick embed function using default client."""
    return _get_default_client().embed(text, **kwargs)


def generate_image(prompt: str, **kwargs) -> ImageResponse:
    """Quick image generation function using default client."""
    return _get_default_client().generate_image(prompt, **kwargs)
