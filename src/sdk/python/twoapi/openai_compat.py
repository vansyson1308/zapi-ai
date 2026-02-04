"""
2api.ai SDK - OpenAI-Compatible Interface

Provides an OpenAI-like API surface for easy migration.

Example:
    from twoapi import TwoAPI

    # OpenAI-compatible usage
    client = TwoAPI(api_key="2api_xxx")

    response = client.chat.completions.create(
        model="openai/gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ],
        temperature=0.7,
        max_tokens=100
    )

    print(response.choices[0].message.content)

    # Streaming
    stream = client.chat.completions.create(
        model="openai/gpt-4o",
        messages=[{"role": "user", "content": "Tell me a story"}],
        stream=True
    )

    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="")
"""

from __future__ import annotations

import json
from typing import (
    Any,
    Dict,
    Generator,
    List,
    Optional,
    Union,
    AsyncGenerator,
    Iterator,
    AsyncIterator,
    Literal,
    overload,
)
from dataclasses import dataclass, field


# ============================================================
# OpenAI-Compatible Response Models
# ============================================================

@dataclass
class FunctionCall:
    """Function call in a message."""
    name: str
    arguments: str


@dataclass
class ToolCallFunction:
    """Function details in a tool call."""
    name: str
    arguments: str


@dataclass
class OpenAIToolCall:
    """Tool call in OpenAI format."""
    id: str
    type: str
    function: ToolCallFunction

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OpenAIToolCall":
        func = data.get("function", {})
        return cls(
            id=data.get("id", ""),
            type=data.get("type", "function"),
            function=ToolCallFunction(
                name=func.get("name", ""),
                arguments=func.get("arguments", "{}")
            )
        )


@dataclass
class ChatCompletionMessage:
    """Message in a chat completion response."""
    role: str
    content: Optional[str] = None
    function_call: Optional[FunctionCall] = None
    tool_calls: Optional[List[OpenAIToolCall]] = None
    refusal: Optional[str] = None


@dataclass
class Choice:
    """A choice in the completion response."""
    index: int
    message: ChatCompletionMessage
    finish_reason: str
    logprobs: Optional[Any] = None


@dataclass
class CompletionUsage:
    """Usage statistics for a completion."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class ChatCompletion:
    """
    OpenAI-compatible chat completion response.

    Matches the OpenAI API response format exactly.
    """
    id: str
    object: str
    created: int
    model: str
    choices: List[Choice]
    usage: CompletionUsage
    system_fingerprint: Optional[str] = None
    service_tier: Optional[str] = None

    # 2api.ai extensions
    provider: str = ""
    _2api: Optional[Dict[str, Any]] = None

    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> "ChatCompletion":
        """Create from 2api.ai API response."""
        choices = []
        for i, choice_data in enumerate(data.get("choices", [])):
            msg_data = choice_data.get("message", {})

            tool_calls = None
            if msg_data.get("tool_calls"):
                tool_calls = [
                    OpenAIToolCall.from_dict(tc)
                    for tc in msg_data["tool_calls"]
                ]

            message = ChatCompletionMessage(
                role=msg_data.get("role", "assistant"),
                content=msg_data.get("content"),
                tool_calls=tool_calls
            )

            choices.append(Choice(
                index=i,
                message=message,
                finish_reason=choice_data.get("finish_reason", "stop"),
                logprobs=choice_data.get("logprobs")
            ))

        usage_data = data.get("usage", {})

        return cls(
            id=data.get("id", ""),
            object="chat.completion",
            created=data.get("created", 0),
            model=data.get("model", ""),
            choices=choices,
            usage=CompletionUsage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0)
            ),
            system_fingerprint=data.get("system_fingerprint"),
            provider=data.get("provider", ""),
            _2api=data.get("_2api")
        )


# ============================================================
# Streaming Response Models
# ============================================================

@dataclass
class ChoiceDelta:
    """Delta content in a streaming chunk."""
    role: Optional[str] = None
    content: Optional[str] = None
    function_call: Optional[FunctionCall] = None
    tool_calls: Optional[List[OpenAIToolCall]] = None
    refusal: Optional[str] = None


@dataclass
class StreamChoice:
    """A choice in a streaming response chunk."""
    index: int
    delta: ChoiceDelta
    finish_reason: Optional[str] = None
    logprobs: Optional[Any] = None


@dataclass
class ChatCompletionChunk:
    """
    OpenAI-compatible streaming chunk.

    Matches the OpenAI API streaming response format.
    """
    id: str
    object: str
    created: int
    model: str
    choices: List[StreamChoice]
    system_fingerprint: Optional[str] = None
    service_tier: Optional[str] = None
    usage: Optional[CompletionUsage] = None

    @classmethod
    def from_sse_data(cls, data: Dict[str, Any]) -> "ChatCompletionChunk":
        """Create from SSE data chunk."""
        choices = []
        for choice_data in data.get("choices", []):
            delta_data = choice_data.get("delta", {})

            tool_calls = None
            if delta_data.get("tool_calls"):
                tool_calls = [
                    OpenAIToolCall.from_dict(tc)
                    for tc in delta_data["tool_calls"]
                ]

            delta = ChoiceDelta(
                role=delta_data.get("role"),
                content=delta_data.get("content"),
                tool_calls=tool_calls
            )

            choices.append(StreamChoice(
                index=choice_data.get("index", 0),
                delta=delta,
                finish_reason=choice_data.get("finish_reason"),
                logprobs=choice_data.get("logprobs")
            ))

        usage = None
        if data.get("usage"):
            usage_data = data["usage"]
            usage = CompletionUsage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0)
            )

        return cls(
            id=data.get("id", ""),
            object="chat.completion.chunk",
            created=data.get("created", 0),
            model=data.get("model", ""),
            choices=choices,
            system_fingerprint=data.get("system_fingerprint"),
            usage=usage
        )


# ============================================================
# OpenAI-Compatible Interface Classes
# ============================================================

class ChatCompletions:
    """
    OpenAI-compatible chat completions interface.

    Usage:
        client.chat.completions.create(
            model="openai/gpt-4o",
            messages=[{"role": "user", "content": "Hello"}]
        )
    """

    def __init__(self, client: "TwoAPIOpenAICompat"):
        self._client = client

    @overload
    def create(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        stream: Literal[False] = False,
        **kwargs
    ) -> ChatCompletion: ...

    @overload
    def create(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        stream: Literal[True],
        **kwargs
    ) -> Iterator[ChatCompletionChunk]: ...

    def create(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        stream: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        n: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        response_format: Optional[Dict[str, str]] = None,
        seed: Optional[int] = None,
        user: Optional[str] = None,
        **kwargs
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        """
        Create a chat completion.

        Matches OpenAI's chat.completions.create() signature.

        Args:
            model: Model ID to use
            messages: List of messages in the conversation
            stream: Whether to stream the response
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            frequency_penalty: Frequency penalty (-2 to 2)
            presence_penalty: Presence penalty (-2 to 2)
            stop: Stop sequences
            n: Number of completions to generate
            tools: List of tools available
            tool_choice: How to select tools
            response_format: Response format specification
            seed: Random seed for reproducibility
            user: End-user identifier

        Returns:
            ChatCompletion or Iterator of ChatCompletionChunk if streaming
        """
        # Build request payload
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
        }

        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if top_p is not None:
            payload["top_p"] = top_p
        if frequency_penalty is not None:
            payload["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            payload["presence_penalty"] = presence_penalty
        if stop is not None:
            payload["stop"] = stop
        if n is not None:
            payload["n"] = n
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if response_format is not None:
            payload["response_format"] = response_format
        if seed is not None:
            payload["seed"] = seed
        if user is not None:
            payload["user"] = user

        # Add any 2api-specific kwargs
        if "routing" in kwargs:
            payload["routing"] = kwargs.pop("routing")

        if stream:
            return self._create_stream(payload)
        else:
            return self._create_sync(payload)

    def _create_sync(self, payload: Dict[str, Any]) -> ChatCompletion:
        """Create a non-streaming completion."""
        response = self._client._request_with_retry(
            "POST", "/chat/completions", json=payload
        )
        return ChatCompletion.from_api_response(response)

    def _create_stream(
        self, payload: Dict[str, Any]
    ) -> Iterator[ChatCompletionChunk]:
        """Create a streaming completion."""
        payload["stream"] = True

        with self._client._client.stream(
            "POST",
            "/chat/completions",
            json=payload
        ) as response:
            self._client._check_response(response)

            for line in response.iter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk_data = json.loads(data)
                        yield ChatCompletionChunk.from_sse_data(chunk_data)
                    except json.JSONDecodeError:
                        continue


class AsyncChatCompletions:
    """Async version of ChatCompletions."""

    def __init__(self, client: "AsyncTwoAPIOpenAICompat"):
        self._client = client

    @overload
    async def create(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        stream: Literal[False] = False,
        **kwargs
    ) -> ChatCompletion: ...

    @overload
    async def create(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        stream: Literal[True],
        **kwargs
    ) -> AsyncIterator[ChatCompletionChunk]: ...

    async def create(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        stream: bool = False,
        **kwargs
    ) -> Union[ChatCompletion, AsyncIterator[ChatCompletionChunk]]:
        """Create a chat completion asynchronously."""
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
        }

        # Copy all optional parameters
        for key in [
            "temperature", "max_tokens", "top_p",
            "frequency_penalty", "presence_penalty", "stop",
            "n", "tools", "tool_choice", "response_format",
            "seed", "user", "routing"
        ]:
            if key in kwargs and kwargs[key] is not None:
                payload[key] = kwargs[key]

        if stream:
            return self._create_stream(payload)
        else:
            return await self._create_sync(payload)

    async def _create_sync(self, payload: Dict[str, Any]) -> ChatCompletion:
        """Create a non-streaming completion."""
        response = await self._client._request_with_retry(
            "POST", "/chat/completions", json=payload
        )
        return ChatCompletion.from_api_response(response)

    async def _create_stream(
        self, payload: Dict[str, Any]
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Create a streaming completion."""
        payload["stream"] = True

        client = await self._client._get_client()

        async with client.stream(
            "POST",
            "/chat/completions",
            json=payload
        ) as response:
            await self._client._check_response(response)

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk_data = json.loads(data)
                        yield ChatCompletionChunk.from_sse_data(chunk_data)
                    except json.JSONDecodeError:
                        continue


class Chat:
    """Chat namespace for OpenAI compatibility."""

    def __init__(self, client: "TwoAPIOpenAICompat"):
        self.completions = ChatCompletions(client)


class AsyncChat:
    """Async chat namespace for OpenAI compatibility."""

    def __init__(self, client: "AsyncTwoAPIOpenAICompat"):
        self.completions = AsyncChatCompletions(client)


# ============================================================
# Embeddings Interface
# ============================================================

@dataclass
class EmbeddingData:
    """Single embedding result."""
    index: int
    embedding: List[float]
    object: str = "embedding"


@dataclass
class EmbeddingResponse:
    """OpenAI-compatible embedding response."""
    object: str
    data: List[EmbeddingData]
    model: str
    usage: CompletionUsage

    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> "EmbeddingResponse":
        """Create from API response."""
        embeddings = []
        for i, item in enumerate(data.get("data", [])):
            embeddings.append(EmbeddingData(
                index=i,
                embedding=item.get("embedding", [])
            ))

        usage_data = data.get("usage", {})

        return cls(
            object="list",
            data=embeddings,
            model=data.get("model", ""),
            usage=CompletionUsage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=0,
                total_tokens=usage_data.get("total_tokens", 0)
            )
        )


class Embeddings:
    """OpenAI-compatible embeddings interface."""

    def __init__(self, client: "TwoAPIOpenAICompat"):
        self._client = client

    def create(
        self,
        *,
        model: str,
        input: Union[str, List[str]],
        dimensions: Optional[int] = None,
        encoding_format: Optional[str] = None,
        user: Optional[str] = None,
        **kwargs
    ) -> EmbeddingResponse:
        """Create embeddings for text."""
        payload: Dict[str, Any] = {
            "model": model,
            "input": input,
        }

        if dimensions is not None:
            payload["dimensions"] = dimensions
        if encoding_format is not None:
            payload["encoding_format"] = encoding_format
        if user is not None:
            payload["user"] = user

        response = self._client._request_with_retry(
            "POST", "/embeddings", json=payload
        )
        return EmbeddingResponse.from_api_response(response)


class AsyncEmbeddings:
    """Async embeddings interface."""

    def __init__(self, client: "AsyncTwoAPIOpenAICompat"):
        self._client = client

    async def create(
        self,
        *,
        model: str,
        input: Union[str, List[str]],
        dimensions: Optional[int] = None,
        **kwargs
    ) -> EmbeddingResponse:
        """Create embeddings asynchronously."""
        payload: Dict[str, Any] = {
            "model": model,
            "input": input,
        }

        if dimensions is not None:
            payload["dimensions"] = dimensions

        response = await self._client._request_with_retry(
            "POST", "/embeddings", json=payload
        )
        return EmbeddingResponse.from_api_response(response)


# Type hint imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .client import TwoAPI as TwoAPIOpenAICompat
    from .async_client import AsyncTwoAPI as AsyncTwoAPIOpenAICompat
