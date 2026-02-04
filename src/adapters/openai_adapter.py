"""
2api.ai - OpenAI Provider Adapter

Adapter for OpenAI's API (GPT-4, GPT-4o, DALL-E, etc.)
"""

import json
import time
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from .base import AdapterConfig, BaseAdapter, ProviderHealth
from ..core.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    FinishReason,
    FunctionCall,
    ImageData,
    ImageGenerationRequest,
    ImageGenerationResponse,
    Message,
    ModelInfo,
    ModelPricing,
    Provider,
    ProviderError,
    APIError,
    Role,
    ToolCall,
    Usage,
)


class OpenAIAdapter(BaseAdapter):
    """
    Adapter for OpenAI API.
    
    Supports:
    - Chat completions (GPT-4, GPT-4o, GPT-4o-mini)
    - Embeddings (text-embedding-3-small, text-embedding-3-large)
    - Image generation (DALL-E 3)
    - Vision (GPT-4o)
    - Tool/Function calling
    - Streaming
    """
    
    provider = Provider.OPENAI
    DEFAULT_BASE_URL = "https://api.openai.com/v1"

    # Model catalog with pricing (as of 2024)
    MODELS: List[ModelInfo] = [
        ModelInfo(
            id="openai/gpt-4o",
            provider=Provider.OPENAI,
            name="gpt-4o",
            capabilities=["chat", "vision", "tools"],
            context_window=128000,
            max_output_tokens=16384,
            pricing=ModelPricing(input_per_1m_tokens=2.50, output_per_1m_tokens=10.00)
        ),
        ModelInfo(
            id="openai/gpt-4o-mini",
            provider=Provider.OPENAI,
            name="gpt-4o-mini",
            capabilities=["chat", "vision", "tools"],
            context_window=128000,
            max_output_tokens=16384,
            pricing=ModelPricing(input_per_1m_tokens=0.15, output_per_1m_tokens=0.60)
        ),
        ModelInfo(
            id="openai/gpt-4-turbo",
            provider=Provider.OPENAI,
            name="gpt-4-turbo",
            capabilities=["chat", "vision", "tools"],
            context_window=128000,
            max_output_tokens=4096,
            pricing=ModelPricing(input_per_1m_tokens=10.00, output_per_1m_tokens=30.00)
        ),
        ModelInfo(
            id="openai/gpt-4",
            provider=Provider.OPENAI,
            name="gpt-4",
            capabilities=["chat", "tools"],
            context_window=8192,
            max_output_tokens=8192,
            pricing=ModelPricing(input_per_1m_tokens=30.00, output_per_1m_tokens=60.00)
        ),
        ModelInfo(
            id="openai/gpt-3.5-turbo",
            provider=Provider.OPENAI,
            name="gpt-3.5-turbo",
            capabilities=["chat", "tools"],
            context_window=16385,
            max_output_tokens=4096,
            pricing=ModelPricing(input_per_1m_tokens=0.50, output_per_1m_tokens=1.50)
        ),
        ModelInfo(
            id="openai/text-embedding-3-small",
            provider=Provider.OPENAI,
            name="text-embedding-3-small",
            capabilities=["embedding"],
            context_window=8191,
            max_output_tokens=0,
            pricing=ModelPricing(input_per_1m_tokens=0.02, output_per_1m_tokens=0)
        ),
        ModelInfo(
            id="openai/text-embedding-3-large",
            provider=Provider.OPENAI,
            name="text-embedding-3-large",
            capabilities=["embedding"],
            context_window=8191,
            max_output_tokens=0,
            pricing=ModelPricing(input_per_1m_tokens=0.13, output_per_1m_tokens=0)
        ),
        ModelInfo(
            id="openai/dall-e-3",
            provider=Provider.OPENAI,
            name="dall-e-3",
            capabilities=["image"],
            context_window=4000,
            max_output_tokens=0,
            pricing=ModelPricing(input_per_1m_tokens=0, output_per_1m_tokens=0)  # Per image pricing
        ),
    ]

    def __init__(self, config: AdapterConfig):
        super().__init__(config)
        self.base_url = config.base_url or self.DEFAULT_BASE_URL
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
            timeout=config.timeout
        )

    async def chat_completion(
        self,
        request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Generate a chat completion using OpenAI."""
        
        # Build OpenAI-specific request
        payload = self._build_chat_payload(request)
        
        start_time = time.time()
        
        try:
            response = await self.client.post("/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as e:
            raise self._handle_error(e)
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        # Convert to unified response
        return self._parse_chat_response(data, request.model_name, latency_ms)

    async def chat_completion_stream(
        self,
        request: ChatCompletionRequest
    ) -> AsyncIterator[str]:
        """Generate a streaming chat completion."""
        
        payload = self._build_chat_payload(request)
        payload["stream"] = True
        
        async with self.client.stream(
            "POST",
            "/chat/completions",
            json=payload
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        yield "data: [DONE]\n\n"
                        break
                    yield f"data: {data}\n\n"

    async def embedding(
        self,
        request: EmbeddingRequest
    ) -> EmbeddingResponse:
        """Create embeddings using OpenAI."""
        
        model_name = request.model.split("/")[-1] if "/" in request.model else request.model
        
        payload: Dict[str, Any] = {
            "model": model_name,
            "input": request.input,
            "encoding_format": request.encoding_format,
        }
        
        if request.dimensions:
            payload["dimensions"] = request.dimensions
        
        try:
            response = await self.client.post("/embeddings", json=payload)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as e:
            raise self._handle_error(e)
        
        return EmbeddingResponse(
            data=[
                EmbeddingData(
                    embedding=item["embedding"],
                    index=item["index"]
                )
                for item in data["data"]
            ],
            model=data["model"],
            usage=Usage(
                prompt_tokens=data["usage"]["prompt_tokens"],
                completion_tokens=0,
                total_tokens=data["usage"]["total_tokens"]
            )
        )

    async def image_generation(
        self,
        request: ImageGenerationRequest
    ) -> ImageGenerationResponse:
        """Generate images using DALL-E."""
        
        model_name = request.model.split("/")[-1] if "/" in request.model else request.model
        
        payload: Dict[str, Any] = {
            "model": model_name,
            "prompt": request.prompt,
            "n": request.n,
            "size": request.size,
            "quality": request.quality,
            "style": request.style,
            "response_format": request.response_format,
        }
        
        try:
            response = await self.client.post("/images/generations", json=payload)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as e:
            raise self._handle_error(e)
        
        return ImageGenerationResponse(
            created=data["created"],
            data=[
                ImageData(
                    url=item.get("url"),
                    b64_json=item.get("b64_json"),
                    revised_prompt=item.get("revised_prompt")
                )
                for item in data["data"]
            ]
        )

    def list_models(self) -> List[ModelInfo]:
        """List available OpenAI models."""
        return self.MODELS

    async def health_check(self) -> ProviderHealth:
        """Check OpenAI API health."""
        try:
            start = time.time()
            response = await self.client.get("/models")
            latency = int((time.time() - start) * 1000)
            
            return ProviderHealth(
                provider=self.provider,
                is_healthy=response.status_code == 200,
                avg_latency_ms=latency
            )
        except Exception as e:
            return ProviderHealth(
                provider=self.provider,
                is_healthy=False,
                last_error=str(e)
            )

    # ============================================================
    # Private helper methods
    # ============================================================

    def _build_chat_payload(
        self,
        request: ChatCompletionRequest
    ) -> Dict[str, Any]:
        """Build OpenAI-specific chat payload."""
        
        model_name = request.model_name
        
        payload: Dict[str, Any] = {
            "model": model_name,
            "messages": self._convert_messages(request.messages),
        }
        
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.tools:
            payload["tools"] = self._normalize_tools(request)
        if request.tool_choice:
            payload["tool_choice"] = self._convert_tool_choice(request.tool_choice)
        
        return payload

    def _convert_messages(
        self,
        messages: List[Message]
    ) -> List[Dict[str, Any]]:
        """Convert unified messages to OpenAI format."""
        result = []
        
        for msg in messages:
            openai_msg: Dict[str, Any] = {"role": msg.role.value}
            
            # Handle content
            if msg.content is not None:
                if isinstance(msg.content, str):
                    openai_msg["content"] = msg.content
                else:
                    # Multimodal content
                    openai_msg["content"] = []
                    for part in msg.content:
                        if hasattr(part, 'text'):
                            openai_msg["content"].append({
                                "type": "text",
                                "text": part.text
                            })
                        elif hasattr(part, 'image_url'):
                            openai_msg["content"].append({
                                "type": "image_url",
                                "image_url": {
                                    "url": part.image_url.url,
                                    "detail": part.image_url.detail
                                }
                            })
            
            # Handle tool-related fields
            if msg.name:
                openai_msg["name"] = msg.name
            if msg.tool_call_id:
                openai_msg["tool_call_id"] = msg.tool_call_id
            if msg.tool_calls:
                openai_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in msg.tool_calls
                ]
            
            result.append(openai_msg)
        
        return result

    def _convert_tool_choice(self, tool_choice: Any) -> Any:
        """Convert tool choice to OpenAI format."""
        if isinstance(tool_choice, str):
            return tool_choice
        return {
            "type": tool_choice.type,
            "function": {"name": tool_choice.function.name}
        }

    def _parse_chat_response(
        self,
        data: Dict[str, Any],
        model: str,
        latency_ms: int
    ) -> ChatCompletionResponse:
        """Parse OpenAI response to unified format."""
        
        choice = data["choices"][0]
        message_data = choice["message"]
        
        # Parse tool calls if present
        tool_calls = None
        if "tool_calls" in message_data:
            tool_calls = [
                ToolCall(
                    id=tc["id"],
                    type=tc["type"],
                    function=FunctionCall(
                        name=tc["function"]["name"],
                        arguments=tc["function"]["arguments"]
                    )
                )
                for tc in message_data["tool_calls"]
            ]
        
        # Map finish reason
        finish_reason_map = {
            "stop": FinishReason.STOP,
            "length": FinishReason.LENGTH,
            "tool_calls": FinishReason.TOOL_CALLS,
            "content_filter": FinishReason.CONTENT_FILTER,
        }
        finish_reason = finish_reason_map.get(
            choice.get("finish_reason", "stop"),
            FinishReason.STOP
        )
        
        # Build usage
        usage = Usage(
            prompt_tokens=data["usage"]["prompt_tokens"],
            completion_tokens=data["usage"]["completion_tokens"],
            total_tokens=data["usage"]["total_tokens"]
        )
        
        return ChatCompletionResponse(
            id=data["id"],
            created=data["created"],
            model=data["model"],
            provider=self.provider.value,
            choices=[
                Choice(
                    index=choice["index"],
                    message=Message(
                        role=Role.ASSISTANT,
                        content=message_data.get("content"),
                        tool_calls=tool_calls
                    ),
                    finish_reason=finish_reason
                )
            ],
            usage=usage
        )

    def _handle_error(self, error: httpx.HTTPStatusError) -> ProviderError:
        """Convert HTTP error to ProviderError."""
        try:
            error_data = error.response.json()
            error_info = error_data.get("error", {})
            return ProviderError(
                APIError(
                    code=error_info.get("code", "unknown"),
                    message=error_info.get("message", str(error)),
                    type="provider_error",
                    provider="openai"
                )
            )
        except Exception:
            return ProviderError(
                APIError(
                    code="unknown",
                    message=str(error),
                    type="provider_error",
                    provider="openai"
                )
            )

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
