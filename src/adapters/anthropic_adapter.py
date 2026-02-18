"""
2api.ai - Anthropic Provider Adapter

Adapter for Anthropic's Claude API (Claude 3.5, Claude 3, etc.)
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
    EmbeddingRequest,
    EmbeddingResponse,
    FinishReason,
    FunctionCall,
    ImageGenerationRequest,
    ImageGenerationResponse,
    Message,
    ModelInfo,
    ModelPricing,
    Provider,
    Role,
    ToolCall,
    Usage,
    ContentPart,
    TextContent,
    ImageContent,
)
from ..core.errors import (
    TwoApiException,
    handle_anthropic_error,
    StreamInterruptedError,
)


class AnthropicAdapter(BaseAdapter):
    """
    Adapter for Anthropic Claude API.
    
    Supports:
    - Chat completions (Claude 3.5 Sonnet, Claude 3 Opus/Sonnet/Haiku)
    - Vision (all Claude 3+ models)
    - Tool/Function calling
    - Streaming
    
    Note: Anthropic does not support embeddings or image generation.
    """
    
    provider = Provider.ANTHROPIC
    DEFAULT_BASE_URL = "https://api.anthropic.com"
    API_VERSION = "2023-06-01"

    # Model catalog with pricing (as of 2024)
    MODELS: List[ModelInfo] = [
        ModelInfo(
            id="anthropic/claude-3-5-sonnet",
            provider=Provider.ANTHROPIC,
            name="claude-3-5-sonnet-20241022",
            capabilities=["chat", "vision", "tools"],
            context_window=200000,
            max_output_tokens=8192,
            pricing=ModelPricing(input_per_1m_tokens=3.00, output_per_1m_tokens=15.00)
        ),
        ModelInfo(
            id="anthropic/claude-3-opus",
            provider=Provider.ANTHROPIC,
            name="claude-3-opus-20240229",
            capabilities=["chat", "vision", "tools"],
            context_window=200000,
            max_output_tokens=4096,
            pricing=ModelPricing(input_per_1m_tokens=15.00, output_per_1m_tokens=75.00)
        ),
        ModelInfo(
            id="anthropic/claude-3-sonnet",
            provider=Provider.ANTHROPIC,
            name="claude-3-sonnet-20240229",
            capabilities=["chat", "vision", "tools"],
            context_window=200000,
            max_output_tokens=4096,
            pricing=ModelPricing(input_per_1m_tokens=3.00, output_per_1m_tokens=15.00)
        ),
        ModelInfo(
            id="anthropic/claude-3-haiku",
            provider=Provider.ANTHROPIC,
            name="claude-3-haiku-20240307",
            capabilities=["chat", "vision", "tools"],
            context_window=200000,
            max_output_tokens=4096,
            pricing=ModelPricing(input_per_1m_tokens=0.25, output_per_1m_tokens=1.25)
        ),
    ]
    
    # Mapping from short names to full model IDs
    MODEL_ALIASES = {
        "claude-3-5-sonnet": "claude-3-5-sonnet-20241022",
        "claude-3.5-sonnet": "claude-3-5-sonnet-20241022",
        "claude-3-opus": "claude-3-opus-20240229",
        "claude-3-sonnet": "claude-3-sonnet-20240229",
        "claude-3-haiku": "claude-3-haiku-20240307",
    }

    def __init__(self, config: AdapterConfig):
        super().__init__(config)
        self.base_url = config.base_url or self.DEFAULT_BASE_URL
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "x-api-key": config.api_key,
                "anthropic-version": self.API_VERSION,
                "Content-Type": "application/json",
            },
            timeout=config.timeout
        )

    async def chat_completion(
        self,
        request: ChatCompletionRequest,
        request_id: str = ""
    ) -> ChatCompletionResponse:
        """Generate a chat completion using Claude."""

        payload = self._build_chat_payload(request)

        start_time = time.time()

        try:
            response = await self.client.post("/v1/messages", json=payload)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            raise handle_anthropic_error(e, request_id)

        latency_ms = int((time.time() - start_time) * 1000)

        return self._parse_chat_response(data, request.model_name, latency_ms)

    async def chat_completion_stream(
        self,
        request: ChatCompletionRequest,
        request_id: str = ""
    ) -> AsyncIterator[str]:
        """
        Generate a streaming chat completion.

        IMPORTANT: Implements "no semantic drift" rule:
        - Errors before any content: raise exception (caller can retry/fallback)
        - Errors after content started: yield error chunk with partial_content, then stop
        """

        payload = self._build_chat_payload(request)
        payload["stream"] = True

        partial_content = ""
        content_started = False
        message_id = None

        try:
            async with self.client.stream(
                "POST",
                "/v1/messages",
                json=payload
            ) as response:
                # Check for HTTP errors before streaming starts
                if response.status_code >= 400:
                    error_body = await response.aread()
                    class FakeError(httpx.HTTPStatusError):
                        def __init__(self):
                            self.response = response
                            self.request = response.request
                    raise handle_anthropic_error(FakeError(), request_id)

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "":
                            continue

                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        event_type = data.get("type")

                        # Handle Anthropic streaming errors
                        if event_type == "error":
                            error_data = data.get("error", {})
                            error_msg = error_data.get("message", "Unknown streaming error")
                            if content_started:
                                raise StreamInterruptedError(
                                    provider="anthropic",
                                    partial_content=partial_content,
                                    request_id=request_id
                                )
                            else:
                                from ..core.errors import InfraError, ErrorDetails, ErrorType
                                raise InfraError(
                                    ErrorDetails(
                                        code="stream_error",
                                        message=error_msg,
                                        type=ErrorType.INFRA,
                                        provider="anthropic",
                                        request_id=request_id,
                                        retryable=True
                                    ),
                                    status_code=500
                                )

                        if event_type == "message_start":
                            message_id = data.get("message", {}).get("id")
                        elif event_type == "content_block_delta":
                            delta = data.get("delta", {})
                            if delta.get("type") == "text_delta":
                                text = delta.get("text", "")
                                content_started = True
                                partial_content += text
                                # Convert to OpenAI-compatible SSE format
                                chunk = {
                                    "id": message_id,
                                    "object": "chat.completion.chunk",
                                    "choices": [{
                                        "index": 0,
                                        "delta": {"content": text},
                                        "finish_reason": None
                                    }]
                                }
                                yield f"data: {json.dumps(chunk)}\n\n"
                        elif event_type == "message_stop":
                            chunk = {
                                "id": message_id,
                                "object": "chat.completion.chunk",
                                "choices": [{
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": "stop"
                                }]
                            }
                            yield f"data: {json.dumps(chunk)}\n\n"
                            yield "data: [DONE]\n\n"

        except TwoApiException:
            if content_started:
                raise StreamInterruptedError(
                    provider="anthropic",
                    partial_content=partial_content,
                    request_id=request_id
                )
            raise

        except Exception as e:
            error = handle_anthropic_error(e, request_id)
            if content_started:
                raise StreamInterruptedError(
                    provider="anthropic",
                    partial_content=partial_content,
                    request_id=request_id
                )
            raise error

    async def embedding(
        self,
        request: EmbeddingRequest,
        request_id: str = ""
    ) -> EmbeddingResponse:
        """
        Create embeddings - NOT SUPPORTED by Anthropic.

        Raises:
            SemanticError: Anthropic does not support embeddings
        """
        from ..core.errors import SemanticError, ErrorDetails, ErrorType
        raise SemanticError(
            ErrorDetails(
                code="unsupported_operation",
                message="Anthropic does not support embeddings. Use OpenAI or Google.",
                type=ErrorType.SEMANTIC,
                provider="anthropic",
                request_id=request_id,
                retryable=False
            ),
            status_code=400
        )

    async def image_generation(
        self,
        request: ImageGenerationRequest,
        request_id: str = ""
    ) -> ImageGenerationResponse:
        """
        Generate images - NOT SUPPORTED by Anthropic.

        Raises:
            SemanticError: Anthropic does not support image generation
        """
        from ..core.errors import SemanticError, ErrorDetails, ErrorType
        raise SemanticError(
            ErrorDetails(
                code="unsupported_operation",
                message="Anthropic does not support image generation. Use OpenAI (DALL-E).",
                type=ErrorType.SEMANTIC,
                provider="anthropic",
                request_id=request_id,
                retryable=False
            ),
            status_code=400
        )

    def list_models(self) -> List[ModelInfo]:
        """List available Anthropic models."""
        return self.MODELS

    async def health_check(self) -> ProviderHealth:
        """Check Anthropic API health."""
        try:
            start = time.time()
            # Anthropic doesn't have a simple health endpoint,
            # so we'll make a minimal request
            response = await self.client.post(
                "/v1/messages",
                json={
                    "model": "claude-3-haiku-20240307",
                    "max_tokens": 1,
                    "messages": [{"role": "user", "content": "Hi"}]
                }
            )
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
        """Build Anthropic-specific chat payload."""
        
        # Resolve model name
        model_name = request.model_name
        model_name = self.MODEL_ALIASES.get(model_name, model_name)
        
        # Extract system message (Anthropic handles it separately)
        system_content, messages = self._extract_system_message(request.messages)
        
        payload: Dict[str, Any] = {
            "model": model_name,
            "messages": self._convert_messages(messages),
            "max_tokens": request.max_tokens or 4096,  # Anthropic requires max_tokens
        }
        
        if system_content:
            payload["system"] = system_content
        
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        
        if request.tools:
            payload["tools"] = self._convert_tools(request)
        
        if request.tool_choice:
            payload["tool_choice"] = self._convert_tool_choice(request.tool_choice)
        
        return payload

    def _extract_system_message(
        self,
        messages: List[Message]
    ) -> tuple[Optional[str], List[Message]]:
        """
        Extract system message from message list.
        Anthropic requires system message as a separate parameter.
        """
        system_content = None
        filtered_messages = []
        
        for msg in messages:
            if msg.role == Role.SYSTEM:
                if isinstance(msg.content, str):
                    system_content = msg.content
                # Skip system messages in the regular list
            else:
                filtered_messages.append(msg)
        
        return system_content, filtered_messages

    def _convert_messages(
        self,
        messages: List[Message]
    ) -> List[Dict[str, Any]]:
        """Convert unified messages to Anthropic format."""
        result = []
        
        for msg in messages:
            anthropic_msg: Dict[str, Any] = {"role": msg.role.value}
            
            # Handle content
            if msg.content is not None:
                if isinstance(msg.content, str):
                    anthropic_msg["content"] = msg.content
                else:
                    # Multimodal content - Anthropic uses different format
                    anthropic_msg["content"] = []
                    for part in msg.content:
                        if hasattr(part, 'text'):
                            anthropic_msg["content"].append({
                                "type": "text",
                                "text": part.text
                            })
                        elif hasattr(part, 'image_url'):
                            # Anthropic expects base64 images
                            url = part.image_url.url
                            if url.startswith("data:"):
                                # Parse data URL
                                media_type = url.split(";")[0].split(":")[1]
                                data = url.split(",")[1]
                                anthropic_msg["content"].append({
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": data
                                    }
                                })
                            else:
                                # URL-based image (need to fetch and convert)
                                anthropic_msg["content"].append({
                                    "type": "image",
                                    "source": {
                                        "type": "url",
                                        "url": url
                                    }
                                })
            
            # Handle tool results
            if msg.role == Role.TOOL and msg.tool_call_id:
                anthropic_msg = {
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id,
                        "content": msg.content if isinstance(msg.content, str) else ""
                    }]
                }
            
            result.append(anthropic_msg)
        
        return result

    def _convert_tools(
        self,
        request: ChatCompletionRequest
    ) -> List[Dict[str, Any]]:
        """Convert tools to Anthropic format."""
        if not request.tools:
            return []
        
        return [
            {
                "name": tool.function.name,
                "description": tool.function.description,
                "input_schema": tool.function.parameters
            }
            for tool in request.tools
        ]

    def _convert_tool_choice(self, tool_choice: Any) -> Dict[str, Any]:
        """Convert tool choice to Anthropic format."""
        if isinstance(tool_choice, str):
            if tool_choice == "auto":
                return {"type": "auto"}
            elif tool_choice == "none":
                return {"type": "none"}
            elif tool_choice == "required":
                return {"type": "any"}
        return {
            "type": "tool",
            "name": tool_choice.function.name
        }

    def _parse_chat_response(
        self,
        data: Dict[str, Any],
        model: str,
        latency_ms: int
    ) -> ChatCompletionResponse:
        """Parse Anthropic response to unified format."""
        
        # Extract content and tool calls
        content_text = ""
        tool_calls = []
        
        for block in data.get("content", []):
            if block["type"] == "text":
                content_text += block["text"]
            elif block["type"] == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block["id"],
                        type="function",
                        function=FunctionCall(
                            name=block["name"],
                            arguments=json.dumps(block["input"])
                        )
                    )
                )
        
        # Map stop reason
        stop_reason = data.get("stop_reason", "end_turn")
        finish_reason_map = {
            "end_turn": FinishReason.STOP,
            "max_tokens": FinishReason.LENGTH,
            "tool_use": FinishReason.TOOL_CALLS,
            "stop_sequence": FinishReason.STOP,
        }
        finish_reason = finish_reason_map.get(stop_reason, FinishReason.STOP)
        
        # Build usage
        usage_data = data.get("usage", {})
        usage = Usage(
            prompt_tokens=usage_data.get("input_tokens", 0),
            completion_tokens=usage_data.get("output_tokens", 0)
        )
        
        return ChatCompletionResponse(
            id=data["id"],
            model=data["model"],
            provider=self.provider.value,
            choices=[
                Choice(
                    index=0,
                    message=Message(
                        role=Role.ASSISTANT,
                        content=content_text if content_text else None,
                        tool_calls=tool_calls if tool_calls else None
                    ),
                    finish_reason=finish_reason
                )
            ],
            usage=usage
        )

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
