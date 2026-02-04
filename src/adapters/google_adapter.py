"""
2api.ai - Google Gemini Provider Adapter

Adapter for Google's Gemini API (Gemini 1.5 Pro, Gemini 1.5 Flash, etc.)
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


class GoogleAdapter(BaseAdapter):
    """
    Adapter for Google Gemini API.
    
    Supports:
    - Chat completions (Gemini 1.5 Pro, Gemini 1.5 Flash, Gemini 1.0 Pro)
    - Vision (all Gemini 1.5 models)
    - Embeddings (text-embedding-004)
    - Tool/Function calling
    - Streaming
    
    Note: Image generation is available via Imagen API (not included here).
    """
    
    provider = Provider.GOOGLE
    DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    # Model catalog with pricing (as of 2024)
    MODELS: List[ModelInfo] = [
        ModelInfo(
            id="google/gemini-1.5-pro",
            provider=Provider.GOOGLE,
            name="gemini-1.5-pro",
            capabilities=["chat", "vision", "tools"],
            context_window=2000000,  # 2M tokens!
            max_output_tokens=8192,
            pricing=ModelPricing(input_per_1m_tokens=1.25, output_per_1m_tokens=5.00)
        ),
        ModelInfo(
            id="google/gemini-1.5-flash",
            provider=Provider.GOOGLE,
            name="gemini-1.5-flash",
            capabilities=["chat", "vision", "tools"],
            context_window=1000000,  # 1M tokens
            max_output_tokens=8192,
            pricing=ModelPricing(input_per_1m_tokens=0.075, output_per_1m_tokens=0.30)
        ),
        ModelInfo(
            id="google/gemini-1.5-flash-8b",
            provider=Provider.GOOGLE,
            name="gemini-1.5-flash-8b",
            capabilities=["chat", "vision", "tools"],
            context_window=1000000,
            max_output_tokens=8192,
            pricing=ModelPricing(input_per_1m_tokens=0.0375, output_per_1m_tokens=0.15)
        ),
        ModelInfo(
            id="google/gemini-1.0-pro",
            provider=Provider.GOOGLE,
            name="gemini-1.0-pro",
            capabilities=["chat", "tools"],
            context_window=32760,
            max_output_tokens=8192,
            pricing=ModelPricing(input_per_1m_tokens=0.50, output_per_1m_tokens=1.50)
        ),
        ModelInfo(
            id="google/text-embedding-004",
            provider=Provider.GOOGLE,
            name="text-embedding-004",
            capabilities=["embedding"],
            context_window=2048,
            max_output_tokens=0,
            pricing=ModelPricing(input_per_1m_tokens=0.00, output_per_1m_tokens=0)  # Free tier available
        ),
    ]

    def __init__(self, config: AdapterConfig):
        super().__init__(config)
        self.api_key = config.api_key
        self.base_url = config.base_url or self.DEFAULT_BASE_URL
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=config.timeout
        )

    async def chat_completion(
        self,
        request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Generate a chat completion using Gemini."""
        
        model_name = self._resolve_model_name(request.model_name)
        payload = self._build_chat_payload(request)
        
        url = f"/models/{model_name}:generateContent?key={self.api_key}"
        
        start_time = time.time()
        
        try:
            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as e:
            raise self._handle_error(e)
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        return self._parse_chat_response(data, model_name, latency_ms)

    async def chat_completion_stream(
        self,
        request: ChatCompletionRequest
    ) -> AsyncIterator[str]:
        """Generate a streaming chat completion."""
        
        model_name = self._resolve_model_name(request.model_name)
        payload = self._build_chat_payload(request)
        
        url = f"/models/{model_name}:streamGenerateContent?key={self.api_key}&alt=sse"
        
        async with self.client.stream(
            "POST",
            url,
            json=payload
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "":
                        continue
                    
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    
                    # Convert to OpenAI-compatible format
                    candidates = data.get("candidates", [])
                    if candidates:
                        content = candidates[0].get("content", {})
                        parts = content.get("parts", [])
                        if parts and "text" in parts[0]:
                            text = parts[0]["text"]
                            chunk = {
                                "id": f"chatcmpl-{int(time.time())}",
                                "object": "chat.completion.chunk",
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": text},
                                    "finish_reason": None
                                }]
                            }
                            yield f"data: {json.dumps(chunk)}\n\n"
                        
                        # Check for finish
                        finish_reason = candidates[0].get("finishReason")
                        if finish_reason:
                            chunk = {
                                "id": f"chatcmpl-{int(time.time())}",
                                "object": "chat.completion.chunk",
                                "choices": [{
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": "stop"
                                }]
                            }
                            yield f"data: {json.dumps(chunk)}\n\n"
            
            yield "data: [DONE]\n\n"

    async def embedding(
        self,
        request: EmbeddingRequest
    ) -> EmbeddingResponse:
        """Create embeddings using Google's embedding model."""
        
        model_name = self._resolve_model_name(
            request.model.split("/")[-1] if "/" in request.model else request.model
        )
        
        # Handle single or batch input
        inputs = request.input if isinstance(request.input, list) else [request.input]
        
        embeddings = []
        total_tokens = 0
        
        for i, text in enumerate(inputs):
            url = f"/models/{model_name}:embedContent?key={self.api_key}"
            payload = {
                "model": f"models/{model_name}",
                "content": {
                    "parts": [{"text": text}]
                }
            }
            
            try:
                response = await self.client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
            except httpx.HTTPStatusError as e:
                raise self._handle_error(e)
            
            embedding_values = data.get("embedding", {}).get("values", [])
            embeddings.append(
                EmbeddingData(
                    embedding=embedding_values,
                    index=i
                )
            )
            # Estimate tokens (rough approximation)
            total_tokens += len(text.split()) * 1.3
        
        return EmbeddingResponse(
            data=embeddings,
            model=model_name,
            usage=Usage(
                prompt_tokens=int(total_tokens),
                completion_tokens=0
            )
        )

    async def image_generation(
        self,
        request: ImageGenerationRequest
    ) -> ImageGenerationResponse:
        """
        Generate images - Limited support via Gemini.
        
        Note: For full image generation, use OpenAI DALL-E or Google Imagen.
        """
        raise NotImplementedError(
            "Google Gemini does not directly support image generation. "
            "Use OpenAI (DALL-E) for image generation requests."
        )

    def list_models(self) -> List[ModelInfo]:
        """List available Google models."""
        return self.MODELS

    async def health_check(self) -> ProviderHealth:
        """Check Google API health."""
        try:
            start = time.time()
            url = f"/models?key={self.api_key}"
            response = await self.client.get(url)
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

    def _resolve_model_name(self, model: str) -> str:
        """Resolve model aliases to actual model names."""
        aliases = {
            "gemini-pro": "gemini-1.0-pro",
            "gemini-1.5": "gemini-1.5-pro",
            "gemini-flash": "gemini-1.5-flash",
        }
        return aliases.get(model, model)

    def _build_chat_payload(
        self,
        request: ChatCompletionRequest
    ) -> Dict[str, Any]:
        """Build Gemini-specific chat payload."""
        
        # Convert messages to Gemini format
        contents = self._convert_messages(request.messages)
        
        payload: Dict[str, Any] = {
            "contents": contents,
        }
        
        # Generation config
        generation_config: Dict[str, Any] = {}
        
        if request.temperature is not None:
            generation_config["temperature"] = request.temperature
        if request.max_tokens is not None:
            generation_config["maxOutputTokens"] = request.max_tokens
        
        if generation_config:
            payload["generationConfig"] = generation_config
        
        # Tools
        if request.tools:
            payload["tools"] = [{"functionDeclarations": self._convert_tools(request)}]
        
        # System instruction (if first message is system)
        system_instruction = self._extract_system_instruction(request.messages)
        if system_instruction:
            payload["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }
        
        return payload

    def _extract_system_instruction(
        self,
        messages: List[Message]
    ) -> Optional[str]:
        """Extract system instruction from messages."""
        for msg in messages:
            if msg.role == Role.SYSTEM and isinstance(msg.content, str):
                return msg.content
        return None

    def _convert_messages(
        self,
        messages: List[Message]
    ) -> List[Dict[str, Any]]:
        """Convert unified messages to Gemini format."""
        result = []
        
        for msg in messages:
            # Skip system messages (handled separately)
            if msg.role == Role.SYSTEM:
                continue
            
            # Map roles
            role = "user" if msg.role in [Role.USER, Role.TOOL] else "model"
            
            parts = []
            
            # Handle content
            if msg.content is not None:
                if isinstance(msg.content, str):
                    parts.append({"text": msg.content})
                else:
                    # Multimodal content
                    for part in msg.content:
                        if hasattr(part, 'text'):
                            parts.append({"text": part.text})
                        elif hasattr(part, 'image_url'):
                            url = part.image_url.url
                            if url.startswith("data:"):
                                # Parse data URL
                                media_type = url.split(";")[0].split(":")[1]
                                data = url.split(",")[1]
                                parts.append({
                                    "inlineData": {
                                        "mimeType": media_type,
                                        "data": data
                                    }
                                })
                            else:
                                parts.append({
                                    "fileData": {
                                        "fileUri": url,
                                        "mimeType": "image/jpeg"  # Default
                                    }
                                })
            
            # Handle tool calls in assistant messages
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    parts.append({
                        "functionCall": {
                            "name": tc.function.name,
                            "args": json.loads(tc.function.arguments)
                        }
                    })
            
            # Handle tool results
            if msg.role == Role.TOOL and msg.tool_call_id:
                parts = [{
                    "functionResponse": {
                        "name": msg.tool_call_id,  # Gemini uses name, not ID
                        "response": {
                            "content": msg.content if isinstance(msg.content, str) else ""
                        }
                    }
                }]
            
            if parts:
                result.append({
                    "role": role,
                    "parts": parts
                })
        
        return result

    def _convert_tools(
        self,
        request: ChatCompletionRequest
    ) -> List[Dict[str, Any]]:
        """Convert tools to Gemini format."""
        if not request.tools:
            return []
        
        return [
            {
                "name": tool.function.name,
                "description": tool.function.description,
                "parameters": tool.function.parameters
            }
            for tool in request.tools
        ]

    def _parse_chat_response(
        self,
        data: Dict[str, Any],
        model: str,
        latency_ms: int
    ) -> ChatCompletionResponse:
        """Parse Gemini response to unified format."""
        
        candidates = data.get("candidates", [])
        if not candidates:
            return ChatCompletionResponse(
                id=f"chatcmpl-{int(time.time())}",
                model=model,
                provider=self.provider.value,
                choices=[
                    Choice(
                        index=0,
                        message=Message(role=Role.ASSISTANT, content=""),
                        finish_reason=FinishReason.ERROR
                    )
                ],
                usage=Usage()
            )
        
        candidate = candidates[0]
        content = candidate.get("content", {})
        parts = content.get("parts", [])
        
        # Extract text and tool calls
        text_content = ""
        tool_calls = []
        
        for part in parts:
            if "text" in part:
                text_content += part["text"]
            elif "functionCall" in part:
                fc = part["functionCall"]
                tool_calls.append(
                    ToolCall(
                        id=f"call_{int(time.time())}_{fc['name']}",
                        type="function",
                        function=FunctionCall(
                            name=fc["name"],
                            arguments=json.dumps(fc.get("args", {}))
                        )
                    )
                )
        
        # Map finish reason
        finish_reason_map = {
            "STOP": FinishReason.STOP,
            "MAX_TOKENS": FinishReason.LENGTH,
            "SAFETY": FinishReason.CONTENT_FILTER,
            "RECITATION": FinishReason.CONTENT_FILTER,
        }
        gemini_finish = candidate.get("finishReason", "STOP")
        finish_reason = finish_reason_map.get(gemini_finish, FinishReason.STOP)
        
        if tool_calls:
            finish_reason = FinishReason.TOOL_CALLS
        
        # Parse usage
        usage_metadata = data.get("usageMetadata", {})
        usage = Usage(
            prompt_tokens=usage_metadata.get("promptTokenCount", 0),
            completion_tokens=usage_metadata.get("candidatesTokenCount", 0),
            total_tokens=usage_metadata.get("totalTokenCount", 0)
        )
        
        return ChatCompletionResponse(
            id=f"chatcmpl-{int(time.time())}",
            model=model,
            provider=self.provider.value,
            choices=[
                Choice(
                    index=0,
                    message=Message(
                        role=Role.ASSISTANT,
                        content=text_content if text_content else None,
                        tool_calls=tool_calls if tool_calls else None
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
                    code=str(error_info.get("code", "unknown")),
                    message=error_info.get("message", str(error)),
                    type="provider_error",
                    provider="google"
                )
            )
        except Exception:
            return ProviderError(
                APIError(
                    code="unknown",
                    message=str(error),
                    type="provider_error",
                    provider="google"
                )
            )

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
