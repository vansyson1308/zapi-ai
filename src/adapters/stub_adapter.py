"""
2api.ai - Stub Provider Adapter

Deterministic in-process adapter used for smoke/integration testing.
No network calls, no external provider keys required.
"""

import json
import time
from typing import AsyncIterator, List

from .base import AdapterConfig, BaseAdapter, ProviderHealth
from ..core.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    FinishReason,
    ImageData,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ModelInfo,
    ModelPricing,
    Provider,
    Usage,
)


class StubAdapter(BaseAdapter):
    """Deterministic adapter for tests/smoke checks."""

    provider = Provider.OPENAI

    MODELS = [
        ModelInfo(
            id="openai/gpt-4o-mini",
            provider=Provider.OPENAI,
            name="gpt-4o-mini",
            capabilities=["chat", "tools"],
            context_window=128000,
            max_output_tokens=4096,
            pricing=ModelPricing(input_per_1m_tokens=0.15, output_per_1m_tokens=0.60),
        ),
        ModelInfo(
            id="openai/text-embedding-3-small",
            provider=Provider.OPENAI,
            name="text-embedding-3-small",
            capabilities=["embedding"],
            context_window=8192,
            max_output_tokens=0,
            pricing=ModelPricing(input_per_1m_tokens=0.02, output_per_1m_tokens=0.0),
        ),
    ]

    def __init__(self, config: AdapterConfig):
        super().__init__(config)

    async def chat_completion(
        self,
        request: ChatCompletionRequest,
        request_id: str = "",
    ) -> ChatCompletionResponse:
        usage = Usage(prompt_tokens=8, completion_tokens=6)
        model_name = request.model_name or "gpt-4o-mini"
        return ChatCompletionResponse.create(
            content="stub: deterministic response",
            model=model_name,
            provider=self.provider.value,
            usage=usage,
            finish_reason=FinishReason.STOP,
        )

    async def chat_completion_stream(
        self,
        request: ChatCompletionRequest,
        request_id: str = "",
    ) -> AsyncIterator[str]:
        stream_id = "chatcmpl-stub123"
        created = int(time.time())

        first = {
            "id": stream_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model_name or "gpt-4o-mini",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "stub:"}, "finish_reason": None}],
        }
        second = {
            "id": stream_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model_name or "gpt-4o-mini",
            "choices": [{"index": 0, "delta": {"content": " deterministic stream"}, "finish_reason": None}],
        }
        final = {
            "id": stream_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model_name or "gpt-4o-mini",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }

        yield f"data: {json.dumps(first)}\n\n"
        yield f"data: {json.dumps(second)}\n\n"
        yield f"data: {json.dumps(final)}\n\n"
        yield "data: [DONE]\n\n"

    async def embedding(
        self,
        request: EmbeddingRequest,
        request_id: str = "",
    ) -> EmbeddingResponse:
        return EmbeddingResponse(
            data=[EmbeddingData(embedding=[0.01, 0.02, 0.03], index=0)],
            model=request.model,
            usage=Usage(prompt_tokens=3, completion_tokens=0),
        )

    async def image_generation(
        self,
        request: ImageGenerationRequest,
        request_id: str = "",
    ) -> ImageGenerationResponse:
        return ImageGenerationResponse(
            data=[ImageData(url="https://example.test/stub.png", revised_prompt=request.prompt)]
        )

    def list_models(self) -> List[ModelInfo]:
        return self.MODELS

    async def health_check(self) -> ProviderHealth:
        return ProviderHealth(provider=self.provider, is_healthy=True, avg_latency_ms=5)

    async def close(self):
        return
