"""
OpenRouter passthrough adapter.

Why this adapter exists:
  AgentOps positions itself as the OPERATIONS LAYER on top of any LLM
  gateway, including OpenRouter itself. This adapter lets a tenant route
  their requests to OpenRouter while still benefiting from Guardian
  (kill switches, cost attribution, PII redaction, audit logs).

Implementation:
  OpenRouter exposes an OpenAI-compatible Chat Completions endpoint. We
  delegate the heavy lifting (request shape, streaming, error mapping) to
  OpenAIAdapter by overriding `DEFAULT_BASE_URL` and the model catalog.

  Model id format: `openrouter/<provider>/<model>` (e.g.
  `openrouter/anthropic/claude-3.5-sonnet`). We strip the leading
  `openrouter/` before forwarding so OpenRouter receives the form it
  expects (`anthropic/claude-3.5-sonnet`).
"""

from __future__ import annotations

from typing import AsyncIterator, List, Optional

import httpx

from .base import AdapterConfig, ProviderHealth
from .openai_adapter import OpenAIAdapter
from ..core.errors import handle_openai_error
from ..core.models import (
    ChatCompletionRequest,
    EmbeddingRequest,
    EmbeddingResponse,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ModelInfo,
    ModelPricing,
    Provider,
)


class OpenRouterAdapter(OpenAIAdapter):
    """
    OpenAI-compatible adapter pointed at OpenRouter.

    The model catalog here is intentionally curated (top ~10 production-grade
    models). OpenRouter has 400+; we don't try to mirror them all because
    most are noise for production and the routing strategies here would
    have to score 400 candidates per request (perf hit).
    """

    provider = Provider.OPENROUTER
    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

    # Curated catalog. Pricing matches OpenRouter list price as of May 2026.
    # Source: https://openrouter.ai/models — keep this short and update when
    # we onboard a new design partner who needs a specific model.
    MODELS: List[ModelInfo] = [
        ModelInfo(
            id="openrouter/anthropic/claude-3.5-sonnet",
            provider=Provider.OPENROUTER,
            name="anthropic/claude-3.5-sonnet",
            capabilities=["chat", "vision", "tools"],
            context_window=200_000,
            max_output_tokens=8192,
            pricing=ModelPricing(input_per_1m_tokens=3.00, output_per_1m_tokens=15.00),
        ),
        ModelInfo(
            id="openrouter/openai/gpt-4o",
            provider=Provider.OPENROUTER,
            name="openai/gpt-4o",
            capabilities=["chat", "vision", "tools"],
            context_window=128_000,
            max_output_tokens=16384,
            pricing=ModelPricing(input_per_1m_tokens=2.50, output_per_1m_tokens=10.00),
        ),
        ModelInfo(
            id="openrouter/openai/gpt-4o-mini",
            provider=Provider.OPENROUTER,
            name="openai/gpt-4o-mini",
            capabilities=["chat", "vision", "tools"],
            context_window=128_000,
            max_output_tokens=16384,
            pricing=ModelPricing(input_per_1m_tokens=0.15, output_per_1m_tokens=0.60),
        ),
        ModelInfo(
            id="openrouter/google/gemini-2.0-flash",
            provider=Provider.OPENROUTER,
            name="google/gemini-2.0-flash",
            capabilities=["chat", "vision", "tools"],
            context_window=1_000_000,
            max_output_tokens=8192,
            pricing=ModelPricing(input_per_1m_tokens=0.10, output_per_1m_tokens=0.40),
        ),
        ModelInfo(
            id="openrouter/meta-llama/llama-3.3-70b-instruct",
            provider=Provider.OPENROUTER,
            name="meta-llama/llama-3.3-70b-instruct",
            capabilities=["chat", "tools"],
            context_window=131_072,
            max_output_tokens=4096,
            pricing=ModelPricing(input_per_1m_tokens=0.23, output_per_1m_tokens=0.40),
        ),
        ModelInfo(
            id="openrouter/deepseek/deepseek-v3",
            provider=Provider.OPENROUTER,
            name="deepseek/deepseek-v3",
            capabilities=["chat", "tools"],
            context_window=64_000,
            max_output_tokens=8192,
            pricing=ModelPricing(input_per_1m_tokens=0.27, output_per_1m_tokens=1.10),
        ),
        ModelInfo(
            id="openrouter/qwen/qwen-2.5-72b-instruct",
            provider=Provider.OPENROUTER,
            name="qwen/qwen-2.5-72b-instruct",
            capabilities=["chat", "tools"],
            context_window=131_072,
            max_output_tokens=4096,
            pricing=ModelPricing(input_per_1m_tokens=0.40, output_per_1m_tokens=0.40),
        ),
    ]

    def __init__(self, config: AdapterConfig):
        # OpenAIAdapter.__init__ already wires up self.client with proper headers
        # and base_url override. We additionally set OpenRouter-recommended
        # headers (HTTP-Referer + X-Title) so the upstream attribution
        # dashboard shows our app correctly.
        super().__init__(config)
        # Re-create client with the extra headers OpenRouter expects.
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
                # These are advisory, not required.
                "HTTP-Referer": "https://2api.ai",
                "X-Title": "2api.ai (AgentOps)",
            },
            timeout=config.timeout,
        )

    # ------------------------------------------------------------------
    # model name plumbing
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_provider_prefix(model: str) -> str:
        """`openrouter/<provider>/<model>` → `<provider>/<model>`."""
        if model.startswith("openrouter/"):
            return model[len("openrouter/"):]
        return model

    def _build_chat_payload(self, request: ChatCompletionRequest):  # type: ignore[override]
        # Reuse the parent payload shaping then rewrite the model field.
        payload = super()._build_chat_payload(request)
        payload["model"] = self._strip_provider_prefix(payload["model"])
        return payload

    # ------------------------------------------------------------------
    # we override embedding/image to match the same model rewrite, since
    # OpenRouter currently proxies embeddings via OpenAI-style format too.
    # ------------------------------------------------------------------

    async def embedding(
        self,
        request: EmbeddingRequest,
        request_id: str = "",
    ) -> EmbeddingResponse:
        # Rewrite model id then delegate
        if "/" in request.model:
            request = EmbeddingRequest(
                model=self._strip_provider_prefix(request.model),
                input=request.input,
                encoding_format=request.encoding_format,
                dimensions=request.dimensions,
            )
        return await super().embedding(request, request_id)

    async def image_generation(
        self,
        request: ImageGenerationRequest,
        request_id: str = "",
    ) -> ImageGenerationResponse:
        if "/" in request.model:
            request = ImageGenerationRequest(
                model=self._strip_provider_prefix(request.model),
                prompt=request.prompt,
                n=request.n,
                size=request.size,
                quality=request.quality,
                response_format=request.response_format,
            )
        return await super().image_generation(request, request_id)

    async def chat_completion_stream(
        self,
        request: ChatCompletionRequest,
        request_id: str = "",
    ) -> AsyncIterator[str]:
        # Same rewrite for streaming
        async for chunk in super().chat_completion_stream(request, request_id):
            yield chunk

    async def health_check(self) -> ProviderHealth:
        """Lightweight health check — call /models endpoint."""
        try:
            response = await self.client.get("/models", timeout=5.0)
            healthy = response.status_code == 200
            return ProviderHealth(provider=self.provider, is_healthy=healthy)
        except Exception as exc:
            return ProviderHealth(
                provider=self.provider,
                is_healthy=False,
                last_error=str(exc),
            )
