"""
2api.ai - Provider Adapter Base

Abstract base class for AI provider adapters.
Each provider (OpenAI, Anthropic, Google) implements this interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional

from ..core.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ModelInfo,
    Provider,
    Usage,
)


@dataclass
class AdapterConfig:
    """Configuration for a provider adapter."""
    api_key: str
    base_url: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3


@dataclass
class ProviderHealth:
    """Health status of a provider."""
    provider: Provider
    is_healthy: bool
    avg_latency_ms: Optional[int] = None
    error_rate: Optional[float] = None
    last_error: Optional[str] = None


class BaseAdapter(ABC):
    """
    Abstract base class for AI provider adapters.
    
    Each provider adapter must implement:
    - chat_completion: Generate chat responses
    - chat_completion_stream: Generate streaming chat responses
    - embedding: Create embeddings
    - image_generation: Generate images
    - list_models: List available models
    - health_check: Check provider health
    
    The adapter is responsible for:
    1. Converting 2api.ai unified format → provider-specific format
    2. Making the API call to the provider
    3. Converting provider response → 2api.ai unified format
    4. Handling provider-specific errors
    """
    
    provider: Provider
    
    def __init__(self, config: AdapterConfig):
        self.config = config
        self._models_cache: Optional[List[ModelInfo]] = None

    @abstractmethod
    async def chat_completion(
        self,
        request: ChatCompletionRequest,
        request_id: str = ""
    ) -> ChatCompletionResponse:
        """
        Generate a chat completion.

        Args:
            request: Unified chat completion request
            request_id: Request ID for error tracking

        Returns:
            Unified chat completion response
        """
        pass

    @abstractmethod
    async def chat_completion_stream(
        self,
        request: ChatCompletionRequest,
        request_id: str = ""
    ) -> AsyncIterator[str]:
        """
        Generate a streaming chat completion.

        Args:
            request: Unified chat completion request
            request_id: Request ID for error tracking

        Yields:
            Server-sent events (SSE) formatted chunks.
            On error after content started: yields error chunk with partial_content.
        """
        pass

    @abstractmethod
    async def embedding(
        self,
        request: EmbeddingRequest,
        request_id: str = ""
    ) -> EmbeddingResponse:
        """
        Create embeddings for input text.

        Args:
            request: Unified embedding request
            request_id: Request ID for error tracking

        Returns:
            Unified embedding response
        """
        pass

    @abstractmethod
    async def image_generation(
        self,
        request: ImageGenerationRequest,
        request_id: str = ""
    ) -> ImageGenerationResponse:
        """
        Generate images from a prompt.

        Args:
            request: Unified image generation request
            request_id: Request ID for error tracking

        Returns:
            Unified image generation response
        """
        pass

    @abstractmethod
    def list_models(self) -> List[ModelInfo]:
        """
        List available models for this provider.
        
        Returns:
            List of model information
        """
        pass

    @abstractmethod
    async def health_check(self) -> ProviderHealth:
        """
        Check the health of this provider.
        
        Returns:
            Provider health status
        """
        pass

    def supports_capability(self, capability: str) -> bool:
        """Check if this provider supports a capability."""
        models = self.list_models()
        return any(model.supports(capability) for model in models)

    def get_model(self, model_name: str) -> Optional[ModelInfo]:
        """Get model info by name."""
        models = self.list_models()
        for model in models:
            if model.name == model_name or model.id.endswith(f"/{model_name}"):
                return model
        return None

    def calculate_cost(
        self,
        model_name: str,
        usage: Usage
    ) -> float:
        """Calculate cost for a request."""
        model = self.get_model(model_name)
        if model:
            return model.calculate_cost(
                usage.prompt_tokens,
                usage.completion_tokens
            )
        return 0.0

    # ============================================================
    # Helper methods for subclasses
    # ============================================================

    def _normalize_messages(
        self,
        request: ChatCompletionRequest
    ) -> List[Dict[str, Any]]:
        """
        Convert unified messages to provider-specific format.
        Override in subclass if needed.
        """
        from ..core.models import message_to_dict
        return [message_to_dict(msg) for msg in request.messages]

    def _normalize_tools(
        self,
        request: ChatCompletionRequest
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Convert unified tools to provider-specific format.
        Override in subclass if needed.
        """
        if not request.tools:
            return None
        
        return [
            {
                "type": tool.type,
                "function": {
                    "name": tool.function.name,
                    "description": tool.function.description,
                    "parameters": tool.function.parameters
                }
            }
            for tool in request.tools
        ]
