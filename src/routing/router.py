"""
2api.ai - Router Service

Intelligent routing between AI providers with:
- Fallback handling
- Load balancing
- Cost optimization
- Latency optimization
"""

import asyncio
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from ..core.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ModelInfo,
    Provider,
    RoutingConfig,
    RoutingDecision,
    RoutingStrategy,
    TwoApiMetadata,
    TwoApiError,
    Usage,
)
from ..adapters.base import BaseAdapter, ProviderHealth


@dataclass
class ProviderStats:
    """Real-time statistics for a provider."""
    provider: Provider
    total_requests: int = 0
    failed_requests: int = 0
    total_latency_ms: int = 0
    last_error: Optional[str] = None
    last_success_time: Optional[float] = None
    is_healthy: bool = True
    
    @property
    def avg_latency_ms(self) -> int:
        if self.total_requests == 0:
            return 0
        return self.total_latency_ms // self.total_requests
    
    @property
    def error_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests
    
    def record_success(self, latency_ms: int):
        self.total_requests += 1
        self.total_latency_ms += latency_ms
        self.last_success_time = time.time()
        self.is_healthy = True
    
    def record_failure(self, error: str):
        self.total_requests += 1
        self.failed_requests += 1
        self.last_error = error
        # Mark as unhealthy if error rate > 50%
        if self.error_rate > 0.5:
            self.is_healthy = False


@dataclass
class RoutingResult:
    """Result of routing decision."""
    selected_provider: Provider
    selected_model: str
    adapter: BaseAdapter
    decision: RoutingDecision


class Router:
    """
    Intelligent router for AI requests.
    
    Routing strategies:
    - COST: Choose the cheapest provider/model
    - LATENCY: Choose the fastest provider/model
    - QUALITY: Choose the highest quality model
    
    Features:
    - Automatic fallback on provider failure
    - Load balancing across providers
    - Health tracking
    - Cost/latency optimization
    """
    
    def __init__(self, adapters: Dict[Provider, BaseAdapter]):
        """
        Initialize router with provider adapters.
        
        Args:
            adapters: Dictionary mapping Provider enum to adapter instance
        """
        self.adapters = adapters
        self.stats: Dict[Provider, ProviderStats] = {
            provider: ProviderStats(provider=provider)
            for provider in adapters.keys()
        }
        
        # Model registry (built from all adapters)
        self._model_registry: Dict[str, Tuple[Provider, ModelInfo]] = {}
        self._build_model_registry()
    
    def _build_model_registry(self):
        """Build a unified model registry from all adapters."""
        for provider, adapter in self.adapters.items():
            for model in adapter.list_models():
                self._model_registry[model.id] = (provider, model)
                # Also register by short name
                self._model_registry[f"{provider.value}/{model.name}"] = (provider, model)
    
    async def route_chat(
        self,
        request: ChatCompletionRequest
    ) -> Tuple[ChatCompletionResponse, RoutingDecision]:
        """
        Route a chat completion request.
        
        Args:
            request: Chat completion request
            
        Returns:
            Tuple of (response, routing_decision)
        """
        start_time = time.time()
        
        # Determine routing
        routing_result = self._select_provider(
            request=request,
            capability="chat"
        )
        
        # Try primary provider
        try:
            response = await routing_result.adapter.chat_completion(request)
            latency_ms = int((time.time() - start_time) * 1000)
            
            # Record success
            self.stats[routing_result.selected_provider].record_success(latency_ms)
            
            # Calculate cost
            cost = routing_result.adapter.calculate_cost(
                routing_result.selected_model,
                response.usage
            )
            
            # Add 2api metadata
            response._2api = TwoApiMetadata(
                request_id=response.id,
                latency_ms=latency_ms,
                cost_usd=cost,
                routing_decision=routing_result.decision
            )
            
            return response, routing_result.decision
            
        except Exception as e:
            # Record failure
            self.stats[routing_result.selected_provider].record_failure(str(e))
            
            # Try fallback if configured
            if request.routing and request.routing.fallback:
                return await self._try_fallback(
                    request=request,
                    failed_provider=routing_result.selected_provider,
                    fallback_chain=request.routing.fallback,
                    original_error=e
                )
            
            # No fallback available
            raise
    
    async def route_embedding(
        self,
        request: EmbeddingRequest
    ) -> Tuple[EmbeddingResponse, RoutingDecision]:
        """Route an embedding request."""
        
        # Only OpenAI and Google support embeddings
        routing_result = self._select_provider(
            request=None,
            capability="embedding",
            model_hint=request.model
        )
        
        start_time = time.time()
        
        try:
            response = await routing_result.adapter.embedding(request)
            latency_ms = int((time.time() - start_time) * 1000)
            
            self.stats[routing_result.selected_provider].record_success(latency_ms)
            
            decision = routing_result.decision
            return response, decision
            
        except Exception as e:
            self.stats[routing_result.selected_provider].record_failure(str(e))
            raise
    
    async def route_image(
        self,
        request: ImageGenerationRequest
    ) -> Tuple[ImageGenerationResponse, RoutingDecision]:
        """Route an image generation request."""
        
        # Only OpenAI supports image generation
        routing_result = self._select_provider(
            request=None,
            capability="image",
            model_hint=request.model
        )
        
        start_time = time.time()
        
        try:
            response = await routing_result.adapter.image_generation(request)
            latency_ms = int((time.time() - start_time) * 1000)
            
            self.stats[routing_result.selected_provider].record_success(latency_ms)
            
            return response, routing_result.decision
            
        except Exception as e:
            self.stats[routing_result.selected_provider].record_failure(str(e))
            raise
    
    def _select_provider(
        self,
        request: Optional[ChatCompletionRequest],
        capability: str,
        model_hint: Optional[str] = None
    ) -> RoutingResult:
        """
        Select the best provider for a request.
        
        Selection logic:
        1. If explicit model specified → use that provider
        2. If auto routing → apply strategy
        3. Otherwise → use default
        """
        candidates_evaluated = []
        
        # Get model from request or hint
        model_str = None
        if request:
            model_str = request.model
        elif model_hint:
            model_str = model_hint
        
        # Case 1: Explicit model specified
        if model_str and model_str.lower() != "auto" and "/" in model_str:
            provider_name = model_str.split("/")[0]
            model_name = model_str.split("/")[1]
            
            try:
                provider = Provider(provider_name)
                if provider in self.adapters:
                    return RoutingResult(
                        selected_provider=provider,
                        selected_model=model_name,
                        adapter=self.adapters[provider],
                        decision=RoutingDecision(
                            strategy_used="explicit",
                            candidates_evaluated=[model_str],
                            fallback_used=False
                        )
                    )
            except ValueError:
                pass
        
        # Case 2: Auto routing with strategy
        strategy = RoutingStrategy.COST  # Default
        if request and request.routing and request.routing.strategy:
            strategy = request.routing.strategy
        
        # Find candidates that support the capability
        candidates: List[Tuple[Provider, ModelInfo]] = []
        for model_id, (provider, model_info) in self._model_registry.items():
            if model_info.supports(capability) and self.stats[provider].is_healthy:
                candidates.append((provider, model_info))
                candidates_evaluated.append(model_id)
        
        if not candidates:
            # Fallback to any available provider
            for provider in self.adapters.keys():
                if self.stats[provider].is_healthy:
                    models = self.adapters[provider].list_models()
                    if models:
                        return RoutingResult(
                            selected_provider=provider,
                            selected_model=models[0].name,
                            adapter=self.adapters[provider],
                            decision=RoutingDecision(
                                strategy_used="fallback_any",
                                candidates_evaluated=[],
                                fallback_used=True
                            )
                        )
            raise RuntimeError("No healthy providers available")
        
        # Apply strategy
        selected = self._apply_strategy(candidates, strategy)
        
        return RoutingResult(
            selected_provider=selected[0],
            selected_model=selected[1].name,
            adapter=self.adapters[selected[0]],
            decision=RoutingDecision(
                strategy_used=strategy.value,
                candidates_evaluated=candidates_evaluated[:5],  # Limit for response size
                fallback_used=False
            )
        )
    
    def _apply_strategy(
        self,
        candidates: List[Tuple[Provider, ModelInfo]],
        strategy: RoutingStrategy
    ) -> Tuple[Provider, ModelInfo]:
        """Apply routing strategy to select best candidate."""
        
        if strategy == RoutingStrategy.COST:
            # Sort by input token price (cheapest first)
            candidates.sort(key=lambda x: x[1].pricing.input_per_1m_tokens)
            return candidates[0]
        
        elif strategy == RoutingStrategy.LATENCY:
            # Sort by average latency (fastest first)
            candidates.sort(key=lambda x: self.stats[x[0]].avg_latency_ms)
            return candidates[0]
        
        elif strategy == RoutingStrategy.QUALITY:
            # Use a quality score (higher context = better, higher price = better quality assumption)
            def quality_score(c: Tuple[Provider, ModelInfo]) -> float:
                _, model = c
                return model.pricing.output_per_1m_tokens * model.context_window
            
            candidates.sort(key=quality_score, reverse=True)
            return candidates[0]
        
        # Default: random selection for load balancing
        return random.choice(candidates)
    
    async def _try_fallback(
        self,
        request: ChatCompletionRequest,
        failed_provider: Provider,
        fallback_chain: List[str],
        original_error: Exception
    ) -> Tuple[ChatCompletionResponse, RoutingDecision]:
        """Try fallback providers in order."""
        
        for fallback_model in fallback_chain:
            if "/" in fallback_model:
                provider_name = fallback_model.split("/")[0]
                model_name = fallback_model.split("/")[1]
            else:
                # Assume it's just a provider name
                provider_name = fallback_model
                model_name = None
            
            try:
                provider = Provider(provider_name)
            except ValueError:
                continue
            
            if provider not in self.adapters or provider == failed_provider:
                continue
            
            if not self.stats[provider].is_healthy:
                continue
            
            # Create modified request with fallback model
            fallback_request = ChatCompletionRequest(
                model=fallback_model if model_name else f"{provider_name}/auto",
                messages=request.messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=request.stream,
                tools=request.tools,
                tool_choice=request.tool_choice,
                metadata=request.metadata
            )
            
            try:
                start_time = time.time()
                adapter = self.adapters[provider]
                
                # Get actual model if not specified
                if not model_name:
                    models = adapter.list_models()
                    chat_models = [m for m in models if m.supports("chat")]
                    if chat_models:
                        model_name = chat_models[0].name
                
                response = await adapter.chat_completion(fallback_request)
                latency_ms = int((time.time() - start_time) * 1000)
                
                self.stats[provider].record_success(latency_ms)
                
                cost = adapter.calculate_cost(model_name or "", response.usage)
                
                decision = RoutingDecision(
                    strategy_used="fallback",
                    candidates_evaluated=fallback_chain,
                    fallback_used=True
                )
                
                response._2api = TwoApiMetadata(
                    request_id=response.id,
                    latency_ms=latency_ms,
                    cost_usd=cost,
                    routing_decision=decision
                )
                
                return response, decision
                
            except Exception as e:
                self.stats[provider].record_failure(str(e))
                continue
        
        # All fallbacks failed
        raise original_error
    
    async def check_all_health(self) -> Dict[Provider, ProviderHealth]:
        """Check health of all providers."""
        results = {}
        
        tasks = [
            adapter.health_check()
            for adapter in self.adapters.values()
        ]
        
        health_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for adapter, result in zip(self.adapters.values(), health_results):
            if isinstance(result, Exception):
                results[adapter.provider] = ProviderHealth(
                    provider=adapter.provider,
                    is_healthy=False,
                    last_error=str(result)
                )
                self.stats[adapter.provider].is_healthy = False
            else:
                results[adapter.provider] = result
                self.stats[adapter.provider].is_healthy = result.is_healthy
        
        return results
    
    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get routing statistics for all providers."""
        return {
            provider.value: {
                "total_requests": stats.total_requests,
                "failed_requests": stats.failed_requests,
                "error_rate": stats.error_rate,
                "avg_latency_ms": stats.avg_latency_ms,
                "is_healthy": stats.is_healthy,
                "last_error": stats.last_error
            }
            for provider, stats in self.stats.items()
        }
    
    def list_all_models(self) -> List[ModelInfo]:
        """List all available models across all providers."""
        models = []
        seen_ids = set()
        
        for provider, adapter in self.adapters.items():
            for model in adapter.list_models():
                if model.id not in seen_ids:
                    models.append(model)
                    seen_ids.add(model.id)
        
        return models
