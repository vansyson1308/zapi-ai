"""
2api.ai - Router Service

Intelligent routing between AI providers with:
- Circuit breaker pattern for fault tolerance
- Multiple routing strategies (cost, latency, quality)
- Fallback chains with semantic drift protection
- Real-time health tracking and scoring
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, AsyncIterator

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
    Usage,
)
from ..core.errors import (
    TwoApiException,
    InfraError,
    AllProvidersFailedError,
    StreamInterruptedError,
    ErrorDetails,
    ErrorType,
    create_stream_error_chunk,
)
from ..adapters.base import BaseAdapter, ProviderHealth

from .circuit_breaker import CircuitBreakerRegistry, CircuitBreakerConfig
from .strategies import (
    get_strategy,
    ProviderMetrics,
    RoutingConstraints,
    ScoredCandidate,
)
from .fallback import (
    FallbackCoordinator,
    FallbackChainConfig,
    create_fallback_chain,
    RequestPhaseTracker,
)
from .health import HealthRegistry
from ..streaming.normalizer import StreamNormalizer
from ..streaming.tool_calls import ToolCallStreamTracker


@dataclass
class ProviderStats:
    """Real-time statistics for a provider (legacy compatibility)."""
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
        if self.error_rate > 0.5:
            self.is_healthy = False


@dataclass
class RoutingResult:
    """Result of routing decision."""
    selected_provider: Provider
    selected_model: str
    adapter: BaseAdapter
    decision: RoutingDecision
    score_breakdown: Optional[Dict[str, float]] = None


class Router:
    """
    Intelligent router for AI requests.

    Features:
    - Circuit breaker for fault tolerance
    - Strategy-based routing (cost, latency, quality)
    - Fallback chains with semantic drift protection
    - Real-time health tracking
    """

    def __init__(
        self,
        adapters: Dict[Provider, BaseAdapter],
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        fallback_config: Optional[FallbackChainConfig] = None
    ):
        """
        Initialize router with provider adapters.

        Args:
            adapters: Dictionary mapping Provider enum to adapter instance
            circuit_breaker_config: Configuration for circuit breakers
            fallback_config: Configuration for fallback behavior
        """
        self.adapters = adapters

        # Legacy stats (for backward compatibility)
        self.stats: Dict[Provider, ProviderStats] = {
            provider: ProviderStats(provider=provider)
            for provider in adapters.keys()
        }

        # New components
        self._circuit_breakers = CircuitBreakerRegistry(circuit_breaker_config)
        self._health_registry = HealthRegistry()
        self._fallback_coordinator = FallbackCoordinator(fallback_config)

        # Model registry
        self._model_registry: Dict[str, Tuple[Provider, ModelInfo]] = {}
        self._build_model_registry()

        # Initialize circuit breakers for all providers
        for provider in adapters.keys():
            self._circuit_breakers.get_breaker(provider.value)

    def _build_model_registry(self):
        """Build a unified model registry from all adapters."""
        for provider, adapter in self.adapters.items():
            for model in adapter.list_models():
                self._model_registry[model.id] = (provider, model)
                self._model_registry[f"{provider.value}/{model.name}"] = (provider, model)

    def _get_provider_metrics(self, provider: Provider) -> ProviderMetrics:
        """Get current metrics for a provider."""
        tracker = self._health_registry.get_tracker(provider)
        snapshot = tracker.get_snapshot()

        return ProviderMetrics(
            provider=provider,
            avg_latency_ms=snapshot.latency_stats.avg_ms,
            error_rate=snapshot.error_rate,
            total_requests=snapshot.total_requests,
            is_available=(
                self._circuit_breakers.is_provider_available(provider.value) and
                snapshot.is_healthy
            ),
            p99_latency_ms=snapshot.latency_stats.p99_ms
        )

    def _record_success(self, provider: Provider, latency_ms: int):
        """Record a successful request across all tracking systems."""
        # Legacy stats
        self.stats[provider].record_success(latency_ms)

        # New systems
        self._circuit_breakers.record_success(provider.value)
        self._health_registry.record_success(provider, latency_ms)

    def _record_failure(self, provider: Provider, error: str, latency_ms: Optional[int] = None):
        """Record a failed request across all tracking systems."""
        # Legacy stats
        self.stats[provider].record_failure(error)

        # New systems
        self._circuit_breakers.record_failure(provider.value, error)
        self._health_registry.record_failure(provider, error, latency_ms)

    async def route_chat(
        self,
        request: ChatCompletionRequest,
        request_id: Optional[str] = None
    ) -> Tuple[ChatCompletionResponse, RoutingDecision]:
        """
        Route a chat completion request.

        Args:
            request: Chat completion request
            request_id: Optional request ID for tracking

        Returns:
            Tuple of (response, routing_decision)
        """
        start_time = time.time()

        # Create phase tracker for semantic drift protection
        tracker = self._fallback_coordinator.create_tracker(
            request_id or f"req_{int(time.time() * 1000)}"
        )

        try:
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
                self._record_success(routing_result.selected_provider, latency_ms)

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

            except TwoApiException as e:
                # Record failure
                latency_ms = int((time.time() - start_time) * 1000)
                self._record_failure(
                    routing_result.selected_provider,
                    str(e),
                    latency_ms
                )

                # Check if fallback is allowed
                if tracker.can_fallback() and request.routing and request.routing.fallback:
                    return await self._try_fallback(
                        request=request,
                        failed_provider=routing_result.selected_provider,
                        fallback_chain=request.routing.fallback,
                        original_error=e,
                        tracker=tracker
                    )
                raise

            except Exception as e:
                # Record failure
                latency_ms = int((time.time() - start_time) * 1000)
                self._record_failure(
                    routing_result.selected_provider,
                    str(e),
                    latency_ms
                )

                # Try fallback
                if tracker.can_fallback() and request.routing and request.routing.fallback:
                    return await self._try_fallback(
                        request=request,
                        failed_provider=routing_result.selected_provider,
                        fallback_chain=request.routing.fallback,
                        original_error=e,
                        tracker=tracker
                    )
                raise

        finally:
            self._fallback_coordinator.cleanup_tracker(tracker.request_id)

    async def route_chat_stream(
        self,
        request: ChatCompletionRequest,
        request_id: str
    ) -> AsyncIterator[str]:
        """
        Route a streaming chat completion request through unified orchestration.

        This is the single source of truth for streaming routing/fallback semantics.
        """
        tracker = self._fallback_coordinator.create_tracker(request_id)

        try:
            routing_result = self._select_provider(request=request, capability="chat")

            async for event in self._stream_with_provider(
                request=request,
                request_id=request_id,
                routing_result=routing_result,
                tracker=tracker,
            ):
                yield event

        except TwoApiException as e:
            # fallback only allowed before first content
            if tracker.can_fallback() and request.routing and request.routing.fallback:
                async for event in self._try_fallback_stream(
                    request=request,
                    failed_provider=routing_result.selected_provider,
                    fallback_chain=request.routing.fallback,
                    tracker=tracker,
                    request_id=request_id,
                ):
                    yield event
            else:
                partial = tracker.get_partial_content() or ""
                yield create_stream_error_chunk(e, partial)

        except Exception as e:
            error = StreamInterruptedError(
                provider=routing_result.selected_provider.value if 'routing_result' in locals() else "unknown",
                partial_content=tracker.get_partial_content() or "",
                request_id=request_id,
            )
            yield create_stream_error_chunk(error, tracker.get_partial_content() or "")

        finally:
            self._fallback_coordinator.cleanup_tracker(request_id)

    async def _stream_with_provider(
        self,
        request: ChatCompletionRequest,
        request_id: str,
        routing_result: RoutingResult,
        tracker: RequestPhaseTracker,
    ) -> AsyncIterator[str]:
        """Stream using one provider and normalize all chunks consistently."""
        start_time = time.time()
        adapter = routing_result.adapter
        provider_name = routing_result.selected_provider.value
        normalizer = StreamNormalizer(model=request.model_name, provider=provider_name, request_id=request_id)
        tool_tracker = ToolCallStreamTracker()

        try:
            async for raw_chunk in adapter.chat_completion_stream(request, request_id):
                normalized_events = self._normalize_raw_stream_chunk(raw_chunk, normalizer, provider_name, tool_tracker)

                for evt in normalized_events:
                    if evt.startswith("data: [DONE]"):
                        yield evt
                        latency_ms = int((time.time() - start_time) * 1000)
                        self._record_success(routing_result.selected_provider, latency_ms)
                        tracker.mark_completed()
                        return

                    # inspect payload for content start tracking
                    try:
                        payload = evt[len("data: "):].strip()
                        parsed = json.loads(payload)
                        choice = (parsed.get("choices") or [{}])[0]
                        delta = choice.get("delta", {})
                        content = delta.get("content")
                        if content:
                            if tracker.can_fallback():
                                tracker.mark_content_started(content)
                            else:
                                tracker.append_content(content)

                        for tc in delta.get("tool_calls", []) or []:
                            f = tc.get("function", {})
                            tool_tracker.update_call(
                                index=tc.get("index", 0),
                                id=tc.get("id"),
                                function_name=f.get("name"),
                                arguments_delta=f.get("arguments", ""),
                            )
                    except Exception:
                        pass

                    yield evt

            # adapters may return without [DONE] - enforce termination
            yield normalizer.create_done_event()
            latency_ms = int((time.time() - start_time) * 1000)
            self._record_success(routing_result.selected_provider, latency_ms)
            tracker.mark_completed()

        except TwoApiException:
            latency_ms = int((time.time() - start_time) * 1000)
            self._record_failure(routing_result.selected_provider, "stream_error", latency_ms)
            raise
        except Exception:
            latency_ms = int((time.time() - start_time) * 1000)
            self._record_failure(routing_result.selected_provider, "stream_error", latency_ms)
            raise

    def _normalize_raw_stream_chunk(
        self,
        raw_chunk: str,
        normalizer: StreamNormalizer,
        provider_name: str,
        tool_tracker: ToolCallStreamTracker,
    ) -> List[str]:
        """Normalize provider chunk(s) into OpenAI-compatible SSE events."""
        events: List[str] = []

        if not isinstance(raw_chunk, str):
            return events

        lines = [ln.strip() for ln in raw_chunk.splitlines() if ln.strip()]
        for line in lines:
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                events.append("data: [DONE]\n\n")
                continue

            try:
                data = json.loads(payload)
            except Exception:
                continue

            chunk = None
            if provider_name == "anthropic" and "type" in data:
                chunk = normalizer.normalize_anthropic_event(data.get("type"), data)
            elif provider_name == "google" and "candidates" in data:
                chunk = normalizer.normalize_google_chunk(data)
            else:
                chunk = normalizer.normalize_openai_chunk(data)

            if chunk is not None:
                events.append(chunk.to_sse())

        return events

    async def route_embedding(
        self,
        request: EmbeddingRequest
    ) -> Tuple[EmbeddingResponse, RoutingDecision]:
        """Route an embedding request."""
        routing_result = self._select_provider(
            request=None,
            capability="embedding",
            model_hint=request.model
        )

        start_time = time.time()

        try:
            response = await routing_result.adapter.embedding(request)
            latency_ms = int((time.time() - start_time) * 1000)

            self._record_success(routing_result.selected_provider, latency_ms)

            return response, routing_result.decision

        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            self._record_failure(routing_result.selected_provider, str(e), latency_ms)
            raise

    async def route_image(
        self,
        request: ImageGenerationRequest
    ) -> Tuple[ImageGenerationResponse, RoutingDecision]:
        """Route an image generation request."""
        routing_result = self._select_provider(
            request=None,
            capability="image",
            model_hint=request.model
        )

        start_time = time.time()

        try:
            response = await routing_result.adapter.image_generation(request)
            latency_ms = int((time.time() - start_time) * 1000)

            self._record_success(routing_result.selected_provider, latency_ms)

            return response, routing_result.decision

        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            self._record_failure(routing_result.selected_provider, str(e), latency_ms)
            raise

    def _select_provider(
        self,
        request: Optional[ChatCompletionRequest],
        capability: str,
        model_hint: Optional[str] = None
    ) -> RoutingResult:
        """
        Select the best provider for a request using strategy-based scoring.

        Selection logic:
        1. If explicit model specified → use that provider (if available)
        2. If auto routing → apply strategy with scoring
        3. Apply constraints (max_latency, max_cost, excluded providers)
        """
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
                    # Check if available
                    if self._circuit_breakers.is_provider_available(provider_name):
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

        # Case 2: Strategy-based routing
        strategy = RoutingStrategy.COST
        if request and request.routing and request.routing.strategy:
            strategy = request.routing.strategy

        # Build constraints
        constraints = RoutingConstraints()
        if request and request.routing:
            constraints.max_latency_ms = request.routing.max_latency_ms
            constraints.max_cost_per_request = request.routing.max_cost
        constraints.required_capabilities = [capability]

        # Find candidates
        candidates: List[Tuple[Provider, ModelInfo]] = []
        for model_id, (provider, model_info) in self._model_registry.items():
            if not model_info.supports(capability):
                continue
            if not self._circuit_breakers.is_provider_available(provider.value):
                continue
            if provider not in self.adapters:
                continue

            candidates.append((provider, model_info))

        if not candidates:
            # Try any available provider
            for provider in self.adapters.keys():
                if self._circuit_breakers.is_provider_available(provider.value):
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
            raise AllProvidersFailedError(
                providers=[p.value for p in self.adapters.keys()],
                request_id=""
            )

        # Get metrics for scoring
        metrics = {
            provider: self._get_provider_metrics(provider)
            for provider, _ in candidates
        }

        # Apply strategy
        strategy_impl = get_strategy(strategy)
        scored = strategy_impl.select_best(
            candidates=candidates,
            metrics=metrics,
            constraints=constraints
        )

        if scored is None:
            raise AllProvidersFailedError(
                providers=[p.value for p, _ in candidates],
                request_id=""
            )

        return RoutingResult(
            selected_provider=scored.provider,
            selected_model=scored.model.name,
            adapter=self.adapters[scored.provider],
            decision=RoutingDecision(
                strategy_used=strategy.value,
                candidates_evaluated=[f"{p.value}/{m.name}" for p, m in candidates[:5]],
                fallback_used=False
            ),
            score_breakdown=scored.breakdown
        )

    async def _try_fallback(
        self,
        request: ChatCompletionRequest,
        failed_provider: Provider,
        fallback_chain: List[str],
        original_error: Exception,
        tracker: RequestPhaseTracker
    ) -> Tuple[ChatCompletionResponse, RoutingDecision]:
        """Try fallback providers in order."""
        chain = create_fallback_chain(
            primary=f"{failed_provider.value}/failed",
            fallback_list=fallback_chain
        )

        # Skip the first (primary already failed)
        chain.get_next()

        while True:
            next_option = chain.get_next(exclude=[failed_provider.value])
            if next_option is None:
                break

            provider_name, model_name = next_option

            try:
                provider = Provider(provider_name)
            except ValueError:
                continue

            if provider not in self.adapters:
                continue

            if not self._circuit_breakers.is_provider_available(provider_name):
                continue

            # Create modified request
            fallback_request = ChatCompletionRequest(
                model=f"{provider_name}/{model_name}" if model_name else f"{provider_name}/auto",
                messages=request.messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=request.stream,
                tools=request.tools,
                tool_choice=request.tool_choice,
                metadata=request.metadata
            )

            start_time = time.time()

            try:
                adapter = self.adapters[provider]

                if not model_name:
                    models = adapter.list_models()
                    chat_models = [m for m in models if m.supports("chat")]
                    if chat_models:
                        model_name = chat_models[0].name

                response = await adapter.chat_completion(fallback_request)
                latency_ms = int((time.time() - start_time) * 1000)

                self._record_success(provider, latency_ms)

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

                chain.record_attempt(provider_name, model_name or "", "success", latency_ms)

                return response, decision

            except Exception as e:
                latency_ms = int((time.time() - start_time) * 1000)
                self._record_failure(provider, str(e), latency_ms)
                chain.record_attempt(provider_name, model_name or "", str(e), latency_ms)
                continue

        # All fallbacks failed
        raise AllProvidersFailedError(
            providers=fallback_chain,
            request_id=tracker.request_id
        )

    async def _try_fallback_stream(
        self,
        request: ChatCompletionRequest,
        failed_provider: Provider,
        fallback_chain: List[str],
        tracker: RequestPhaseTracker,
        request_id: str
    ) -> AsyncIterator[str]:
        """Try fallback providers for streaming request before content starts."""
        chain = create_fallback_chain(
            primary=f"{failed_provider.value}/failed",
            fallback_list=fallback_chain
        )

        chain.get_next()  # Skip primary

        while True:
            next_option = chain.get_next(exclude=[failed_provider.value])
            if next_option is None:
                break

            provider_name, model_name = next_option

            try:
                provider = Provider(provider_name)
            except ValueError:
                continue

            if provider not in self.adapters:
                continue

            if not self._circuit_breakers.is_provider_available(provider_name):
                continue

            fallback_request = ChatCompletionRequest(
                model=f"{provider_name}/{model_name}" if model_name else f"{provider_name}/auto",
                messages=request.messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=True,
                tools=request.tools,
                tool_choice=request.tool_choice,
                metadata=request.metadata
            )

            routing_result = RoutingResult(
                selected_provider=provider,
                selected_model=model_name or "auto",
                adapter=self.adapters[provider],
                decision=RoutingDecision(strategy_used="fallback", candidates_evaluated=fallback_chain, fallback_used=True),
            )

            try:
                async for evt in self._stream_with_provider(
                    request=fallback_request,
                    request_id=request_id,
                    routing_result=routing_result,
                    tracker=tracker,
                ):
                    yield evt
                return

            except Exception as e:
                self._record_failure(provider, str(e))
                if not tracker.can_fallback():
                    err = StreamInterruptedError(
                        provider=provider_name,
                        partial_content=tracker.get_partial_content() or "",
                        request_id=request_id,
                    )
                    yield create_stream_error_chunk(err, tracker.get_partial_content() or "")
                    return
                continue

        # All fallbacks failed
        err = AllProvidersFailedError(providers=fallback_chain, request_id=request_id)
        yield create_stream_error_chunk(err)

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
                self._circuit_breakers.record_failure(adapter.provider.value, str(result))
            else:
                results[adapter.provider] = result
                self.stats[adapter.provider].is_healthy = result.is_healthy
                if result.is_healthy:
                    self._circuit_breakers.record_success(adapter.provider.value)

        return results

    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get routing statistics for all providers."""
        result = {}

        for provider, stats in self.stats.items():
            # Get health snapshot
            health_snapshot = self._health_registry.get_tracker(provider).get_snapshot()
            circuit_status = self._circuit_breakers.get_breaker(provider.value).get_status()

            result[provider.value] = {
                "total_requests": stats.total_requests,
                "failed_requests": stats.failed_requests,
                "error_rate": round(stats.error_rate, 4),
                "avg_latency_ms": stats.avg_latency_ms,
                "is_healthy": stats.is_healthy,
                "last_error": stats.last_error,
                "health_score": health_snapshot.health_score.total,
                "health_grade": health_snapshot.health_score.grade,
                "circuit_state": circuit_status["state"],
            }

        return result

    def get_detailed_health(self) -> Dict[str, Any]:
        """Get detailed health information."""
        return {
            "providers": self.get_stats(),
            "circuit_breakers": self._circuit_breakers.get_all_status(),
            "health_snapshots": {
                k: {
                    "score": v.health_score.total,
                    "grade": v.health_score.grade,
                    "latency": {
                        "avg": v.latency_stats.avg_ms,
                        "p95": v.latency_stats.p95_ms,
                        "p99": v.latency_stats.p99_ms,
                    },
                    "error_rate": v.error_rate,
                    "is_healthy": v.is_healthy,
                }
                for k, v in self._health_registry.get_all_snapshots().items()
            }
        }

    def list_all_models(self) -> List[ModelInfo]:
        """List all available models across all providers."""
        models = []
        seen_ids = set()

        for provider, adapter in self.adapters.items():
            # Only include from healthy providers
            if not self._circuit_breakers.is_provider_available(provider.value):
                continue

            for model in adapter.list_models():
                if model.id not in seen_ids:
                    models.append(model)
                    seen_ids.add(model.id)

        return models
