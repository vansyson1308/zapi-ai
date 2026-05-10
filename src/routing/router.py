"""
2api.ai - Router (facade)

Public entry point for routing AI requests. Coordinates three internal
components:

  * ProviderSelector       — picks the best provider/model
  * StreamingManager       — owns the streaming pipeline + chunk normalization
  * FallbackOrchestrator   — walks the fallback chain on primary failure

This file used to contain ~880 LOC of mixed concerns. The behavior is
preserved exactly; the public Router class API (route_chat, route_chat_stream,
route_embedding, route_image, check_all_health, get_stats, list_all_models,
adapters, stats, _circuit_breakers, _health_registry, _fallback_coordinator)
is unchanged so existing tests, server.py, and SDK callers keep working.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from ..adapters.base import BaseAdapter, ProviderHealth
from ..core.errors import (
    StreamInterruptedError,
    TwoApiException,
    create_stream_error_chunk,
)
from ..core.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ModelInfo,
    Provider,
    RoutingDecision,
    TwoApiMetadata,
)

from .circuit_breaker import CircuitBreakerConfig, CircuitBreakerRegistry
from .fallback import FallbackChainConfig, FallbackCoordinator, RequestPhaseTracker
from .fallback_orchestrator import FallbackOrchestrator
from .health import HealthRegistry
from .provider_selector import ProviderSelector, RoutingResult
from .streaming_manager import StreamingManager


# ============================================================
# Legacy stats (preserved for backward compat — tests check this)
# ============================================================


@dataclass
class ProviderStats:
    """Real-time per-provider stats. Kept for backward compatibility with old callers."""

    provider: Provider
    total_requests: int = 0
    failed_requests: int = 0
    total_latency_ms: int = 0
    last_error: Optional[str] = None
    last_success_time: Optional[float] = None
    is_healthy: bool = True

    @property
    def avg_latency_ms(self) -> int:
        return self.total_latency_ms // self.total_requests if self.total_requests else 0

    @property
    def error_rate(self) -> float:
        return self.failed_requests / self.total_requests if self.total_requests else 0.0

    def record_success(self, latency_ms: int) -> None:
        self.total_requests += 1
        self.total_latency_ms += latency_ms
        self.last_success_time = time.time()
        self.is_healthy = True

    def record_failure(self, error: str) -> None:
        self.total_requests += 1
        self.failed_requests += 1
        self.last_error = error
        if self.error_rate > 0.5:
            self.is_healthy = False


# ============================================================
# Router
# ============================================================


class Router:
    """
    Top-level router. Holds the registry of adapters + delegates to internal
    components. Public API matches the pre-refactor implementation 1:1.
    """

    def __init__(
        self,
        adapters: Dict[Provider, BaseAdapter],
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        fallback_config: Optional[FallbackChainConfig] = None,
    ) -> None:
        self.adapters = adapters

        # Legacy stats dict (still consumed by get_stats and check_all_health).
        self.stats: Dict[Provider, ProviderStats] = {
            provider: ProviderStats(provider=provider) for provider in adapters.keys()
        }

        # Shared state
        self._circuit_breakers = CircuitBreakerRegistry(circuit_breaker_config)
        self._health_registry = HealthRegistry()
        self._fallback_coordinator = FallbackCoordinator(fallback_config)

        # Pre-arm circuit breakers
        for provider in adapters.keys():
            self._circuit_breakers.get_breaker(provider.value)

        # Internal components
        self._selector = ProviderSelector(
            adapters=adapters,
            circuit_breakers=self._circuit_breakers,
            health_registry=self._health_registry,
        )
        self._streaming = StreamingManager(
            record_success=self._record_success,
            record_failure=self._record_failure,
        )
        self._fallback_orchestrator = FallbackOrchestrator(
            adapters=adapters,
            circuit_breakers=self._circuit_breakers,
            record_success=self._record_success,
            record_failure=self._record_failure,
            streaming_manager=self._streaming,
        )

    # ------------------------------------------------------------------
    # state recording (called by streaming manager + fallback orchestrator)
    # ------------------------------------------------------------------

    def _record_success(self, provider: Provider, latency_ms: int) -> None:
        self.stats[provider].record_success(latency_ms)
        self._circuit_breakers.record_success(provider.value)
        self._health_registry.record_success(provider, latency_ms)

    def _record_failure(self, provider: Provider, error: str, latency_ms: Optional[int] = None) -> None:
        self.stats[provider].record_failure(error)
        self._circuit_breakers.record_failure(provider.value, error)
        self._health_registry.record_failure(provider, error, latency_ms)

    # ------------------------------------------------------------------
    # backward-compat shim: tests call _select_provider directly
    # ------------------------------------------------------------------

    def _select_provider(
        self,
        request: Optional[ChatCompletionRequest],
        capability: str,
        model_hint: Optional[str] = None,
    ) -> RoutingResult:
        return self._selector.select(request=request, capability=capability, model_hint=model_hint)

    # ------------------------------------------------------------------
    # chat completions (non-streaming)
    # ------------------------------------------------------------------

    async def route_chat(
        self,
        request: ChatCompletionRequest,
        request_id: Optional[str] = None,
    ) -> Tuple[ChatCompletionResponse, RoutingDecision]:
        start_time = time.time()
        tracker = self._fallback_coordinator.create_tracker(
            request_id or f"req_{int(time.time() * 1000)}"
        )

        try:
            routing_result = self._selector.select(request=request, capability="chat")

            try:
                response = await routing_result.adapter.chat_completion(request)
                latency_ms = int((time.time() - start_time) * 1000)
                self._record_success(routing_result.selected_provider, latency_ms)

                cost = routing_result.adapter.calculate_cost(
                    routing_result.selected_model, response.usage
                )
                response._2api = TwoApiMetadata(
                    request_id=response.id,
                    latency_ms=latency_ms,
                    cost_usd=cost,
                    routing_decision=routing_result.decision,
                )
                return response, routing_result.decision

            except TwoApiException as e:
                latency_ms = int((time.time() - start_time) * 1000)
                self._record_failure(routing_result.selected_provider, str(e), latency_ms)
                if tracker.can_fallback() and request.routing and request.routing.fallback:
                    return await self._fallback_orchestrator.try_chat(
                        request=request,
                        failed_provider=routing_result.selected_provider,
                        fallback_chain=request.routing.fallback,
                        tracker=tracker,
                    )
                raise

            except Exception as e:
                latency_ms = int((time.time() - start_time) * 1000)
                self._record_failure(routing_result.selected_provider, str(e), latency_ms)
                if tracker.can_fallback() and request.routing and request.routing.fallback:
                    return await self._fallback_orchestrator.try_chat(
                        request=request,
                        failed_provider=routing_result.selected_provider,
                        fallback_chain=request.routing.fallback,
                        tracker=tracker,
                    )
                raise

        finally:
            self._fallback_coordinator.cleanup_tracker(tracker.request_id)

    # ------------------------------------------------------------------
    # chat completions (streaming)
    # ------------------------------------------------------------------

    async def route_chat_stream(
        self,
        request: ChatCompletionRequest,
        request_id: str,
    ) -> AsyncIterator[str]:
        tracker = self._fallback_coordinator.create_tracker(request_id)
        routing_result: Optional[RoutingResult] = None

        try:
            routing_result = self._selector.select(request=request, capability="chat")

            async for event in self._streaming.stream(
                adapter=routing_result.adapter,
                provider=routing_result.selected_provider,
                model=routing_result.selected_model,
                request=request,
                request_id=request_id,
                tracker=tracker,
            ):
                yield event

        except TwoApiException as e:
            if (
                routing_result is not None
                and tracker.can_fallback()
                and request.routing
                and request.routing.fallback
            ):
                async for event in self._fallback_orchestrator.try_chat_stream(
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

        except Exception:
            provider_label = (
                routing_result.selected_provider.value if routing_result is not None else "unknown"
            )
            error = StreamInterruptedError(
                provider=provider_label,
                partial_content=tracker.get_partial_content() or "",
                request_id=request_id,
            )
            yield create_stream_error_chunk(error, tracker.get_partial_content() or "")

        finally:
            self._fallback_coordinator.cleanup_tracker(request_id)

    # ------------------------------------------------------------------
    # embeddings + images (no fallback yet — kept identical to old behavior)
    # ------------------------------------------------------------------

    async def route_embedding(
        self,
        request: EmbeddingRequest,
    ) -> Tuple[EmbeddingResponse, RoutingDecision]:
        routing_result = self._selector.select(
            request=None,
            capability="embedding",
            model_hint=request.model,
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
        request: ImageGenerationRequest,
    ) -> Tuple[ImageGenerationResponse, RoutingDecision]:
        routing_result = self._selector.select(
            request=None,
            capability="image",
            model_hint=request.model,
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

    # ------------------------------------------------------------------
    # health & stats
    # ------------------------------------------------------------------

    async def check_all_health(self) -> Dict[Provider, ProviderHealth]:
        results: Dict[Provider, ProviderHealth] = {}
        tasks = [adapter.health_check() for adapter in self.adapters.values()]
        health_results = await asyncio.gather(*tasks, return_exceptions=True)

        for adapter, result in zip(self.adapters.values(), health_results):
            if isinstance(result, Exception):
                results[adapter.provider] = ProviderHealth(
                    provider=adapter.provider,
                    is_healthy=False,
                    last_error=str(result),
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
        result: Dict[str, Dict[str, Any]] = {}
        for provider, stats in self.stats.items():
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
            },
        }

    def list_all_models(self) -> List[ModelInfo]:
        return self._selector.list_all_models()
