"""
FallbackOrchestrator — walks a fallback chain when the primary provider fails.

Two flavors:
  * try_chat: returns a single ChatCompletionResponse (non-streaming)
  * try_chat_stream: yields SSE chunks (streaming, with semantic-safety
    constraint that we only switch providers BEFORE first content emission)

Extracted from router.py so retry-budget logic and chain construction can be
unit-tested independently of selection/streaming.
"""

from __future__ import annotations

import time
from typing import AsyncIterator, Callable, Dict, List, Optional, Tuple

from ..adapters.base import BaseAdapter
from ..core.errors import (
    AllProvidersFailedError,
    StreamInterruptedError,
    TwoApiException,
    create_stream_error_chunk,
)
from ..core.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Provider,
    RoutingDecision,
    TwoApiMetadata,
)

from .circuit_breaker import CircuitBreakerRegistry
from .fallback import RequestPhaseTracker, create_fallback_chain
from .provider_selector import RoutingResult
from .streaming_manager import StreamingManager


RecordSuccess = Callable[[Provider, int], None]
RecordFailure = Callable[[Provider, str, Optional[int]], None]


class FallbackOrchestrator:
    """Try the fallback chain in order. Stops at first success."""

    def __init__(
        self,
        adapters: Dict[Provider, BaseAdapter],
        circuit_breakers: CircuitBreakerRegistry,
        record_success: RecordSuccess,
        record_failure: RecordFailure,
        streaming_manager: StreamingManager,
    ) -> None:
        self.adapters = adapters
        self._circuit_breakers = circuit_breakers
        self._record_success = record_success
        self._record_failure = record_failure
        self._streaming_manager = streaming_manager

    # ------------------------------------------------------------------
    # non-streaming
    # ------------------------------------------------------------------

    async def try_chat(
        self,
        request: ChatCompletionRequest,
        failed_provider: Provider,
        fallback_chain: List[str],
        tracker: RequestPhaseTracker,
    ) -> Tuple[ChatCompletionResponse, RoutingDecision]:
        chain = create_fallback_chain(
            primary=f"{failed_provider.value}/failed",
            fallback_list=fallback_chain,
        )
        chain.get_next()  # skip primary (already failed)

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

            adapter = self.adapters[provider]
            if not model_name:
                models = adapter.list_models()
                chat_models = [m for m in models if m.supports("chat")]
                if chat_models:
                    model_name = chat_models[0].name

            fallback_request = ChatCompletionRequest(
                model=f"{provider_name}/{model_name}" if model_name else f"{provider_name}/auto",
                messages=request.messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=request.stream,
                tools=request.tools,
                tool_choice=request.tool_choice,
                metadata=request.metadata,
            )

            start_time = time.time()
            try:
                response = await adapter.chat_completion(fallback_request)
                latency_ms = int((time.time() - start_time) * 1000)
                self._record_success(provider, latency_ms)

                cost = adapter.calculate_cost(model_name or "", response.usage)

                decision = RoutingDecision(
                    strategy_used="fallback",
                    candidates_evaluated=fallback_chain,
                    fallback_used=True,
                )
                response._2api = TwoApiMetadata(
                    request_id=response.id,
                    latency_ms=latency_ms,
                    cost_usd=cost,
                    routing_decision=decision,
                )
                chain.record_attempt(provider_name, model_name or "", "success", latency_ms)
                return response, decision

            except Exception as e:
                latency_ms = int((time.time() - start_time) * 1000)
                self._record_failure(provider, str(e), latency_ms)
                chain.record_attempt(provider_name, model_name or "", str(e), latency_ms)
                continue

        raise AllProvidersFailedError(
            providers=fallback_chain,
            request_id=tracker.request_id,
        )

    # ------------------------------------------------------------------
    # streaming
    # ------------------------------------------------------------------

    async def try_chat_stream(
        self,
        request: ChatCompletionRequest,
        failed_provider: Provider,
        fallback_chain: List[str],
        tracker: RequestPhaseTracker,
        request_id: str,
    ) -> AsyncIterator[str]:
        chain = create_fallback_chain(
            primary=f"{failed_provider.value}/failed",
            fallback_list=fallback_chain,
        )
        chain.get_next()  # skip primary

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
                metadata=request.metadata,
            )

            try:
                async for evt in self._streaming_manager.stream(
                    adapter=self.adapters[provider],
                    provider=provider,
                    model=model_name or "auto",
                    request=fallback_request,
                    request_id=request_id,
                    tracker=tracker,
                ):
                    yield evt
                return  # stream completed normally

            except Exception as e:
                self._record_failure(provider, str(e), None)
                # If content already streamed to client, we cannot try another provider.
                if not tracker.can_fallback():
                    err = StreamInterruptedError(
                        provider=provider_name,
                        partial_content=tracker.get_partial_content() or "",
                        request_id=request_id,
                    )
                    yield create_stream_error_chunk(err, tracker.get_partial_content() or "")
                    return
                continue

        # All fallbacks exhausted
        err = AllProvidersFailedError(providers=fallback_chain, request_id=request_id)
        yield create_stream_error_chunk(err)
