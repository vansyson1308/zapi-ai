"""
ProviderSelector — picks the best provider/model for a request.

Extracted from router.py so selection logic can be unit-tested in isolation
and reused by future call sites (e.g. AgentOps batch routing) without
dragging the streaming + fallback machinery along.

Public surface mirrors what the old `Router._select_provider` exposed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..adapters.base import BaseAdapter
from ..core.errors import AllProvidersFailedError
from ..core.models import (
    ChatCompletionRequest,
    ModelInfo,
    Provider,
    RoutingDecision,
    RoutingStrategy,
)

from .circuit_breaker import CircuitBreakerRegistry
from .health import HealthRegistry
from .strategies import (
    ProviderMetrics,
    RoutingConstraints,
    get_strategy,
)

_logger = logging.getLogger(__name__)


@dataclass
class RoutingResult:
    """Result of routing decision (returned by ProviderSelector)."""

    selected_provider: Provider
    selected_model: str
    adapter: BaseAdapter
    decision: RoutingDecision
    score_breakdown: Optional[Dict[str, float]] = None


class ProviderSelector:
    """
    Stateless-ish selector: holds references to shared registries and adapters
    but mutates only the model_registry on init.
    """

    def __init__(
        self,
        adapters: Dict[Provider, BaseAdapter],
        circuit_breakers: CircuitBreakerRegistry,
        health_registry: HealthRegistry,
    ) -> None:
        self.adapters = adapters
        self._circuit_breakers = circuit_breakers
        self._health_registry = health_registry

        self._model_registry: Dict[str, Tuple[Provider, ModelInfo]] = {}
        self._build_model_registry()

    # ------------------------------------------------------------------
    # registry management
    # ------------------------------------------------------------------

    def _build_model_registry(self) -> None:
        """Populate model_registry from all adapters' list_models().

        If an adapter's list_models() raises (e.g. transient HTTP failure on
        startup), we skip that adapter and keep the registry partial rather
        than failing the entire Router init. The provider's circuit breaker
        will eventually surface the issue if requests can't route.
        """
        for provider, adapter in self.adapters.items():
            try:
                models = adapter.list_models()
            except Exception as exc:
                _logger.warning(
                    "list_models() failed for provider=%s; skipping registry entries",
                    provider.value,
                    extra={"error": str(exc)},
                )
                continue
            for model in models:
                self._model_registry[model.id] = (provider, model)
                self._model_registry[f"{provider.value}/{model.name}"] = (provider, model)

    @property
    def model_registry(self) -> Dict[str, Tuple[Provider, ModelInfo]]:
        return self._model_registry

    # ------------------------------------------------------------------
    # metrics view (kept here because selection consults health/CB state)
    # ------------------------------------------------------------------

    def get_provider_metrics(self, provider: Provider) -> ProviderMetrics:
        tracker = self._health_registry.get_tracker(provider)
        snapshot = tracker.get_snapshot()
        return ProviderMetrics(
            provider=provider,
            avg_latency_ms=snapshot.latency_stats.avg_ms,
            error_rate=snapshot.error_rate,
            total_requests=snapshot.total_requests,
            is_available=(
                self._circuit_breakers.is_provider_available(provider.value)
                and snapshot.is_healthy
            ),
            p99_latency_ms=snapshot.latency_stats.p99_ms,
        )

    # ------------------------------------------------------------------
    # core: select_provider
    # ------------------------------------------------------------------

    def select(
        self,
        request: Optional[ChatCompletionRequest],
        capability: str,
        model_hint: Optional[str] = None,
    ) -> RoutingResult:
        """
        Select the best provider for a request.

        Selection order:
          1. Explicit `provider/model` form on the request → that provider (if circuit allows).
          2. Strategy-based scoring across candidates that support `capability`.
          3. Last-resort fallback to any healthy provider.
        """
        model_str: Optional[str] = None
        if request:
            model_str = request.model
        elif model_hint:
            model_str = model_hint

        # Case 1: explicit provider/model
        if model_str and model_str.lower() != "auto" and "/" in model_str:
            provider_name, _, model_name = model_str.partition("/")
            try:
                provider = Provider(provider_name)
                if (
                    provider in self.adapters
                    and self._circuit_breakers.is_provider_available(provider_name)
                ):
                    return RoutingResult(
                        selected_provider=provider,
                        selected_model=model_name,
                        adapter=self.adapters[provider],
                        decision=RoutingDecision(
                            strategy_used="explicit",
                            candidates_evaluated=[model_str],
                            fallback_used=False,
                        ),
                    )
            except ValueError:
                pass  # unknown provider → fall through

        # Case 2: strategy-based selection
        strategy = RoutingStrategy.COST
        if request and request.routing and request.routing.strategy:
            strategy = request.routing.strategy

        constraints = RoutingConstraints()
        if request and request.routing:
            constraints.max_latency_ms = request.routing.max_latency_ms
            constraints.max_cost_per_request = request.routing.max_cost
        constraints.required_capabilities = [capability]

        candidates: List[Tuple[Provider, ModelInfo]] = []
        for _, (provider, model_info) in self._model_registry.items():
            if not model_info.supports(capability):
                continue
            if not self._circuit_breakers.is_provider_available(provider.value):
                continue
            if provider not in self.adapters:
                continue
            candidates.append((provider, model_info))

        if not candidates:
            # Case 3: last-resort fallback to ANY healthy provider
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
                                fallback_used=True,
                            ),
                        )
            raise AllProvidersFailedError(
                providers=[p.value for p in self.adapters.keys()],
                request_id="",
            )

        metrics = {provider: self.get_provider_metrics(provider) for provider, _ in candidates}

        strategy_impl = get_strategy(strategy)
        scored = strategy_impl.select_best(
            candidates=candidates,
            metrics=metrics,
            constraints=constraints,
        )

        if scored is None:
            raise AllProvidersFailedError(
                providers=[p.value for p, _ in candidates],
                request_id="",
            )

        return RoutingResult(
            selected_provider=scored.provider,
            selected_model=scored.model.name,
            adapter=self.adapters[scored.provider],
            decision=RoutingDecision(
                strategy_used=strategy.value,
                candidates_evaluated=[f"{p.value}/{m.name}" for p, m in candidates[:5]],
                fallback_used=False,
            ),
            score_breakdown=scored.breakdown,
        )

    def list_all_models(self) -> List[ModelInfo]:
        """List all unique models across healthy providers."""
        models: List[ModelInfo] = []
        seen_ids = set()
        for provider, adapter in self.adapters.items():
            if not self._circuit_breakers.is_provider_available(provider.value):
                continue
            for model in adapter.list_models():
                if model.id not in seen_ids:
                    models.append(model)
                    seen_ids.add(model.id)
        return models
