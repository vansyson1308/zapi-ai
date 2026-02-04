"""
2api.ai - Routing Strategies

Implements intelligent routing strategies for provider selection:
- COST: Minimize cost while meeting quality requirements
- LATENCY: Minimize response time
- QUALITY: Maximize output quality (context, capabilities)

Each strategy uses a scoring system that considers:
- Provider health and availability
- Historical performance metrics
- Cost and latency constraints
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

from ..core.models import ModelInfo, Provider, RoutingStrategy


@dataclass
class ProviderMetrics:
    """Real-time metrics for provider scoring."""
    provider: Provider
    avg_latency_ms: int = 0
    error_rate: float = 0.0
    total_requests: int = 0
    is_available: bool = True
    p99_latency_ms: int = 0


@dataclass
class RoutingConstraints:
    """Constraints for filtering candidates."""
    max_latency_ms: Optional[int] = None
    max_cost_per_request: Optional[float] = None
    required_capabilities: Optional[List[str]] = None
    excluded_providers: Optional[List[str]] = None
    preferred_providers: Optional[List[str]] = None


@dataclass
class ScoredCandidate:
    """A candidate with its calculated score."""
    provider: Provider
    model: ModelInfo
    score: float
    breakdown: Dict[str, float]  # Score breakdown for debugging


class BaseStrategy(ABC):
    """Base class for routing strategies."""

    @abstractmethod
    def score_candidate(
        self,
        provider: Provider,
        model: ModelInfo,
        metrics: ProviderMetrics,
        constraints: RoutingConstraints,
        estimated_tokens: int = 1000
    ) -> Optional[ScoredCandidate]:
        """
        Calculate score for a candidate.

        Returns None if candidate should be excluded.
        Higher score = better candidate.
        """
        pass

    def select_best(
        self,
        candidates: List[Tuple[Provider, ModelInfo]],
        metrics: Dict[Provider, ProviderMetrics],
        constraints: RoutingConstraints,
        estimated_tokens: int = 1000
    ) -> Optional[ScoredCandidate]:
        """
        Select the best candidate from a list.

        Args:
            candidates: List of (provider, model) tuples
            metrics: Provider metrics for scoring
            constraints: Routing constraints
            estimated_tokens: Estimated token count for cost calculation

        Returns:
            Best scored candidate or None if no valid candidates
        """
        scored = []

        for provider, model in candidates:
            provider_metrics = metrics.get(
                provider,
                ProviderMetrics(provider=provider)
            )

            result = self.score_candidate(
                provider, model, provider_metrics, constraints, estimated_tokens
            )

            if result is not None:
                scored.append(result)

        if not scored:
            return None

        # Sort by score (highest first)
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[0]


class CostStrategy(BaseStrategy):
    """
    Cost-optimized routing strategy.

    Prioritizes lowest cost while considering:
    - Model pricing (input + output tokens)
    - Provider availability bonus
    - Error rate penalty
    """

    def score_candidate(
        self,
        provider: Provider,
        model: ModelInfo,
        metrics: ProviderMetrics,
        constraints: RoutingConstraints,
        estimated_tokens: int = 1000
    ) -> Optional[ScoredCandidate]:
        breakdown = {}

        # Check availability
        if not metrics.is_available:
            return None

        # Check excluded providers
        if constraints.excluded_providers:
            if provider.value in constraints.excluded_providers:
                return None

        # Calculate estimated cost
        # Assume 70% input, 30% output for typical request
        input_tokens = int(estimated_tokens * 0.7)
        output_tokens = int(estimated_tokens * 0.3)
        estimated_cost = model.calculate_cost(input_tokens, output_tokens)

        # Check cost constraint
        if constraints.max_cost_per_request:
            if estimated_cost > constraints.max_cost_per_request:
                return None

        # Check capabilities
        if constraints.required_capabilities:
            for cap in constraints.required_capabilities:
                if not model.supports(cap):
                    return None

        # Base score: inverse of cost (lower cost = higher score)
        # Normalize to 0-100 scale (assuming max $0.10 per request)
        max_cost = 0.10
        cost_score = max(0, (1 - estimated_cost / max_cost)) * 100
        breakdown["cost_score"] = cost_score

        # Availability bonus
        availability_score = 10 if metrics.is_available else 0
        breakdown["availability_score"] = availability_score

        # Error rate penalty
        error_penalty = metrics.error_rate * 20
        breakdown["error_penalty"] = -error_penalty

        # Preferred provider bonus
        preferred_bonus = 0
        if constraints.preferred_providers:
            if provider.value in constraints.preferred_providers:
                preferred_bonus = 15
        breakdown["preferred_bonus"] = preferred_bonus

        total_score = cost_score + availability_score - error_penalty + preferred_bonus

        return ScoredCandidate(
            provider=provider,
            model=model,
            score=total_score,
            breakdown=breakdown
        )


class LatencyStrategy(BaseStrategy):
    """
    Latency-optimized routing strategy.

    Prioritizes lowest response time while considering:
    - Historical latency metrics
    - Provider availability
    - Error rate (failures increase effective latency)
    """

    def score_candidate(
        self,
        provider: Provider,
        model: ModelInfo,
        metrics: ProviderMetrics,
        constraints: RoutingConstraints,
        estimated_tokens: int = 1000
    ) -> Optional[ScoredCandidate]:
        breakdown = {}

        # Check availability
        if not metrics.is_available:
            return None

        # Check excluded providers
        if constraints.excluded_providers:
            if provider.value in constraints.excluded_providers:
                return None

        # Check latency constraint
        if constraints.max_latency_ms:
            if metrics.avg_latency_ms > constraints.max_latency_ms:
                return None

        # Check capabilities
        if constraints.required_capabilities:
            for cap in constraints.required_capabilities:
                if not model.supports(cap):
                    return None

        # Base score: inverse of latency (lower latency = higher score)
        # Normalize to 0-100 scale (assuming max 5000ms)
        max_latency = 5000
        effective_latency = metrics.avg_latency_ms if metrics.avg_latency_ms > 0 else 500

        latency_score = max(0, (1 - effective_latency / max_latency)) * 100
        breakdown["latency_score"] = latency_score

        # Availability bonus
        availability_score = 10 if metrics.is_available else 0
        breakdown["availability_score"] = availability_score

        # Error rate penalty (errors mean retries = more latency)
        error_penalty = metrics.error_rate * 30
        breakdown["error_penalty"] = -error_penalty

        # Consistency bonus (low variance in latency)
        # Use p99 vs avg ratio as proxy for consistency
        if metrics.p99_latency_ms > 0 and metrics.avg_latency_ms > 0:
            consistency_ratio = metrics.p99_latency_ms / metrics.avg_latency_ms
            # Good consistency: p99 < 2x avg
            if consistency_ratio < 2:
                consistency_bonus = 10
            elif consistency_ratio < 3:
                consistency_bonus = 5
            else:
                consistency_bonus = 0
        else:
            consistency_bonus = 5  # No data, assume average
        breakdown["consistency_bonus"] = consistency_bonus

        # Preferred provider bonus
        preferred_bonus = 0
        if constraints.preferred_providers:
            if provider.value in constraints.preferred_providers:
                preferred_bonus = 10
        breakdown["preferred_bonus"] = preferred_bonus

        total_score = (
            latency_score + availability_score - error_penalty +
            consistency_bonus + preferred_bonus
        )

        return ScoredCandidate(
            provider=provider,
            model=model,
            score=total_score,
            breakdown=breakdown
        )


class QualityStrategy(BaseStrategy):
    """
    Quality-optimized routing strategy.

    Prioritizes model capabilities and output quality:
    - Context window size (larger = better for complex tasks)
    - Model pricing (assumption: higher price = higher quality)
    - Provider reliability (low error rate)
    """

    def score_candidate(
        self,
        provider: Provider,
        model: ModelInfo,
        metrics: ProviderMetrics,
        constraints: RoutingConstraints,
        estimated_tokens: int = 1000
    ) -> Optional[ScoredCandidate]:
        breakdown = {}

        # Check availability
        if not metrics.is_available:
            return None

        # Check excluded providers
        if constraints.excluded_providers:
            if provider.value in constraints.excluded_providers:
                return None

        # Check capabilities
        if constraints.required_capabilities:
            for cap in constraints.required_capabilities:
                if not model.supports(cap):
                    return None

        # Context window score (larger is better)
        # Normalize to 0-40 scale (max 200k context)
        max_context = 200000
        context_score = min(model.context_window / max_context, 1.0) * 40
        breakdown["context_score"] = context_score

        # Price as quality proxy (more expensive = better model assumption)
        # Normalize to 0-30 scale (max $60 per 1M tokens)
        max_price = 60.0
        price_score = min(
            model.pricing.output_per_1m_tokens / max_price, 1.0
        ) * 30
        breakdown["price_score"] = price_score

        # Capabilities score
        # More capabilities = more versatile model
        capabilities_score = min(len(model.capabilities) * 5, 15)
        breakdown["capabilities_score"] = capabilities_score

        # Reliability bonus (low error rate)
        reliability_score = (1 - metrics.error_rate) * 10
        breakdown["reliability_score"] = reliability_score

        # Preferred provider bonus
        preferred_bonus = 0
        if constraints.preferred_providers:
            if provider.value in constraints.preferred_providers:
                preferred_bonus = 10
        breakdown["preferred_bonus"] = preferred_bonus

        total_score = (
            context_score + price_score + capabilities_score +
            reliability_score + preferred_bonus
        )

        return ScoredCandidate(
            provider=provider,
            model=model,
            score=total_score,
            breakdown=breakdown
        )


class BalancedStrategy(BaseStrategy):
    """
    Balanced strategy combining cost, latency, and quality.

    Weights:
    - Cost: 40%
    - Latency: 30%
    - Quality: 30%
    """

    def __init__(self):
        self._cost_strategy = CostStrategy()
        self._latency_strategy = LatencyStrategy()
        self._quality_strategy = QualityStrategy()

    def score_candidate(
        self,
        provider: Provider,
        model: ModelInfo,
        metrics: ProviderMetrics,
        constraints: RoutingConstraints,
        estimated_tokens: int = 1000
    ) -> Optional[ScoredCandidate]:
        # Get scores from each strategy
        cost_result = self._cost_strategy.score_candidate(
            provider, model, metrics, constraints, estimated_tokens
        )
        latency_result = self._latency_strategy.score_candidate(
            provider, model, metrics, constraints, estimated_tokens
        )
        quality_result = self._quality_strategy.score_candidate(
            provider, model, metrics, constraints, estimated_tokens
        )

        # If any strategy rejects, candidate is invalid
        if cost_result is None or latency_result is None or quality_result is None:
            return None

        # Weighted combination
        cost_weight = 0.4
        latency_weight = 0.3
        quality_weight = 0.3

        total_score = (
            cost_result.score * cost_weight +
            latency_result.score * latency_weight +
            quality_result.score * quality_weight
        )

        breakdown = {
            "cost_component": cost_result.score * cost_weight,
            "latency_component": latency_result.score * latency_weight,
            "quality_component": quality_result.score * quality_weight,
        }

        return ScoredCandidate(
            provider=provider,
            model=model,
            score=total_score,
            breakdown=breakdown
        )


def get_strategy(strategy: RoutingStrategy) -> BaseStrategy:
    """Factory function to get a strategy instance."""
    strategies = {
        RoutingStrategy.COST: CostStrategy(),
        RoutingStrategy.LATENCY: LatencyStrategy(),
        RoutingStrategy.QUALITY: QualityStrategy(),
    }
    return strategies.get(strategy, CostStrategy())
