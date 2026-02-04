"""
2api.ai - Routing System Tests

Comprehensive tests for EPIC C - Routing Engine + Fallback Chain.
Verifies:
- Circuit breaker state transitions
- Routing strategies (cost, latency, quality)
- Fallback chain with semantic drift protection
- Health tracking and scoring
"""

import pytest
import time
from unittest.mock import MagicMock, AsyncMock, patch

from src.routing.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerRegistry,
    CircuitState,
)
from src.routing.strategies import (
    CostStrategy,
    LatencyStrategy,
    QualityStrategy,
    BalancedStrategy,
    ProviderMetrics,
    RoutingConstraints,
    get_strategy,
)
from src.routing.fallback import (
    FallbackChain,
    FallbackChainConfig,
    FallbackCoordinator,
    FallbackPhase,
    RequestPhaseTracker,
    create_fallback_chain,
    check_fallback_eligibility,
)
from src.routing.health import (
    HealthTracker,
    HealthRegistry,
    HealthScore,
)
from src.core.models import (
    Provider,
    RoutingStrategy,
    ModelInfo,
    ModelPricing,
)


# ============================================================
# Circuit Breaker Tests
# ============================================================

class TestCircuitBreaker:
    """Test circuit breaker state transitions."""

    def test_initial_state_is_closed(self):
        """Circuit should start in CLOSED state."""
        cb = CircuitBreaker("test_provider")
        assert cb.state == CircuitState.CLOSED
        assert cb.is_available is True

    def test_success_keeps_circuit_closed(self):
        """Success should keep circuit CLOSED."""
        cb = CircuitBreaker("test_provider")
        cb.record_success()
        cb.record_success()
        assert cb.state == CircuitState.CLOSED
        assert cb.is_available is True

    def test_failures_open_circuit(self):
        """Enough failures should OPEN the circuit."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            failure_window_seconds=60
        )
        cb = CircuitBreaker("test_provider", config)

        # Record failures
        for _ in range(3):
            cb.record_failure("test error")

        assert cb.state == CircuitState.OPEN
        assert cb.is_available is False

    def test_open_circuit_blocks_requests(self):
        """OPEN circuit should block requests."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout_seconds=30
        )
        cb = CircuitBreaker("test_provider", config)

        cb.record_failure("error")
        cb.record_failure("error")

        assert cb.state == CircuitState.OPEN
        assert cb.is_available is False

    def test_recovery_timeout_transitions_to_half_open(self):
        """After recovery timeout, should transition to HALF_OPEN."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout_seconds=0.1  # 100ms for testing
        )
        cb = CircuitBreaker("test_provider", config)

        cb.record_failure("error")
        cb.record_failure("error")
        assert cb.state == CircuitState.OPEN

        # Wait for recovery timeout
        time.sleep(0.15)

        # Checking availability should transition to HALF_OPEN
        assert cb.is_available is True
        assert cb.state == CircuitState.HALF_OPEN

    def test_half_open_success_closes_circuit(self):
        """Success in HALF_OPEN should CLOSE circuit."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout_seconds=0.05,
            success_threshold=1
        )
        cb = CircuitBreaker("test_provider", config)

        # Open circuit
        cb.record_failure("error")
        cb.record_failure("error")

        # Wait and check to transition to half-open
        time.sleep(0.1)
        _ = cb.is_available

        # Success should close
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_half_open_failure_reopens_circuit(self):
        """Failure in HALF_OPEN should reopen circuit."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout_seconds=0.05
        )
        cb = CircuitBreaker("test_provider", config)

        # Open circuit
        cb.record_failure("error")
        cb.record_failure("error")

        # Wait and transition to half-open
        time.sleep(0.1)
        _ = cb.is_available
        assert cb.state == CircuitState.HALF_OPEN

        # Failure should reopen
        cb.record_failure("error")
        assert cb.state == CircuitState.OPEN

    def test_get_status(self):
        """get_status should return correct information."""
        cb = CircuitBreaker("test_provider")
        cb.record_success()
        cb.record_failure("error")

        status = cb.get_status()
        assert status["provider"] == "test_provider"
        assert status["state"] == "closed"
        assert status["stats"]["total_requests"] == 2
        assert status["stats"]["successful_requests"] == 1
        assert status["stats"]["failed_requests"] == 1


class TestCircuitBreakerRegistry:
    """Test circuit breaker registry."""

    def test_creates_breakers_on_demand(self):
        """Registry should create breakers on demand."""
        registry = CircuitBreakerRegistry()

        breaker = registry.get_breaker("provider1")
        assert breaker is not None
        assert breaker.provider_name == "provider1"

    def test_returns_same_breaker(self):
        """Registry should return same breaker for same provider."""
        registry = CircuitBreakerRegistry()

        b1 = registry.get_breaker("provider1")
        b2 = registry.get_breaker("provider1")
        assert b1 is b2

    def test_get_available_providers(self):
        """Should return only available providers."""
        config = CircuitBreakerConfig(failure_threshold=1)
        registry = CircuitBreakerRegistry(config)

        # Make provider1 unavailable
        registry.record_failure("provider1", "error")

        # provider2 should still be available
        registry.get_breaker("provider2")

        available = registry.get_available_providers()
        assert "provider2" in available
        assert "provider1" not in available


# ============================================================
# Routing Strategy Tests
# ============================================================

class TestCostStrategy:
    """Test cost-based routing strategy."""

    def create_model(
        self,
        provider: Provider,
        input_price: float,
        output_price: float
    ) -> ModelInfo:
        """Helper to create model info."""
        return ModelInfo(
            id=f"{provider.value}/test-model",
            provider=provider,
            name="test-model",
            capabilities=["chat"],
            context_window=128000,
            max_output_tokens=4096,
            pricing=ModelPricing(
                input_per_1m_tokens=input_price,
                output_per_1m_tokens=output_price
            )
        )

    def test_selects_cheapest_model(self):
        """Should select cheapest model."""
        strategy = CostStrategy()

        candidates = [
            (Provider.OPENAI, self.create_model(Provider.OPENAI, 5.0, 15.0)),
            (Provider.ANTHROPIC, self.create_model(Provider.ANTHROPIC, 3.0, 12.0)),
            (Provider.GOOGLE, self.create_model(Provider.GOOGLE, 7.0, 21.0)),
        ]

        metrics = {
            Provider.OPENAI: ProviderMetrics(Provider.OPENAI, is_available=True),
            Provider.ANTHROPIC: ProviderMetrics(Provider.ANTHROPIC, is_available=True),
            Provider.GOOGLE: ProviderMetrics(Provider.GOOGLE, is_available=True),
        }

        result = strategy.select_best(candidates, metrics, RoutingConstraints())

        assert result is not None
        assert result.provider == Provider.ANTHROPIC  # Cheapest

    def test_excludes_unavailable_providers(self):
        """Should exclude unavailable providers."""
        strategy = CostStrategy()

        candidates = [
            (Provider.OPENAI, self.create_model(Provider.OPENAI, 5.0, 15.0)),
            (Provider.ANTHROPIC, self.create_model(Provider.ANTHROPIC, 1.0, 3.0)),  # Cheapest but unavailable
        ]

        metrics = {
            Provider.OPENAI: ProviderMetrics(Provider.OPENAI, is_available=True),
            Provider.ANTHROPIC: ProviderMetrics(Provider.ANTHROPIC, is_available=False),
        }

        result = strategy.select_best(candidates, metrics, RoutingConstraints())

        assert result is not None
        assert result.provider == Provider.OPENAI  # Only available one

    def test_respects_max_cost_constraint(self):
        """Should respect max cost constraint."""
        strategy = CostStrategy()

        # Create expensive model
        expensive = self.create_model(Provider.OPENAI, 50.0, 150.0)
        cheap = self.create_model(Provider.ANTHROPIC, 1.0, 3.0)

        candidates = [
            (Provider.OPENAI, expensive),
            (Provider.ANTHROPIC, cheap),
        ]

        metrics = {
            Provider.OPENAI: ProviderMetrics(Provider.OPENAI, is_available=True),
            Provider.ANTHROPIC: ProviderMetrics(Provider.ANTHROPIC, is_available=True),
        }

        constraints = RoutingConstraints(max_cost_per_request=0.01)

        result = strategy.select_best(candidates, metrics, constraints)

        # Only cheap model should pass cost constraint
        assert result is not None
        assert result.provider == Provider.ANTHROPIC


class TestLatencyStrategy:
    """Test latency-based routing strategy."""

    def create_model(self, provider: Provider) -> ModelInfo:
        return ModelInfo(
            id=f"{provider.value}/test",
            provider=provider,
            name="test",
            capabilities=["chat"],
            context_window=128000,
            max_output_tokens=4096,
            pricing=ModelPricing(5.0, 15.0)
        )

    def test_selects_lowest_latency(self):
        """Should select provider with lowest latency."""
        strategy = LatencyStrategy()

        candidates = [
            (Provider.OPENAI, self.create_model(Provider.OPENAI)),
            (Provider.ANTHROPIC, self.create_model(Provider.ANTHROPIC)),
            (Provider.GOOGLE, self.create_model(Provider.GOOGLE)),
        ]

        metrics = {
            Provider.OPENAI: ProviderMetrics(
                Provider.OPENAI, avg_latency_ms=500, is_available=True
            ),
            Provider.ANTHROPIC: ProviderMetrics(
                Provider.ANTHROPIC, avg_latency_ms=200, is_available=True
            ),
            Provider.GOOGLE: ProviderMetrics(
                Provider.GOOGLE, avg_latency_ms=800, is_available=True
            ),
        }

        result = strategy.select_best(candidates, metrics, RoutingConstraints())

        assert result is not None
        assert result.provider == Provider.ANTHROPIC  # Lowest latency

    def test_respects_max_latency_constraint(self):
        """Should exclude providers exceeding max latency."""
        strategy = LatencyStrategy()

        candidates = [
            (Provider.OPENAI, self.create_model(Provider.OPENAI)),
            (Provider.ANTHROPIC, self.create_model(Provider.ANTHROPIC)),
        ]

        metrics = {
            Provider.OPENAI: ProviderMetrics(
                Provider.OPENAI, avg_latency_ms=500, is_available=True
            ),
            Provider.ANTHROPIC: ProviderMetrics(
                Provider.ANTHROPIC, avg_latency_ms=2000, is_available=True
            ),
        }

        constraints = RoutingConstraints(max_latency_ms=1000)

        result = strategy.select_best(candidates, metrics, constraints)

        assert result is not None
        assert result.provider == Provider.OPENAI  # Only one under limit


class TestQualityStrategy:
    """Test quality-based routing strategy."""

    def test_selects_highest_quality(self):
        """Should select highest quality model."""
        strategy = QualityStrategy()

        # Higher price and context = higher quality score
        high_quality = ModelInfo(
            id="openai/gpt-4o",
            provider=Provider.OPENAI,
            name="gpt-4o",
            capabilities=["chat", "vision", "tools"],
            context_window=128000,
            max_output_tokens=16384,
            pricing=ModelPricing(5.0, 15.0)
        )

        low_quality = ModelInfo(
            id="google/gemini-1.5-flash",
            provider=Provider.GOOGLE,
            name="gemini-1.5-flash",
            capabilities=["chat"],
            context_window=32000,
            max_output_tokens=4096,
            pricing=ModelPricing(0.5, 1.5)
        )

        candidates = [
            (Provider.OPENAI, high_quality),
            (Provider.GOOGLE, low_quality),
        ]

        metrics = {
            Provider.OPENAI: ProviderMetrics(Provider.OPENAI, is_available=True),
            Provider.GOOGLE: ProviderMetrics(Provider.GOOGLE, is_available=True),
        }

        result = strategy.select_best(candidates, metrics, RoutingConstraints())

        assert result is not None
        assert result.provider == Provider.OPENAI  # Higher quality


class TestGetStrategy:
    """Test strategy factory function."""

    def test_returns_correct_strategy(self):
        """Should return correct strategy instance."""
        assert isinstance(get_strategy(RoutingStrategy.COST), CostStrategy)
        assert isinstance(get_strategy(RoutingStrategy.LATENCY), LatencyStrategy)
        assert isinstance(get_strategy(RoutingStrategy.QUALITY), QualityStrategy)


# ============================================================
# Fallback Chain Tests
# ============================================================

class TestRequestPhaseTracker:
    """Test request phase tracking for semantic drift protection."""

    def test_initial_phase_is_pre_content(self):
        """Initial phase should be PRE_CONTENT."""
        tracker = RequestPhaseTracker("req_123")
        assert tracker.phase == FallbackPhase.PRE_CONTENT
        assert tracker.can_fallback() is True

    def test_mark_content_started_blocks_fallback(self):
        """After content starts, fallback should be blocked."""
        tracker = RequestPhaseTracker("req_123")

        tracker.mark_content_started("Hello")

        assert tracker.phase == FallbackPhase.CONTENT_STARTED
        assert tracker.can_fallback() is False

    def test_tracks_partial_content(self):
        """Should track partial content for error reporting."""
        tracker = RequestPhaseTracker("req_123")

        tracker.mark_content_started("Hello")
        tracker.append_content(" world")
        tracker.append_content("!")

        assert tracker.get_partial_content() == "Hello world!"
        assert tracker.chunks_delivered == 3

    def test_completed_phase(self):
        """Should track completed phase."""
        tracker = RequestPhaseTracker("req_123")
        tracker.mark_content_started("test")
        tracker.mark_completed()

        assert tracker.phase == FallbackPhase.COMPLETED


class TestFallbackChain:
    """Test fallback chain execution."""

    def test_parse_provider_model(self):
        """Should correctly parse provider/model strings."""
        chain = FallbackChain(["openai/gpt-4o", "anthropic/claude-3-opus"])

        p, m = chain.parse_provider_model("openai/gpt-4o")
        assert p == "openai"
        assert m == "gpt-4o"

        p, m = chain.parse_provider_model("anthropic")
        assert p == "anthropic"
        assert m is None

    def test_get_next_returns_fallbacks(self):
        """get_next should return fallbacks in order."""
        chain = FallbackChain([
            "openai/gpt-4o",
            "anthropic/claude-3-opus",
            "google/gemini-1.5-pro"
        ])

        p1, m1 = chain.get_next()
        assert p1 == "openai"

        p2, m2 = chain.get_next()
        assert p2 == "anthropic"

        p3, m3 = chain.get_next()
        assert p3 == "google"

        result = chain.get_next()
        assert result is None  # Chain exhausted

    def test_get_next_excludes_providers(self):
        """get_next should exclude specified providers."""
        chain = FallbackChain([
            "openai/gpt-4o",
            "anthropic/claude-3",
            "google/gemini"
        ])

        result = chain.get_next(exclude=["openai"])
        assert result[0] == "anthropic"  # OpenAI excluded

    def test_is_exhausted(self):
        """Should correctly report when chain is exhausted."""
        chain = FallbackChain(["openai/gpt-4o"])

        assert chain.is_exhausted() is False
        chain.get_next()
        assert chain.is_exhausted() is True


class TestFallbackCoordinator:
    """Test fallback coordinator."""

    def test_creates_and_retrieves_tracker(self):
        """Should create and retrieve trackers."""
        coord = FallbackCoordinator()

        tracker = coord.create_tracker("req_123")
        assert tracker is not None

        retrieved = coord.get_tracker("req_123")
        assert retrieved is tracker

    def test_cleanup_removes_tracker(self):
        """cleanup_tracker should remove tracker."""
        coord = FallbackCoordinator()

        coord.create_tracker("req_123")
        coord.cleanup_tracker("req_123")

        assert coord.get_tracker("req_123") is None

    def test_can_attempt_fallback_pre_content(self):
        """Should allow fallback in pre-content phase."""
        coord = FallbackCoordinator()
        tracker = coord.create_tracker("req_123")
        chain = FallbackChain(["openai/gpt-4o", "anthropic/claude"])

        can, reason, partial = coord.can_attempt_fallback("req_123", chain)

        assert can is True
        assert reason == "eligible"

    def test_cannot_fallback_after_content(self):
        """Should block fallback after content."""
        coord = FallbackCoordinator()
        tracker = coord.create_tracker("req_123")
        tracker.mark_content_started("Hello")

        chain = FallbackChain(["openai/gpt-4o"])

        can, reason, partial = coord.can_attempt_fallback("req_123", chain)

        assert can is False
        assert reason == "semantic_drift_blocked"
        assert partial == "Hello"


class TestCheckFallbackEligibility:
    """Test fallback eligibility checking."""

    def test_eligible_pre_content(self):
        """Should be eligible in pre-content phase."""
        tracker = RequestPhaseTracker("req_123")
        config = FallbackChainConfig()

        eligible, reason = check_fallback_eligibility(tracker, config)

        assert eligible is True
        assert reason == "pre_content_phase"

    def test_ineligible_after_content(self):
        """Should be ineligible after content delivery."""
        tracker = RequestPhaseTracker("req_123")
        tracker.mark_content_started("test")
        config = FallbackChainConfig()

        eligible, reason = check_fallback_eligibility(tracker, config)

        assert eligible is False
        assert reason == "content_already_delivered"


# ============================================================
# Health Tracking Tests
# ============================================================

class TestHealthTracker:
    """Test health tracking."""

    def test_record_success_updates_stats(self):
        """record_success should update statistics."""
        tracker = HealthTracker(Provider.OPENAI)

        tracker.record_success(100)
        tracker.record_success(200)

        snapshot = tracker.get_snapshot()
        assert snapshot.total_requests == 2
        assert snapshot.successful_requests == 2
        assert snapshot.failed_requests == 0

    def test_record_failure_updates_stats(self):
        """record_failure should update statistics."""
        tracker = HealthTracker(Provider.OPENAI)

        tracker.record_success(100)
        tracker.record_failure("timeout")

        snapshot = tracker.get_snapshot()
        assert snapshot.total_requests == 2
        assert snapshot.successful_requests == 1
        assert snapshot.failed_requests == 1
        assert snapshot.last_error == "timeout"

    def test_latency_stats_calculation(self):
        """Should calculate latency statistics correctly."""
        tracker = HealthTracker(Provider.OPENAI)

        # Add latencies: 100, 200, 300
        tracker.record_success(100)
        tracker.record_success(200)
        tracker.record_success(300)

        snapshot = tracker.get_snapshot()
        assert snapshot.latency_stats.avg_ms == 200
        assert snapshot.latency_stats.min_ms == 100
        assert snapshot.latency_stats.max_ms == 300

    def test_health_score_calculation(self):
        """Should calculate health score."""
        tracker = HealthTracker(Provider.OPENAI)

        # Perfect performance
        for _ in range(10):
            tracker.record_success(100)

        snapshot = tracker.get_snapshot()
        assert snapshot.health_score.total >= 90
        assert snapshot.health_score.grade == "A"

    def test_consecutive_failures_affect_health(self):
        """Consecutive failures should affect health."""
        tracker = HealthTracker(Provider.OPENAI)

        # 5 consecutive failures
        for _ in range(5):
            tracker.record_failure("error")

        snapshot = tracker.get_snapshot()
        assert snapshot.is_healthy is False
        assert snapshot.consecutive_failures == 5


class TestHealthRegistry:
    """Test health registry."""

    def test_creates_tracker_on_demand(self):
        """Should create tracker on demand."""
        registry = HealthRegistry()

        tracker = registry.get_tracker(Provider.OPENAI)
        assert tracker is not None

    def test_record_success(self):
        """Should record success through registry."""
        registry = HealthRegistry()

        registry.record_success(Provider.OPENAI, 100)

        tracker = registry.get_tracker(Provider.OPENAI)
        assert tracker.get_snapshot().successful_requests == 1

    def test_get_healthy_providers(self):
        """Should return healthy providers."""
        registry = HealthRegistry()

        # Make OpenAI healthy
        for _ in range(5):
            registry.record_success(Provider.OPENAI, 100)

        # Make Anthropic unhealthy
        for _ in range(5):
            registry.record_failure(Provider.ANTHROPIC, "error")

        healthy = registry.get_healthy_providers()
        assert Provider.OPENAI in healthy
        assert Provider.ANTHROPIC not in healthy

    def test_get_ranked_providers(self):
        """Should return providers ranked by health score."""
        registry = HealthRegistry()

        # OpenAI: all success
        for _ in range(10):
            registry.record_success(Provider.OPENAI, 100)

        # Anthropic: some failures
        for _ in range(7):
            registry.record_success(Provider.ANTHROPIC, 100)
        for _ in range(3):
            registry.record_failure(Provider.ANTHROPIC, "error")

        rankings = registry.get_ranked_providers()

        # OpenAI should rank higher
        assert rankings[0][0] == Provider.OPENAI


class TestHealthScore:
    """Test health score calculation."""

    def test_calculate_grade(self):
        """Should calculate correct letter grade."""
        assert HealthScore.calculate_grade(95) == "A"
        assert HealthScore.calculate_grade(85) == "B"
        assert HealthScore.calculate_grade(75) == "C"
        assert HealthScore.calculate_grade(65) == "D"
        assert HealthScore.calculate_grade(50) == "F"


# ============================================================
# Run Tests
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
