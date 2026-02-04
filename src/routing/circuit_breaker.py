"""
2api.ai - Circuit Breaker

Implements the Circuit Breaker pattern to prevent cascading failures
when providers become unhealthy.

States:
- CLOSED: Normal operation, requests pass through
- OPEN: Provider is down, requests fail fast
- HALF_OPEN: Testing if provider recovered

Transitions:
- CLOSED -> OPEN: When failure threshold exceeded
- OPEN -> HALF_OPEN: After recovery timeout
- HALF_OPEN -> CLOSED: On successful test request
- HALF_OPEN -> OPEN: On failed test request
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional
from threading import Lock


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing fast
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    # Failure threshold to trip the breaker
    failure_threshold: int = 5

    # Time window for counting failures (seconds)
    failure_window_seconds: float = 60.0

    # Time to wait before testing recovery (seconds)
    recovery_timeout_seconds: float = 30.0

    # Number of successful requests needed to close circuit
    success_threshold: int = 2

    # Minimum requests before evaluating failure rate
    min_requests_for_evaluation: int = 3


@dataclass
class CircuitStats:
    """Statistics for a circuit breaker."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    state_changed_at: float = field(default_factory=time.time)

    # Rolling window for failure tracking
    recent_failures: list = field(default_factory=list)

    @property
    def failure_rate(self) -> float:
        """Calculate current failure rate."""
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests

    def add_failure(self, timestamp: float):
        """Record a failure."""
        self.total_requests += 1
        self.failed_requests += 1
        self.consecutive_failures += 1
        self.consecutive_successes = 0
        self.last_failure_time = timestamp
        self.recent_failures.append(timestamp)

    def add_success(self, timestamp: float):
        """Record a success."""
        self.total_requests += 1
        self.successful_requests += 1
        self.consecutive_successes += 1
        self.consecutive_failures = 0
        self.last_success_time = timestamp

    def count_recent_failures(self, window_seconds: float) -> int:
        """Count failures within the time window."""
        now = time.time()
        cutoff = now - window_seconds

        # Clean old failures
        self.recent_failures = [t for t in self.recent_failures if t > cutoff]

        return len(self.recent_failures)

    def reset(self):
        """Reset statistics for new state."""
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self.recent_failures = []
        self.state_changed_at = time.time()


class CircuitBreaker:
    """
    Circuit breaker for a single provider.

    Prevents repeated calls to a failing provider by:
    1. Tracking failure rates
    2. Opening the circuit when failures exceed threshold
    3. Testing recovery after a timeout period
    """

    def __init__(
        self,
        provider_name: str,
        config: Optional[CircuitBreakerConfig] = None
    ):
        self.provider_name = provider_name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.stats = CircuitStats()
        self._lock = Lock()

    @property
    def is_available(self) -> bool:
        """Check if requests can be made through this circuit."""
        with self._lock:
            return self._check_availability()

    def _check_availability(self) -> bool:
        """Internal availability check (must hold lock)."""
        now = time.time()

        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            time_in_open = now - self.stats.state_changed_at
            if time_in_open >= self.config.recovery_timeout_seconds:
                # Transition to half-open for testing
                self._transition_to(CircuitState.HALF_OPEN)
                return True
            return False

        if self.state == CircuitState.HALF_OPEN:
            # Allow limited requests in half-open state
            return True

        return False

    def record_success(self):
        """Record a successful request."""
        with self._lock:
            now = time.time()
            self.stats.add_success(now)

            if self.state == CircuitState.HALF_OPEN:
                # Check if enough successes to close circuit
                if self.stats.consecutive_successes >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)

    def record_failure(self, error: Optional[str] = None):
        """Record a failed request."""
        with self._lock:
            now = time.time()
            self.stats.add_failure(now)

            if self.state == CircuitState.HALF_OPEN:
                # Any failure in half-open immediately opens circuit
                self._transition_to(CircuitState.OPEN)
                return

            if self.state == CircuitState.CLOSED:
                # Check if we should open the circuit
                recent_failures = self.stats.count_recent_failures(
                    self.config.failure_window_seconds
                )

                if recent_failures >= self.config.failure_threshold:
                    self._transition_to(CircuitState.OPEN)

    def _transition_to(self, new_state: CircuitState):
        """Transition to a new state."""
        old_state = self.state
        self.state = new_state
        self.stats.state_changed_at = time.time()

        if new_state == CircuitState.CLOSED:
            self.stats.reset()

    def get_status(self) -> Dict:
        """Get current circuit breaker status."""
        with self._lock:
            return {
                "provider": self.provider_name,
                "state": self.state.value,
                "is_available": self._check_availability(),
                "stats": {
                    "total_requests": self.stats.total_requests,
                    "successful_requests": self.stats.successful_requests,
                    "failed_requests": self.stats.failed_requests,
                    "failure_rate": round(self.stats.failure_rate, 4),
                    "consecutive_failures": self.stats.consecutive_failures,
                    "consecutive_successes": self.stats.consecutive_successes,
                },
                "time_in_current_state": round(
                    time.time() - self.stats.state_changed_at, 2
                )
            }

    def force_open(self):
        """Manually open the circuit (for testing or emergency)."""
        with self._lock:
            self._transition_to(CircuitState.OPEN)

    def force_close(self):
        """Manually close the circuit (for testing or recovery)."""
        with self._lock:
            self._transition_to(CircuitState.CLOSED)


class CircuitBreakerRegistry:
    """
    Registry managing circuit breakers for all providers.
    """

    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        self.config = config or CircuitBreakerConfig()
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = Lock()

    def get_breaker(self, provider_name: str) -> CircuitBreaker:
        """Get or create a circuit breaker for a provider."""
        with self._lock:
            if provider_name not in self._breakers:
                self._breakers[provider_name] = CircuitBreaker(
                    provider_name, self.config
                )
            return self._breakers[provider_name]

    def is_provider_available(self, provider_name: str) -> bool:
        """Check if a provider is available (circuit not open)."""
        return self.get_breaker(provider_name).is_available

    def record_success(self, provider_name: str):
        """Record a successful request to a provider."""
        self.get_breaker(provider_name).record_success()

    def record_failure(self, provider_name: str, error: Optional[str] = None):
        """Record a failed request to a provider."""
        self.get_breaker(provider_name).record_failure(error)

    def get_all_status(self) -> Dict[str, Dict]:
        """Get status of all circuit breakers."""
        with self._lock:
            return {
                name: breaker.get_status()
                for name, breaker in self._breakers.items()
            }

    def get_available_providers(self) -> list:
        """Get list of providers with available circuits."""
        with self._lock:
            return [
                name for name, breaker in self._breakers.items()
                if breaker.is_available
            ]
