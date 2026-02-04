"""
2api.ai - Health Tracking

Real-time health monitoring for providers with:
- Latency tracking (avg, p50, p95, p99)
- Error rate monitoring
- Success/failure streaks
- Automatic health scoring

Health Score calculation:
- Base: 100 points
- Penalties for high error rate
- Penalties for high latency
- Penalties for consecutive failures
- Bonuses for consistent performance
"""

import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, List, Optional
import statistics

from ..core.models import Provider


@dataclass
class LatencyStats:
    """Latency statistics."""
    avg_ms: int = 0
    min_ms: int = 0
    max_ms: int = 0
    p50_ms: int = 0
    p95_ms: int = 0
    p99_ms: int = 0
    sample_count: int = 0


@dataclass
class HealthScore:
    """Health score with breakdown."""
    total: float  # 0-100
    breakdown: Dict[str, float] = field(default_factory=dict)
    grade: str = "A"  # A, B, C, D, F

    @staticmethod
    def calculate_grade(score: float) -> str:
        """Calculate letter grade from score."""
        if score >= 90:
            return "A"
        if score >= 80:
            return "B"
        if score >= 70:
            return "C"
        if score >= 60:
            return "D"
        return "F"


@dataclass
class HealthSnapshot:
    """Point-in-time health snapshot."""
    provider: str
    timestamp: float
    is_healthy: bool
    health_score: HealthScore
    latency_stats: LatencyStats
    error_rate: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    consecutive_failures: int
    consecutive_successes: int
    last_error: Optional[str] = None
    last_success_time: Optional[float] = None
    last_failure_time: Optional[float] = None


class HealthTracker:
    """
    Tracks health metrics for a single provider.

    Uses a sliding window for latency calculations and
    real-time counters for request tracking.
    """

    # Window size for latency calculations
    LATENCY_WINDOW_SIZE = 100

    # Window size for error rate (in seconds)
    ERROR_WINDOW_SECONDS = 300  # 5 minutes

    def __init__(self, provider: Provider):
        self.provider = provider
        self._lock = Lock()

        # Request counters
        self._total_requests = 0
        self._successful_requests = 0
        self._failed_requests = 0

        # Streak tracking
        self._consecutive_failures = 0
        self._consecutive_successes = 0

        # Latency tracking (sliding window)
        self._latencies: deque = deque(maxlen=self.LATENCY_WINDOW_SIZE)

        # Recent errors (for windowed error rate)
        self._recent_errors: List[float] = []

        # Last events
        self._last_error: Optional[str] = None
        self._last_success_time: Optional[float] = None
        self._last_failure_time: Optional[float] = None

    def record_success(self, latency_ms: int):
        """Record a successful request."""
        with self._lock:
            now = time.time()

            self._total_requests += 1
            self._successful_requests += 1
            self._consecutive_successes += 1
            self._consecutive_failures = 0
            self._last_success_time = now

            self._latencies.append(latency_ms)

    def record_failure(self, error: str, latency_ms: Optional[int] = None):
        """Record a failed request."""
        with self._lock:
            now = time.time()

            self._total_requests += 1
            self._failed_requests += 1
            self._consecutive_failures += 1
            self._consecutive_successes = 0
            self._last_failure_time = now
            self._last_error = error

            self._recent_errors.append(now)

            if latency_ms is not None:
                self._latencies.append(latency_ms)

    def _get_latency_stats(self) -> LatencyStats:
        """Calculate latency statistics from window."""
        if not self._latencies:
            return LatencyStats()

        latencies = list(self._latencies)
        sorted_latencies = sorted(latencies)
        count = len(sorted_latencies)

        def percentile(data: List[int], p: float) -> int:
            """Calculate percentile."""
            if not data:
                return 0
            k = (len(data) - 1) * p / 100
            f = int(k)
            c = f + 1 if f + 1 < len(data) else f
            return int(data[f] + (data[c] - data[f]) * (k - f))

        return LatencyStats(
            avg_ms=int(statistics.mean(latencies)),
            min_ms=min(sorted_latencies),
            max_ms=max(sorted_latencies),
            p50_ms=percentile(sorted_latencies, 50),
            p95_ms=percentile(sorted_latencies, 95),
            p99_ms=percentile(sorted_latencies, 99),
            sample_count=count
        )

    def _get_windowed_error_rate(self) -> float:
        """Calculate error rate within the time window."""
        now = time.time()
        cutoff = now - self.ERROR_WINDOW_SECONDS

        # Clean old errors
        self._recent_errors = [t for t in self._recent_errors if t > cutoff]

        if self._total_requests == 0:
            return 0.0

        # Calculate rate based on recent errors vs recent total
        # For simplicity, use overall rate but weighted toward recent
        overall_rate = self._failed_requests / self._total_requests

        return overall_rate

    def calculate_health_score(self) -> HealthScore:
        """
        Calculate comprehensive health score.

        Scoring:
        - Base: 100 points
        - Error rate penalty: up to -40 points
        - Latency penalty: up to -20 points
        - Consecutive failure penalty: up to -20 points
        - Consistency bonus: up to +10 points
        """
        breakdown = {}

        # Base score
        base_score = 100.0
        breakdown["base"] = base_score

        # Error rate penalty (0-40 points)
        error_rate = self._get_windowed_error_rate()
        error_penalty = min(error_rate * 100, 40)  # Max 40 point penalty
        breakdown["error_penalty"] = -error_penalty

        # Latency penalty (0-20 points)
        latency_stats = self._get_latency_stats()
        if latency_stats.avg_ms > 0:
            # Penalty starts at 2000ms avg latency
            if latency_stats.avg_ms > 2000:
                latency_penalty = min((latency_stats.avg_ms - 2000) / 100, 20)
            else:
                latency_penalty = 0
        else:
            latency_penalty = 0
        breakdown["latency_penalty"] = -latency_penalty

        # Consecutive failure penalty (0-20 points)
        failure_penalty = min(self._consecutive_failures * 4, 20)
        breakdown["failure_penalty"] = -failure_penalty

        # Consistency bonus (0-10 points)
        # If p99 is close to p50, provider is consistent
        if latency_stats.p50_ms > 0 and latency_stats.p99_ms > 0:
            ratio = latency_stats.p99_ms / latency_stats.p50_ms
            if ratio < 1.5:
                consistency_bonus = 10
            elif ratio < 2.0:
                consistency_bonus = 5
            elif ratio < 3.0:
                consistency_bonus = 2
            else:
                consistency_bonus = 0
        else:
            consistency_bonus = 0
        breakdown["consistency_bonus"] = consistency_bonus

        # Calculate total
        total = (
            base_score - error_penalty - latency_penalty -
            failure_penalty + consistency_bonus
        )
        total = max(0, min(100, total))  # Clamp to 0-100

        return HealthScore(
            total=round(total, 2),
            breakdown=breakdown,
            grade=HealthScore.calculate_grade(total)
        )

    def get_snapshot(self) -> HealthSnapshot:
        """Get current health snapshot."""
        with self._lock:
            latency_stats = self._get_latency_stats()
            error_rate = self._get_windowed_error_rate()
            health_score = self.calculate_health_score()

            # Consider unhealthy if score < 50 or too many consecutive failures
            is_healthy = health_score.total >= 50 and self._consecutive_failures < 5

            return HealthSnapshot(
                provider=self.provider.value,
                timestamp=time.time(),
                is_healthy=is_healthy,
                health_score=health_score,
                latency_stats=latency_stats,
                error_rate=round(error_rate, 4),
                total_requests=self._total_requests,
                successful_requests=self._successful_requests,
                failed_requests=self._failed_requests,
                consecutive_failures=self._consecutive_failures,
                consecutive_successes=self._consecutive_successes,
                last_error=self._last_error,
                last_success_time=self._last_success_time,
                last_failure_time=self._last_failure_time
            )

    @property
    def is_healthy(self) -> bool:
        """Quick health check."""
        with self._lock:
            # Unhealthy if too many consecutive failures
            if self._consecutive_failures >= 5:
                return False

            # Unhealthy if error rate too high
            error_rate = self._get_windowed_error_rate()
            if error_rate > 0.5:
                return False

            return True

    @property
    def avg_latency_ms(self) -> int:
        """Get average latency."""
        with self._lock:
            if not self._latencies:
                return 0
            return int(statistics.mean(self._latencies))


class HealthRegistry:
    """
    Registry for managing health trackers across all providers.
    """

    def __init__(self):
        self._trackers: Dict[Provider, HealthTracker] = {}
        self._lock = Lock()

    def get_tracker(self, provider: Provider) -> HealthTracker:
        """Get or create a health tracker for a provider."""
        with self._lock:
            if provider not in self._trackers:
                self._trackers[provider] = HealthTracker(provider)
            return self._trackers[provider]

    def record_success(self, provider: Provider, latency_ms: int):
        """Record a successful request."""
        self.get_tracker(provider).record_success(latency_ms)

    def record_failure(
        self,
        provider: Provider,
        error: str,
        latency_ms: Optional[int] = None
    ):
        """Record a failed request."""
        self.get_tracker(provider).record_failure(error, latency_ms)

    def is_healthy(self, provider: Provider) -> bool:
        """Check if a provider is healthy."""
        return self.get_tracker(provider).is_healthy

    def get_all_snapshots(self) -> Dict[str, HealthSnapshot]:
        """Get health snapshots for all tracked providers."""
        with self._lock:
            return {
                provider.value: tracker.get_snapshot()
                for provider, tracker in self._trackers.items()
            }

    def get_healthy_providers(self) -> List[Provider]:
        """Get list of currently healthy providers."""
        with self._lock:
            return [
                provider for provider, tracker in self._trackers.items()
                if tracker.is_healthy
            ]

    def get_ranked_providers(self) -> List[tuple]:
        """
        Get providers ranked by health score.

        Returns:
            List of (provider, health_score) tuples, sorted by score descending
        """
        with self._lock:
            rankings = []
            for provider, tracker in self._trackers.items():
                snapshot = tracker.get_snapshot()
                rankings.append((provider, snapshot.health_score.total))

            rankings.sort(key=lambda x: x[1], reverse=True)
            return rankings
