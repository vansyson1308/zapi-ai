"""
2api.ai - Fallback Chain

Implements intelligent fallback handling with the critical
"No Semantic Drift" rule:

RULE: Once any content has been streamed to the client,
      NO fallback or retry is allowed.

This prevents:
- Duplicate content delivery
- Inconsistent responses
- Client confusion from mixed provider outputs

Fallback is ONLY allowed:
- Before any content is streamed (pre-content phase)
- For non-streaming requests that fail completely
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Callable, Any
import time

from ..core.models import Provider


class FallbackPhase(str, Enum):
    """
    Request phases for fallback eligibility.

    PRE_CONTENT: Before any content sent - fallback allowed
    CONTENT_STARTED: Content streaming begun - NO fallback
    COMPLETED: Request finished - NO fallback
    """
    PRE_CONTENT = "pre_content"
    CONTENT_STARTED = "content_started"
    COMPLETED = "completed"


@dataclass
class FallbackAttempt:
    """Record of a fallback attempt."""
    provider: str
    model: str
    error: str
    timestamp: float
    duration_ms: int


@dataclass
class FallbackResult:
    """Result of fallback chain execution."""
    success: bool
    final_provider: Optional[str] = None
    final_model: Optional[str] = None
    attempts: List[FallbackAttempt] = field(default_factory=list)
    total_attempts: int = 0
    total_duration_ms: int = 0
    phase_blocked: bool = False  # True if blocked due to semantic drift rule
    partial_content: Optional[str] = None  # Content delivered before failure


@dataclass
class FallbackChainConfig:
    """Configuration for fallback behavior."""
    # Maximum number of fallback attempts
    max_attempts: int = 3

    # Timeout per attempt in seconds
    timeout_per_attempt: float = 30.0

    # Whether to allow fallback after partial content (should always be False!)
    allow_semantic_drift: bool = False  # NEVER set this to True in production


class RequestPhaseTracker:
    """
    Tracks the phase of a request for semantic drift protection.

    Usage:
        tracker = RequestPhaseTracker(request_id)

        # Before making request
        if tracker.can_fallback():
            # Safe to try fallback

        # When first content chunk arrives
        tracker.mark_content_started(chunk)

        # If error after content
        if not tracker.can_fallback():
            # Must return error with partial_content, no retry!
    """

    def __init__(self, request_id: str):
        self.request_id = request_id
        self.phase = FallbackPhase.PRE_CONTENT
        self.content_started_at: Optional[float] = None
        self.partial_content: str = ""
        self.chunks_delivered: int = 0

    def can_fallback(self) -> bool:
        """
        Check if fallback is allowed in current phase.

        Returns True ONLY if no content has been delivered.
        """
        return self.phase == FallbackPhase.PRE_CONTENT

    def mark_content_started(self, initial_content: str = ""):
        """
        Mark that content streaming has begun.

        After this point, NO fallback is allowed.
        """
        if self.phase == FallbackPhase.PRE_CONTENT:
            self.phase = FallbackPhase.CONTENT_STARTED
            self.content_started_at = time.time()
            self.partial_content = initial_content
            self.chunks_delivered = 1

    def append_content(self, content: str):
        """Track additional content delivered."""
        self.partial_content += content
        self.chunks_delivered += 1

    def mark_completed(self):
        """Mark request as completed."""
        self.phase = FallbackPhase.COMPLETED

    def get_partial_content(self) -> Optional[str]:
        """Get content delivered before failure (for error response)."""
        if self.partial_content:
            return self.partial_content
        return None


class FallbackChain:
    """
    Manages fallback execution across providers.

    Implements the fallback chain with strict semantic drift protection.
    """

    def __init__(
        self,
        chain: List[str],
        config: Optional[FallbackChainConfig] = None
    ):
        """
        Initialize fallback chain.

        Args:
            chain: List of provider/model strings in fallback order
                   e.g., ["openai/gpt-4o", "anthropic/claude-3-opus", "google/gemini-1.5-pro"]
            config: Fallback configuration
        """
        self.chain = chain
        self.config = config or FallbackChainConfig()
        self.attempts: List[FallbackAttempt] = []
        self.current_index = 0

    def parse_provider_model(self, entry: str) -> tuple:
        """Parse a chain entry into (provider, model)."""
        if "/" in entry:
            parts = entry.split("/", 1)
            return parts[0], parts[1]
        return entry, None

    def get_next(self, exclude: Optional[List[str]] = None) -> Optional[tuple]:
        """
        Get the next provider/model to try.

        Args:
            exclude: List of providers to skip

        Returns:
            (provider, model) tuple or None if chain exhausted
        """
        exclude = exclude or []

        while self.current_index < len(self.chain):
            entry = self.chain[self.current_index]
            provider, model = self.parse_provider_model(entry)

            self.current_index += 1

            if provider not in exclude:
                return provider, model

        return None

    def record_attempt(
        self,
        provider: str,
        model: str,
        error: str,
        duration_ms: int
    ):
        """Record a failed fallback attempt."""
        self.attempts.append(FallbackAttempt(
            provider=provider,
            model=model or "auto",
            error=error,
            timestamp=time.time(),
            duration_ms=duration_ms
        ))

    def get_result(
        self,
        success: bool,
        final_provider: Optional[str] = None,
        final_model: Optional[str] = None,
        phase_blocked: bool = False,
        partial_content: Optional[str] = None
    ) -> FallbackResult:
        """Build the fallback result."""
        total_duration = sum(a.duration_ms for a in self.attempts)

        return FallbackResult(
            success=success,
            final_provider=final_provider,
            final_model=final_model,
            attempts=self.attempts,
            total_attempts=len(self.attempts),
            total_duration_ms=total_duration,
            phase_blocked=phase_blocked,
            partial_content=partial_content
        )

    def is_exhausted(self) -> bool:
        """Check if all fallback options have been tried."""
        return self.current_index >= len(self.chain)

    def remaining_count(self) -> int:
        """Get number of remaining fallback options."""
        return max(0, len(self.chain) - self.current_index)


def create_fallback_chain(
    primary: str,
    fallback_list: Optional[List[str]] = None,
    config: Optional[FallbackChainConfig] = None
) -> FallbackChain:
    """
    Create a fallback chain starting with primary provider.

    Args:
        primary: Primary provider/model (e.g., "openai/gpt-4o")
        fallback_list: Additional fallback providers
        config: Fallback configuration

    Returns:
        Configured FallbackChain
    """
    chain = [primary]
    if fallback_list:
        chain.extend(fallback_list)

    return FallbackChain(chain, config)


def check_fallback_eligibility(
    phase_tracker: RequestPhaseTracker,
    config: FallbackChainConfig
) -> tuple:
    """
    Check if fallback is allowed based on request phase.

    Returns:
        (is_eligible, reason)
    """
    if config.allow_semantic_drift:
        # DANGER: This should never be True in production!
        return True, "semantic_drift_allowed"

    if phase_tracker.can_fallback():
        return True, "pre_content_phase"

    return False, "content_already_delivered"


class FallbackCoordinator:
    """
    Coordinates fallback execution with semantic drift protection.

    This is the main interface for routing to use for fallback handling.
    """

    def __init__(self, config: Optional[FallbackChainConfig] = None):
        self.config = config or FallbackChainConfig()
        self._active_trackers: Dict[str, RequestPhaseTracker] = {}

    def create_tracker(self, request_id: str) -> RequestPhaseTracker:
        """Create a phase tracker for a request."""
        tracker = RequestPhaseTracker(request_id)
        self._active_trackers[request_id] = tracker
        return tracker

    def get_tracker(self, request_id: str) -> Optional[RequestPhaseTracker]:
        """Get the tracker for a request."""
        return self._active_trackers.get(request_id)

    def cleanup_tracker(self, request_id: str):
        """Remove tracker after request completes."""
        self._active_trackers.pop(request_id, None)

    def can_attempt_fallback(
        self,
        request_id: str,
        fallback_chain: FallbackChain
    ) -> tuple:
        """
        Check if fallback can be attempted.

        Returns:
            (can_fallback, reason, partial_content)
        """
        tracker = self.get_tracker(request_id)

        if tracker is None:
            return False, "no_tracker", None

        if not tracker.can_fallback():
            return False, "semantic_drift_blocked", tracker.get_partial_content()

        if fallback_chain.is_exhausted():
            return False, "chain_exhausted", None

        return True, "eligible", None

    def build_error_with_partial_content(
        self,
        request_id: str,
        original_error: Exception
    ) -> Dict[str, Any]:
        """
        Build error response including partial content (if any).

        Used when fallback is blocked due to semantic drift.
        """
        tracker = self.get_tracker(request_id)
        partial = tracker.get_partial_content() if tracker else None

        return {
            "error": str(original_error),
            "partial_content": partial,
            "fallback_blocked": True,
            "reason": "semantic_drift_protection",
            "chunks_delivered": tracker.chunks_delivered if tracker else 0
        }
