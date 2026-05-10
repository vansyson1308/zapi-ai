"""
EventBus — pluggable event broadcast for circuit breaker, Guardian kill events,
drift alerts, etc.

Why this exists:
- Many systems quietly degrade (circuit opens, budget exceeded, PII detected)
  but only show up in logs. We want to push them to operators (Slack/PagerDuty/
  webhook) without coupling each call site to a specific transport.
- Handlers run in best-effort mode: a slow webhook must never block the request
  path or break a downstream subscriber.

Usage:

    bus = EventBus()
    bus.subscribe("kill_switch_triggered", slack_handler)
    await bus.publish(Event(name="kill_switch_triggered", payload={"agent_id": ...}))

The default global bus is intentionally process-local; in multi-instance
deployments, individual subscribers (e.g. WebhookHandler) are responsible for
making outbound calls if they want broadcast semantics. We do not bake in
distributed pub/sub here.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


# ============================================================
# Event taxonomy
# ============================================================


# Names are stable string identifiers. Adding a new event = add a constant.
EVENT_KILL_SWITCH_TRIGGERED = "kill_switch_triggered"
EVENT_BUDGET_WARNING = "budget_warning"  # soft limit (e.g. 80%) crossed
EVENT_BUDGET_EXCEEDED = "budget_exceeded"  # hard limit hit
EVENT_CIRCUIT_OPENED = "circuit_opened"
EVENT_CIRCUIT_CLOSED = "circuit_closed"
EVENT_PII_DETECTED = "pii_detected"
EVENT_PROVIDER_DRIFT_DETECTED = "provider_drift_detected"
EVENT_FALLBACK_USED = "fallback_used"


@dataclass
class Event:
    """
    An event flowing through the bus.

    `name` should be one of the EVENT_* constants. `payload` is freeform JSON-
    safe data (must be serializable for webhook handlers to work).
    """

    name: str
    payload: Dict[str, Any] = field(default_factory=dict)
    severity: str = "info"  # info | warning | critical
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "payload": self.payload,
            "severity": self.severity,
            "timestamp": self.timestamp,
        }


# ============================================================
# Handler types
# ============================================================


# Handlers may be sync or async. We always invoke them in async context but
# accept either, dispatching sync handlers via run_in_executor (rarely used).
SyncHandler = Callable[[Event], None]
AsyncHandler = Callable[[Event], Awaitable[None]]
Handler = Union[SyncHandler, AsyncHandler]


# ============================================================
# Bus implementation
# ============================================================


class EventBus:
    """
    Process-local async event bus.

    - subscribe(name, handler): register; supports wildcard "*" to receive all events.
    - publish(event): dispatch to matching handlers. Handlers are run concurrently
      with `asyncio.gather(return_exceptions=True)` so one failing handler never
      breaks another.
    - Handlers MUST NOT block the request path. publish() does NOT await
      handlers in flight if `fire_and_forget=True` (the default for safety).
    """

    WILDCARD = "*"

    def __init__(self, default_fire_and_forget: bool = True) -> None:
        self._handlers: Dict[str, List[Handler]] = {}
        self._default_fire_and_forget = default_fire_and_forget
        # Track in-flight tasks so close() can drain them. We use a set and
        # rely on `add_done_callback` to remove finished tasks, so the set
        # never grows unbounded under bursty publishes.
        self._inflight: "set[asyncio.Task[Any]]" = set()
        self._closed = False

    # ------------------------------------------------------------------
    # subscription management
    # ------------------------------------------------------------------

    def subscribe(self, event_name: str, handler: Handler) -> None:
        """Register a handler for a specific event name (or "*" for all)."""
        self._handlers.setdefault(event_name, []).append(handler)

    def unsubscribe(self, event_name: str, handler: Handler) -> bool:
        """Remove a handler. Returns True if it was registered."""
        handlers = self._handlers.get(event_name)
        if not handlers:
            return False
        try:
            handlers.remove(handler)
            return True
        except ValueError:
            return False

    def clear(self) -> None:
        """Remove all handlers. Useful for tests."""
        self._handlers.clear()

    # ------------------------------------------------------------------
    # publishing
    # ------------------------------------------------------------------

    async def publish(self, event: Event, fire_and_forget: Optional[bool] = None) -> None:
        """
        Dispatch event to all matching handlers.

        Args:
            event: Event to publish
            fire_and_forget: If True (default), schedule handlers and return
                immediately. If False, await all handlers before returning
                (used in tests + ordered scenarios).
        """
        if self._closed:
            return

        f_and_f = self._default_fire_and_forget if fire_and_forget is None else fire_and_forget
        targets = self._collect_handlers(event.name)
        if not targets:
            return

        if f_and_f:
            # Schedule but don't await. Track in a set + auto-remove on done so
            # the structure never grows unbounded under bursty publishes.
            task = asyncio.create_task(self._run_handlers(event, targets))
            self._inflight.add(task)
            task.add_done_callback(self._inflight.discard)
        else:
            await self._run_handlers(event, targets)

    def _collect_handlers(self, name: str) -> List[Handler]:
        targets: List[Handler] = []
        targets.extend(self._handlers.get(name, []))
        targets.extend(self._handlers.get(self.WILDCARD, []))
        return targets

    async def _run_handlers(self, event: Event, handlers: List[Handler]) -> None:
        coros = []
        for handler in handlers:
            coros.append(self._invoke(handler, event))
        results = await asyncio.gather(*coros, return_exceptions=True)
        for res, handler in zip(results, handlers):
            if isinstance(res, Exception):
                logger.warning(
                    "EventBus handler raised: handler=%s event=%s error=%s",
                    getattr(handler, "__qualname__", repr(handler)),
                    event.name,
                    res,
                )

    async def _invoke(self, handler: Handler, event: Event) -> None:
        if asyncio.iscoroutinefunction(handler):
            await handler(event)  # type: ignore[arg-type]
            return
        # Sync handler: run in default executor so it can't block event loop
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, handler, event)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    async def drain(self, timeout: float = 5.0) -> None:
        """Wait for all in-flight handler tasks to complete (best-effort)."""
        if not self._inflight:
            return
        # Snapshot now since the set mutates as tasks complete via callback.
        pending = list(self._inflight)
        try:
            await asyncio.wait_for(
                asyncio.gather(*pending, return_exceptions=True),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("EventBus.drain timed out after %.1fs; cancelling stragglers", timeout)
            for t in pending:
                if not t.done():
                    t.cancel()

    async def close(self) -> None:
        """Stop accepting new publishes and drain in-flight handlers."""
        self._closed = True
        await self.drain()
        self._handlers.clear()


# ============================================================
# Built-in handlers
# ============================================================


class WebhookHandler:
    """
    Handler that POSTs the event to a webhook URL.

    Uses httpx for non-blocking HTTP. Failures are logged but never raised.
    """

    def __init__(self, url: str, timeout: float = 5.0, headers: Optional[Dict[str, str]] = None) -> None:
        self.url = url
        self.timeout = timeout
        self.headers = headers or {"Content-Type": "application/json"}

    async def __call__(self, event: Event) -> None:
        # Lazy import so production code that doesn't use this handler isn't
        # forced to pay the import cost.
        try:
            import httpx  # noqa: F401
        except ImportError:
            logger.error("WebhookHandler requires httpx; install it to enable webhook delivery")
            return

        import httpx as _httpx

        body = event.to_dict()
        try:
            async with _httpx.AsyncClient(timeout=self.timeout) as client:
                await client.post(self.url, json=body, headers=self.headers)
        except Exception as exc:
            logger.warning("WebhookHandler post failed url=%s error=%s", self.url, exc)


class SlackHandler(WebhookHandler):
    """
    Slack-specific webhook handler that formats the event as a Slack message.

    Pass an Incoming Webhook URL. Severity drives emoji prefix.
    """

    SEVERITY_EMOJI = {
        "info": ":information_source:",
        "warning": ":warning:",
        "critical": ":rotating_light:",
    }

    async def __call__(self, event: Event) -> None:  # type: ignore[override]
        try:
            import httpx as _httpx
        except ImportError:
            logger.error("SlackHandler requires httpx")
            return

        emoji = self.SEVERITY_EMOJI.get(event.severity, ":bell:")
        text_lines = [f"{emoji} *{event.name}*"]
        for k, v in event.payload.items():
            text_lines.append(f"• `{k}`: {v}")
        body = {"text": "\n".join(text_lines)}

        try:
            async with _httpx.AsyncClient(timeout=self.timeout) as client:
                await client.post(self.url, json=body)
        except Exception as exc:
            logger.warning("SlackHandler post failed url=%s error=%s", self.url, exc)


# ============================================================
# Process-wide singleton (mirrors redis_client.py pattern)
# ============================================================


_default_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Return process-wide EventBus, creating one if needed."""
    global _default_bus
    if _default_bus is None:
        _default_bus = EventBus()
    return _default_bus


def set_event_bus(bus: Optional[EventBus]) -> None:
    """Override the process-wide event bus."""
    global _default_bus
    _default_bus = bus
