"""
StreamingManager — owns the per-provider stream loop and SSE chunk normalization.

Extracted from router.py so the streaming pipeline can be unit-tested in
isolation. The manager intentionally does NOT know about routing decisions
or fallback chains — it streams from one adapter and reports outcomes to
its caller through callbacks (record_success/record_failure).

Semantic-safety contract (preserved from old router):
- Tracker.mark_content_started() is called the first time content arrives,
  which flips the "no further fallback" flag. Once that flag is set, the
  manager continues streaming the SAME provider's output to completion or
  yields a stream-error chunk; it never silently restarts.
"""

from __future__ import annotations

import json
import time
from typing import AsyncIterator, Awaitable, Callable, List, Optional

from ..adapters.base import BaseAdapter
from ..core.errors import TwoApiException
from ..core.models import ChatCompletionRequest, Provider
from ..streaming.normalizer import StreamNormalizer
from ..streaming.tool_calls import ToolCallStreamTracker

from .fallback import RequestPhaseTracker


# Callback signatures shared with the rest of the routing layer.
RecordSuccess = Callable[[Provider, int], None]  # (provider, latency_ms)
RecordFailure = Callable[[Provider, str, Optional[int]], None]  # (provider, error, latency_ms)


class StreamingManager:
    """Stream chat completions from a single adapter and normalize the output."""

    def __init__(
        self,
        record_success: RecordSuccess,
        record_failure: RecordFailure,
    ) -> None:
        self._record_success = record_success
        self._record_failure = record_failure

    async def stream(
        self,
        adapter: BaseAdapter,
        provider: Provider,
        model: str,
        request: ChatCompletionRequest,
        request_id: str,
        tracker: RequestPhaseTracker,
    ) -> AsyncIterator[str]:
        """
        Stream from `adapter`. Yields fully-normalized SSE chunks.

        On success: yields a final `data: [DONE]\\n\\n` (either from the
        provider or synthesized).

        On exception: re-raises a TwoApiException so the caller can decide
        whether fallback is allowed.
        """
        start_time = time.time()
        provider_name = provider.value
        normalizer = StreamNormalizer(model=model, provider=provider_name, request_id=request_id)
        tool_tracker = ToolCallStreamTracker()

        try:
            async for raw_chunk in adapter.chat_completion_stream(request, request_id):
                events = self._normalize_chunk(raw_chunk, normalizer, provider_name, tool_tracker)

                for evt in events:
                    if evt.startswith("data: [DONE]"):
                        yield evt
                        latency_ms = int((time.time() - start_time) * 1000)
                        self._record_success(provider, latency_ms)
                        tracker.mark_completed()
                        return

                    self._inspect_event_for_phase_tracking(evt, tracker, tool_tracker)
                    yield evt

            # Adapter ended without [DONE] — synthesize one and report success.
            yield normalizer.create_done_event()
            latency_ms = int((time.time() - start_time) * 1000)
            self._record_success(provider, latency_ms)
            tracker.mark_completed()

        except TwoApiException:
            latency_ms = int((time.time() - start_time) * 1000)
            self._record_failure(provider, "stream_error", latency_ms)
            raise
        except Exception:
            latency_ms = int((time.time() - start_time) * 1000)
            self._record_failure(provider, "stream_error", latency_ms)
            raise

    # ------------------------------------------------------------------
    # internal: chunk normalization (provider-format-aware)
    # ------------------------------------------------------------------

    def _normalize_chunk(
        self,
        raw_chunk: str,
        normalizer: StreamNormalizer,
        provider_name: str,
        tool_tracker: ToolCallStreamTracker,
    ) -> List[str]:
        """Convert one (or many) provider chunks into OpenAI-compatible SSE events."""
        events: List[str] = []
        if not isinstance(raw_chunk, str):
            return events

        for line in (ln.strip() for ln in raw_chunk.splitlines()):
            if not line or not line.startswith("data: "):
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

    def _inspect_event_for_phase_tracking(
        self,
        evt: str,
        tracker: RequestPhaseTracker,
        tool_tracker: ToolCallStreamTracker,
    ) -> None:
        """Update phase tracker + tool tracker by looking at the SSE payload."""
        try:
            payload = evt[len("data: "):].strip()
            parsed = json.loads(payload)
            # Guard against `{"choices": []}` which would raise IndexError on [0].
            choices = parsed.get("choices") or []
            if not choices:
                return
            choice = choices[0] or {}
            delta = choice.get("delta", {}) or {}
            content = delta.get("content")
            if content:
                if tracker.can_fallback():
                    tracker.mark_content_started(content)
                else:
                    tracker.append_content(content)

            for tc in delta.get("tool_calls", []) or []:
                f = tc.get("function", {}) or {}
                tool_tracker.update_call(
                    index=tc.get("index", 0),
                    id=tc.get("id"),
                    function_name=f.get("name"),
                    arguments_delta=f.get("arguments", ""),
                )
        except Exception:
            # Non-JSON or malformed events are safe to ignore for tracking purposes.
            return
