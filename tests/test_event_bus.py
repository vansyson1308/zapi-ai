"""
Tests for src/observability/event_bus.py.

Covers:
  - subscribe / publish (sync + async handlers)
  - wildcard handlers
  - one failing handler does not break the others
  - fire_and_forget vs await semantics
  - drain + close
"""

from __future__ import annotations

import asyncio
from typing import List

import pytest

from src.observability.event_bus import (
    EVENT_KILL_SWITCH_TRIGGERED,
    Event,
    EventBus,
)


# ============================================================
# basic dispatch
# ============================================================


@pytest.mark.asyncio
async def test_async_handler_receives_event():
    bus = EventBus(default_fire_and_forget=False)
    received: List[Event] = []

    async def handler(evt: Event):
        received.append(evt)

    bus.subscribe("hello", handler)
    await bus.publish(Event(name="hello", payload={"x": 1}))
    assert len(received) == 1
    assert received[0].payload == {"x": 1}


@pytest.mark.asyncio
async def test_sync_handler_receives_event():
    bus = EventBus(default_fire_and_forget=False)
    received: List[Event] = []

    def handler(evt: Event):
        received.append(evt)

    bus.subscribe("hello", handler)
    await bus.publish(Event(name="hello"))
    assert len(received) == 1


@pytest.mark.asyncio
async def test_no_handler_for_event_is_noop():
    bus = EventBus(default_fire_and_forget=False)
    # Should not raise
    await bus.publish(Event(name="orphan"))


# ============================================================
# wildcard
# ============================================================


@pytest.mark.asyncio
async def test_wildcard_handler_receives_all_events():
    bus = EventBus(default_fire_and_forget=False)
    received: List[Event] = []

    async def all_handler(evt: Event):
        received.append(evt)

    bus.subscribe("*", all_handler)
    await bus.publish(Event(name="a"))
    await bus.publish(Event(name="b"))
    assert [e.name for e in received] == ["a", "b"]


@pytest.mark.asyncio
async def test_wildcard_runs_alongside_named_handler():
    bus = EventBus(default_fire_and_forget=False)
    named: List[Event] = []
    wildcard: List[Event] = []

    async def n(evt: Event):
        named.append(evt)

    async def w(evt: Event):
        wildcard.append(evt)

    bus.subscribe("topic", n)
    bus.subscribe("*", w)
    await bus.publish(Event(name="topic"))
    assert len(named) == 1
    assert len(wildcard) == 1


# ============================================================
# resilience
# ============================================================


@pytest.mark.asyncio
async def test_failing_handler_does_not_break_others():
    bus = EventBus(default_fire_and_forget=False)
    survived: List[Event] = []

    async def bad(evt: Event):
        raise RuntimeError("boom")

    async def good(evt: Event):
        survived.append(evt)

    bus.subscribe("topic", bad)
    bus.subscribe("topic", good)
    # Should not raise
    await bus.publish(Event(name="topic"))
    assert len(survived) == 1


# ============================================================
# fire and forget
# ============================================================


@pytest.mark.asyncio
async def test_fire_and_forget_returns_quickly():
    bus = EventBus(default_fire_and_forget=True)
    completed = asyncio.Event()

    async def slow(evt: Event):
        await asyncio.sleep(0.05)
        completed.set()

    bus.subscribe("topic", slow)

    # publish must not block on the slow handler
    await bus.publish(Event(name="topic"))
    # handler is still running
    assert not completed.is_set()

    # drain waits for handlers
    await bus.drain()
    assert completed.is_set()


@pytest.mark.asyncio
async def test_fire_and_forget_can_be_overridden_per_call():
    bus = EventBus(default_fire_and_forget=True)
    done = []

    async def h(evt: Event):
        await asyncio.sleep(0.01)
        done.append(evt.name)

    bus.subscribe("topic", h)
    await bus.publish(Event(name="topic"), fire_and_forget=False)
    assert done == ["topic"]


# ============================================================
# unsubscribe + clear
# ============================================================


@pytest.mark.asyncio
async def test_unsubscribe_removes_handler():
    bus = EventBus(default_fire_and_forget=False)
    received: List[Event] = []

    async def h(evt: Event):
        received.append(evt)

    bus.subscribe("t", h)
    assert bus.unsubscribe("t", h) is True
    await bus.publish(Event(name="t"))
    assert received == []


@pytest.mark.asyncio
async def test_unsubscribe_unknown_returns_false():
    bus = EventBus(default_fire_and_forget=False)

    async def h(evt: Event):
        pass

    assert bus.unsubscribe("nope", h) is False


@pytest.mark.asyncio
async def test_clear_drops_all_handlers():
    bus = EventBus(default_fire_and_forget=False)
    received: List[Event] = []

    async def h(evt: Event):
        received.append(evt)

    bus.subscribe("t", h)
    bus.clear()
    await bus.publish(Event(name="t"))
    assert received == []


# ============================================================
# close
# ============================================================


@pytest.mark.asyncio
async def test_close_stops_new_publishes():
    bus = EventBus(default_fire_and_forget=False)
    received: List[Event] = []

    async def h(evt: Event):
        received.append(evt)

    bus.subscribe("t", h)
    await bus.close()
    await bus.publish(Event(name="t"))
    assert received == []
