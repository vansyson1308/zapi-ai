"""
Tests for src/core/redis_client.py — InMemoryRedisClient interface conformance.

We don't run RealRedisClient in CI (no live Redis); the in-memory
implementation is the SUT for behavior tests, and is also what production
uses when REDIS_URL is unset.
"""

from __future__ import annotations

import asyncio

import pytest

from src.core.redis_client import (
    InMemoryRedisClient,
    RedisClientError,
    get_redis_client,
    set_redis_client,
)


# ============================================================
# basic kv
# ============================================================


@pytest.mark.asyncio
async def test_set_and_get_roundtrip():
    client = InMemoryRedisClient()
    assert await client.set("k1", "hello")
    assert await client.get("k1") == "hello"


@pytest.mark.asyncio
async def test_get_missing_returns_none():
    client = InMemoryRedisClient()
    assert await client.get("missing") is None


@pytest.mark.asyncio
async def test_set_nx_only_writes_when_absent():
    client = InMemoryRedisClient()
    assert await client.set("lock", "owner1", only_if_absent=True)
    # second set with NX must fail
    assert not await client.set("lock", "owner2", only_if_absent=True)
    assert await client.get("lock") == "owner1"


@pytest.mark.asyncio
async def test_delete_returns_count():
    client = InMemoryRedisClient()
    await client.set("a", "1")
    await client.set("b", "2")
    assert await client.delete("a", "b", "c") == 2
    assert await client.get("a") is None


# ============================================================
# ttl semantics
# ============================================================


@pytest.mark.asyncio
async def test_ttl_returns_minus_two_when_missing():
    client = InMemoryRedisClient()
    assert await client.ttl("nope") == -2


@pytest.mark.asyncio
async def test_ttl_returns_minus_one_without_expiry():
    client = InMemoryRedisClient()
    await client.set("eternal", "x")
    assert await client.ttl("eternal") == -1


@pytest.mark.asyncio
async def test_ttl_decreases_with_time():
    client = InMemoryRedisClient()
    await client.set("eph", "x", ttl_seconds=5)
    ttl = await client.ttl("eph")
    assert 0 <= ttl <= 5


@pytest.mark.asyncio
async def test_expire_extends_ttl():
    client = InMemoryRedisClient()
    await client.set("k", "v")
    assert await client.expire("k", 60)
    ttl = await client.ttl("k")
    assert 50 <= ttl <= 60


@pytest.mark.asyncio
async def test_expire_on_missing_returns_false():
    client = InMemoryRedisClient()
    assert await client.expire("ghost", 10) is False


# ============================================================
# integer counters
# ============================================================


@pytest.mark.asyncio
async def test_incrby_starts_from_zero():
    client = InMemoryRedisClient()
    assert await client.incrby("n", 5) == 5
    assert await client.incrby("n", 3) == 8


@pytest.mark.asyncio
async def test_get_int_returns_zero_when_missing():
    client = InMemoryRedisClient()
    assert await client.get_int("nope") == 0


@pytest.mark.asyncio
async def test_incrby_sets_ttl_on_first_write_only():
    client = InMemoryRedisClient()
    await client.incrby("counter", 1, ttl_seconds=60)
    ttl1 = await client.ttl("counter")
    # second incrby with a TTL — the in-memory shim doesn't refresh it.
    await client.incrby("counter", 1, ttl_seconds=120)
    ttl2 = await client.ttl("counter")
    # We allow ±1s drift for clock granularity
    assert ttl2 <= ttl1 + 1


@pytest.mark.asyncio
async def test_incrby_concurrency_safe():
    client = InMemoryRedisClient()
    N = 200
    await asyncio.gather(*(client.incrby("c", 1) for _ in range(N)))
    assert await client.get_int("c") == N


# ============================================================
# float counters
# ============================================================


@pytest.mark.asyncio
async def test_incrbyfloat_accumulates():
    client = InMemoryRedisClient()
    assert (await client.incrbyfloat("f", 1.5)) == 1.5
    assert (await client.incrbyfloat("f", 0.25)) == pytest.approx(1.75)


@pytest.mark.asyncio
async def test_incrby_on_non_int_value_raises():
    client = InMemoryRedisClient()
    await client.set("not_int", "abc")
    with pytest.raises(RedisClientError):
        await client.incrby("not_int", 1)


# ============================================================
# sorted sets
# ============================================================


@pytest.mark.asyncio
async def test_zadd_then_zcard():
    client = InMemoryRedisClient()
    assert await client.zadd_score("z", "m1", 1.0) == 1
    assert await client.zadd_score("z", "m2", 2.0) == 1
    # update existing
    assert await client.zadd_score("z", "m1", 3.0) == 0
    assert await client.zcard("z") == 2


@pytest.mark.asyncio
async def test_zremrangebyscore():
    client = InMemoryRedisClient()
    for i in range(5):
        await client.zadd_score("z", f"m{i}", float(i))
    removed = await client.zremrangebyscore("z", 1.0, 3.0)
    assert removed == 3
    assert await client.zcard("z") == 2


# ============================================================
# streams (DLQ)
# ============================================================


@pytest.mark.asyncio
async def test_xadd_xrange_xdel():
    client = InMemoryRedisClient()
    id1 = await client.xadd("s", {"a": "1"})
    id2 = await client.xadd("s", {"a": "2"})
    entries = await client.xrange("s")
    assert len(entries) == 2
    assert entries[0][0] == id1
    assert entries[0][1] == {"a": "1"}

    removed = await client.xdel("s", id1)
    assert removed == 1
    assert len(await client.xrange("s")) == 1


@pytest.mark.asyncio
async def test_xadd_with_maxlen_trims_oldest():
    client = InMemoryRedisClient()
    for i in range(10):
        await client.xadd("s", {"i": str(i)}, maxlen=3)
    entries = await client.xrange("s")
    assert len(entries) == 3
    # The first remaining entry should be "i=7"
    assert entries[0][1]["i"] == "7"


# ============================================================
# global accessors
# ============================================================


@pytest.mark.asyncio
async def test_get_redis_client_returns_in_memory_when_no_url(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    set_redis_client(None)
    client = get_redis_client()
    assert isinstance(client, InMemoryRedisClient)


@pytest.mark.asyncio
async def test_set_redis_client_overrides_resolution():
    custom = InMemoryRedisClient()
    set_redis_client(custom)
    assert get_redis_client() is custom
    set_redis_client(None)


# ============================================================
# close is idempotent
# ============================================================


@pytest.mark.asyncio
async def test_close_in_memory_is_noop():
    client = InMemoryRedisClient()
    await client.close()
    # Should still be usable (close is a no-op for in-memory)
    await client.set("post", "x")
    assert await client.get("post") == "x"
