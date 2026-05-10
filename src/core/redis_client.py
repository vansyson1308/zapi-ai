"""
Redis client + state primitives for distributed AgentOps state.

Provides a thin, dependency-injectable async Redis pool plus an in-memory
fallback so the rest of the system can write `await client.incr(...)`
identically whether REDIS_URL is configured or not. Tests use
InMemoryRedisClient by default; production wires up RealRedisClient.

Design goals:
- One implementation interface; two backends (Redis, in-memory).
- Atomic primitives we actually use for AgentOps:
  * incrby + expire for budget counters
  * SET NX with TTL for distributed locks
  * pubsub for event bus broadcasts (best-effort; subscribers OPT-IN)
- Safe at import time: zero side effects, no eager connection.
- Closeable so tests don't leak.

NOT goals:
- Generic Redis library wrapper. We expose only what we need.
- Cluster/sharding - single instance is fine for the AgentOps wedge.
"""

from __future__ import annotations

import asyncio
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple


class RedisClientError(RuntimeError):
    """Raised on unrecoverable Redis client errors."""


class RedisClient(ABC):
    """Async Redis interface used by AgentOps modules."""

    @abstractmethod
    async def incrby(self, key: str, amount: int, ttl_seconds: Optional[int] = None) -> int:
        """Atomically add `amount` to integer value at `key`. Sets TTL on first write only."""

    @abstractmethod
    async def incrbyfloat(self, key: str, amount: float, ttl_seconds: Optional[int] = None) -> float:
        """Atomically add `amount` (float) to value at `key`."""

    @abstractmethod
    async def get_int(self, key: str) -> int:
        """Get integer value, returning 0 if missing or invalid."""

    @abstractmethod
    async def get_float(self, key: str) -> float:
        """Get float value, returning 0.0 if missing or invalid."""

    @abstractmethod
    async def set(
        self, key: str, value: str, ttl_seconds: Optional[int] = None, only_if_absent: bool = False
    ) -> bool:
        """Set a string value. Returns True if value was written (False if NX failed)."""

    @abstractmethod
    async def get(self, key: str) -> Optional[str]:
        """Get a string value, or None if missing."""

    @abstractmethod
    async def delete(self, *keys: str) -> int:
        """Delete keys. Returns number of keys deleted."""

    @abstractmethod
    async def expire(self, key: str, ttl_seconds: int) -> bool:
        """Set TTL on key. Returns True if key exists."""

    @abstractmethod
    async def ttl(self, key: str) -> int:
        """Return TTL in seconds. -1 if no expiry, -2 if missing."""

    @abstractmethod
    async def zadd_score(self, key: str, member: str, score: float) -> int:
        """Add member to sorted set with `score`. Returns 1 if added, 0 if updated."""

    @abstractmethod
    async def zremrangebyscore(self, key: str, min_score: float, max_score: float) -> int:
        """Remove sorted-set members with score in [min_score, max_score]. Returns count."""

    @abstractmethod
    async def zcard(self, key: str) -> int:
        """Cardinality of sorted set."""

    @abstractmethod
    async def xadd(self, stream: str, fields: Dict[str, str], maxlen: Optional[int] = None) -> str:
        """Append to a stream. Returns generated ID."""

    @abstractmethod
    async def xrange(self, stream: str, count: int = 100) -> List[Tuple[str, Dict[str, str]]]:
        """Read up to `count` entries from start of stream. Returns [(id, fields), ...]."""

    @abstractmethod
    async def xdel(self, stream: str, *ids: str) -> int:
        """Delete entries from a stream by ID."""

    @abstractmethod
    async def close(self) -> None:
        """Close any underlying connections."""


# ============================================================
# In-memory implementation (default; used in tests and local dev
# when REDIS_URL is not set)
# ============================================================


@dataclass
class _ZSetEntry:
    member: str
    score: float


@dataclass
class _StreamEntry:
    id: str
    fields: Dict[str, str]


@dataclass
class _Bucket:
    """A simple value container with optional expiry timestamp."""

    value: Any
    expires_at: Optional[float] = None  # absolute unix timestamp

    def is_expired(self) -> bool:
        return self.expires_at is not None and time.time() >= self.expires_at


class InMemoryRedisClient(RedisClient):
    """
    Pure-Python in-memory implementation for tests + zero-config dev runs.

    Safe under asyncio concurrency via a single internal lock. Not suitable
    for multi-process — that's what RealRedisClient is for.
    """

    def __init__(self) -> None:
        self._kv: Dict[str, _Bucket] = {}
        self._zsets: Dict[str, List[_ZSetEntry]] = {}
        self._streams: Dict[str, List[_StreamEntry]] = {}
        self._stream_seq: Dict[str, int] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _purge_if_expired(self, key: str) -> None:
        bucket = self._kv.get(key)
        if bucket is not None and bucket.is_expired():
            self._kv.pop(key, None)

    def _set_bucket(self, key: str, value: Any, ttl_seconds: Optional[int]) -> None:
        expires_at = time.time() + ttl_seconds if ttl_seconds is not None else None
        self._kv[key] = _Bucket(value=value, expires_at=expires_at)

    # ------------------------------------------------------------------
    # interface
    # ------------------------------------------------------------------

    async def incrby(self, key: str, amount: int, ttl_seconds: Optional[int] = None) -> int:
        async with self._lock:
            self._purge_if_expired(key)
            bucket = self._kv.get(key)
            if bucket is None:
                self._set_bucket(key, int(amount), ttl_seconds)
                return int(amount)
            try:
                current = int(bucket.value)
            except (TypeError, ValueError):
                raise RedisClientError(f"value at {key} is not an integer")
            new_value = current + int(amount)
            bucket.value = new_value
            return new_value

    async def incrbyfloat(self, key: str, amount: float, ttl_seconds: Optional[int] = None) -> float:
        async with self._lock:
            self._purge_if_expired(key)
            bucket = self._kv.get(key)
            if bucket is None:
                self._set_bucket(key, float(amount), ttl_seconds)
                return float(amount)
            try:
                current = float(bucket.value)
            except (TypeError, ValueError):
                raise RedisClientError(f"value at {key} is not a float")
            new_value = current + float(amount)
            bucket.value = new_value
            return new_value

    async def get_int(self, key: str) -> int:
        async with self._lock:
            self._purge_if_expired(key)
            bucket = self._kv.get(key)
            if bucket is None:
                return 0
            try:
                return int(bucket.value)
            except (TypeError, ValueError):
                return 0

    async def get_float(self, key: str) -> float:
        async with self._lock:
            self._purge_if_expired(key)
            bucket = self._kv.get(key)
            if bucket is None:
                return 0.0
            try:
                return float(bucket.value)
            except (TypeError, ValueError):
                return 0.0

    async def set(
        self, key: str, value: str, ttl_seconds: Optional[int] = None, only_if_absent: bool = False
    ) -> bool:
        async with self._lock:
            self._purge_if_expired(key)
            if only_if_absent and key in self._kv:
                return False
            self._set_bucket(key, value, ttl_seconds)
            return True

    async def get(self, key: str) -> Optional[str]:
        async with self._lock:
            self._purge_if_expired(key)
            bucket = self._kv.get(key)
            if bucket is None:
                return None
            return str(bucket.value)

    async def delete(self, *keys: str) -> int:
        async with self._lock:
            removed = 0
            for k in keys:
                self._purge_if_expired(k)
                if k in self._kv:
                    self._kv.pop(k)
                    removed += 1
            return removed

    async def expire(self, key: str, ttl_seconds: int) -> bool:
        async with self._lock:
            self._purge_if_expired(key)
            bucket = self._kv.get(key)
            if bucket is None:
                return False
            bucket.expires_at = time.time() + ttl_seconds
            return True

    async def ttl(self, key: str) -> int:
        async with self._lock:
            self._purge_if_expired(key)
            bucket = self._kv.get(key)
            if bucket is None:
                return -2
            if bucket.expires_at is None:
                return -1
            remaining = int(bucket.expires_at - time.time())
            return max(0, remaining)

    async def zadd_score(self, key: str, member: str, score: float) -> int:
        async with self._lock:
            zset = self._zsets.setdefault(key, [])
            for entry in zset:
                if entry.member == member:
                    entry.score = score
                    return 0
            zset.append(_ZSetEntry(member=member, score=score))
            return 1

    async def zremrangebyscore(self, key: str, min_score: float, max_score: float) -> int:
        async with self._lock:
            zset = self._zsets.get(key)
            if not zset:
                return 0
            before = len(zset)
            self._zsets[key] = [
                e for e in zset if not (min_score <= e.score <= max_score)
            ]
            return before - len(self._zsets[key])

    async def zcard(self, key: str) -> int:
        async with self._lock:
            return len(self._zsets.get(key, []))

    async def xadd(self, stream: str, fields: Dict[str, str], maxlen: Optional[int] = None) -> str:
        async with self._lock:
            seq = self._stream_seq.get(stream, 0) + 1
            self._stream_seq[stream] = seq
            entry_id = f"{int(time.time() * 1000)}-{seq}"
            entries = self._streams.setdefault(stream, [])
            entries.append(_StreamEntry(id=entry_id, fields=dict(fields)))
            # Match Redis semantics: XADD MAXLEN keeps the NEWEST `maxlen`
            # entries (i.e. drops the oldest). Real Redis docs:
            # https://redis.io/commands/xadd/ — MAXLEN trims from the head
            # (oldest end), preserving the most recent items.
            if maxlen is not None and len(entries) > maxlen:
                self._streams[stream] = entries[-maxlen:]
            return entry_id

    async def xrange(self, stream: str, count: int = 100) -> List[Tuple[str, Dict[str, str]]]:
        async with self._lock:
            entries = self._streams.get(stream, [])
            return [(e.id, dict(e.fields)) for e in entries[:count]]

    async def xdel(self, stream: str, *ids: str) -> int:
        async with self._lock:
            entries = self._streams.get(stream)
            if not entries:
                return 0
            id_set = set(ids)
            removed = sum(1 for e in entries if e.id in id_set)
            self._streams[stream] = [e for e in entries if e.id not in id_set]
            return removed

    async def close(self) -> None:
        # nothing to close
        return None


# ============================================================
# Real Redis-backed implementation
# ============================================================


class RealRedisClient(RedisClient):
    """
    Async Redis implementation using `redis.asyncio`. Imported lazily so the
    rest of the system can run with only the in-memory client when redis
    isn't installed.
    """

    def __init__(self, url: str) -> None:
        try:
            import redis.asyncio as redis_async  # type: ignore[import]
        except ImportError as exc:  # pragma: no cover - exercised in prod env only
            raise RedisClientError(
                "redis package is required for RealRedisClient; install with `pip install redis`"
            ) from exc
        self._redis_async = redis_async
        self._client = redis_async.from_url(url, encoding="utf-8", decode_responses=True)

    async def incrby(self, key: str, amount: int, ttl_seconds: Optional[int] = None) -> int:
        # Use a pipeline for atomic increment + conditional expire on first write.
        async with self._client.pipeline(transaction=True) as pipe:
            pipe.incrby(key, amount)
            if ttl_seconds is not None:
                pipe.expire(key, ttl_seconds, nx=True)  # only set TTL if not already set
            results = await pipe.execute()
        return int(results[0])

    async def incrbyfloat(self, key: str, amount: float, ttl_seconds: Optional[int] = None) -> float:
        async with self._client.pipeline(transaction=True) as pipe:
            pipe.incrbyfloat(key, amount)
            if ttl_seconds is not None:
                pipe.expire(key, ttl_seconds, nx=True)
            results = await pipe.execute()
        return float(results[0])

    async def get_int(self, key: str) -> int:
        value = await self._client.get(key)
        try:
            return int(value) if value is not None else 0
        except (TypeError, ValueError):
            return 0

    async def get_float(self, key: str) -> float:
        value = await self._client.get(key)
        try:
            return float(value) if value is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    async def set(
        self, key: str, value: str, ttl_seconds: Optional[int] = None, only_if_absent: bool = False
    ) -> bool:
        kwargs: Dict[str, Any] = {}
        if ttl_seconds is not None:
            kwargs["ex"] = ttl_seconds
        if only_if_absent:
            kwargs["nx"] = True
        result = await self._client.set(key, value, **kwargs)
        return bool(result)

    async def get(self, key: str) -> Optional[str]:
        return await self._client.get(key)

    async def delete(self, *keys: str) -> int:
        if not keys:
            return 0
        return int(await self._client.delete(*keys))

    async def expire(self, key: str, ttl_seconds: int) -> bool:
        return bool(await self._client.expire(key, ttl_seconds))

    async def ttl(self, key: str) -> int:
        return int(await self._client.ttl(key))

    async def zadd_score(self, key: str, member: str, score: float) -> int:
        return int(await self._client.zadd(key, {member: score}))

    async def zremrangebyscore(self, key: str, min_score: float, max_score: float) -> int:
        return int(await self._client.zremrangebyscore(key, min_score, max_score))

    async def zcard(self, key: str) -> int:
        return int(await self._client.zcard(key))

    async def xadd(self, stream: str, fields: Dict[str, str], maxlen: Optional[int] = None) -> str:
        kwargs: Dict[str, Any] = {}
        if maxlen is not None:
            kwargs["maxlen"] = maxlen
            kwargs["approximate"] = True
        return str(await self._client.xadd(stream, fields, **kwargs))

    async def xrange(self, stream: str, count: int = 100) -> List[Tuple[str, Dict[str, str]]]:
        # XRANGE returns a list of (id, fields-dict)
        raw = await self._client.xrange(stream, count=count)
        return [(str(item[0]), {str(k): str(v) for k, v in item[1].items()}) for item in raw]

    async def xdel(self, stream: str, *ids: str) -> int:
        if not ids:
            return 0
        return int(await self._client.xdel(stream, *ids))

    async def close(self) -> None:
        try:
            await self._client.aclose()  # type: ignore[attr-defined]
        except AttributeError:
            await self._client.close()


# ============================================================
# Module-level singleton accessors
# ============================================================


_default_client: Optional[RedisClient] = None


def get_redis_client() -> RedisClient:
    """
    Return the process-wide RedisClient.

    Resolution order:
      1. If a client has been set via `set_redis_client`, return it.
      2. Else if REDIS_URL env var is non-empty, build RealRedisClient.
      3. Else fall back to InMemoryRedisClient.
    """
    global _default_client
    if _default_client is not None:
        return _default_client

    url = os.getenv("REDIS_URL", "").strip()
    if url:
        try:
            _default_client = RealRedisClient(url)
        except RedisClientError:
            # If redis package is not installed, fall back to memory.
            _default_client = InMemoryRedisClient()
    else:
        _default_client = InMemoryRedisClient()

    return _default_client


def set_redis_client(client: Optional[RedisClient]) -> None:
    """Override the process-wide client. Pass None to reset to default-resolution."""
    global _default_client
    _default_client = client


async def reset_redis_client_for_tests() -> None:
    """Test helper: close existing client and reset, so next get_redis_client() rebuilds."""
    global _default_client
    if _default_client is not None:
        try:
            await _default_client.close()
        except Exception:
            pass
        _default_client = None
