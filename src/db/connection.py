"""
2api.ai - Database Connection Pool

Async PostgreSQL connection pool using asyncpg.
"""

import os
from typing import Optional
from contextlib import asynccontextmanager

import asyncpg


class DatabasePool:
    """
    Async PostgreSQL connection pool.

    Usage:
        pool = DatabasePool()
        await pool.connect()

        async with pool.acquire() as conn:
            result = await conn.fetch("SELECT * FROM tenants")

        await pool.close()
    """

    _instance: Optional["DatabasePool"] = None
    _pool: Optional[asyncpg.Pool] = None

    def __init__(
        self,
        dsn: Optional[str] = None,
        min_size: int = 2,
        max_size: int = 10,
    ):
        """
        Initialize database pool configuration.

        Args:
            dsn: PostgreSQL connection string. If None, reads from DATABASE_URL env.
            min_size: Minimum number of connections in pool.
            max_size: Maximum number of connections in pool.
        """
        self.dsn = dsn or os.getenv(
            "DATABASE_URL",
            "postgresql://postgres:postgres@localhost:5432/twoapi"
        )
        self.min_size = min_size
        self.max_size = max_size

    async def connect(self) -> None:
        """Create the connection pool."""
        if self._pool is not None:
            return

        self._pool = await asyncpg.create_pool(
            self.dsn,
            min_size=self.min_size,
            max_size=self.max_size,
            command_timeout=60,
        )

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    @asynccontextmanager
    async def acquire(self):
        """
        Acquire a connection from the pool.

        Usage:
            async with pool.acquire() as conn:
                await conn.execute(...)
        """
        if self._pool is None:
            raise RuntimeError("Database pool not connected. Call connect() first.")

        async with self._pool.acquire() as connection:
            yield connection

    async def execute(self, query: str, *args) -> str:
        """Execute a query and return status."""
        async with self.acquire() as conn:
            return await conn.execute(query, *args)

    async def fetch(self, query: str, *args) -> list:
        """Execute a query and return all rows."""
        async with self.acquire() as conn:
            return await conn.fetch(query, *args)

    async def fetchrow(self, query: str, *args) -> Optional[asyncpg.Record]:
        """Execute a query and return one row."""
        async with self.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def fetchval(self, query: str, *args):
        """Execute a query and return single value."""
        async with self.acquire() as conn:
            return await conn.fetchval(query, *args)

    @property
    def is_connected(self) -> bool:
        """Check if pool is connected."""
        return self._pool is not None

    @classmethod
    def get_instance(cls) -> "DatabasePool":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


# Global pool instance for dependency injection
_db_pool: Optional[DatabasePool] = None


async def init_db(dsn: Optional[str] = None) -> DatabasePool:
    """
    Initialize the global database pool.

    Call this once at application startup.
    """
    global _db_pool
    _db_pool = DatabasePool(dsn=dsn)
    await _db_pool.connect()
    return _db_pool


async def close_db() -> None:
    """Close the global database pool."""
    global _db_pool
    if _db_pool is not None:
        await _db_pool.close()
        _db_pool = None


def get_db() -> DatabasePool:
    """
    Get the global database pool.

    Raises:
        RuntimeError: If database not initialized.
    """
    if _db_pool is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return _db_pool


def get_db_optional() -> Optional[DatabasePool]:
    """
    Get the global database pool, or None if not initialized.

    Useful for local mode where DB is optional.
    """
    return _db_pool
