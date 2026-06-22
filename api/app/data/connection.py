import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import anyio
from asyncpg import Pool, Record, create_pool
from asyncpg.pool import PoolConnectionProxy

from app.constants.db import SCHEMA_PATH

logger = logging.getLogger(__name__)


class Database:
    """
    Manage an asyncpg connection pool, queries, and schema migrations.
    """

    def __init__(
        self,
        dsn: str,
        min_size: int = 1,
        max_size: int = 10,
    ) -> None:
        """
        Create a Database manager.
        """
        self.dsn: str = dsn
        self.min_size: int = min_size
        self.max_size: int = max_size
        self.pool: Pool | None = None

    async def init(self) -> None:
        """
        Initialize the asyncpg pool if not already done.
        """
        if self.pool is None:
            logger.info("Initializing database pool")
            try:
                self.pool = await create_pool(
                    dsn=self.dsn,
                    min_size=self.min_size,
                    max_size=self.max_size,
                )
            except Exception:
                logger.exception("Failed to initialize database pool")
                raise
            else:
                logger.info("Database pool initialized successfully")
        else:
            logger.debug("Database pool already initialized")

    async def disconnect(self) -> None:
        """
        Close and clean up the connection pool.
        """
        if self.pool:
            logger.info("Closing database connection pool")
            await self.pool.close()
            self.pool = None
            logger.info("Database pool closed")

    async def _ensure_pool(self) -> Pool:
        """
        Ensure the pool is up, initializing it if necessary.
        """
        if self.pool is None:
            logger.warning("Pool not initialized, calling init()")
            await self.init()

        if self.pool is None:
            msg = "Database pool is None after init()"
            logger.error(msg)
            raise RuntimeError(msg)

        return self.pool

    async def fetch(self, query: str, *args: object) -> list[Record]:
        """
        Run a SELECT query and return all rows.
        """
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            return await conn.fetch(query, *args)

    async def fetchrow(self, query: str, *args: object) -> Record | None:
        """
        Run a SELECT query and return the first row (or None).
        """
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def fetchval(
        self,
        query: str,
        *args: object,
        column: int = 0,
    ) -> object:
        """
        Run a query and return a single value from the first row.
        """
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            return await conn.fetchval(query, *args, column=column)

    async def execute(self, query: str, *args: object) -> str:
        """
        Run an INSERT/UPDATE/DELETE/DDL command.
        """
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            return await conn.execute(query, *args)

    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[PoolConnectionProxy[Record]]:
        """Acquire a connection and open a transaction for multi-statement atomic work."""
        pool = await self._ensure_pool()
        async with pool.acquire() as conn, conn.transaction():
            yield conn

    async def run_migrations(self) -> None:
        """
        Read the SQL in SCHEMA_PATH and execute it as one big transaction.
        """
        pool = await self._ensure_pool()

        p = anyio.Path(SCHEMA_PATH)
        if not await p.is_file():
            msg = f"Schema file not found at {SCHEMA_PATH}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        sql = (await p.read_text()).strip()
        if not sql:
            logger.warning("Schema file is empty; skipping migrations")
            return

        logger.info("Running migrations from %s", SCHEMA_PATH)
        async with pool.acquire() as conn:
            try:
                await conn.execute(sql)
            except Exception:
                logger.exception("Error executing migrations")
                raise
            else:
                logger.info("Migrations applied successfully")
