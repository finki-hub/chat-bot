import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import anyio
from asyncpg import Pool, PostgresError, Record, create_pool
from asyncpg.pool import PoolConnectionProxy

from app.constants.db import MIGRATIONS_PATH

logger = logging.getLogger(__name__)

_MIGRATION_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version TEXT PRIMARY KEY,
    applied_at TIMESTAMP NOT NULL DEFAULT NOW()
)
"""

_APPLIED_MIGRATIONS_SQL = "SELECT version FROM schema_migrations ORDER BY version ASC"

_RECORD_MIGRATION_SQL = "INSERT INTO schema_migrations (version) VALUES ($1)"

_MIGRATION_ADVISORY_LOCK_KEY: Final[int] = 0x46494E4B4D494752
_MIGRATION_ADVISORY_LOCK_SQL: Final[str] = "SELECT pg_advisory_lock($1)"
_MIGRATION_ADVISORY_UNLOCK_SQL: Final[str] = "SELECT pg_advisory_unlock($1)"


@dataclass(frozen=True, slots=True)
class Migration:
    """A versioned SQL migration file ready to apply."""

    version: str
    sql: str


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
        Apply versioned SQL migrations that have not been recorded yet.
        """
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            lock_acquired = False
            try:
                await conn.execute(
                    _MIGRATION_ADVISORY_LOCK_SQL,
                    _MIGRATION_ADVISORY_LOCK_KEY,
                )
                lock_acquired = True

                migrations = await _load_migrations(MIGRATIONS_PATH)
                if not migrations:
                    logger.warning(
                        "No database migrations found in %s",
                        MIGRATIONS_PATH,
                    )
                    return

                logger.info("Running database migrations from %s", MIGRATIONS_PATH)
                await conn.execute(_MIGRATION_TABLE_SQL)
                applied = {
                    row["version"] for row in await conn.fetch(_APPLIED_MIGRATIONS_SQL)
                }
                pending = [m for m in migrations if m.version not in applied]

                if not pending:
                    logger.info("Database schema is up to date")
                    return

                for migration in pending:
                    logger.info("Applying database migration %s", migration.version)
                    try:
                        async with conn.transaction():
                            await conn.execute(migration.sql)
                            await conn.execute(_RECORD_MIGRATION_SQL, migration.version)
                    except PostgresError:
                        logger.exception(
                            "Failed to apply database migration %s",
                            migration.version,
                        )
                        raise

                logger.info("Applied %d database migration(s)", len(pending))
            finally:
                if lock_acquired:
                    with anyio.CancelScope(shield=True):
                        await conn.execute(
                            _MIGRATION_ADVISORY_UNLOCK_SQL,
                            _MIGRATION_ADVISORY_LOCK_KEY,
                        )


async def _load_migrations(path: Path) -> list[Migration]:
    """Load non-empty .sql files from the migration directory in filename order."""
    migration_dir = anyio.Path(path)
    if not await migration_dir.is_dir():
        msg = f"Migration directory not found at {path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    migrations: list[Migration] = []
    for migration_path in _sql_migration_paths(path):
        sql = (await anyio.Path(migration_path).read_text()).strip()
        if not sql:
            logger.warning("Skipping empty migration file %s", migration_path)
            continue
        migrations.append(Migration(version=str(migration_path.name), sql=sql))

    return migrations


def _sql_migration_paths(path: Path) -> list[Path]:
    """Return SQL migration paths in deterministic filename order."""
    return sorted(path.glob("*.sql"))
