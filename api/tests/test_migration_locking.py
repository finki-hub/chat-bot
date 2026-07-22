from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
import pytest
from anyio.lowlevel import checkpoint
from asyncpg import PostgresError

from app.data import connection as connection_module
from app.data.connection import (
    _MIGRATION_ADVISORY_LOCK_KEY,
    _MIGRATION_ADVISORY_LOCK_SQL,
    _MIGRATION_ADVISORY_UNLOCK_SQL,
    Database,
)


class FakeTransaction:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> bool:
        return False


class MigrationConnection:
    def __init__(self) -> None:
        self.applied: set[str] = set()
        self.executed: list[tuple[str, tuple[str | int, ...]]] = []
        self.lock_held = False

    async def execute(self, sql: str, *args: str | int) -> str:
        await checkpoint()
        self.executed.append((sql, args))
        if sql == _MIGRATION_ADVISORY_LOCK_SQL:
            assert not self.lock_held
            self.lock_held = True
        elif sql == _MIGRATION_ADVISORY_UNLOCK_SQL:
            assert self.lock_held
            self.lock_held = False
        elif sql.startswith("INSERT INTO schema_migrations"):
            version = args[0]
            assert isinstance(version, str)
            self.applied.add(version)
        return "EXECUTE"

    async def fetch(self, sql: str) -> list[dict[str, str]]:
        await checkpoint()
        assert "FROM schema_migrations" in sql
        return [{"version": version} for version in sorted(self.applied)]

    def transaction(self) -> FakeTransaction:
        return FakeTransaction()


class FailingMigrationConnection(MigrationConnection):
    async def execute(self, sql: str, *args: str | int) -> str:
        result = await super().execute(sql, *args)
        if sql == "SELECT 1;":
            raise PostgresError("migration failed")
        return result


class BlockingMigrationConnection(MigrationConnection):
    def __init__(self) -> None:
        super().__init__()
        self.block_migration = True
        self.migration_started = anyio.Event()

    async def execute(self, sql: str, *args: str | int) -> str:
        result = await super().execute(sql, *args)
        if sql == "SELECT 1;" and self.block_migration:
            self.migration_started.set()
            await anyio.sleep_forever()
        return result


class FakePool:
    def __init__(self, connection: MigrationConnection) -> None:
        self.connection = connection

    @asynccontextmanager
    async def acquire(self) -> AsyncGenerator[MigrationConnection]:
        yield self.connection


def test_run_migrations_releases_advisory_lock_after_postgres_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def run() -> FailingMigrationConnection:
        # Given: a migration execution fails with a PostgreSQL error.
        await anyio.Path(tmp_path / "0001_failing.sql").write_text("SELECT 1;")
        monkeypatch.setattr(connection_module, "MIGRATIONS_PATH", tmp_path)
        connection = FailingMigrationConnection()
        database = Database("postgresql://unused")
        monkeypatch.setattr(database, "pool", FakePool(connection))

        # When: the migration runner propagates that error.
        with pytest.raises(PostgresError):
            await database.run_migrations()
        with pytest.raises(PostgresError):
            await database.run_migrations()
        return connection

    connection = anyio.run(run)

    # Then: it releases the session lock before returning the connection to the pool.
    assert connection.executed[-1] == (
        _MIGRATION_ADVISORY_UNLOCK_SQL,
        (_MIGRATION_ADVISORY_LOCK_KEY,),
    )


def test_run_migrations_releases_advisory_lock_when_cancelled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def run() -> BlockingMigrationConnection:
        # Given: migration work is blocked after the session lock is acquired.
        await anyio.Path(tmp_path / "0001_blocking.sql").write_text("SELECT 1;")
        monkeypatch.setattr(connection_module, "MIGRATIONS_PATH", tmp_path)
        connection = BlockingMigrationConnection()
        database = Database("postgresql://unused")
        monkeypatch.setattr(database, "pool", FakePool(connection))

        # When: the caller cancels the migration runner.
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(database.run_migrations)
            await connection.migration_started.wait()
            task_group.cancel_scope.cancel()

        connection.block_migration = False
        await database.run_migrations()
        return connection

    connection = anyio.run(run)

    # Then: shielded cleanup releases the lock despite cancellation.
    assert connection.lock_held is False
    assert connection.executed[-1] == (
        _MIGRATION_ADVISORY_UNLOCK_SQL,
        (_MIGRATION_ADVISORY_LOCK_KEY,),
    )
