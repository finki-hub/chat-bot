from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
from anyio.lowlevel import checkpoint

from app.data import connection as connection_module
from app.data.connection import (
    _MIGRATION_ADVISORY_LOCK_KEY,
    _MIGRATION_ADVISORY_LOCK_SQL,
    _MIGRATION_ADVISORY_UNLOCK_SQL,
    Database,
    Migration,
    _load_migrations,
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


class FakeConnection:
    def __init__(self, applied: set[str]) -> None:
        self.applied = applied
        self.executed: list[tuple[str, tuple[str | int, ...]]] = []

    async def execute(self, sql: str, *args: str | int) -> str:
        await checkpoint()
        self.executed.append((sql, args))
        if sql.startswith("INSERT INTO schema_migrations"):
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


class BlockingLockConnection(FakeConnection):
    def __init__(self, applied: set[str]) -> None:
        super().__init__(applied)
        self.lock_waiting = anyio.Event()
        self.release_lock = anyio.Event()

    async def execute(self, sql: str, *args: str | int) -> str:
        if sql == _MIGRATION_ADVISORY_LOCK_SQL:
            self.executed.append((sql, args))
            self.lock_waiting.set()
            await self.release_lock.wait()
            return "LOCK"
        return await super().execute(sql, *args)


class FakePool:
    def __init__(self, conn: FakeConnection) -> None:
        self.conn = conn

    @asynccontextmanager
    async def acquire(self) -> AsyncGenerator[FakeConnection]:
        yield self.conn


def test_load_migrations_returns_non_empty_sql_files_in_filename_order(
    tmp_path: Path,
) -> None:
    async def run() -> list[Migration]:
        # Given: migration files out of order, plus an empty file.
        await anyio.Path(tmp_path / "0002_second.sql").write_text("SELECT 2;")
        await anyio.Path(tmp_path / "0001_first.sql").write_text("SELECT 1;")
        await anyio.Path(tmp_path / "0003_empty.sql").write_text("\n")

        # When: migrations are loaded from disk.
        return await _load_migrations(tmp_path)

    migrations = anyio.run(run)

    # Then: only executable migrations are returned in deterministic order.
    assert migrations == [
        Migration(version="0001_first.sql", sql="SELECT 1;"),
        Migration(version="0002_second.sql", sql="SELECT 2;"),
    ]


def test_chat_user_migration_allows_supported_auth_providers() -> None:
    # Given: forward migrations own the persisted provider whitelist updates.
    microsoft_migration = Path(
        "resources/migrations/0003_allow_microsoft_chat_user_provider.sql",
    ).read_text()
    discord_migration = Path(
        "resources/migrations/0006_allow_discord_chat_user_provider.sql",
    ).read_text()

    # When / Then: every provider accepted by the web BFF and Discord bot is allowed.
    assert "provider IN ('google', 'microsoft-entra-id')" in microsoft_migration
    assert (
        "provider IN ('google', 'microsoft-entra-id', 'discord')" in discord_migration
    )


def test_chat_message_parts_migration_adds_nullable_jsonb_column() -> None:
    # Given: persisted chat messages need durable AI SDK UI parts.
    migration = Path(
        "resources/migrations/0005_add_chat_message_parts.sql",
    ).read_text()

    # When / Then: the forward migration keeps legacy rows valid while adding parts.
    assert "ADD COLUMN IF NOT EXISTS parts JSONB" in migration
    assert "NOT NULL" not in migration


def test_embedding_invalidation_migration_declares_lifecycle_contract() -> None:
    # Given: all corpus invalidation behavior is owned by one forward migration.
    migration_path = Path("resources/migrations/0010_add_embedding_invalidation.sql")
    assert migration_path.is_file(), "missing 0010 lifecycle DDL migration"
    migration = migration_path.read_text()

    # When / Then: each corpus receives durable lifecycle state and safe triggers.
    assert migration.count("embedding_revision BIGINT NOT NULL DEFAULT 1") == 4
    for table in ("question", "chunk", "diploma", "professor_document"):
        assert f"ALTER TABLE {table}" in migration
        assert f"embedding_notify_dirty('{table}')" in migration
    assert "embedding_bge_m3_version TEXT" in migration
    assert "embedding_bge_m3_updated_at TIMESTAMP" in migration
    assert "IS DISTINCT FROM" in migration
    assert "pg_notify('embedding_dirty'" in migration
    assert "document_title_invalidate_chunks" in migration


def test_run_migrations_applies_only_pending_versions(
    monkeypatch,
    tmp_path: Path,
) -> None:
    async def run() -> FakeConnection:
        # Given: one migration is already recorded and one is pending.
        await anyio.Path(tmp_path / "0001_first.sql").write_text("SELECT 1;")
        await anyio.Path(tmp_path / "0002_second.sql").write_text("SELECT 2;")
        monkeypatch.setattr(connection_module, "MIGRATIONS_PATH", tmp_path)

        conn = FakeConnection(applied={"0001_first.sql"})
        db = Database("postgresql://unused")
        monkeypatch.setattr(db, "pool", FakePool(conn))

        # When: migrations run.
        await db.run_migrations()
        return conn

    conn = anyio.run(run)

    # Then: the runner ensures the history table and applies only the pending file.
    assert conn.executed == [
        (_MIGRATION_ADVISORY_LOCK_SQL, (_MIGRATION_ADVISORY_LOCK_KEY,)),
        (
            """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version TEXT PRIMARY KEY,
    applied_at TIMESTAMP NOT NULL DEFAULT NOW()
)
""",
            (),
        ),
        ("SELECT 2;", ()),
        ("INSERT INTO schema_migrations (version) VALUES ($1)", ("0002_second.sql",)),
        (_MIGRATION_ADVISORY_UNLOCK_SQL, (_MIGRATION_ADVISORY_LOCK_KEY,)),
    ]


def test_run_migrations_is_idempotent(monkeypatch, tmp_path: Path) -> None:
    async def run() -> FakeConnection:
        # Given: every migration file has already been recorded.
        await anyio.Path(tmp_path / "0001_first.sql").write_text("SELECT 1;")
        monkeypatch.setattr(connection_module, "MIGRATIONS_PATH", tmp_path)

        conn = FakeConnection(applied={"0001_first.sql"})
        db = Database("postgresql://unused")
        monkeypatch.setattr(db, "pool", FakePool(conn))

        # When: migrations run again.
        await db.run_migrations()
        return conn

    conn = anyio.run(run)

    # Then: no migration SQL is re-executed.
    assert conn.executed == [
        (_MIGRATION_ADVISORY_LOCK_SQL, (_MIGRATION_ADVISORY_LOCK_KEY,)),
        (
            """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version TEXT PRIMARY KEY,
    applied_at TIMESTAMP NOT NULL DEFAULT NOW()
)
""",
            (),
        ),
        (_MIGRATION_ADVISORY_UNLOCK_SQL, (_MIGRATION_ADVISORY_LOCK_KEY,)),
    ]


def test_run_migrations_cancels_while_waiting_for_advisory_lock(
    monkeypatch,
    tmp_path: Path,
) -> None:
    async def run() -> BlockingLockConnection:
        # Given: one pending migration and a runner blocked on advisory-lock wait.
        await anyio.Path(tmp_path / "0001_first.sql").write_text("SELECT 1;")
        monkeypatch.setattr(connection_module, "MIGRATIONS_PATH", tmp_path)

        conn = BlockingLockConnection(applied=set())
        db = Database("postgresql://unused")
        monkeypatch.setattr(db, "pool", FakePool(conn))

        # When: cancellation arrives before PostgreSQL grants the advisory lock.
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(db.run_migrations)
            await conn.lock_waiting.wait()
            task_group.cancel_scope.cancel()
            conn.release_lock.set()
            await checkpoint()

        return conn

    conn = anyio.run(run)

    # Then: cancellation unwinds without running migration SQL or unlocking a lock
    # the task never acquired.
    assert conn.executed == [
        (_MIGRATION_ADVISORY_LOCK_SQL, (_MIGRATION_ADVISORY_LOCK_KEY,)),
    ]
