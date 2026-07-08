from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
from anyio.lowlevel import checkpoint

from app.data import connection as connection_module
from app.data.connection import Database, Migration, _load_migrations


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
        self.executed: list[tuple[str, tuple[str, ...]]] = []

    async def execute(self, sql: str, *args: str) -> str:
        await checkpoint()
        self.executed.append((sql, args))
        if sql.startswith("INSERT INTO schema_migrations"):
            self.applied.add(args[0])
        return "EXECUTE"

    async def fetch(self, sql: str) -> list[dict[str, str]]:
        await checkpoint()
        assert "FROM schema_migrations" in sql
        return [{"version": version} for version in sorted(self.applied)]

    def transaction(self) -> FakeTransaction:
        return FakeTransaction()


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
    # Given: a forward migration owns the persisted provider whitelist update.
    migration = Path(
        "resources/migrations/0003_allow_microsoft_chat_user_provider.sql",
    ).read_text()

    # When / Then: both Auth.js providers accepted by the web BFF are allowed.
    assert "provider IN ('google', 'microsoft-entra-id')" in migration


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
        (
            """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version TEXT PRIMARY KEY,
    applied_at TIMESTAMP NOT NULL DEFAULT NOW()
)
""",
            (),
        ),
    ]
