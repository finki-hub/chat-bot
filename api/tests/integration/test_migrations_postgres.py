import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import anyio
import asyncpg
import pytest
from anyio import TASK_STATUS_IGNORED
from anyio.abc import TaskStatus

from app.data import connection as connection_module
from app.data.connection import Database
from app.data.questions import create_question_query, update_question_query
from app.schemas.questions import CreateQuestionSchema, UpdateQuestionSchema

DATABASE_URL = os.getenv("TEST_DATABASE_URL")

pytestmark = pytest.mark.skipif(
    DATABASE_URL is None,
    reason="set TEST_DATABASE_URL to run real-PostgreSQL migration tests",
)


def _test_database_url() -> str:
    if DATABASE_URL is None:
        pytest.skip("set TEST_DATABASE_URL to run real-PostgreSQL migration tests")
    return DATABASE_URL


def _database_url_for(database_name: str) -> str:
    database_url = _test_database_url()
    parts = urlsplit(database_url)
    return urlunsplit(
        (parts.scheme, parts.netloc, f"/{database_name}", parts.query, parts.fragment),
    )


@asynccontextmanager
async def _temporary_database() -> AsyncIterator[str]:
    database_name = f"migration_test_{uuid4().hex}"
    admin_connection = await asyncpg.connect(_test_database_url(), database="postgres")
    database_created = False
    try:
        await admin_connection.execute(f"CREATE DATABASE {database_name}")
        database_created = True
        yield _database_url_for(database_name)
    finally:
        if database_created:
            await admin_connection.execute(
                "SELECT pg_terminate_backend(pid) "
                "FROM pg_stat_activity "
                "WHERE datname = $1 AND pid <> pg_backend_pid()",
                database_name,
            )
            await admin_connection.execute(
                f"DROP DATABASE IF EXISTS {database_name}",
            )
        await admin_connection.close()


def test_real_postgres_concurrent_migrations_apply_each_version_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def run() -> list[str]:
        # Given: two independent pools target an empty disposable PostgreSQL database.
        await anyio.Path(tmp_path / "0001_create_probe.sql").write_text(
            "CREATE TABLE migration_race_probe (id INTEGER PRIMARY KEY);",
        )
        await anyio.Path(tmp_path / "0002_seed_probe.sql").write_text(
            "INSERT INTO migration_race_probe (id) VALUES (1);",
        )
        monkeypatch.setattr(connection_module, "MIGRATIONS_PATH", tmp_path)
        start_migrations = anyio.Event()

        async def run_migrations_when_started(
            database: Database,
            *,
            task_status: TaskStatus[None] = TASK_STATUS_IGNORED,
        ) -> None:
            task_status.started()
            await start_migrations.wait()
            await database.run_migrations()

        async with _temporary_database() as database_url:
            first_database = Database(database_url, min_size=1, max_size=1)
            second_database = Database(database_url, min_size=1, max_size=1)
            await first_database.init()
            await second_database.init()
            try:
                # When: both runners begin migration execution at the same time.
                async with anyio.create_task_group() as task_group:
                    await task_group.start(run_migrations_when_started, first_database)
                    await task_group.start(run_migrations_when_started, second_database)
                    start_migrations.set()

                verification_connection = await asyncpg.connect(database_url)
                try:
                    rows = await verification_connection.fetch(
                        "SELECT version FROM schema_migrations ORDER BY version",
                    )
                    return [row["version"] for row in rows]
                finally:
                    await verification_connection.close()
            finally:
                await first_database.disconnect()
                await second_database.disconnect()

    versions = anyio.run(run)

    # Then: both calls succeed and every migration is recorded exactly once.
    assert versions == ["0001_create_probe.sql", "0002_seed_probe.sql"]


def test_question_links_round_trip_through_real_postgres() -> None:
    async def run() -> tuple[dict[str, str], dict[str, str]]:
        # Given: a migrated disposable database and a readable Discord profile link.
        async with _temporary_database() as database_url:
            database = Database(database_url, min_size=1, max_size=1)
            await database.init()
            try:
                await database.run_migrations()
                created = await create_question_query(
                    database,
                    CreateQuestionSchema.model_validate(
                        {
                            "name": "Discord identity integration",
                            "content": "Started by Delemangi.",
                            "links": {
                                "Discord profile": (
                                    "https://discord.com/users/198249751001563136"
                                ),
                            },
                        },
                    ),
                )
                assert created is not None

                # When: the link is replaced through the partial update path.
                updated = await update_question_query(
                    database,
                    created.name,
                    UpdateQuestionSchema.model_validate(
                        {
                            "links": {
                                "Discord server": "https://discord.gg/finki-studenti",
                            },
                        },
                    ),
                )
                assert updated is not None

                # Then: both writes decode from JSONB as the expected URL maps.
                return (
                    {label: str(url) for label, url in (created.links or {}).items()},
                    {label: str(url) for label, url in (updated.links or {}).items()},
                )
            finally:
                await database.disconnect()

    created_links, updated_links = anyio.run(run)

    assert created_links == {
        "Discord profile": "https://discord.com/users/198249751001563136",
    }
    assert updated_links == {
        "Discord server": "https://discord.gg/finki-studenti",
    }
