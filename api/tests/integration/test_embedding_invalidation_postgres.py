import os
from datetime import datetime

import anyio
import pytest

from app.data.connection import Database
from tests.integration.embedding_postgres_test_support import database as _database
from tests.integration.embedding_postgres_test_support import (
    lifecycle_states as _states,
)
from tests.integration.embedding_postgres_test_support import (
    notifications as _notifications,
)
from tests.integration.embedding_postgres_test_support import (
    seed_current_rows as _seed_current_rows,
)

DATABASE_URL = os.getenv("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    DATABASE_URL is None,
    reason="set TEST_DATABASE_URL to run real-PostgreSQL embedding invalidation tests",
)

SEED_UPDATED_AT = datetime.fromisoformat("2000-01-01T00:00:00")
QUESTION_ID = "00000000-0000-0000-0000-000000000001"
DOCUMENT_ID = "00000000-0000-0000-0000-000000000002"
CHUNK_ONE_ID = "00000000-0000-0000-0000-000000000003"
CHUNK_TWO_ID = "00000000-0000-0000-0000-000000000004"
DIPLOMA_ID = "00000000-0000-0000-0000-000000000005"
PROFESSOR_DOCUMENT_ID = "00000000-0000-0000-0000-000000000006"
DIRTY_INSERT_PAYLOADS = [
    f"question:{QUESTION_ID}",
    f"chunk:{CHUNK_ONE_ID}",
    f"chunk:{CHUNK_TWO_ID}",
    f"diploma:{DIPLOMA_ID}",
    f"professor_document:{PROFESSOR_DOCUMENT_ID}",
]
INVALIDATION_PAYLOADS = [
    f"question:{QUESTION_ID}",
    f"chunk:{CHUNK_ONE_ID}",
    f"diploma:{DIPLOMA_ID}",
    f"professor_document:{PROFESSOR_DOCUMENT_ID}",
    f"chunk:{DOCUMENT_ID}",
]
LIFECYCLE_COLUMNS = (
    ("embedding_revision", "bigint", "NO", "1"),
    ("embedding_bge_m3_version", "text", "YES", None),
    ("embedding_bge_m3_updated_at", "timestamp without time zone", "YES", None),
)


async def _seed_dirty_rows(database: Database) -> None:
    await database.execute(
        """
        INSERT INTO document (id, name, title) VALUES ('00000000-0000-0000-0000-000000000002', 'document', 'Original');
        INSERT INTO question (id, name, content) VALUES ('00000000-0000-0000-0000-000000000001', 'question', 'original');
        INSERT INTO chunk (id, document_id, chunk_index, section, content) VALUES ('00000000-0000-0000-0000-000000000003', '00000000-0000-0000-0000-000000000002', 0, 'One', 'original one');
        INSERT INTO chunk (id, document_id, chunk_index, section, content) VALUES ('00000000-0000-0000-0000-000000000004', '00000000-0000-0000-0000-000000000002', 1, 'Two', 'original two');
        INSERT INTO diploma (id, external_id, title, description, mentor, status) VALUES ('00000000-0000-0000-0000-000000000005', 'diploma', 'Original', 'original', 'Mentor', 'defended');
        INSERT INTO professor_document (id, external_id, title, abstract) VALUES ('00000000-0000-0000-0000-000000000006', 'professor-document', 'Original', 'original');
        """,
    )


def test_real_postgres_adds_exact_lifecycle_columns_when_migrations_run() -> None:
    async def run() -> None:
        # Given: a clean disposable PostgreSQL database.
        async with _database() as database:
            # When: migrations are applied through the final migration.
            rows = await database.fetch(
                """
                SELECT table_name, column_name, data_type, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_name IN ('question', 'chunk', 'diploma', 'professor_document')
                  AND column_name IN ('embedding_revision', 'embedding_bge_m3_version', 'embedding_bge_m3_updated_at')
                """,
            )
            applied_version = await database.fetchval(
                "SELECT MAX(version) FROM schema_migrations",
            )

            # Then: each corpus owns the same durable lifecycle metadata.
            expected: set[tuple[str, str, str, str, str | None]] = {
                (table, column, data_type, nullable, default)
                for table in ("question", "chunk", "diploma", "professor_document")
                for column, data_type, nullable, default in LIFECYCLE_COLUMNS
            }
            assert {tuple(row) for row in rows} == expected
            assert applied_version == "0010_add_embedding_invalidation.sql"

    anyio.run(run)


def test_real_postgres_invalidates_all_corpora_when_sources_change() -> None:
    async def run() -> None:
        # Given: current BGE-M3 lifecycle state and a listener on the dirty channel.
        async with _database() as database, _notifications() as notifications:
            await _seed_current_rows(database)
            await notifications.wait_for_count(5)
            notifications.payloads.clear()

            # When: each corpus changes its embedding input and the document title changes.
            async with database.transaction() as connection:
                await connection.execute(
                    "UPDATE question SET content = 'changed' WHERE id = $1",
                    QUESTION_ID,
                )
                await connection.execute(
                    "UPDATE chunk SET content = 'changed' WHERE id = $1",
                    CHUNK_ONE_ID,
                )
                await connection.execute(
                    "UPDATE diploma SET description = 'changed' WHERE id = $1",
                    DIPLOMA_ID,
                )
                await connection.execute(
                    "UPDATE professor_document SET abstract = 'changed' WHERE id = $1",
                    PROFESSOR_DOCUMENT_ID,
                )
                await connection.execute(
                    "UPDATE document SET title = 'Changed' WHERE id = $1",
                    DOCUMENT_ID,
                )
            await notifications.wait_for_count(5)

            # Then: each committed edit dirties its row and produces the exact wake payload.
            assert sorted(notifications.payloads) == sorted(INVALIDATION_PAYLOADS)
            states = {state[0]: state[1:] for state in await _states(database)}
            assert [
                states[name][0]
                for name in (
                    "question",
                    "chunk-one",
                    "chunk-two",
                    "diploma",
                    "professor_document",
                )
            ] == [2, 3, 2, 2, 2]
            assert all(state[1:4] == (True, None, None) for state in states.values())
            assert all(state[4] > SEED_UPDATED_AT for state in states.values())
            document_updated_at = await database.fetchval(
                "SELECT updated_at FROM document WHERE id = $1",
                DOCUMENT_ID,
            )
            assert isinstance(document_updated_at, datetime)
            assert document_updated_at > SEED_UPDATED_AT

    anyio.run(run)


def test_real_postgres_notifies_stale_version_inserts_when_rows_have_vectors() -> None:
    async def run() -> None:
        # Given: default-dirty inserts and a listener on the dirty channel.
        async with _database() as database, _notifications() as notifications:
            await _seed_dirty_rows(database)
            await notifications.wait_for_count(5)
            assert sorted(notifications.payloads) == sorted(DIRTY_INSERT_PAYLOADS)
            notifications.payloads.clear()

            # When: each corpus receives an inserted vector with an old spec version.
            await database.execute(
                """
                INSERT INTO question (id, name, content, embedding_bge_m3, embedding_bge_m3_version) VALUES ('00000000-0000-0000-0000-000000000007', 'stale-question', 'stale', ('[' || repeat('0,', 1023) || '0]')::vector, 'old-bge-spec');
                INSERT INTO chunk (id, document_id, chunk_index, content, embedding_bge_m3, embedding_bge_m3_version) VALUES ('00000000-0000-0000-0000-000000000008', '00000000-0000-0000-0000-000000000002', 2, 'stale', ('[' || repeat('0,', 1023) || '0]')::vector, 'old-bge-spec');
                INSERT INTO diploma (id, external_id, title, description, mentor, status, embedding_bge_m3, embedding_bge_m3_version) VALUES ('00000000-0000-0000-0000-000000000009', 'stale-diploma', 'stale', 'stale', 'Mentor', 'defended', ('[' || repeat('0,', 1023) || '0]')::vector, 'old-bge-spec');
                INSERT INTO professor_document (id, external_id, title, abstract, embedding_bge_m3, embedding_bge_m3_version) VALUES ('00000000-0000-0000-0000-000000000010', 'stale-professor', 'stale', 'stale', ('[' || repeat('0,', 1023) || '0]')::vector, 'old-bge-spec');
                """,
            )
            await notifications.wait_for_count(4)

            # Then: every stale vector insert wakes its exact corpus resource.
            assert sorted(notifications.payloads) == sorted(
                [
                    "question:00000000-0000-0000-0000-000000000007",
                    "chunk:00000000-0000-0000-0000-000000000008",
                    "diploma:00000000-0000-0000-0000-000000000009",
                    "professor_document:00000000-0000-0000-0000-000000000010",
                ],
            )

    anyio.run(run)
