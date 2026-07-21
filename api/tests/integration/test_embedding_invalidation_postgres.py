import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime

import anyio
import pytest
from asyncpg import connect

from app.data.connection import Database

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
LIFECYCLE_COLUMNS = (
    ("embedding_revision", "bigint", "NO", "1"),
    ("embedding_bge_m3_version", "text", "YES", None),
    ("embedding_bge_m3_updated_at", "timestamp without time zone", "YES", None),
)
type LifecycleState = tuple[str, int, bool, str | None, datetime | None, datetime]


class RollbackRequestedError(Exception):
    pass


class NotificationCollector:
    def __init__(self) -> None:
        self.payloads: list[str] = []
        self._event = anyio.Event()

    def receive(self, _connection, _process_id, _channel, payload) -> None:
        self.payloads.append(payload)
        self._event.set()

    async def wait_for_count(self, expected_count: int) -> None:
        while len(self.payloads) < expected_count:
            event = self._event
            with anyio.fail_after(2):
                await event.wait()
            if event is self._event:
                self._event = anyio.Event()
        self._event = anyio.Event()

    async def assert_no_notification(self) -> None:
        previous_count = len(self.payloads)
        event = self._event
        with anyio.move_on_after(0.25) as scope:
            await event.wait()
        assert scope.cancelled_caught
        assert len(self.payloads) == previous_count


def _test_database_url() -> str:
    return os.environ["TEST_DATABASE_URL"]


@asynccontextmanager
async def _database() -> AsyncIterator[Database]:
    database = Database(_test_database_url(), min_size=1, max_size=4)
    await database.init()
    await database.run_migrations()
    await database.execute(
        "TRUNCATE question, document, diploma, professor_document CASCADE",
    )
    try:
        yield database
    finally:
        await database.disconnect()


@asynccontextmanager
async def _notifications() -> AsyncIterator[NotificationCollector]:
    connection = await connect(_test_database_url())
    collector = NotificationCollector()
    await connection.add_listener("embedding_dirty", collector.receive)
    try:
        yield collector
    finally:
        await connection.remove_listener("embedding_dirty", collector.receive)
        await connection.close()


async def _seed_current_rows(database: Database) -> None:
    await database.execute(
        """
        INSERT INTO document (id, name, title, updated_at) VALUES ('00000000-0000-0000-0000-000000000002', 'document', 'Original', TIMESTAMP '2000-01-01');
        INSERT INTO question (id, name, content, embedding_bge_m3, embedding_bge_m3_version, embedding_bge_m3_updated_at, updated_at) VALUES ('00000000-0000-0000-0000-000000000001', 'question', 'original', ('[' || repeat('0,', 1023) || '0]')::vector, 'test-bge-m3-v1', TIMESTAMP '2000-01-01', TIMESTAMP '2000-01-01');
        INSERT INTO chunk (id, document_id, chunk_index, section, content, embedding_bge_m3, embedding_bge_m3_version, embedding_bge_m3_updated_at, updated_at) VALUES ('00000000-0000-0000-0000-000000000003', '00000000-0000-0000-0000-000000000002', 0, 'One', 'original one', ('[' || repeat('0,', 1023) || '0]')::vector, 'test-bge-m3-v1', TIMESTAMP '2000-01-01', TIMESTAMP '2000-01-01');
        INSERT INTO chunk (id, document_id, chunk_index, section, content, embedding_bge_m3, embedding_bge_m3_version, embedding_bge_m3_updated_at, updated_at) VALUES ('00000000-0000-0000-0000-000000000004', '00000000-0000-0000-0000-000000000002', 1, 'Two', 'original two', ('[' || repeat('0,', 1023) || '0]')::vector, 'test-bge-m3-v1', TIMESTAMP '2000-01-01', TIMESTAMP '2000-01-01');
        INSERT INTO diploma (id, external_id, title, description, mentor, status, embedding_bge_m3, embedding_bge_m3_version, embedding_bge_m3_updated_at, updated_at) VALUES ('00000000-0000-0000-0000-000000000005', 'diploma', 'Original', 'original', 'Mentor', 'defended', ('[' || repeat('0,', 1023) || '0]')::vector, 'test-bge-m3-v1', TIMESTAMP '2000-01-01', TIMESTAMP '2000-01-01');
        INSERT INTO professor_document (id, external_id, title, abstract, embedding_bge_m3, embedding_bge_m3_version, embedding_bge_m3_updated_at, updated_at) VALUES ('00000000-0000-0000-0000-000000000006', 'professor-document', 'Original', 'original', ('[' || repeat('0,', 1023) || '0]')::vector, 'test-bge-m3-v1', TIMESTAMP '2000-01-01', TIMESTAMP '2000-01-01');
        """,
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


async def _states(database: Database) -> list[LifecycleState]:
    rows = await database.fetch(
        """
        SELECT 'question', embedding_revision, embedding_bge_m3 IS NULL, embedding_bge_m3_version, embedding_bge_m3_updated_at, updated_at FROM question WHERE id = '00000000-0000-0000-0000-000000000001'
        UNION ALL SELECT 'chunk-one', embedding_revision, embedding_bge_m3 IS NULL, embedding_bge_m3_version, embedding_bge_m3_updated_at, updated_at FROM chunk WHERE id = '00000000-0000-0000-0000-000000000003'
        UNION ALL SELECT 'chunk-two', embedding_revision, embedding_bge_m3 IS NULL, embedding_bge_m3_version, embedding_bge_m3_updated_at, updated_at FROM chunk WHERE id = '00000000-0000-0000-0000-000000000004'
        UNION ALL SELECT 'diploma', embedding_revision, embedding_bge_m3 IS NULL, embedding_bge_m3_version, embedding_bge_m3_updated_at, updated_at FROM diploma WHERE id = '00000000-0000-0000-0000-000000000005'
        UNION ALL SELECT 'professor_document', embedding_revision, embedding_bge_m3 IS NULL, embedding_bge_m3_version, embedding_bge_m3_updated_at, updated_at FROM professor_document WHERE id = '00000000-0000-0000-0000-000000000006'
        """,
    )
    return [tuple(row) for row in rows]


async def _rollback_question(database: Database) -> None:
    async with database.transaction() as connection:
        await connection.execute(
            "UPDATE question SET name = 'changed' WHERE id = $1",
            QUESTION_ID,
        )
        raise RollbackRequestedError


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
            assert sorted(notifications.payloads) == sorted(DIRTY_INSERT_PAYLOADS)
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


def test_real_postgres_preserves_current_state_when_irrelevant_fields_change() -> None:
    async def run() -> None:
        # Given: current BGE-M3 lifecycle state before the listener is attached.
        async with _database() as database:
            await _seed_current_rows(database)
            async with _notifications() as notifications:
                before = await _states(database)

                # When: only non-embedding fields change for every corpus.
                async with database.transaction() as connection:
                    await connection.execute(
                        "UPDATE question SET user_id = 'user' WHERE id = $1",
                        QUESTION_ID,
                    )
                    await connection.execute(
                        'UPDATE chunk SET metadata = \'{"source": "test"}\'::jsonb WHERE id = $1',
                        CHUNK_ONE_ID,
                    )
                    await connection.execute(
                        "UPDATE diploma SET mentor = 'Other' WHERE id = $1",
                        DIPLOMA_ID,
                    )
                    await connection.execute(
                        "UPDATE professor_document SET year = 2026 WHERE id = $1",
                        PROFESSOR_DOCUMENT_ID,
                    )
                await notifications.assert_no_notification()

                # Then: the vector, version, timestamp, revision, and update timestamp are untouched.
                assert await _states(database) == before

    anyio.run(run)


def test_real_postgres_preserves_state_when_relevant_edit_rolls_back() -> None:
    async def run() -> None:
        # Given: a current BGE-M3 question and a listener before a transaction begins.
        async with _database() as database, _notifications() as notifications:
            await _seed_current_rows(database)
            await notifications.wait_for_count(5)
            notifications.payloads.clear()
            before = await _states(database)

            # When: a relevant source edit rolls back instead of committing.
            with pytest.raises(RollbackRequestedError):
                await _rollback_question(database)
            await notifications.assert_no_notification()

            # Then: neither durable lifecycle state nor the wakeup escaped the rollback.
            assert await _states(database) == before

    anyio.run(run)
