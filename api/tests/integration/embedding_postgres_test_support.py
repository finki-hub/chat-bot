from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime

import anyio
from asyncpg import connect

from app.data.connection import Database

type LifecycleState = tuple[str, int, bool, str | None, datetime | None, datetime]


class NotificationCollector:
    def __init__(self) -> None:
        self.payloads: list[str] = []
        self._event = anyio.Event()

    def receive(
        self,
        _connection,
        _process_id,
        _channel,
        payload,
    ) -> None:
        self.payloads.append(payload)
        self._event.set()

    async def wait_for_count(self, expected_count: int) -> None:
        while len(self.payloads) < expected_count:
            event = self._event
            with anyio.fail_after(2):
                await event.wait()
            self._event = anyio.Event()
        self._event = anyio.Event()

    async def assert_no_notification(self) -> None:
        previous_count = len(self.payloads)
        event = self._event
        with anyio.move_on_after(0.25) as scope:
            await event.wait()
        assert scope.cancelled_caught
        assert len(self.payloads) == previous_count


def database_url() -> str:
    return os.environ["TEST_DATABASE_URL"]


@asynccontextmanager
async def database() -> AsyncIterator[Database]:
    current_database = Database(database_url(), min_size=1, max_size=4)
    await current_database.init()
    await current_database.run_migrations()
    await current_database.execute(
        "TRUNCATE question, document, diploma, professor_document CASCADE",
    )
    try:
        yield current_database
    finally:
        await current_database.disconnect()


@asynccontextmanager
async def notifications() -> AsyncIterator[NotificationCollector]:
    connection = await connect(database_url())
    collector = NotificationCollector()
    await connection.add_listener("embedding_dirty", collector.receive)
    try:
        yield collector
    finally:
        await connection.remove_listener("embedding_dirty", collector.receive)
        await connection.close()


async def seed_current_rows(database: Database) -> None:
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


async def lifecycle_states(database: Database) -> list[LifecycleState]:
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
