import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID

import anyio
import pytest
from asyncpg import connect

from app.data.connection import Database
from app.data.embedding_lifecycle import (
    BGE_M3_DIMENSIONS,
    EmbeddingBatch,
    EmbeddingCorpus,
    fetch_dirty_embeddings,
    persist_embedding_batch,
    rebuild_embedding_lifecycle_in_transaction,
)

DATABASE_URL = os.getenv("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    DATABASE_URL is None,
    reason="set TEST_DATABASE_URL to run real-PostgreSQL embedding rebuild tests",
)

QUESTION_ID = UUID("00000000-0000-0000-0000-000000000107")


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


def _database_url() -> str:
    return os.environ["TEST_DATABASE_URL"]


@asynccontextmanager
async def _database() -> AsyncIterator[Database]:
    database = Database(_database_url())
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
    connection = await connect(_database_url())
    collector = NotificationCollector()
    await connection.add_listener("embedding_dirty", collector.receive)
    try:
        yield collector
    finally:
        await connection.remove_listener("embedding_dirty", collector.receive)
        await connection.close()


def test_real_postgres_rebuild_rolls_back_metadata_and_notification() -> None:
    async def run() -> None:
        # Given: a current question state and a listener before a rebuild transaction.
        async with _database() as database:
            await database.execute(
                "INSERT INTO question (id, name, content) VALUES ($1, 'Question', 'Answer')",
                QUESTION_ID,
            )
            candidate = (
                await fetch_dirty_embeddings(
                    database,
                    EmbeddingCorpus.QUESTION,
                    limit=1,
                )
            )[0]
            await persist_embedding_batch(
                database,
                EmbeddingBatch(
                    corpus=EmbeddingCorpus.QUESTION,
                    candidates=(candidate,),
                    vectors=((0.0,) * BGE_M3_DIMENSIONS,),
                ),
            )
            before = await database.fetchrow(
                "SELECT embedding_revision, embedding_bge_m3 IS NULL AS dirty, embedding_bge_m3_version FROM question WHERE id = $1",
                QUESTION_ID,
            )
            async with _notifications() as notifications:

                async def abort_rebuild() -> None:
                    async with database.transaction() as connection:
                        await rebuild_embedding_lifecycle_in_transaction(connection)
                        raise RollbackRequestedError

                # When: the caller-owned rebuild transaction aborts.
                with pytest.raises(RollbackRequestedError):
                    await abort_rebuild()
                await database.execute(
                    "SELECT pg_notify('embedding_dirty', $1)",
                    "rollback-sentinel",
                )
                await notifications.wait_for_count(1)
                after = await database.fetchrow(
                    "SELECT embedding_revision, embedding_bge_m3 IS NULL AS dirty, embedding_bge_m3_version FROM question WHERE id = $1",
                    QUESTION_ID,
                )

                # Then: neither state nor the wake notification escapes the rollback.
                assert notifications.payloads == ["rollback-sentinel"]
                assert before is not None
                assert after is not None
                assert tuple(after) == tuple(before)

    anyio.run(run)
