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
    EMBEDDING_WAKE_PAYLOAD,
    EmbeddingBatch,
    EmbeddingCorpus,
    fetch_dirty_embeddings,
    lifecycle_counts,
    persist_embedding_batch,
    rebuild_embedding_lifecycle,
    wake_embedding_worker,
)
from app.llms.models import BGE_M3_EMBEDDING_SPEC_VERSION

DATABASE_URL = os.getenv("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    DATABASE_URL is None,
    reason="set TEST_DATABASE_URL to run real-PostgreSQL embedding lifecycle tests",
)

QUESTION_ID = UUID("00000000-0000-0000-0000-000000000101")
QUESTION_TWO_ID = UUID("00000000-0000-0000-0000-000000000102")
DOCUMENT_ID = UUID("00000000-0000-0000-0000-000000000103")
CHUNK_ID = UUID("00000000-0000-0000-0000-000000000104")
DIPLOMA_ID = UUID("00000000-0000-0000-0000-000000000105")
PROFESSOR_DOCUMENT_ID = UUID("00000000-0000-0000-0000-000000000106")


def _database_url() -> str:
    return os.environ["TEST_DATABASE_URL"]


@asynccontextmanager
async def _database() -> AsyncIterator[Database]:
    database = Database(_database_url(), min_size=1, max_size=4)
    await database.init()
    await database.run_migrations()
    await database.execute(
        "TRUNCATE question, document, diploma, professor_document CASCADE",
    )
    try:
        yield database
    finally:
        await database.disconnect()


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


async def _seed_dirty_rows(database: Database) -> None:
    async with database.transaction() as connection:
        await connection.execute(
            "INSERT INTO document (id, name, title) VALUES ($1, 'document', 'Document')",
            DOCUMENT_ID,
        )
        await connection.execute(
            "INSERT INTO question (id, name, content) VALUES ($1, 'Question', 'Answer')",
            QUESTION_ID,
        )
        await connection.execute(
            "INSERT INTO question (id, name, content) VALUES ($1, 'Question two', 'Answer two')",
            QUESTION_TWO_ID,
        )
        await connection.execute(
            "INSERT INTO chunk (id, document_id, chunk_index, section, content) VALUES ($1, $2, 0, 'Section', 'Chunk')",
            CHUNK_ID,
            DOCUMENT_ID,
        )
        await connection.execute(
            "INSERT INTO diploma (id, external_id, title, description, mentor, status) VALUES ($1, 'diploma', 'Diploma', 'Description', 'Mentor', 'defended')",
            DIPLOMA_ID,
        )
        await connection.execute(
            "INSERT INTO professor_document (id, external_id, title, abstract) VALUES ($1, 'professor', 'Paper', 'Abstract')",
            PROFESSOR_DOCUMENT_ID,
        )


def test_real_postgres_fetches_dirty_rows_and_reports_all_corpus_counts() -> None:
    async def run() -> None:
        # Given: one dirty row in every corpus and a second question for ordering.
        async with _database() as database:
            await _seed_dirty_rows(database)

            # When: lifecycle batches and counts are fetched.
            questions = await fetch_dirty_embeddings(
                database,
                EmbeddingCorpus.QUESTION,
                limit=1,
            )
            chunks = await fetch_dirty_embeddings(
                database,
                EmbeddingCorpus.CHUNK,
                limit=4,
            )
            counts = await lifecycle_counts(database)

            # Then: selections are bounded/deterministic and all corpora report dirty work.
            assert [candidate.id for candidate in questions] == [QUESTION_ID]
            assert questions[0].text == "Наслов: Question\nСодржина: Answer"
            assert chunks[0].text == "Наслов: Document (Section)\nСодржина: Chunk"
            assert tuple(
                (count.corpus, count.ready, count.dirty) for count in counts
            ) == (
                (EmbeddingCorpus.QUESTION, 0, 2),
                (EmbeddingCorpus.CHUNK, 0, 1),
                (EmbeddingCorpus.DIPLOMA, 0, 1),
                (EmbeddingCorpus.PROFESSOR_DOCUMENT, 0, 1),
            )

    anyio.run(run)


def test_real_postgres_persists_guardedly_and_rejects_stale_revisions() -> None:
    async def run() -> None:
        # Given: a captured dirty question revision.
        async with _database() as database:
            await _seed_dirty_rows(database)
            candidate = (
                await fetch_dirty_embeddings(
                    database,
                    EmbeddingCorpus.QUESTION,
                    limit=1,
                )
            )[0]
            batch = EmbeddingBatch(
                corpus=EmbeddingCorpus.QUESTION,
                candidates=(candidate,),
                vectors=((0.0,) * BGE_M3_DIMENSIONS,),
            )

            # When: the current revision is persisted, then the source advances before reuse.
            current_result = await persist_embedding_batch(database, batch)
            await database.execute(
                "UPDATE question SET content = 'new answer' WHERE id = $1",
                QUESTION_ID,
            )
            stale_result = await persist_embedding_batch(database, batch)
            row = await database.fetchrow(
                "SELECT embedding_bge_m3 IS NULL AS dirty, embedding_bge_m3_version FROM question WHERE id = $1",
                QUESTION_ID,
            )

            # Then: current writes are complete while stale writes apply zero rows.
            assert current_result.valid
            assert current_result.updated == 1
            assert stale_result.valid
            assert stale_result.updated == 0
            assert row is not None
            assert row["dirty"]
            assert row["embedding_bge_m3_version"] is None

    anyio.run(run)


def test_real_postgres_malformed_batch_writes_nothing() -> None:
    async def run() -> None:
        # Given: two dirty question candidates and a cardinality-mismatched provider batch.
        async with _database() as database:
            await _seed_dirty_rows(database)
            candidates = await fetch_dirty_embeddings(
                database,
                EmbeddingCorpus.QUESTION,
                limit=2,
            )
            malformed = EmbeddingBatch(
                corpus=EmbeddingCorpus.QUESTION,
                candidates=candidates,
                vectors=((0.0,) * BGE_M3_DIMENSIONS,),
            )

            # When: the malformed batch reaches the lifecycle persistence boundary.
            result = await persist_embedding_batch(database, malformed)
            dirty = await database.fetchval(
                "SELECT COUNT(*) FROM question WHERE embedding_bge_m3 IS NULL",
            )

            # Then: validation rejects it before any source row is written.
            assert not result.valid
            assert result.updated == 0
            assert dirty == 2

    anyio.run(run)


def test_real_postgres_rebuild_and_wake_notify_without_payload_work() -> None:
    async def run() -> None:
        # Given: current metadata and captured revisions for every corpus.
        async with _database() as database:
            await _seed_dirty_rows(database)
            for corpus in EmbeddingCorpus:
                candidate = (await fetch_dirty_embeddings(database, corpus, limit=1))[0]
                await persist_embedding_batch(
                    database,
                    EmbeddingBatch(
                        corpus=corpus,
                        candidates=(candidate,),
                        vectors=((0.0,) * BGE_M3_DIMENSIONS,),
                    ),
                )
            before = await database.fetch(
                """
                SELECT 'question' AS resource, embedding_revision FROM question WHERE id = $1
                UNION ALL SELECT 'chunk', embedding_revision FROM chunk WHERE id = $2
                UNION ALL SELECT 'diploma', embedding_revision FROM diploma WHERE id = $3
                UNION ALL SELECT 'professor_document', embedding_revision FROM professor_document WHERE id = $4
                """,
                QUESTION_ID,
                CHUNK_ID,
                DIPLOMA_ID,
                PROFESSOR_DOCUMENT_ID,
            )
            async with _notifications() as notifications:
                # When: a bare wake and a transactional lifecycle rebuild are requested.
                await wake_embedding_worker(database)
                counts = await rebuild_embedding_lifecycle(database)
                await notifications.wait_for_count(2)
            after = await database.fetch(
                """
                SELECT 'question' AS resource, embedding_revision, embedding_bge_m3 IS NULL AS dirty, embedding_bge_m3_version, embedding_bge_m3_updated_at FROM question WHERE id = $1
                UNION ALL SELECT 'chunk', embedding_revision, embedding_bge_m3 IS NULL, embedding_bge_m3_version, embedding_bge_m3_updated_at FROM chunk WHERE id = $2
                UNION ALL SELECT 'diploma', embedding_revision, embedding_bge_m3 IS NULL, embedding_bge_m3_version, embedding_bge_m3_updated_at FROM diploma WHERE id = $3
                UNION ALL SELECT 'professor_document', embedding_revision, embedding_bge_m3 IS NULL, embedding_bge_m3_version, embedding_bge_m3_updated_at FROM professor_document WHERE id = $4
                """,
                QUESTION_ID,
                CHUNK_ID,
                DIPLOMA_ID,
                PROFESSOR_DOCUMENT_ID,
            )

            # Then: every corpus is dirty, advanced, cleared, and emits only wake hints.
            assert notifications.payloads == [EMBEDDING_WAKE_PAYLOAD] * 2
            assert tuple(
                (count.corpus, count.ready, count.dirty) for count in counts
            ) == (
                (EmbeddingCorpus.QUESTION, 0, 2),
                (EmbeddingCorpus.CHUNK, 0, 1),
                (EmbeddingCorpus.DIPLOMA, 0, 1),
                (EmbeddingCorpus.PROFESSOR_DOCUMENT, 0, 1),
            )
            revisions = {row["resource"]: row["embedding_revision"] for row in before}
            assert all(
                row["embedding_revision"] == revisions[row["resource"]] + 1
                and row["dirty"]
                for row in after
            )
            assert all(
                row["embedding_bge_m3_version"] is None
                and row["embedding_bge_m3_updated_at"] is None
                for row in after
            )
            assert BGE_M3_EMBEDDING_SPEC_VERSION == "bge-m3-v1"

    anyio.run(run)
