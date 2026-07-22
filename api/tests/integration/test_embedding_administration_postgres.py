import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID

import anyio
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.data.connection import Database
from app.data.embedding_lifecycle import (
    BGE_M3_DIMENSIONS,
    EMBEDDING_WAKE_PAYLOAD,
    EmbeddingBatch,
    EmbeddingCandidate,
    EmbeddingCorpus,
    fetch_dirty_embeddings,
    persist_embedding_batch,
)
from app.main import make_app
from app.utils.settings import Settings
from tests.integration.embedding_postgres_test_support import (
    database_url as _database_url,
)
from tests.integration.embedding_postgres_test_support import (
    notifications as _notifications,
)

DATABASE_URL = os.getenv("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    DATABASE_URL is None,
    reason="set TEST_DATABASE_URL to run real-PostgreSQL embedding administration tests",
)

API_KEY = "task-5-api-key"
QUESTION_ID = UUID("00000000-0000-0000-0000-000000000501")
QUESTION_TWO_ID = UUID("00000000-0000-0000-0000-000000000502")
DOCUMENT_ID = UUID("00000000-0000-0000-0000-000000000503")
CHUNK_ID = UUID("00000000-0000-0000-0000-000000000504")
DIPLOMA_ID = UUID("00000000-0000-0000-0000-000000000505")
PROFESSOR_DOCUMENT_ID = UUID("00000000-0000-0000-0000-000000000506")


@asynccontextmanager
async def _app() -> AsyncIterator[FastAPI]:
    database = Database(_database_url(), min_size=1, max_size=4)
    await database.init()
    await database.run_migrations()
    await database.execute(
        "TRUNCATE question, document, diploma, professor_document CASCADE",
    )
    await database.disconnect()
    app = make_app(
        Settings(
            API_KEY=API_KEY,
            CREDENTIAL_ENCRYPTION_KEY="test-credential-key",
            MCP_API_KEY="test-mcp-key",
            DATABASE_URL=_database_url(),
        ),
    )
    async with app.router.lifespan_context(app):
        yield app


async def _seed_and_fill(database: Database) -> EmbeddingCandidate:
    async with database.transaction() as connection:
        await connection.execute(
            "INSERT INTO document (id, name, title) VALUES ($1, 'document', 'Document')",
            DOCUMENT_ID,
        )
        await connection.executemany(
            "INSERT INTO question (id, name, content) VALUES ($1, $2, 'Answer')",
            [(QUESTION_ID, "Question"), (QUESTION_TWO_ID, "Question two")],
        )
        await connection.execute(
            "INSERT INTO chunk (id, document_id, chunk_index, content) VALUES ($1, $2, 0, 'Chunk')",
            CHUNK_ID,
            DOCUMENT_ID,
        )
        await connection.execute(
            "INSERT INTO diploma (id, external_id, title, description, mentor, status) VALUES ($1, 'diploma', 'Diploma', 'Description', 'Mentor', 'defended')",
            DIPLOMA_ID,
        )
        await connection.execute(
            "INSERT INTO professor_document (id, external_id, title) VALUES ($1, 'professor', 'Paper')",
            PROFESSOR_DOCUMENT_ID,
        )
    question_candidate = (
        await fetch_dirty_embeddings(database, EmbeddingCorpus.QUESTION, limit=1)
    )[0]
    for corpus in EmbeddingCorpus:
        candidates = await fetch_dirty_embeddings(database, corpus, limit=10)
        result = await persist_embedding_batch(
            database,
            EmbeddingBatch(
                corpus=corpus,
                candidates=candidates,
                vectors=tuple((0.0,) * BGE_M3_DIMENSIONS for _ in candidates),
            ),
        )
        assert result.valid
        assert result.updated == len(candidates)
    return question_candidate


async def _lifecycle_rows(
    database: Database,
) -> list[tuple[str, int, bool, str | None, bool]]:
    rows = await database.fetch(
        """
        SELECT 'question' AS corpus, embedding_revision, embedding_bge_m3 IS NULL AS dirty, embedding_bge_m3_version, embedding_bge_m3_updated_at IS NULL AS timestamp_cleared FROM question
        UNION ALL SELECT 'chunk', embedding_revision, embedding_bge_m3 IS NULL, embedding_bge_m3_version, embedding_bge_m3_updated_at IS NULL FROM chunk
        UNION ALL SELECT 'diploma', embedding_revision, embedding_bge_m3 IS NULL, embedding_bge_m3_version, embedding_bge_m3_updated_at IS NULL FROM diploma
        UNION ALL SELECT 'professor_document', embedding_revision, embedding_bge_m3 IS NULL, embedding_bge_m3_version, embedding_bge_m3_updated_at IS NULL FROM professor_document
        ORDER BY corpus, embedding_revision
        """,
    )
    return [
        (
            str(row["corpus"]),
            int(row["embedding_revision"]),
            bool(row["dirty"]),
            row["embedding_bge_m3_version"],
            bool(row["timestamp_cleared"]),
        )
        for row in rows
    ]


def test_embedding_administration_rejects_missing_and_wrong_api_keys() -> None:
    async def run() -> None:
        # Given: the real application and every administration route.
        async with _app() as app:
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport,
                base_url="https://test",
            ) as client:
                # When: each operation receives absent and incorrect credentials.
                for method, path in (
                    (client.get, "/embeddings/health"),
                    (client.post, "/embeddings/fill-dirty"),
                    (client.post, "/embeddings/rebuild"),
                ):
                    missing = await method(path)
                    wrong = await method(path, headers={"x-api-key": "wrong"})

                    # Then: repository-standard auth failures are returned before lifecycle work.
                    assert missing.status_code == 401
                    assert missing.json() == {"detail": "Missing API Key"}
                    assert wrong.status_code == 401
                    assert wrong.json() == {"detail": "Invalid API Key"}

    anyio.run(run)


def test_embedding_administration_wakes_and_rebuilds_all_corpora() -> None:
    async def run() -> None:
        # Given: current BGE metadata, captured revisions, and a real PostgreSQL listener.
        async with _app() as app:
            database: Database = app.state.db
            stale_candidate = await _seed_and_fill(database)
            before_fill = await _lifecycle_rows(database)
            transport = ASGITransport(app=app)
            async with (
                _notifications() as notifications,
                AsyncClient(
                    transport=transport,
                    base_url="https://test",
                ) as client,
            ):
                health_response = await client.get(
                    "/embeddings/health",
                    headers={"x-api-key": API_KEY},
                )

                # When: fill-dirty sends a wake without changing rows.
                fill_response = await client.post(
                    "/embeddings/fill-dirty",
                    headers={"x-api-key": API_KEY},
                )
                await notifications.wait_for_count(1)

                # Then: counts are deterministic and the source/lifecycle state is unchanged.
                assert health_response.status_code == 200
                assert health_response.json() == {
                    "counts": {
                        "question": {"ready": 2, "dirty": 0},
                        "chunk": {"ready": 1, "dirty": 0},
                        "diploma": {"ready": 1, "dirty": 0},
                        "professor_document": {"ready": 1, "dirty": 0},
                    },
                }
                assert fill_response.status_code == 200
                assert fill_response.json() == {
                    "counts": {
                        "question": {"ready": 2, "dirty": 0},
                        "chunk": {"ready": 1, "dirty": 0},
                        "diploma": {"ready": 1, "dirty": 0},
                        "professor_document": {"ready": 1, "dirty": 0},
                    },
                }
                assert await _lifecycle_rows(database) == before_fill

                # When: rebuild invalidates every corpus through the authenticated HTTP surface.
                rebuild_response = await client.post(
                    "/embeddings/rebuild",
                    headers={"x-api-key": API_KEY},
                )
                await notifications.wait_for_count(2)

            # Then: all rows are dirty, advanced, cleared, and reject an old guarded write.
            assert notifications.payloads == [EMBEDDING_WAKE_PAYLOAD] * 2
            assert rebuild_response.status_code == 200
            assert rebuild_response.json() == {
                "counts": {
                    "question": {"ready": 0, "dirty": 2},
                    "chunk": {"ready": 0, "dirty": 1},
                    "diploma": {"ready": 0, "dirty": 1},
                    "professor_document": {"ready": 0, "dirty": 1},
                },
            }
            after_rebuild = await _lifecycle_rows(database)
            assert all(
                row[1] == before[1] + 1 and row[2] and row[3] is None and row[4]
                for before, row in zip(before_fill, after_rebuild, strict=True)
            )
            stale_result = await persist_embedding_batch(
                database,
                EmbeddingBatch(
                    corpus=EmbeddingCorpus.QUESTION,
                    candidates=(stale_candidate,),
                    vectors=((0.0,) * BGE_M3_DIMENSIONS,),
                ),
            )
            assert stale_result.valid
            assert stale_result.updated == 0
            assert await _lifecycle_rows(database) == after_rebuild

    anyio.run(run)
