import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID

import anyio
import pytest

from app.data.connection import Database
from app.data.diplomas import (
    get_backtest_population,
    get_closest_diplomas,
    get_diplomas_without_embeddings,
)
from app.data.documents import fetch_chunk_rows_for_fill, get_closest_chunks
from app.data.professor_documents import (
    fetch_professor_document_rows_for_fill,
    get_closest_professor_documents,
)
from app.data.questions import (
    get_closest_questions,
    get_questions_without_embeddings_query,
)
from app.llms.embeddings import stream_fill_embeddings
from app.llms.models import BGE_M3_EMBEDDING_SPEC_VERSION, Model

DATABASE_URL = os.getenv("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    DATABASE_URL is None,
    reason="set TEST_DATABASE_URL to run real-PostgreSQL embedding data-path tests",
)


@asynccontextmanager
async def _database() -> AsyncIterator[Database]:
    database = Database(os.environ["TEST_DATABASE_URL"], min_size=1, max_size=4)
    await database.init()
    await database.run_migrations()
    await database.execute(
        "TRUNCATE question, document, diploma, professor_document CASCADE",
    )
    try:
        yield database
    finally:
        await database.disconnect()


async def _seed_current_and_old_rows(database: Database) -> None:
    vector = "[1," + "0," * 1022 + "0]"
    current = BGE_M3_EMBEDDING_SPEC_VERSION
    old = "bge-m3-old"
    document_id = UUID("00000000-0000-0000-0000-000000000301")
    async with database.transaction() as connection:
        await connection.execute(
            "INSERT INTO document (id, name, title) VALUES ($1, 'document', 'Document')",
            document_id,
        )
        await connection.executemany(
            "INSERT INTO question (id, name, content, embedding_bge_m3, embedding_bge_m3_version) VALUES ($1, $2, 'answer', $3::vector, $4)",
            [
                (
                    UUID("00000000-0000-0000-0000-000000000302"),
                    "current-question",
                    vector,
                    current,
                ),
                (
                    UUID("00000000-0000-0000-0000-000000000303"),
                    "old-question",
                    vector,
                    old,
                ),
            ],
        )
        await connection.executemany(
            "INSERT INTO chunk (id, document_id, chunk_index, content, embedding_bge_m3, embedding_bge_m3_version) VALUES ($1, $2, $3, $4, $5::vector, $6)",
            [
                (
                    UUID("00000000-0000-0000-0000-000000000304"),
                    document_id,
                    0,
                    "current-chunk",
                    vector,
                    current,
                ),
                (
                    UUID("00000000-0000-0000-0000-000000000305"),
                    document_id,
                    1,
                    "old-chunk",
                    vector,
                    old,
                ),
            ],
        )
        await connection.executemany(
            "INSERT INTO diploma (id, external_id, title, description, mentor, member1, member2, status, embedding_bge_m3, embedding_bge_m3_version) VALUES ($1, $2, $3, 'description', 'mentor', 'member1', 'member2', 'Одбрана', $4::vector, $5)",
            [
                (
                    UUID("00000000-0000-0000-0000-000000000306"),
                    "current-diploma",
                    "current diploma",
                    vector,
                    current,
                ),
                (
                    UUID("00000000-0000-0000-0000-000000000307"),
                    "old-diploma",
                    "old diploma",
                    vector,
                    old,
                ),
            ],
        )
        await connection.executemany(
            "INSERT INTO professor_document (id, external_id, title, abstract, embedding_bge_m3, embedding_bge_m3_version) VALUES ($1, $2, $3, 'abstract', $4::vector, $5)",
            [
                (
                    UUID("00000000-0000-0000-0000-000000000308"),
                    "current-paper",
                    "current paper",
                    vector,
                    current,
                ),
                (
                    UUID("00000000-0000-0000-0000-000000000309"),
                    "old-paper",
                    "old paper",
                    vector,
                    old,
                ),
            ],
        )


def test_real_postgres_data_paths_only_use_current_bge_vectors_for_both_aliases() -> (
    None
):
    async def run() -> None:
        async with _database() as database:
            await _seed_current_and_old_rows(database)
            for model in (Model.BGE_M3, Model.BGE_M3_LOCAL):
                query_vector = [1.0] + [0.0] * 1023
                questions = await get_closest_questions(database, query_vector, model)
                chunks = await get_closest_chunks(database, query_vector, model)
                diplomas = await get_closest_diplomas(database, query_vector, model)
                papers = await get_closest_professor_documents(
                    database,
                    query_vector,
                    model,
                )
                backtest = await get_backtest_population(database, model)
                dirty_questions = await get_questions_without_embeddings_query(
                    database,
                    model,
                )
                dirty_chunks = await fetch_chunk_rows_for_fill(
                    database,
                    model,
                    None,
                    all_chunks=False,
                )
                dirty_diplomas = await get_diplomas_without_embeddings(database, model)
                dirty_papers = await fetch_professor_document_rows_for_fill(
                    database,
                    model,
                )

                assert [question.name for question in questions] == ["current-question"]
                assert [chunk.content for chunk in chunks] == ["current-chunk"]
                assert [diploma.external_id for diploma in diplomas] == [
                    "current-diploma",
                ]
                assert [paper["external_id"] for paper in papers] == ["current-paper"]
                assert [row["external_id"] for row in backtest] == ["current-diploma"]
                assert [question.name for question in dirty_questions] == [
                    "old-question",
                ]
                assert [row["content"] for row in dirty_chunks] == ["old-chunk"]
                assert [diploma.external_id for diploma in dirty_diplomas] == [
                    "old-diploma",
                ]
                assert [row["title"] for row in dirty_papers] == ["old paper"]

    anyio.run(run)


def test_real_postgres_manual_fill_reports_mixed_guarded_race_outcomes(
    monkeypatch,
) -> None:
    current_id = UUID("00000000-0000-0000-0000-000000000310")
    stale_id = UUID("00000000-0000-0000-0000-000000000311")

    async def run() -> None:
        async with _database() as database:
            async with database.transaction() as connection:
                await connection.executemany(
                    "INSERT INTO question (id, name, content) VALUES ($1, $2, 'answer')",
                    [(current_id, "current"), (stale_id, "stale")],
                )

            async def generate(*_args, **_kwargs) -> list[list[float]]:
                await database.execute(
                    "UPDATE question SET content = 'changed' WHERE id = $1",
                    stale_id,
                )
                return [[1.0] + [0.0] * 1023, [1.0] + [0.0] * 1023]

            monkeypatch.setattr(
                "app.llms.embedding_fills.generate_embeddings",
                generate,
            )
            response = await stream_fill_embeddings(
                database,
                Model.BGE_M3_LOCAL,
                questions=["current", "stale"],
            )
            events: dict[UUID, str] = {}
            async for chunk in response.body_iterator:
                text = chunk if isinstance(chunk, str) else bytes(chunk).decode("utf-8")
                for line in text.splitlines():
                    if line.startswith("data:"):
                        payload = json.loads(line.removeprefix("data:").strip())
                        events[UUID(payload["id"])] = payload["status"]
            rows = await database.fetch(
                "SELECT id, embedding_bge_m3 IS NULL AS dirty, embedding_bge_m3_version FROM question ORDER BY id",
            )

            assert events == {current_id: "ok", stale_id: "error"}
            assert [
                (row["id"], row["dirty"], row["embedding_bge_m3_version"])
                for row in rows
            ] == [
                (current_id, False, BGE_M3_EMBEDDING_SPEC_VERSION),
                (stale_id, True, None),
            ]

    anyio.run(run)
