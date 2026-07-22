import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID

import anyio
import pytest

from app.data.connection import Database
from app.data.documents import get_closest_chunks
from app.llms.context import _chunk_candidate, _expand_and_render
from app.llms.models import BGE_M3_EMBEDDING_SPEC_VERSION, Model
from app.schemas.documents import ChunkSchema


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


@pytest.mark.parametrize("model", [Model.BGE_M3, Model.BGE_M3_LOCAL])
@pytest.mark.parametrize("dirty_version", [None, "bge-m3-old"])
def test_context_excludes_dirty_bge_neighbors(
    model: Model, dirty_version: str | None,
) -> None:
    async def run() -> None:
        vector = "[1," + "0," * 1022 + "0]"
        document_id = UUID("00000000-0000-0000-0000-000000000320")
        async with _database() as database:
            async with database.transaction() as connection:
                await connection.execute(
                    "INSERT INTO document (id, name, title) VALUES ($1, 'context', 'Context')",
                    document_id,
                )
                await connection.executemany(
                    "INSERT INTO chunk (document_id, chunk_index, content, embedding_bge_m3, embedding_bge_m3_version) VALUES ($1, $2, $3, $4::vector, $5)",
                    [
                        (
                            document_id,
                            0,
                            "CURRENT NEIGHBOR",
                            vector,
                            BGE_M3_EMBEDDING_SPEC_VERSION,
                        ),
                        (
                            document_id,
                            1,
                            "CURRENT CENTER",
                            vector,
                            BGE_M3_EMBEDDING_SPEC_VERSION,
                        ),
                        (
                            document_id,
                            2,
                            "DIRTY NEIGHBOR",
                            vector if dirty_version else None,
                            dirty_version,
                        ),
                    ],
                )
            chunks = await get_closest_chunks(database, [1.0] + [0.0] * 1023, model)
            center = next(chunk for chunk in chunks if chunk.chunk_index == 1)
            text = await _expand_and_render(database, [_chunk_candidate(center)], model)
            assert "CURRENT CENTER" in text
            assert "CURRENT NEIGHBOR" in text
            assert "DIRTY NEIGHBOR" not in text

    anyio.run(run)


def test_context_keeps_non_bge_neighbors() -> None:
    async def run() -> None:
        document_id = UUID("00000000-0000-0000-0000-000000000321")
        async with _database() as database:
            async with database.transaction() as connection:
                await connection.execute(
                    "INSERT INTO document (id, name, title) VALUES ($1, 'context-non-bge', 'Context')",
                    document_id,
                )
                await connection.executemany(
                    "INSERT INTO chunk (document_id, chunk_index, content) VALUES ($1, $2, $3)",
                    [
                        (document_id, 0, "NON-BGE NEIGHBOR"),
                        (document_id, 1, "NON-BGE CENTER"),
                    ],
                )
            row = await database.fetchrow("SELECT id FROM chunk WHERE chunk_index = 1")
            assert row is not None
            center = ChunkSchema(
                id=row["id"],
                document_id=document_id,
                document_name="context-non-bge",
                document_title="Context",
                chunk_index=1,
                section=None,
                content="NON-BGE CENTER",
            )
            text = await _expand_and_render(
                database, [_chunk_candidate(center)], Model.TEXT_EMBEDDING_3_LARGE,
            )
            assert "NON-BGE CENTER" in text
            assert "NON-BGE NEIGHBOR" in text

    anyio.run(run)
