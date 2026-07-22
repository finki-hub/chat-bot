import os
from uuid import UUID

import anyio
import pytest

from app.data.embedding_lifecycle import (
    BGE_M3_DIMENSIONS,
    EmbeddingBatch,
    EmbeddingCorpus,
    fetch_dirty_embeddings,
    persist_embedding_batch,
    rebuild_embedding_lifecycle_in_transaction,
)
from tests.integration.embedding_postgres_test_support import database as _database
from tests.integration.embedding_postgres_test_support import (
    notifications as _notifications,
)

DATABASE_URL = os.getenv("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    DATABASE_URL is None,
    reason="set TEST_DATABASE_URL to run real-PostgreSQL embedding rebuild tests",
)

QUESTION_ID = UUID("00000000-0000-0000-0000-000000000107")


class RollbackRequestedError(Exception):
    pass


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
