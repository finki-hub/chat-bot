import os
from uuid import UUID

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
    reason="set TEST_DATABASE_URL to run real-PostgreSQL embedding invalidation state tests",
)

QUESTION_ID = UUID("00000000-0000-0000-0000-000000000001")
CHUNK_ONE_ID = UUID("00000000-0000-0000-0000-000000000003")
DIPLOMA_ID = UUID("00000000-0000-0000-0000-000000000005")
PROFESSOR_DOCUMENT_ID = UUID("00000000-0000-0000-0000-000000000006")


class RollbackRequestedError(Exception):
    pass


async def _rollback_question(database: Database) -> None:
    async with database.transaction() as connection:
        await connection.execute(
            "UPDATE question SET name = 'changed' WHERE id = $1",
            QUESTION_ID,
        )
        raise RollbackRequestedError


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
