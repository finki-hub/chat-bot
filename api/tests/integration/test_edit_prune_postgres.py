import asyncio
import os
from uuid import UUID, uuid4

import pytest

from app.data.chat_state import prune_messages_after_user_message
from app.data.connection import Database
from tests.integration.embedding_postgres_test_support import database as _database

DATABASE_URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    DATABASE_URL is None,
    reason="set TEST_DATABASE_URL to run real-PostgreSQL edit tests",
)


async def _seed(database: Database) -> dict[str, UUID]:
    """A two-turn conversation (question, answer, question, answer), each answer
    rated by its owner on the web."""
    ids = {
        "user": uuid4(),
        "conversation": uuid4(),
        "q1": uuid4(),
        "a1": uuid4(),
        "r1": uuid4(),
        "q2": uuid4(),
        "a2": uuid4(),
        "r2": uuid4(),
    }
    await database.execute(
        """
        INSERT INTO chat_user (id, provider, provider_subject)
        VALUES ($1::uuid, 'google', $1::uuid::text)
        """,
        ids["user"],
    )
    await database.execute(
        "INSERT INTO chat_conversation (id, user_id) VALUES ($1, $2)",
        ids["conversation"],
        ids["user"],
    )
    for offset, (message, role, response) in enumerate(
        [
            ("q1", "user", None),
            ("a1", "assistant", "r1"),
            ("q2", "user", None),
            ("a2", "assistant", "r2"),
        ],
    ):
        await database.execute(
            """
            INSERT INTO chat_message
                (id, conversation_id, role, content, response_id, created_at)
            VALUES ($1, $2, $3, 'synthetic', $4, NOW() + make_interval(secs => $5))
            """,
            ids[message],
            ids["conversation"],
            role,
            None if response is None else ids[response],
            float(offset),
        )
    for response in ("r1", "r2"):
        await database.execute(
            """
            INSERT INTO feedback (response_id, client, user_id, feedback_type)
            VALUES ($1, 'web', $2, 'dislike')
            """,
            ids[response],
            str(ids["user"]),
        )
    return ids


async def _remaining(database: Database, ids: dict[str, UUID]) -> list[UUID]:
    rows = await database.fetch(
        "SELECT id FROM chat_message WHERE conversation_id = $1 ORDER BY created_at",
        ids["conversation"],
    )
    return [row["id"] for row in rows]


async def _rated(database: Database, ids: dict[str, UUID]) -> set[UUID]:
    # Other suites share this database, so only this conversation's rows count.
    rows = await database.fetch(
        "SELECT response_id FROM feedback WHERE response_id = ANY($1::uuid[])",
        [ids["r1"], ids["r2"]],
    )
    return {row["response_id"] for row in rows}


async def _prune(database: Database, ids: dict[str, UUID], message: str) -> bool:
    return await prune_messages_after_user_message(
        database,
        conversation_id=ids["conversation"],
        message_id=ids[message],
        user_id=ids["user"],
    )


def test_editing_the_last_question_drops_only_its_answer() -> None:
    async def run() -> None:
        async with _database() as database:
            ids = await _seed(database)

            assert await _prune(database, ids, "q2")

            assert await _remaining(database, ids) == [ids["q1"], ids["a1"], ids["q2"]]
            assert await _rated(database, ids) == {ids["r1"]}

    asyncio.run(run())


def test_editing_an_earlier_question_drops_everything_after_it() -> None:
    async def run() -> None:
        async with _database() as database:
            ids = await _seed(database)

            assert await _prune(database, ids, "q1")

            assert await _remaining(database, ids) == [ids["q1"]]
            assert await _rated(database, ids) == set()

    asyncio.run(run())


def test_refuses_an_answer_or_a_conversation_with_a_live_stream() -> None:
    async def run() -> None:
        async with _database() as database:
            ids = await _seed(database)

            assert not await _prune(database, ids, "a1")

            await database.execute(
                "UPDATE chat_conversation SET active_stream_id = $2 WHERE id = $1",
                ids["conversation"],
                uuid4(),
            )
            assert not await _prune(database, ids, "q1")

            assert len(await _remaining(database, ids)) == 4
            assert await _rated(database, ids) == {ids["r1"], ids["r2"]}

    asyncio.run(run())


def test_refuses_another_users_conversation() -> None:
    async def run() -> None:
        async with _database() as database:
            ids = await _seed(database)

            assert not await prune_messages_after_user_message(
                database,
                conversation_id=ids["conversation"],
                message_id=ids["q1"],
                user_id=uuid4(),
            )
            assert len(await _remaining(database, ids)) == 4

    asyncio.run(run())
