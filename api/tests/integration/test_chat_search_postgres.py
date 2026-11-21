import os
from datetime import datetime, timedelta
from typing import cast
from uuid import uuid4

import anyio
import pytest

from app.data.chat_persistence import ChatPersistenceDatabase, search_conversations
from app.data.connection import Database

DATABASE_URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    DATABASE_URL is None,
    reason="set TEST_DATABASE_URL to run real-PostgreSQL chat search tests",
)


def test_chat_search_matches_cyrillic_messages_and_scopes_by_user() -> None:
    async def run() -> None:
        assert DATABASE_URL is not None
        database = Database(DATABASE_URL, min_size=1, max_size=2)
        await database.init()
        await database.run_migrations()
        db = cast(ChatPersistenceDatabase, database)

        owner_id = uuid4()
        other_id = uuid4()
        owner_conversation = uuid4()
        other_conversation = uuid4()
        owner_message = uuid4()
        title_only_conversation = uuid4()
        now = datetime.fromisoformat("2026-11-02T12:00:00")

        try:
            await database.execute(
                """
                INSERT INTO chat_user (id, provider, provider_subject)
                VALUES ($1, 'google', $2), ($3, 'google', $4)
                """,
                owner_id,
                f"search-owner-{owner_id}",
                other_id,
                f"search-other-{other_id}",
            )
            await database.execute(
                """
                INSERT INTO chat_conversation (id, user_id, title, updated_at)
                VALUES
                    ($1, $2, 'Испити', $3),
                    ($4, $5, 'Испити', $3),
                    ($6, $2, 'Рокови', $3)
                """,
                owner_conversation,
                owner_id,
                now,
                other_conversation,
                other_id,
                title_only_conversation,
            )
            await database.execute(
                """
                INSERT INTO chat_message
                    (id, conversation_id, role, content, created_at)
                VALUES
                    ($1, $2, 'assistant', 'Рокот за испити е објавен.', $3),
                    ($4, $5, 'assistant', 'Рокот за испити е објавен.', $3),
                    ($6, $7, 'user', 'Кога се објавуваат резултатите?', $3)
                """,
                owner_message,
                owner_conversation,
                now,
                uuid4(),
                other_conversation,
                uuid4(),
                title_only_conversation,
            )

            results = await search_conversations(
                db,
                user_id=owner_id,
                query="испити",
                limit=20,
            )
            title_results = await search_conversations(
                db,
                user_id=owner_id,
                query="рокови",
                limit=20,
            )
            other_results = await search_conversations(
                db,
                user_id=other_id,
                query="испити",
                limit=20,
            )
        finally:
            await database.execute(
                "DELETE FROM chat_user WHERE id = $1 OR id = $2",
                owner_id,
                other_id,
            )
            await database.disconnect()

        assert [result.conversation_id for result in results] == [
            owner_conversation,
        ]
        assert results[0].message is not None
        assert results[0].message.id == owner_message
        assert results[0].snippet is not None
        assert "\x01" in results[0].snippet
        assert "\x02" in results[0].snippet

        assert [result.conversation_id for result in title_results] == [
            title_only_conversation,
        ]
        assert title_results[0].message is None
        assert title_results[0].snippet is None

        assert [result.conversation_id for result in other_results] == [
            other_conversation,
        ]

    anyio.run(run)


def test_chat_search_requires_all_words_matches_prefixes_and_falls_back() -> None:
    async def run() -> None:
        assert DATABASE_URL is not None
        database = Database(DATABASE_URL, min_size=1, max_size=2)
        await database.init()
        await database.run_migrations()
        db = cast(ChatPersistenceDatabase, database)

        owner_id = uuid4()
        both_words = uuid4()
        one_word = uuid4()
        now = datetime.fromisoformat("2026-11-04T12:00:00")

        try:
            await database.execute(
                """
                INSERT INTO chat_user (id, provider, provider_subject)
                VALUES ($1, 'google', $2)
                """,
                owner_id,
                f"search-terms-{owner_id}",
            )
            await database.execute(
                """
                INSERT INTO chat_conversation (id, user_id, title, updated_at)
                VALUES ($1, $3, 'Прв', $4), ($2, $3, 'Втор', $4)
                """,
                both_words,
                one_word,
                owner_id,
                now,
            )
            await database.execute(
                """
                INSERT INTO chat_message
                    (id, conversation_id, role, content, created_at)
                VALUES
                    ($1, $2, 'assistant', 'Менторот го закажа испитот.', $4),
                    ($3, $5, 'assistant', 'Испитот е утре.', $4)
                """,
                uuid4(),
                both_words,
                uuid4(),
                now,
                one_word,
            )

            all_words = await search_conversations(
                db,
                user_id=owner_id,
                query="ментор испит",
            )
            fallback = await search_conversations(
                db,
                user_id=owner_id,
                query="ментор рокови",
            )
            short_words_only = await search_conversations(
                db,
                user_id=owner_id,
                query="е",
            )
        finally:
            await database.execute(
                "DELETE FROM chat_user WHERE id = $1",
                owner_id,
            )
            await database.disconnect()

        assert [result.conversation_id for result in all_words] == [both_words]
        assert [result.conversation_id for result in fallback] == [both_words]
        assert [result.conversation_id for result in short_words_only] == [one_word]

    anyio.run(run)


def test_chat_search_orders_best_message_and_paginates() -> None:
    async def run() -> None:
        assert DATABASE_URL is not None
        database = Database(DATABASE_URL, min_size=1, max_size=2)
        await database.init()
        await database.run_migrations()
        db = cast(ChatPersistenceDatabase, database)

        owner_id = uuid4()
        first_conversation = uuid4()
        second_conversation = uuid4()
        now = datetime.fromisoformat("2026-11-03T12:00:00")

        try:
            await database.execute(
                """
                INSERT INTO chat_user (id, provider, provider_subject)
                VALUES ($1, 'google', $2)
                """,
                owner_id,
                f"search-order-{owner_id}",
            )
            await database.execute(
                """
                INSERT INTO chat_conversation (id, user_id, title, updated_at)
                VALUES
                    ($1, $2, 'Прв разговор', $3),
                    ($4, $2, 'Втор разговор', $5)
                """,
                first_conversation,
                owner_id,
                now,
                second_conversation,
                now - timedelta(minutes=1),
            )
            await database.execute(
                """
                INSERT INTO chat_message
                    (id, conversation_id, role, content, created_at)
                VALUES
                    ($1, $2, 'user', 'испити', $3),
                    ($4, $2, 'assistant', 'испити испити', $3),
                    ($5, $6, 'user', 'испити', $3)
                """,
                uuid4(),
                first_conversation,
                now,
                uuid4(),
                uuid4(),
                second_conversation,
            )

            first_page = await search_conversations(
                db,
                user_id=owner_id,
                query="испити",
                limit=1,
                offset=0,
            )
            second_page = await search_conversations(
                db,
                user_id=owner_id,
                query="испити",
                limit=1,
                offset=1,
            )
        finally:
            await database.execute(
                "DELETE FROM chat_user WHERE id = $1",
                owner_id,
            )
            await database.disconnect()

        assert [result.conversation_id for result in first_page] == [
            first_conversation,
        ]
        assert [result.conversation_id for result in second_page] == [
            second_conversation,
        ]

    anyio.run(run)
