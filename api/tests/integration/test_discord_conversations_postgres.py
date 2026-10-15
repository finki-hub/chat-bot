import asyncio
import os
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit
from uuid import UUID, uuid4

import asyncpg
import pytest

from app.api.discord_storage import DiscordStorage, DiscordWriteRegistry
from app.data.chat_conversation_delete import delete_conversation
from app.data.chat_users import upsert_chat_user
from app.data.connection import Database
from app.data.discord_conversations import (
    DiscordExchange,
    delete_chat_history,
    store_discord_exchange,
)
from app.data.feedback import (
    DiscordFeedbackStaleError,
    DiscordFeedbackUnavailableError,
    upsert_discord_feedback,
    upsert_feedback,
)
from app.schemas.chat_user import ChatUserUpsert
from app.schemas.feedback import FeedbackSchema

DATABASE_URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    DATABASE_URL is None,
    reason="set TEST_DATABASE_URL to run real-PostgreSQL Discord storage tests",
)


def test_exchanges_append_to_one_conversation_in_order():
    async def run():
        async with _database() as database:
            user = await _discord_user(database, "111")
            conversation = uuid4()
            first = _exchange(conversation, user, question="Прво прашање?")
            second = _exchange(conversation, user, question="Второ прашање?")

            assert await store_discord_exchange(database, first)
            assert await store_discord_exchange(database, second)

            title = await database.fetchval(
                "SELECT title FROM chat_conversation WHERE id = $1", conversation
            )
            assert title == "Прво прашање?"
            rows = await database.fetch(
                """
                SELECT role, content, response_id, metadata->>'interface' AS interface
                FROM chat_message WHERE conversation_id = $1
                ORDER BY created_at, id
                """,
                conversation,
            )
            assert [(row["role"], row["content"]) for row in rows] == [
                ("user", "Прво прашање?"),
                ("assistant", "Одговор"),
                ("user", "Второ прашање?"),
                ("assistant", "Одговор"),
            ]
            assert [row["response_id"] for row in rows[1::2]] == [
                first.response_id,
                second.response_id,
            ]
            assert {row["interface"] for row in rows} == {"discord"}

    asyncio.run(run())


def test_exchange_is_not_written_into_another_users_conversation():
    async def run():
        async with _database() as database:
            owner = await _discord_user(database, "111")
            intruder = await _discord_user(database, "222")
            conversation = uuid4()
            assert await store_discord_exchange(
                database, _exchange(conversation, owner)
            )

            assert not await store_discord_exchange(
                database, _exchange(conversation, intruder, question="Туѓо?")
            )
            assert (
                await database.fetchval(
                    "SELECT user_id FROM chat_conversation WHERE id = $1",
                    conversation,
                )
                == owner
            )
            assert (
                await database.fetchval(
                    "SELECT count(*) FROM chat_message WHERE conversation_id = $1",
                    conversation,
                )
                == 2
            )

    asyncio.run(run())


def test_repeated_response_is_stored_once():
    async def run():
        async with _database() as database:
            user = await _discord_user(database, "111")
            exchange = _exchange(uuid4(), user)
            assert await store_discord_exchange(database, exchange)
            assert not await store_discord_exchange(database, exchange)
            assert await database.fetchval("SELECT count(*) FROM chat_message") == 2

    asyncio.run(run())


def test_history_deletion_removes_conversations_and_own_discord_feedback():
    async def run():
        async with _database() as database:
            user = await _discord_user(database, "111")
            other = await _discord_user(database, "222")
            mine = _exchange(uuid4(), user)
            theirs = _exchange(uuid4(), other)
            assert await store_discord_exchange(database, mine)
            assert await store_discord_exchange(database, theirs)
            for discord_id, response_id in (
                ("111", mine.response_id),
                ("222", theirs.response_id),
            ):
                await upsert_feedback(
                    database,
                    FeedbackSchema(
                        response_id=response_id,
                        user_id=discord_id,
                        client="discord",
                        feedback_type="like",
                    ),
                )

            deleted = await delete_chat_history(database, user_id=user)

            assert (deleted.conversations, deleted.feedback) == (1, 1)
            assert (
                await database.fetchval("SELECT count(*) FROM chat_conversation") == 1
            )
            assert await database.fetchval("SELECT user_id FROM feedback") == "222"

    asyncio.run(run())


def _exchange(
    conversation_id: UUID,
    user_id: UUID,
    *,
    question: str = "Кој е декан?",
) -> DiscordExchange:
    return DiscordExchange(
        conversation_id=conversation_id,
        user_id=user_id,
        response_id=uuid4(),
        model="gpt-6-luna",
        question=question,
        answer="Одговор",
        metadata={"interface": "discord", "diagnostics": {"outcome": "completed"}},
        history_generation=0,
    )


async def _discord_user(database: Database, discord_id: str) -> UUID:
    user_id = await database.fetchval(
        """
        INSERT INTO chat_user (provider, provider_subject)
        VALUES ('discord', $1) RETURNING id
        """,
        discord_id,
    )
    assert isinstance(user_id, UUID)
    return user_id


@asynccontextmanager
async def _database():
    assert DATABASE_URL is not None
    if urlsplit(DATABASE_URL).hostname not in {"localhost", "127.0.0.1", "::1"}:
        pytest.skip(
            "Discord storage integration only runs against a local disposable schema"
        )
    schema = f"discord_storage_test_{uuid4().hex}"
    admin = await asyncpg.connect(DATABASE_URL, command_timeout=10)
    database = Database(DATABASE_URL)
    try:
        await admin.execute(f'CREATE SCHEMA "{schema}"')
        database.pool = await asyncpg.create_pool(
            DATABASE_URL,
            min_size=1,
            max_size=4,
            command_timeout=10,
            server_settings={
                "search_path": f'"{schema}"',
                "statement_timeout": "10000",
            },
        )
        # Only this feature's actual DDL. Never create/drop shared extensions or
        # touch the public migration ledger while testing in a disposable schema.
        migrations = Path(__file__).parents[2] / "resources" / "migrations"
        for name in (
            "0002_chat_state.sql",
            "0005_add_chat_message_parts.sql",
            "0006_allow_discord_chat_user_provider.sql",
            "0007_add_chat_conversation_share_token.sql",
            "0009_add_active_chat_replacement.sql",
            "0014_chat_history_fence.sql",
        ):
            await database.execute((migrations / name).read_text(encoding="utf-8"))
        initial = (migrations / "0001_initial_schema.sql").read_text(encoding="utf-8")
        # Feedback is the final section of the initial migration; no vector DDL.
        await database.execute(
            initial[initial.index("CREATE TABLE IF NOT EXISTS feedback (") :]
        )
        yield database
    finally:
        await database.disconnect()
        await admin.execute(f'DROP SCHEMA "{schema}" CASCADE')
        await admin.close()


def test_empty_history_advances_epoch_and_new_requests_work():
    async def run():
        async with _database() as db:
            user = await _discord_user(db, "111")
            old = _exchange(uuid4(), user)
            deleted = await delete_chat_history(db, user_id=user)
            assert (deleted.conversations, deleted.feedback) == (0, 0)
            assert not await store_discord_exchange(db, old)
            assert await store_discord_exchange(db, replace(old, history_generation=1))
            await upsert_chat_user(
                db,
                ChatUserUpsert(
                    provider="discord", provider_subject="111", name="Updated"
                ),
            )
            assert (
                await db.fetchval(
                    "SELECT history_generation FROM chat_user WHERE id = $1", user
                )
                == 1
            )

    asyncio.run(run())


def test_single_delete_tombstones_uuid_and_erases_other_raters_copies():
    async def run():
        async with _database() as db:
            user = await _discord_user(db, "111")
            await _discord_user(db, "222")
            exchange = _exchange(uuid4(), user)
            assert await store_discord_exchange(db, exchange)
            assert await upsert_feedback(
                db,
                FeedbackSchema(
                    response_id=exchange.response_id,
                    user_id="222",
                    client="discord",
                    feedback_type="like",
                    question_text="forged",
                    answer_text="forged",
                ),
            )
            assert (
                await db.fetchval("SELECT question_text FROM feedback")
                == exchange.question
            )
            assert await delete_conversation(
                db, conversation_id=exchange.conversation_id, user_id=user
            )
            assert await db.fetchval("SELECT count(*) FROM feedback") == 0
            assert not await store_discord_exchange(
                db, replace(exchange, response_id=uuid4())
            )
            assert await store_discord_exchange(db, _exchange(uuid4(), user))
            assert (
                await upsert_feedback(
                    db,
                    FeedbackSchema(
                        response_id=exchange.response_id,
                        user_id="222",
                        client="discord",
                        feedback_type="like",
                    ),
                )
                is None
            )

    asyncio.run(run())


def test_empty_history_cleans_orphan_authored_feedback_without_namespace_collisions():
    async def run():
        async with _database() as db:
            user = await _discord_user(db, "111")
            for client, subject in (
                ("discord", "111"),
                ("web", str(user)),
                ("web", "111"),
                ("discord", str(user)),
            ):
                await db.execute(
                    "INSERT INTO feedback (response_id, client, user_id, feedback_type, question_text, answer_text) VALUES ($1,$2,$3,'like','q','a')",
                    uuid4(),
                    client,
                    subject,
                )
            deleted = await delete_chat_history(db, user_id=user)
            assert (deleted.conversations, deleted.feedback) == (0, 2)
            assert await db.fetchval("SELECT count(*) FROM feedback") == 2

    asyncio.run(run())


class _InterceptDatabase:
    """Gate exact SQL boundaries on real, independent pool connections."""

    def __init__(self, db, intercept):
        self.db = db
        self.intercept = intercept

    async def fetch(self, query, *args):
        return await self.db.fetch(query, *args)

    async def fetchrow(self, query, *args):
        return await self.db.fetchrow(query, *args)

    @asynccontextmanager
    async def transaction(self):
        async with self.db.transaction() as connection:
            intercept = self.intercept

            class Connection:
                def __getattr__(self, method):
                    async def call(query, *args):
                        await intercept("before", method, query)
                        result = await getattr(connection, method)(query, *args)
                        await intercept("after", method, query)
                        return result

                    return call

            yield Connection()


def test_history_deletion_rolls_back_epoch_feedback_and_messages_together():
    async def run():
        async with _database() as db:
            user = await _discord_user(db, "111")
            exchange = _exchange(uuid4(), user)
            assert await store_discord_exchange(db, exchange)
            assert await upsert_feedback(
                db,
                FeedbackSchema(
                    response_id=exchange.response_id,
                    user_id="111",
                    client="discord",
                    feedback_type="like",
                ),
            )

            async def fail(when, method, query):
                if (
                    when == "before"
                    and method == "fetch"
                    and "DELETE FROM chat_conversation" in query
                ):
                    raise RuntimeError("synthetic rollback")

            with pytest.raises(RuntimeError, match="synthetic rollback"):
                await delete_chat_history(
                    cast("Database", _InterceptDatabase(db, fail)), user_id=user
                )
            assert (
                await db.fetchval(
                    "SELECT history_generation FROM chat_user WHERE id=$1", user
                )
                == 0
            )
            assert await db.fetchval("SELECT count(*) FROM chat_message") == 2
            assert await db.fetchval("SELECT count(*) FROM feedback") == 1
            assert (
                await db.fetchval("SELECT count(*) FROM chat_conversation_tombstone")
                == 0
            )

    asyncio.run(run())


@pytest.mark.parametrize("first", ["store", "delete"])
@pytest.mark.parametrize("single", [False, True])
def test_account_lock_serializes_old_exchange_and_history_deletion_in_both_orders(
    first,
    single,
):
    async def run():
        async with _database() as db:
            user = await _discord_user(db, "111")
            exchange = _exchange(uuid4(), user)
            if single:
                assert await store_discord_exchange(db, exchange)
                exchange = replace(exchange, response_id=uuid4())
            locked, release, attempted = (
                asyncio.Event(),
                asyncio.Event(),
                asyncio.Event(),
            )

            async def gate(when, method, query):
                if (
                    when == "after"
                    and "FROM chat_user WHERE id = $1 FOR NO KEY UPDATE" in query
                ):
                    locked.set()
                    await release.wait()

            async def observe(when, method, query):
                if (
                    when == "before"
                    and "FROM chat_user WHERE id = $1 FOR NO KEY UPDATE" in query
                ):
                    attempted.set()

            slow, other = (
                cast("Database", _InterceptDatabase(db, gate)),
                cast("Database", _InterceptDatabase(db, observe)),
            )

            async def delete(target):
                if single:
                    return await delete_conversation(
                        target, user_id=user, conversation_id=exchange.conversation_id
                    )
                return await delete_chat_history(target, user_id=user)

            async with asyncio.timeout(5), asyncio.TaskGroup() as tasks:
                if first == "store":
                    leading = tasks.create_task(store_discord_exchange(slow, exchange))
                    await locked.wait()
                    trailing = tasks.create_task(delete(other))
                else:
                    leading = tasks.create_task(delete(slow))
                    await locked.wait()
                    trailing = tasks.create_task(
                        store_discord_exchange(other, exchange)
                    )
                await attempted.wait()
                assert not trailing.done()
                release.set()
                await asyncio.gather(leading, trailing)
            assert await db.fetchval("SELECT count(*) FROM chat_message") == 0
            assert await db.fetchval(
                "SELECT history_generation FROM chat_user WHERE id=$1", user
            ) == (0 if single else 1)
            assert not await store_discord_exchange(db, exchange)

    asyncio.run(run())


@pytest.mark.parametrize("delete_rater", [False, True])
def test_feedback_snapshot_cannot_cross_author_or_rater_deletion(delete_rater):
    async def run():
        async with _database() as db:
            owner = await _discord_user(db, "111")
            rater = await _discord_user(db, "222")
            exchange = _exchange(uuid4(), owner)
            assert await store_discord_exchange(db, exchange)
            captured, release = asyncio.Event(), asyncio.Event()

            class PausedFeedbackDB(_InterceptDatabase):
                async def fetch(self, query, *args):
                    rows = await self.db.fetch(query, *args)
                    captured.set()
                    await release.wait()
                    return rows

            async def noop(*args):
                pass

            paused = PausedFeedbackDB(db, noop)
            feedback = FeedbackSchema(
                response_id=exchange.response_id,
                user_id="222",
                client="discord",
                feedback_type="like",
            )
            async with asyncio.timeout(5), asyncio.TaskGroup() as tasks:
                task = tasks.create_task(
                    upsert_feedback(cast("Database", paused), feedback)
                )
                await captured.wait()
                await delete_chat_history(db, user_id=rater if delete_rater else owner)
                release.set()
                assert await task is None
            assert await db.fetchval("SELECT count(*) FROM feedback") == 0

    asyncio.run(run())


@pytest.mark.parametrize("delete_rater", [False, True])
def test_feedback_committed_while_history_delete_waits_is_removed(delete_rater):
    async def run():
        async with _database() as db:
            owner = await _discord_user(db, "111")
            rater = await _discord_user(db, "222")
            exchange = _exchange(uuid4(), owner)
            assert await store_discord_exchange(db, exchange)
            written, release, attempted = (
                asyncio.Event(),
                asyncio.Event(),
                asyncio.Event(),
            )

            async def gate(when, method, query):
                if when == "after" and "INSERT INTO feedback" in query:
                    written.set()
                    await release.wait()

            async def observe(when, method, query):
                if (
                    when == "before"
                    and "FROM chat_user WHERE id = $1 FOR NO KEY UPDATE" in query
                ):
                    attempted.set()

            payload = FeedbackSchema(
                response_id=exchange.response_id,
                user_id="222",
                client="discord",
                feedback_type="like",
            )
            async with asyncio.timeout(5), asyncio.TaskGroup() as tasks:
                write = tasks.create_task(
                    upsert_feedback(
                        cast("Database", _InterceptDatabase(db, gate)), payload
                    )
                )
                await written.wait()
                delete = tasks.create_task(
                    delete_chat_history(
                        cast("Database", _InterceptDatabase(db, observe)),
                        user_id=rater if delete_rater else owner,
                    )
                )
                await attempted.wait()
                assert not delete.done()
                release.set()
                assert await write is not None
                assert (await delete).feedback == 1
            assert await db.fetchval("SELECT count(*) FROM feedback") == 0

    asyncio.run(run())


def test_two_history_deletions_with_mutual_ratings_use_ordered_copy_locks():
    async def run():
        async with _database() as db:
            a, b = await _discord_user(db, "111"), await _discord_user(db, "222")
            first, second = _exchange(uuid4(), a), _exchange(uuid4(), b)
            assert await store_discord_exchange(db, first)
            assert await store_discord_exchange(db, second)
            for response, subject in (
                (first.response_id, "222"),
                (second.response_id, "111"),
            ):
                assert await upsert_feedback(
                    db,
                    FeedbackSchema(
                        response_id=response,
                        client="discord",
                        user_id=subject,
                        feedback_type="like",
                    ),
                )
            locked: asyncio.Queue[bool] = asyncio.Queue()
            release = asyncio.Event()

            async def gate(when, method, query):
                if (
                    when == "after"
                    and "FROM chat_user WHERE id = $1 FOR NO KEY UPDATE" in query
                ):
                    locked.put_nowait(True)
                    await release.wait()

            held = cast("Database", _InterceptDatabase(db, gate))
            async with asyncio.timeout(5), asyncio.TaskGroup() as tasks:
                one = tasks.create_task(delete_chat_history(held, user_id=a))
                two = tasks.create_task(delete_chat_history(held, user_id=b))
                await locked.get()
                await locked.get()
                release.set()
                results = await asyncio.gather(one, two)
            assert sum(result.feedback for result in results) == 2
            assert await db.fetchval("SELECT count(*) FROM feedback") == 0
            assert await db.fetchval("SELECT count(*) FROM chat_message") == 0

    asyncio.run(run())


@pytest.mark.parametrize("bulk", [False, True])
@pytest.mark.parametrize(
    "interfaces", [("web", "web"), ("web", "discord"), ("discord", "web")]
)
def test_colliding_response_cleanup_preserves_other_owner_and_client_namespace(
    bulk, interfaces
):
    async def run():
        async with _database() as db:
            owner, other = (
                await _discord_user(db, "111"),
                await _discord_user(db, "222"),
            )
            first = _exchange(uuid4(), owner)
            second = replace(_exchange(uuid4(), other), response_id=first.response_id)
            assert await store_discord_exchange(db, first)
            assert await store_discord_exchange(db, second)
            for exchange, interface in (
                (first, interfaces[0]),
                (second, interfaces[1]),
            ):
                await db.execute(
                    "UPDATE chat_message SET metadata = jsonb_set(metadata, '{interface}', to_jsonb($2::text)) WHERE conversation_id = $1 AND role = 'assistant'",
                    exchange.conversation_id,
                    interface,
                )
            mine = (first.response_id, "web", str(owner))
            theirs = (first.response_id, "web", str(other))
            discord_copy = (first.response_id, "discord", "333")
            own_orphan = (uuid4(), "web", str(owner))
            for response_id, client, subject in (
                mine,
                theirs,
                discord_copy,
                own_orphan,
            ):
                await db.execute(
                    "INSERT INTO feedback (response_id, client, user_id, feedback_type) VALUES ($1,$2,$3,'like')",
                    response_id,
                    client,
                    subject,
                )
            if bulk:
                assert (await delete_chat_history(db, user_id=owner)).feedback == (
                    2 + (interfaces[0] == "discord")
                )
            else:
                assert await delete_conversation(
                    db, conversation_id=first.conversation_id, user_id=owner
                )
            remaining = {
                tuple(row.values())
                for row in await db.fetch(
                    "SELECT response_id, client, user_id FROM feedback"
                )
            }
            expected = {theirs}
            if interfaces[0] == "web":
                expected.add(discord_copy)
            if not bulk:
                expected.add(own_orphan)
            assert remaining == expected
            assert (
                await db.fetchval(
                    "SELECT user_id FROM chat_conversation WHERE id=$1",
                    second.conversation_id,
                )
                == other
            )

    asyncio.run(run())


def test_pending_generation_token_cannot_cross_real_rater_deletion():
    async def run():
        async with _database() as db:
            owner, rater = (
                await _discord_user(db, "111"),
                await _discord_user(db, "222"),
            )
            exchange = _exchange(uuid4(), owner)
            feedback = FeedbackSchema(
                response_id=exchange.response_id,
                user_id="222",
                client="discord",
                feedback_type="like",
            )
            with pytest.raises(DiscordFeedbackUnavailableError) as pending:
                await upsert_discord_feedback(db, feedback)
            token = pending.value.generation
            assert token == 0
            await delete_chat_history(db, user_id=rater)
            assert await store_discord_exchange(db, exchange)
            with pytest.raises(DiscordFeedbackStaleError):
                await upsert_discord_feedback(db, feedback, generation=token)
            assert await db.fetchval("SELECT count(*) FROM feedback") == 0

    asyncio.run(run())


def test_shutdown_cancels_pending_sql_write_and_awaits_real_transaction_rollback():
    async def run():
        async with _database() as db:
            owner = await _discord_user(db, "111")
            exchange = _exchange(uuid4(), owner)
            written = asyncio.Event()

            async def pause_after_insert(when, method, query):
                if when == "after" and method == "executemany":
                    written.set()
                    await asyncio.Event().wait()

            held = cast("Database", _InterceptDatabase(db, pause_after_insert))
            registry = DiscordWriteRegistry()
            storage = DiscordStorage(
                held, exchange.conversation_id, owner, [], 0, registry
            )
            try:
                async with asyncio.timeout(5):
                    registry.schedule(storage, exchange)
                    await written.wait()
                    pending = tuple(registry.pending)
                    await registry.drain(grace_seconds=0.01)
                    assert all(task.done() and task.cancelled() for task in pending)
                    assert not registry.pending
                    assert (
                        await db.fetchval("SELECT count(*) FROM chat_conversation") == 0
                    )
                    assert await db.fetchval("SELECT count(*) FROM chat_message") == 0
                    # The pool still serves SQL after rollback, before fixture close.
                    registry.schedule(storage, exchange)
                    assert not registry.pending
                    assert await db.fetchval("SELECT 1") == 1
            finally:
                async with asyncio.timeout(5):
                    await registry.drain(grace_seconds=0)

    asyncio.run(run())
