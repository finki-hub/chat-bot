import asyncio
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import cast
from uuid import UUID, uuid4

import asyncpg
import pytest

from app.data.chat_conversation_delete import delete_conversation, delete_conversations
from app.data.chat_persistence import ChatMessageConflictError, upsert_message
from app.data.chat_state import replace_assistant_message_and_prune_after
from app.data.connection import Database
from app.data.feedback import retract_web_feedback, upsert_feedback, upsert_web_feedback
from app.schemas.chat_persistence import ChatMessageRole, ChatMessageUpsert
from app.schemas.feedback import FeedbackSchema

DATABASE_URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    DATABASE_URL is None,
    reason="set TEST_DATABASE_URL to run real-PostgreSQL deletion tests",
)


@pytest.mark.parametrize("mutation", ["role", "response"])
def test_ordinary_upsert_rejects_detachment_and_feedback_remains_deletable(mutation):
    async def run():
        async with _database() as database:
            case = await _replacement_case(database)
            assert await upsert_web_feedback(database, case.old) is not None
            original = await database.fetchrow(
                "SELECT * FROM chat_message WHERE id = $1", case.message.id
            )
            feedback = await database.fetch("SELECT * FROM feedback ORDER BY id")
            attempted = case.message.model_copy(
                update={
                    "role": ChatMessageRole.USER
                    if mutation == "role"
                    else ChatMessageRole.ASSISTANT,
                    "response_id": case.old.response_id
                    if mutation == "role"
                    else uuid4(),
                }
            )
            with pytest.raises(ChatMessageConflictError):
                await upsert_message(database, attempted)
            assert (
                await database.fetchrow(
                    "SELECT * FROM chat_message WHERE id = $1", case.message.id
                )
                == original
            )
            assert (
                await database.fetch("SELECT * FROM feedback ORDER BY id") == feedback
            )
            assert await delete_conversation(
                database, user_id=case.owner, conversation_id=case.conversation
            )
            assert await database.fetchval("SELECT count(*) FROM feedback") == 0

    asyncio.run(run())


def test_ordinary_upsert_accepts_null_safe_user_edit_in_postgres():
    async def run():
        async with _database() as database:
            _, conversation, _ = await _seed(database)
            user_message_id = await database.fetchval(
                "SELECT id FROM chat_message WHERE role = 'user'"
            )
            edited = await upsert_message(
                database,
                ChatMessageUpsert(
                    id=user_message_id,
                    conversation_id=conversation,
                    role=ChatMessageRole.USER,
                    content="synthetic edited question",
                ),
            )
            assert edited.id == user_message_id
            assert edited.response_id is None
            assert edited.content == "synthetic edited question"

    asyncio.run(run())


@pytest.mark.parametrize("mutation", ["role", "response"])
def test_ordinary_upsert_waiting_on_feedback_cannot_detach_response(mutation):
    async def run():
        async with _database() as database:
            case = await _replacement_case(database)
            attempted = case.message.model_copy(
                update={
                    "role": ChatMessageRole.USER
                    if mutation == "role"
                    else ChatMessageRole.ASSISTANT,
                    "response_id": case.old.response_id
                    if mutation == "role"
                    else uuid4(),
                }
            )

            async def attempt_detachment():
                with pytest.raises(ChatMessageConflictError):
                    await upsert_message(database, attempted)

            assert database.pool is not None
            async with asyncio.TaskGroup() as tasks, database.pool.acquire() as writer:
                async with writer.transaction():
                    assert (
                        await upsert_web_feedback(cast(Database, writer), case.old)
                        is not None
                    )
                    original = await writer.fetchrow(
                        "SELECT * FROM chat_message WHERE id = $1", case.message.id
                    )
                    attempt = tasks.create_task(attempt_detachment())
                    await _wait_for_blocked_transaction(
                        database, writer.get_server_pid()
                    )
                await asyncio.wait_for(attempt, timeout=5)
            assert (
                await database.fetchrow(
                    "SELECT * FROM chat_message WHERE id = $1", case.message.id
                )
                == original
            )
            assert await database.fetchval("SELECT count(*) FROM feedback") == 1
            assert await delete_conversation(
                database, user_id=case.owner, conversation_id=case.conversation
            )
            assert await database.fetchval("SELECT count(*) FROM feedback") == 0

    asyncio.run(run())


@asynccontextmanager
async def _database():
    assert DATABASE_URL is not None
    schema = f"deletion_test_{uuid4().hex}"
    admin = await asyncpg.connect(DATABASE_URL, command_timeout=10)
    database = Database(DATABASE_URL)
    try:
        await admin.execute(f'CREATE SCHEMA "{schema}"')
        database.pool = await asyncpg.create_pool(
            DATABASE_URL,
            min_size=1,
            max_size=6,
            command_timeout=10,
            # Never fall back to another suite's public tables or migration ledger.
            server_settings={
                "search_path": f'"{schema}"',
                "statement_timeout": "10000",
            },
        )
        await database.run_migrations()
        yield database
    finally:
        await database.disconnect()
        await admin.execute(f'DROP SCHEMA "{schema}" CASCADE')
        await admin.close()


async def _seed(database, *, user_id=None):
    user_id = user_id or uuid4()
    conversation_id, response_id = uuid4(), uuid4()
    await database.execute(
        """
        INSERT INTO chat_user (id, provider, provider_subject)
        VALUES ($1::uuid, 'google', $1::uuid::text) ON CONFLICT DO NOTHING
        """,
        user_id,
    )
    await database.execute(
        "INSERT INTO chat_conversation (id, user_id) VALUES ($1, $2)",
        conversation_id,
        user_id,
    )
    await database.execute(
        """
        INSERT INTO chat_message (id, conversation_id, role, content, response_id)
        VALUES ($1, $2, 'user', 'synthetic question', NULL),
               ($3, $2, 'assistant', 'synthetic answer', $4)
        """,
        uuid4(),
        conversation_id,
        uuid4(),
        response_id,
    )
    feedback = FeedbackSchema(
        response_id=response_id,
        user_id=str(user_id),
        client="web",
        feedback_type="like",
    )
    return user_id, conversation_id, feedback


async def _delete(database, user_id, conversation_id, *, bulk):
    if bulk:
        return await delete_conversations(database, user_id=user_id)
    return await delete_conversation(
        database,
        conversation_id=conversation_id,
        user_id=user_id,
    )


@pytest.mark.parametrize("bulk", [False, True])
def test_delete_removes_only_owned_linked_web_feedback(bulk):
    async def run():
        async with _database() as database:
            owner, conversation, feedback = await _seed(database)
            _, other_conversation, other_feedback = await _seed(database, user_id=owner)
            _, _, foreign_feedback = await _seed(database)
            for payload in (feedback, other_feedback, foreign_feedback):
                stored = await upsert_web_feedback(database, payload)
                assert stored is not None
                assert stored.feedback.question_text == "synthetic question"
                assert stored.feedback.answer_text == "synthetic answer"
            # Legacy/external Discord responses have no chat-message FK, and
            # even a coincident response id must not cross the client boundary.
            discord = feedback.model_copy(update={"client": "discord"})
            external = discord.model_copy(update={"response_id": uuid4()})
            foreign = feedback.model_copy(update={"user_id": str(uuid4())})
            for payload in (discord, external, foreign):
                assert await upsert_feedback(database, payload) is not None
            assert await _delete(database, owner, conversation, bulk=bulk)
            expected = {
                (p.response_id, p.client, p.user_id)
                for p in (foreign_feedback, discord, external, foreign)
            }
            if not bulk:
                expected.add((other_feedback.response_id, "web", str(owner)))
            rows = await database.fetch(
                "SELECT response_id, client, user_id FROM feedback"
            )
            assert {tuple(row.values()) for row in rows} == expected
            assert (
                await database.fetchval(
                    "SELECT count(*) FROM chat_message WHERE conversation_id = $1",
                    conversation,
                )
                == 0
            )
            assert await database.fetchval(
                "SELECT count(*) FROM chat_conversation WHERE id = $1",
                other_conversation,
            ) == (0 if bulk else 1)
            assert await upsert_web_feedback(database, feedback) is None
            assert not await retract_web_feedback(
                database,
                response_id=feedback.response_id,
                user_id=str(owner),
            )

    asyncio.run(run())


@dataclass
class _ReplacementCase:
    owner: UUID
    conversation: UUID
    old: FeedbackSchema
    earlier: FeedbackSchema
    pruned: FeedbackSchema
    message: ChatMessageUpsert
    retained: list[UUID]

    async def replace(self, database):
        return await replace_assistant_message_and_prune_after(
            database,
            self.message,
            user_id=self.owner,
            active_stream_id=self.message.response_id,
            retained_message_ids=self.retained,
        )


async def _replacement_case(database):
    owner, conversation, old = await _seed(database)
    target_id = await database.fetchval(
        "SELECT id FROM chat_message WHERE response_id = $1",
        old.response_id,
    )
    earlier_id, pruned_id, new_response = uuid4(), uuid4(), uuid4()
    earlier = old.model_copy(update={"response_id": uuid4()})
    pruned = old.model_copy(update={"response_id": uuid4()})
    await database.execute(
        """
        INSERT INTO chat_message (id, conversation_id, role, content, response_id, created_at)
        VALUES ($1, $2, 'assistant', 'synthetic earlier', $3, NOW() - INTERVAL '1 minute'),
               ($4, $2, 'assistant', 'synthetic later', $5, NOW() + INTERVAL '1 minute')
        """,
        earlier_id,
        conversation,
        earlier.response_id,
        pruned_id,
        pruned.response_id,
    )
    await database.execute(
        "UPDATE chat_conversation SET active_stream_id = $2 WHERE id = $1",
        conversation,
        new_response,
    )
    retained = [
        row["id"]
        for row in await database.fetch(
            "SELECT id FROM chat_message WHERE conversation_id = $1 AND id != $2",
            conversation,
            pruned_id,
        )
    ]
    return _ReplacementCase(
        owner,
        conversation,
        old,
        earlier,
        pruned,
        ChatMessageUpsert(
            id=target_id,
            conversation_id=conversation,
            role=ChatMessageRole.ASSISTANT,
            content="synthetic replacement",
            response_id=new_response,
        ),
        retained,
    )


def test_replacement_prunes_feedback_preserves_boundaries_and_same_response_replay():
    async def run():
        async with _database() as database:
            case = await _replacement_case(database)
            for payload in (case.old, case.earlier, case.pruned):
                assert await upsert_web_feedback(database, payload) is not None
            preserved = [
                payload.model_copy(update={"client": "discord"})
                for payload in (case.old, case.pruned)
            ] + [
                case.old.model_copy(update={"user_id": str(uuid4())}),
                case.pruned.model_copy(update={"user_id": str(uuid4())}),
                case.old.model_copy(
                    update={"client": "discord", "response_id": uuid4()}
                ),
            ]
            for payload in preserved:
                assert await upsert_feedback(database, payload) is not None
            updated = await case.replace(database)
            assert updated is not None
            assert updated.response_id == case.message.response_id
            assert {
                row["id"] for row in await database.fetch("SELECT id FROM chat_message")
            } == set(case.retained)
            expected = {
                (p.response_id, p.client, p.user_id) for p in (*preserved, case.earlier)
            }
            assert {
                tuple(row.values())
                for row in await database.fetch(
                    "SELECT response_id, client, user_id FROM feedback",
                )
            } == expected
            assert await upsert_web_feedback(database, case.old) is None
            assert await upsert_web_feedback(database, case.pruned) is None
            new_feedback = case.old.model_copy(
                update={"response_id": case.message.response_id}
            )
            assert await upsert_web_feedback(database, new_feedback) is not None
            replay = await case.replace(database)
            assert replay is not None
            assert replay.metadata["feedback"] == "like"
            assert (
                await database.fetchval(
                    "SELECT count(*) FROM feedback WHERE response_id = $1",
                    case.message.response_id,
                )
                == 1
            )
            assert await delete_conversation(
                database, user_id=case.owner, conversation_id=case.conversation
            )
            assert {
                tuple(row.values())
                for row in await database.fetch(
                    "SELECT response_id, client, user_id FROM feedback",
                )
            } == {(p.response_id, p.client, p.user_id) for p in preserved}

    asyncio.run(run())


@pytest.mark.parametrize("operation", ["UPDATE", "DELETE"])
def test_replacement_failure_restores_feedback_and_message_state(operation):
    async def run():
        async with _database() as database:
            case = await _replacement_case(database)
            for payload in (case.old, case.earlier, case.pruned):
                assert await upsert_web_feedback(database, payload) is not None
            messages = await database.fetch("SELECT * FROM chat_message ORDER BY id")
            feedback = await database.fetch("SELECT * FROM feedback ORDER BY id")
            await database.execute(
                f"""
                CREATE FUNCTION reject_mutation() RETURNS trigger LANGUAGE plpgsql AS $$
                BEGIN RAISE EXCEPTION 'synthetic replacement failure'; END $$;
                CREATE TRIGGER reject_mutation BEFORE {operation} ON chat_message
                FOR EACH ROW EXECUTE FUNCTION reject_mutation();
                """,
            )
            with pytest.raises(
                asyncpg.RaiseError, match="synthetic replacement failure"
            ):
                await case.replace(database)
            assert (
                await database.fetch("SELECT * FROM chat_message ORDER BY id")
                == messages
            )
            assert (
                await database.fetch("SELECT * FROM feedback ORDER BY id") == feedback
            )

    asyncio.run(run())


@pytest.mark.parametrize("rated", ["old", "pruned"])
@pytest.mark.parametrize("existing", [False, True])
def test_replacement_waits_for_feedback_then_deletes_committed_copy(rated, existing):
    async def run():
        async with _database() as database:
            case = await _replacement_case(database)
            payload = getattr(case, rated)
            if existing:
                assert await upsert_web_feedback(database, payload) is not None
            assert database.pool is not None
            async with asyncio.TaskGroup() as tasks, database.pool.acquire() as writer:
                async with writer.transaction():
                    assert (
                        await upsert_web_feedback(
                            cast(Database, writer),
                            payload.model_copy(update={"feedback_type": "dislike"}),
                        )
                        is not None
                    )
                    replacement = tasks.create_task(case.replace(database))
                    await _wait_for_blocked_transaction(
                        database, writer.get_server_pid()
                    )
                assert await asyncio.wait_for(replacement, timeout=5) is not None
            assert await database.fetchval("SELECT count(*) FROM feedback") == 0

    asyncio.run(run())


@pytest.mark.parametrize("rated", ["old", "pruned"])
@pytest.mark.parametrize("retract", [False, True])
def test_feedback_waiting_for_replacement_cannot_recreate_old_copy(rated, retract):
    async def run():
        async with _database() as database:
            case = await _replacement_case(database)
            payload = getattr(case, rated)
            assert await upsert_web_feedback(database, payload) is not None
            held = _HeldCommitDatabase(database)
            async with asyncio.TaskGroup() as tasks:
                replacement = tasks.create_task(case.replace(held))
                await asyncio.wait_for(held.deleted.wait(), timeout=5)
                write = tasks.create_task(
                    retract_web_feedback(
                        database,
                        response_id=payload.response_id,
                        user_id=str(case.owner),
                    )
                    if retract
                    else upsert_web_feedback(database, payload),
                )
                try:
                    await _wait_for_blocked_transaction(database, held.pid)
                finally:
                    held.commit.set()
                assert await asyncio.wait_for(replacement, timeout=5) is not None
                assert not await asyncio.wait_for(write, timeout=5)
            assert await database.fetchval("SELECT count(*) FROM feedback") == 0

    asyncio.run(run())


@pytest.mark.parametrize("invalid", ["owner", "stream", "missing", "role", "retained"])
def test_invalid_replacement_leaves_all_copies_and_messages_untouched(invalid):
    async def run():
        async with _database() as database:
            case = await _replacement_case(database)
            assert await upsert_web_feedback(database, case.old) is not None
            messages = await database.fetch("SELECT * FROM chat_message ORDER BY id")
            feedback = await database.fetch("SELECT * FROM feedback ORDER BY id")
            target_id = case.message.id
            if invalid == "missing":
                target_id = uuid4()
            elif invalid == "role":
                target_id = await database.fetchval(
                    "SELECT id FROM chat_message WHERE role = 'user'"
                )
            retained = (
                [key for key in case.retained if key != target_id]
                if invalid == "retained"
                else [*case.retained, target_id]
            )
            assert (
                await replace_assistant_message_and_prune_after(
                    database,
                    case.message.model_copy(update={"id": target_id}),
                    user_id=uuid4() if invalid == "owner" else case.owner,
                    active_stream_id=uuid4()
                    if invalid == "stream"
                    else case.message.response_id,
                    retained_message_ids=retained,
                )
                is None
            )
            assert (
                await database.fetch("SELECT * FROM chat_message ORDER BY id")
                == messages
            )
            assert (
                await database.fetch("SELECT * FROM feedback ORDER BY id") == feedback
            )

    asyncio.run(run())


def test_missing_and_wrong_owner_leave_feedback_untouched_and_retraction_works():
    async def run():
        async with _database() as database:
            owner, conversation, feedback = await _seed(database)
            assert await upsert_web_feedback(database, feedback) is not None
            for conversation_id, user_id in ((uuid4(), owner), (conversation, uuid4())):
                assert (
                    await delete_conversation(
                        database,
                        conversation_id=conversation_id,
                        user_id=user_id,
                    )
                    is None
                )
            assert await delete_conversations(database, user_id=uuid4()) == []
            assert await database.fetchval("SELECT count(*) FROM feedback") == 1
            assert not await retract_web_feedback(
                database,
                response_id=feedback.response_id,
                user_id=str(uuid4()),
            )
            assert await retract_web_feedback(
                database,
                response_id=feedback.response_id,
                user_id=str(owner),
            )
            assert await database.fetchval("SELECT count(*) FROM feedback") == 0
            assert (
                await database.fetchval(
                    "SELECT metadata ? 'feedback' FROM chat_message WHERE response_id = $1",
                    feedback.response_id,
                )
                is False
            )
            assert (
                await delete_conversation(
                    database,
                    conversation_id=conversation,
                    user_id=owner,
                )
                is not None
            )

    asyncio.run(run())


@pytest.mark.parametrize("bulk", [False, True])
def test_conversation_delete_failure_rolls_back_feedback_delete(bulk):
    async def run():
        async with _database() as database:
            owner, conversation, feedback = await _seed(database)
            assert await upsert_web_feedback(database, feedback) is not None
            await database.execute(
                """
                CREATE FUNCTION reject_delete() RETURNS trigger LANGUAGE plpgsql AS $$
                BEGIN RAISE EXCEPTION 'synthetic deletion failure'; END $$;
                CREATE TRIGGER reject_delete BEFORE DELETE ON chat_conversation
                FOR EACH ROW EXECUTE FUNCTION reject_delete();
                """,
            )
            with pytest.raises(asyncpg.RaiseError, match="synthetic deletion failure"):
                await _delete(database, owner, conversation, bulk=bulk)
            assert await database.fetchval("SELECT count(*) FROM feedback") == 1
            assert await database.fetchval("SELECT count(*) FROM chat_message") == 2
            assert (
                await database.fetchval("SELECT count(*) FROM chat_conversation") == 1
            )

    asyncio.run(run())


async def _wait_for_blocked_transaction(database, blocker_pid):
    # Observe a real server-side lock wait rather than guessing via a sleep.
    async with asyncio.timeout(5):
        while not await database.fetchval(  # noqa: ASYNC110 - observe PostgreSQL locks across sessions
            """
            SELECT EXISTS (
                SELECT 1 FROM pg_stat_activity
                WHERE $1 = ANY(pg_blocking_pids(pid))
            )
            """,
            blocker_pid,
        ):
            await asyncio.sleep(0.01)


@pytest.mark.parametrize("bulk", [False, True])
@pytest.mark.parametrize("existing", [False, True])
def test_feedback_committing_while_delete_waits_is_removed(bulk, existing):
    async def run():
        async with _database() as database:
            owner, conversation, feedback = await _seed(database)
            if existing:
                assert await upsert_web_feedback(database, feedback) is not None
            assert database.pool is not None
            async with database.pool.acquire() as writer:
                async with writer.transaction():
                    assert (
                        await upsert_web_feedback(
                            cast(Database, writer),
                            feedback.model_copy(update={"feedback_type": "dislike"}),
                        )
                        is not None
                    )
                    deletion = asyncio.create_task(
                        _delete(database, owner, conversation, bulk=bulk),
                    )
                    await _wait_for_blocked_transaction(
                        database, writer.get_server_pid()
                    )
                assert await asyncio.wait_for(deletion, timeout=5)
            assert await database.fetchval("SELECT count(*) FROM feedback") == 0
            assert await database.fetchval("SELECT count(*) FROM chat_message") == 0

    asyncio.run(run())


class _HeldCommitDatabase(Database):
    def __init__(self, database):
        super().__init__(database.dsn)
        self.pool = database.pool
        self.deleted = asyncio.Event()
        self.commit = asyncio.Event()
        self.pid = None

    @asynccontextmanager
    async def transaction(self):
        async with super().transaction() as connection:
            self.pid = connection.get_server_pid()
            yield connection  # noqa: RUF075 - outer transaction rolls back failures
            self.deleted.set()
            await asyncio.wait_for(self.commit.wait(), timeout=5)


@pytest.mark.parametrize("bulk", [False, True])
@pytest.mark.parametrize("retract", [False, True])
def test_feedback_started_during_delete_cannot_recreate_copies(bulk, retract):
    async def run():
        async with _database() as database:
            owner, conversation, feedback = await _seed(database)
            assert await upsert_web_feedback(database, feedback) is not None
            held = _HeldCommitDatabase(database)
            deletion = asyncio.create_task(
                _delete(held, owner, conversation, bulk=bulk)
            )
            await asyncio.wait_for(held.deleted.wait(), timeout=5)
            write = asyncio.create_task(
                retract_web_feedback(
                    database,
                    response_id=feedback.response_id,
                    user_id=str(owner),
                )
                if retract
                else upsert_web_feedback(database, feedback),
            )
            await _wait_for_blocked_transaction(database, held.pid)
            held.commit.set()
            assert await asyncio.wait_for(deletion, timeout=5)
            assert not await asyncio.wait_for(write, timeout=5)
            assert await database.fetchval("SELECT count(*) FROM feedback") == 0
            assert await database.fetchval("SELECT count(*) FROM chat_message") == 0

    asyncio.run(run())
