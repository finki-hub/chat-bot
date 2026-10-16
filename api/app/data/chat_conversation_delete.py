from uuid import UUID

from asyncpg import Record
from asyncpg.pool import PoolConnectionProxy

from app.data.chat_rows import conversation_from_row
from app.data.connection import Database
from app.schemas.chat_persistence import ChatConversation


async def delete_conversation(
    db: Database,
    *,
    conversation_id: UUID,
    user_id: UUID,
) -> ChatConversation | None:
    rows = await _delete_conversations(
        db,
        user_id=user_id,
        conversation_id=conversation_id,
    )
    return rows[0] if rows else None


async def _delete_conversations(
    db: Database,
    *,
    user_id: UUID,
    conversation_id: UUID | None = None,
) -> list[ChatConversation]:
    async with db.transaction() as connection:
        await connection.execute("SET TRANSACTION ISOLATION LEVEL READ COMMITTED")
        rows, _ = await delete_history_on_connection(
            connection,
            user_id=user_id,
            conversation_id=conversation_id,
            erase_authored_feedback=conversation_id is None,
        )
        return rows


async def delete_history_on_connection(
    connection: PoolConnectionProxy[Record],
    *,
    user_id: UUID,
    conversation_id: UUID | None = None,
    erase_authored_feedback: bool = False,
) -> tuple[list[ChatConversation], int]:
    """One transaction, ordered account -> parent -> message locks, fresh DELETE snapshots."""
    # NO KEY UPDATE avoids conflicting with the FK key-share lock taken by web inserts.
    account = await connection.fetchrow(
        "SELECT * FROM chat_user WHERE id = $1 FOR NO KEY UPDATE",
        user_id,
    )
    if account is None:
        return [], 0
    if conversation_id is None:
        await connection.execute(
            "UPDATE chat_user SET history_generation = history_generation + 1 WHERE id = $1",
            user_id,
        )
    rows = await connection.fetch(
        """
        SELECT id FROM chat_conversation
        WHERE user_id = $1 AND ($2::uuid IS NULL OR id = $2)
        ORDER BY id
        FOR UPDATE
        """,
        user_id,
        conversation_id,
    )
    conversation_ids = [row["id"] for row in rows]
    # These statements must stay separate: a DELETE snapshot taken before waiting
    # for message locks can miss feedback committed by a writer we just waited for.
    await connection.fetch(
        """
        SELECT id FROM chat_message
        WHERE conversation_id = ANY($1::uuid[])
        ORDER BY id
        FOR UPDATE
        """,
        conversation_ids,
    )
    # Two users deleting history can share feedback rows (each rated the other's
    # response). Lock those copies in a common order before the fresh DELETE.
    await connection.fetch(
        """
        SELECT stored.id FROM feedback AS stored
        WHERE (stored.client = 'web' AND stored.user_id = $2 AND stored.response_id IN (
            SELECT response_id FROM chat_message
            WHERE conversation_id = ANY($1::uuid[]) AND response_id IS NOT NULL
        )) OR (stored.client = 'discord' AND stored.response_id IN (
            SELECT response_id FROM chat_message
            WHERE conversation_id = ANY($1::uuid[]) AND role = 'assistant'
              AND metadata->>'interface' = 'discord' AND response_id IS NOT NULL
        )) OR ($3::boolean AND (
            (stored.client = 'web' AND stored.user_id = $2)
            OR (stored.client = 'discord' AND $4::text = 'discord' AND stored.user_id = $5)
        ))
        ORDER BY stored.id FOR UPDATE
        """,
        conversation_ids,
        str(user_id),
        erase_authored_feedback,
        account["provider"],
        account["provider_subject"],
    )
    status = await connection.execute(
        """
        DELETE FROM feedback AS stored
        WHERE (stored.client = 'web' AND stored.user_id = $2 AND stored.response_id IN (
            SELECT response_id FROM chat_message
            WHERE conversation_id = ANY($1::uuid[]) AND response_id IS NOT NULL
        )) OR (stored.client = 'discord' AND stored.response_id IN (
            SELECT response_id FROM chat_message
            WHERE conversation_id = ANY($1::uuid[]) AND role = 'assistant'
              AND metadata->>'interface' = 'discord' AND response_id IS NOT NULL
        )) OR ($3::boolean AND (
            (stored.client = 'web' AND stored.user_id = $2)
            OR (stored.client = 'discord' AND $4::text = 'discord' AND stored.user_id = $5)
        ))
        """,
        conversation_ids,
        str(user_id),
        erase_authored_feedback,
        account["provider"],
        account["provider_subject"],
    )
    await connection.execute(
        """
        INSERT INTO chat_conversation_tombstone (id, user_id)
        SELECT id, user_id FROM chat_conversation
        WHERE id = ANY($1::uuid[]) AND user_id = $2
        ON CONFLICT (id) DO NOTHING
        """,
        conversation_ids,
        user_id,
    )
    deleted = await connection.fetch(
        """
        DELETE FROM chat_conversation
        WHERE id = ANY($1::uuid[]) AND user_id = $2
        RETURNING *
        """,
        conversation_ids,
        user_id,
    )
    return [conversation_from_row(dict(row)) for row in deleted], int(
        status.split()[-1]
    )


async def delete_conversations(
    db: Database,
    *,
    user_id: UUID,
) -> list[ChatConversation]:
    return await _delete_conversations(db, user_id=user_id)
